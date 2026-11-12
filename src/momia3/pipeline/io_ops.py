"""IO operations: TIFF import and synthetic-data import.

- ``io.import_tiff`` — read a multi-page ``(t, c, y, x)`` TIFF or an ImageJ
  hyperstack into an OME-Zarr experiment store
  (:class:`momia3.storage.zarr_store.ExperimentStore`).
- ``io.import_synth`` — generate the synthetic rod-bacteria timelapse of
  ``tests/synth.py`` and write it into an experiment store, with the ground
  truth (edges / per-cell records / label stack) as side artifacts for
  demos and benchmarks.

- io.import_mask_dir — import a directory of pre-segmented 2-D masks
  (one TIFF/NPY per frame) into one FOV of an existing store.

Both are registry-only operations: they take no pipeline inputs.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from momia3.pipeline.registry import StepContext, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = [
    "ImportTiffParams",
    "ImportSynthParams",
    "ImportLabelsParams",
    "ImportMaskDirParams",
]


# ---------------------------------------------------------------------------
# labels import (pre-segmented masks)
# ---------------------------------------------------------------------------


class ImportLabelsParams(BaseModel):
    """Parameters of the ``io.import_labels`` operation."""

    path: str = Field(description="Label-mask TIFF to import (absolute or cwd-relative)")
    fov_id: int = Field(0, ge=0, description="FOV id whose labels are replaced")


def _read_labels_tiff(path: Path) -> np.ndarray:
    """Read a label-mask TIFF as a ``(t, y, x)`` integer stack.

    Accepts plain 2D masks and ``(t, y, x)`` stacks (singleton non-spatial
    axes are squeezed). Values are cast to uint32; negative values are
    rejected because 0 is reserved for background.
    """
    import tifffile

    data = np.asarray(tifffile.imread(str(path)))
    for ax in range(data.ndim - 1, 1, -1):
        if data.shape[ax] == 1:
            data = np.squeeze(data, axis=ax)
    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim != 3:
        raise ValueError(
            f"label mask must be 2D or (t, y, x), got shape {data.shape}"
        )
    if np.any(data < 0):
        raise ValueError("label mask contains negative values")
    return data.astype(np.uint32)


@register(
    "io.import_labels",
    params_model=ImportLabelsParams,
    inputs={"experiment": "store"},
    outputs={"experiment": "store"},
    shard=None,
    verify=("fov_*/labels",),
    description=(
        "Import a pre-segmented label-mask TIFF into the labels array of an "
        "existing experiment store (in place). Use when segmentation was "
        "done externally (e.g. Cellpose/Omnipose GUI, Ilastik)."
    ),
)
def run_import_labels(
    inputs: dict[str, Any], params: ImportLabelsParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``io.import_labels`` (see registry description)."""
    del ctx  # store-mutation op; no workspace artifacts
    store_path = Path(inputs["experiment"])
    src = Path(params.path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"label mask not found: {src}")
    labels = _read_labels_tiff(src)
    store = ExperimentStore.open(store_path, mode="a")
    for channel in store.channels:
        shape = store.shape(params.fov_id, channel)
        if labels.shape[-2:] != shape[-2:]:
            raise ValueError(
                f"label mask spatial shape {labels.shape[-2:]} does not match "
                f"channel {channel!r} images {shape[-2:]}"
            )
        if labels.shape[0] not in (1, shape[0]):
            raise ValueError(
                f"label mask has {labels.shape[0]} frame(s); channel {channel!r} "
                f"has {shape[0]} (only 1 or a matching count is supported)"
            )
        break
    store.write_labels(params.fov_id, labels)
    logger.info(
        "io.import_labels: %s -> %s fov=%d shape=%s",
        src,
        store_path,
        params.fov_id,
        labels.shape,
    )
    return {"experiment": store_path}


# ---------------------------------------------------------------------------
# mask-directory import (one file per frame)
# ---------------------------------------------------------------------------


class ImportMaskDirParams(BaseModel):
    """Parameters of the io.import_mask_dir operation."""

    path: str = Field(description="Directory containing one mask per frame")
    pattern: str = Field("*.tif*", description="Glob pattern inside the directory")
    fov_id: int = Field(0, ge=0, description="FOV id used when filename_regex is None")
    filename_regex: str | None = Field(
        None,
        description=(
            "Optional regex with named groups fov and frame; when set, files "
            "are grouped into multiple FOVs instead of natural-sorted into one FOV"
        ),
    )


def _natural_key(path: Path) -> tuple:
    """Natural sort key so frame_2 precedes frame_10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _parse_mask_filename(path: Path, regex: str) -> tuple[int, int]:
    """Parse one mask filename into (fov_id, frame) using named groups."""
    match = re.search(regex, path.name)
    if match is None:
        raise ValueError(
            f"mask filename {path.name!r} does not match filename_regex {regex!r}"
        )
    values = match.groupdict()
    if "fov" not in values or "frame" not in values:
        raise ValueError("filename_regex must define named groups 'fov' and 'frame'")
    try:
        return int(values["fov"]), int(values["frame"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"mask {path.name!r} fov/frame groups are not integers"
        ) from exc


def _group_mask_files(files, regex: str) -> dict[int, dict[int, Path]]:
    """Group files as fov -> frame -> path, rejecting duplicates."""
    grouped: dict[int, dict[int, Path]] = {}
    for file in files:
        fov, frame = _parse_mask_filename(file, regex)
        frames = grouped.setdefault(fov, {})
        if frame in frames:
            raise ValueError(
                f"duplicate frame {frame} for fov {fov}: {frames[frame].name} and {file.name}"
            )
        frames[frame] = file
    return grouped


def _write_mask_group(store, fov_id: int, frames: dict[int, Path]) -> int:
    """Validate and write one FOV mask group; return number of frames."""
    n_frames = store.n_frames(fov_id)
    expected_frames = set(range(n_frames))
    if set(frames) != expected_frames:
        raise ValueError(
            f"fov {fov_id}: mask frames {sorted(frames)} do not cover 0..{n_frames - 1}"
        )
    expected_shape = store.shape(fov_id)[1:]
    masks = []
    for frame in range(n_frames):
        file = frames[frame]
        mask = _read_label_file(file)
        if mask.shape != expected_shape:
            raise ValueError(
                f"mask {file.name} shape {mask.shape} does not match image shape {expected_shape}"
            )
        masks.append(mask)
    store.write_labels(fov_id, np.stack(masks, axis=0).astype(np.uint32, copy=False))
    return n_frames


def _read_label_file(path: Path) -> np.ndarray:
    """Read a single 2-D label mask from a TIFF or NPY file."""
    if path.suffix.lower() == ".npy":
        data = np.asarray(np.load(path))
    else:
        import tifffile

        data = np.asarray(tifffile.imread(str(path)))
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(
            f"mask {path.name} must be 2-D (or a singleton leading axis), "
            f"got shape {data.shape}"
        )
    if np.any(data < 0):
        raise ValueError(f"mask {path.name} contains negative values")
    return data.astype(np.uint32)


@register(
    "io.import_mask_dir",
    params_model=ImportMaskDirParams,
    inputs={"experiment": "store"},
    outputs={"experiment": "store"},
    shard=None,
    verify=("fov_*/labels",),
    description=(
        "Import a directory of pre-segmented 2-D masks. Without filename_regex "
        "the files are natural-sorted into one FOV; with named groups fov and "
        "frame they populate multiple FOVs with strict frame coverage."
    ),
)
def run_import_mask_dir(
    inputs: dict[str, Any], params: ImportMaskDirParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of io.import_mask_dir."""
    del ctx
    root = Path(params.path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"mask directory not found: {root}")
    files = sorted(root.glob(params.pattern), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"no masks matching {params.pattern!r} in {root}")
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="a")
    if params.filename_regex:
        grouped = _group_mask_files(files, params.filename_regex)
        for fov_id in sorted(grouped):
            _write_mask_group(store, fov_id, grouped[fov_id])
        logger.info(
            "io.import_mask_dir: %d masks -> %s fovs=%s",
            len(files),
            store_path,
            sorted(grouped),
        )
    else:
        n_frames = store.n_frames(params.fov_id)
        if len(files) != n_frames:
            raise ValueError(
                f"found {len(files)} mask file(s) but fov {params.fov_id} has "
                f"{n_frames} frame(s); one mask per frame is required"
            )
        frames = {i: file for i, file in enumerate(files)}
        _write_mask_group(store, params.fov_id, frames)
        logger.info(
            "io.import_mask_dir: %d masks -> %s fov=%d",
            len(files),
            store_path,
            params.fov_id,
        )
    return {"experiment": store_path}


# ---------------------------------------------------------------------------
# synth import
# ---------------------------------------------------------------------------


def _load_synth_generator() -> Any:
    """Import ``generate_timelapse`` from the repository's ``tests/synth.py``.

    The test suite is not part of the installed package, so the import
    first tries the ambient ``sys.path`` (pytest puts the tests directory
    on it) and then falls back to the source checkout layout
    (``<repo>/tests`` relative to this file).
    """
    try:
        from synth import generate_timelapse

        return generate_timelapse
    except ImportError:
        pass
    repo_tests = Path(__file__).resolve().parents[3] / "tests"
    if (repo_tests / "synth.py").is_file():
        sys.path.insert(0, str(repo_tests))
        try:
            from synth import generate_timelapse

            return generate_timelapse
        finally:
            sys.path.remove(str(repo_tests))
    raise ImportError(
        "could not import tests/synth.py; io.import_synth requires the "
        "momia3 source checkout (the synthetic generator is not shipped in "
        "the installed package)"
    )


class ImportSynthParams(BaseModel):
    """Parameters of the ``io.import_synth`` operation.

    Mirrors :func:`tests.synth.generate_timelapse`; defaults give a small
    demo timelapse (~12 frames, 256x256, a couple of divisions).
    """

    n_frames: int = Field(12, ge=1, description="Number of time frames")
    shape: tuple[int, int] = Field((256, 256), description="Frame shape (y, x) in px")
    n_initial: int = Field(3, ge=1, description="Founder cells at frame 0")
    pixel_microns: float = Field(0.065, gt=0, description="Physical pixel size (um)")
    growth_rate_um_per_frame: float = Field(
        0.15, ge=0, description="Elongation per frame (um)"
    )
    division_length_um: float = Field(
        2.2, gt=0, description="Length at which a cell divides (um)"
    )
    drift_px_per_frame: float = Field(
        0.6, ge=0, description="Whole-field drift magnitude per frame (px)"
    )
    noise_sigma: float = Field(
        0.02, ge=0, description="Gaussian noise sigma in [0, 1] intensity units"
    )
    vsnap: bool = Field(False, description="V-snapping divisions (large daughter kink)")
    seed: int = Field(7, description="RNG seed; output is deterministic per seed")
    channel: str = Field("phase", description="Channel name written to the store")
    time_interval_s: float | None = Field(
        60.0, gt=0, description="Time interval between frames (s); None skips it"
    )
    write_truth: bool = Field(
        True, description="Also emit ground-truth edges/records/labels outputs"
    )


@register(
    "io.import_synth",
    params_model=ImportSynthParams,
    inputs={},
    outputs={"experiment": "store", "truth": "json", "truth_labels": "array"},
    shard=None,
    description=(
        "Generate the synthetic rod-bacteria timelapse of tests/synth.py "
        "(deterministic per seed) and write it into an experiment zarr store; "
        "ground-truth lineage edges/records and the label stack are emitted "
        "as side artifacts."
    ),
)
def run_import_synth(
    inputs: dict[str, Any], params: ImportSynthParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``io.import_synth`` (see registry description)."""
    del inputs  # registry-only op
    generate_timelapse = _load_synth_generator()
    data = generate_timelapse(
        n_frames=params.n_frames,
        shape=tuple(params.shape),
        n_initial=params.n_initial,
        pixel_microns=params.pixel_microns,
        growth_rate_um_per_frame=params.growth_rate_um_per_frame,
        division_length_um=params.division_length_um,
        drift_px_per_frame=params.drift_px_per_frame,
        noise_sigma=params.noise_sigma,
        vsnap=params.vsnap,
        seed=params.seed,
    )
    store_path = ctx.output_path("experiment")
    store = ExperimentStore.create(
        store_path,
        pixel_microns=params.pixel_microns,
        channels=[params.channel],
        time_interval_s=params.time_interval_s,
        provenance={"op": "io.import_synth", "seed": params.seed},
    )
    # The synth frames are already digitized on a 16-bit grid; store the
    # native float32 [0, 1] values so downstream threshold ops see the same
    # normalized-image convention as momia2 (and the ops test-suite).
    store.write_images(0, params.channel, np.asarray(data["images"], dtype=np.float32))
    truth: dict[str, Any] = {}
    if params.write_truth:
        truth = {
            "edges": [list(e) for e in data["edges"]],
            "records": data["truth"],
            "meta": data["meta"],
        }
    logger.info(
        "io.import_synth: %d frame(s) %s, seed=%d -> %s",
        params.n_frames,
        tuple(params.shape),
        params.seed,
        store_path,
    )
    return {
        "experiment": store_path,
        "truth": truth,
        "truth_labels": np.asarray(data["labels"]),
    }


# ---------------------------------------------------------------------------
# TIFF import
# ---------------------------------------------------------------------------


class ImportTiffParams(BaseModel):
    """Parameters of the ``io.import_tiff`` operation."""

    path: str = Field(description="TIFF file to import (absolute or cwd-relative)")
    pixel_microns: float = Field(1.0, gt=0, description="Physical pixel size (um)")
    channels: list[str] | None = Field(
        None, description="Channel names; None auto-names ch0, ch1, ..."
    )
    fov_id: int = Field(0, ge=0, description="FOV id written into the store")
    axes: str | None = Field(
        None,
        description=(
            "Axis order of the TIFF array (e.g. 'tcyx'); None reads the "
            "ImageJ/OME axis metadata and falls back to guessing"
        ),
    )
    time_interval_s: float | None = Field(
        None, gt=0, description="Time interval between frames (s)"
    )


def _read_tiff_hyperstack(path: Path, axes: str | None) -> tuple[np.ndarray, str]:
    """Read a TIFF into a ``(t, c, y, x)`` array plus the resolved axes.

    Handles multi-page ``(t, c, y, x)`` TIFFs, ImageJ hyperstacks (axis
    order from the series metadata), and plain 2D / ``(t, y, x)`` /
    ``(c, y, x)`` files. Non-spatial singleton axes are squeezed; z-stacks
    with more than one plane are rejected (projection is out of scope).
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        data = series.asarray()
        resolved = (axes or series.axes or "").lower()
    data = np.asarray(data)
    if len(resolved) != data.ndim:
        # Fall back to dimension-count guesses for bare multi-page TIFFs.
        guesses = {2: "yx", 3: "tyx", 4: "tcyx"}
        resolved = guesses.get(data.ndim, "")
        if not resolved:
            raise ValueError(
                f"cannot infer axes for a {data.ndim}D TIFF; pass the 'axes' parameter"
            )
        logger.warning("axes metadata unusable; guessed %r for shape %s", resolved, data.shape)
    unknown = sorted(set(resolved) - set("tcyx"))
    for ax in unknown:
        idx = resolved.index(ax)
        if data.shape[idx] != 1:
            raise ValueError(
                f"unsupported TIFF axis {ax!r} of size {data.shape[idx]}; only "
                "t/c/y/x axes (plus singleton extras) are supported"
            )
        data = np.squeeze(data, axis=idx)
        resolved = resolved.replace(ax, "", 1)
    for ax in "tcyx":
        if ax not in resolved:
            data = np.expand_dims(data, axis=-1)
            resolved = resolved + ax
    perm = [resolved.index(ax) for ax in "tcyx"]
    return np.transpose(data, perm), "tcyx"


@register(
    "io.import_tiff",
    params_model=ImportTiffParams,
    inputs={},
    outputs={"experiment": "store"},
    shard=None,
    description=(
        "Import a multi-page (t, c, y, x) TIFF or ImageJ hyperstack into an "
        "experiment zarr store (one channel stack per c index)."
    ),
)
def run_import_tiff(
    inputs: dict[str, Any], params: ImportTiffParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``io.import_tiff`` (see registry description)."""
    del inputs  # registry-only op
    src = Path(params.path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"TIFF not found: {src}")
    data, _ = _read_tiff_hyperstack(src, params.axes)
    n_t, n_c = int(data.shape[0]), int(data.shape[1])
    channels = params.channels or [f"ch{i}" for i in range(n_c)]
    if len(channels) != n_c:
        raise ValueError(
            f"got {len(channels)} channel name(s) for {n_c} TIFF channel(s)"
        )
    store_path = ctx.output_path("experiment")
    store = ExperimentStore.create(
        store_path,
        pixel_microns=params.pixel_microns,
        channels=list(channels),
        time_interval_s=params.time_interval_s,
        provenance={"op": "io.import_tiff", "source": str(src)},
    )
    for c, channel in enumerate(channels):
        store.write_images(params.fov_id, channel, data[:, c])
    logger.info(
        "io.import_tiff: %s -> %s (t=%d, c=%d, fov=%d)",
        src,
        store_path,
        n_t,
        n_c,
        params.fov_id,
    )
    return {"experiment": store_path}
