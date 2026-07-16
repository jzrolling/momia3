"""IO operations: TIFF import and synthetic-data import.

- ``io.import_tiff`` — read a multi-page ``(t, c, y, x)`` TIFF or an ImageJ
  hyperstack into an OME-Zarr experiment store
  (:class:`momia3.storage.zarr_store.ExperimentStore`).
- ``io.import_synth`` — generate the synthetic rod-bacteria timelapse of
  ``tests/synth.py`` and write it into an experiment store, with the ground
  truth (edges / per-cell records / label stack) as side artifacts for
  demos and benchmarks.

Both are registry-only operations: they take no pipeline inputs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from momia3.pipeline.registry import StepContext, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = ["ImportTiffParams", "ImportSynthParams"]


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
