"""ND2 timelapse import (ported from momia2 momia_IO/_io.py).

momia2's ImageLoader read .nd2 files with the old nd2reader API and served
frames lazily from ND2Reader.get_frame_2D. This port targets the maintained
nd2 package (optional extra io), decodes the whole file into (t, y, x)
stacks once, and writes them into a momia3 experiment store with one zarr
FOV per ND2 position.

Behavior differences vs the momia2 implementation (documented on purpose):

- eager load: ND2File.to_array() reads the file in one pass (momia2 kept
  the file open and read per plane); large multi-position files therefore
  need more RAM. Tracked as a backlog item; the import is a one-off step
  and the store is the working format.
- single Z plane: momia2 could read z planes; this importer requires Z == 1
  or an explicit z_plane selection.
- no nd2reader dependency: the nd2 package is not imported until an import
  actually runs, so the base install stays lean.

Axis order is taken from the ND2File.sizes mapping (insertion order matches
to_array()); positions P map to FOV ids, T to frames, C to channels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from pydantic import BaseModel, Field

from momia3.pipeline.registry import StepContext, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = [
    "ImportNd2Params",
    "Nd2Plan",
    "plan_from_sizes",
    "stacks_from_array",
    "read_nd2",
    "run_import_nd2",
]

_CANONICAL_AXES = ("P", "T", "Z", "C", "Y", "X")
"""Axes this importer understands. P = positions (FOVs)."""


class ImportNd2Params(BaseModel):
    """Parameters of the io.import_nd2 operation."""

    path: str = Field(description="ND2 file to import")
    pixel_microns: float | None = Field(
        None,
        gt=0,
        description=(
            "Physical pixel size override (um); None reads the ND2 voxel "
            "size from metadata and fails when it is unavailable"
        ),
    )
    channels: list[str] | None = Field(
        None,
        description="Channel names; None takes them from ND2 metadata or ch0, ch1, ...",
    )
    z_plane: int = Field(
        0, ge=0, description="Z plane to import when the file has Z > 1"
    )
    fov_start: int = Field(
        0, ge=0, description="First FOV id written to the store"
    )
    time_interval_s: float | None = Field(
        None, gt=0, description="Time interval between frames (s)"
    )


@dataclass(frozen=True)
class Nd2Plan:
    """Decoded axis layout of one ND2 file."""

    raw_order: tuple[str, ...]
    sizes: dict[str, int]
    n_positions: int
    n_frames: int
    n_channels: int
    n_z: int
    shape: tuple[int, int]

    @property
    def fov_count(self) -> int:
        """Number of positions that become FOVs."""
        return self.n_positions


def plan_from_sizes(sizes: Mapping[str, int]) -> Nd2Plan:
    """Derive the import plan from an ND2File.sizes mapping.

    Parameters
    ----------
    sizes : mapping of str to int
        Axis sizes as returned by nd2.ND2File.sizes (insertion order must
        match the array axis order).

    Returns
    -------
    Nd2Plan
        Positions/frames/channels/Z and spatial shape.

    Raises
    ------
    ValueError
        If X/Y are missing, an axis is unknown, or Z handling is unsupported.
    """
    raw = {str(axis).upper(): int(size) for axis, size in sizes.items()}
    for required in ("Y", "X"):
        if required not in raw:
            raise ValueError(
                f"ND2 sizes lack the {required!r} axis: {dict(sizes)!r}"
            )
    unknown = [a for a in raw if a not in _CANONICAL_AXES]
    if unknown:
        raise ValueError(
            f"unsupported ND2 axes {unknown}; supported: {list(_CANONICAL_AXES)}"
        )
    return Nd2Plan(
        raw_order=tuple(raw),
        sizes=dict(raw),
        n_positions=raw.get("P", 1),
        n_frames=raw.get("T", 1),
        n_channels=raw.get("C", 1),
        n_z=raw.get("Z", 1),
        shape=(raw["Y"], raw["X"]),
    )


def stacks_from_array(
    array: np.ndarray,
    plan: Nd2Plan,
    channel_names: list[str],
    *,
    z_plane: int = 0,
) -> dict[int, dict[str, np.ndarray]]:
    """Split a decoded ND2 array into fov -> channel -> (t, y, x).

    Parameters
    ----------
    array : numpy.ndarray
        Whole-file array shaped according to plan.raw_order.
    plan : Nd2Plan
        Layout returned by plan_from_sizes.
    channel_names : list of str
        Names for the C axis; length must equal plan.n_channels.
    z_plane : int
        Plane selected along Z (ignored when n_z == 1).

    Returns
    -------
    dict of int to dict of str to numpy.ndarray
        fov_id -> channel -> (t, y, x) stack.

    Raises
    ------
    ValueError
        On shape, channel-name, or Z-plane mismatches.
    """
    array = np.asarray(array)
    expected = tuple(plan.sizes[axis] for axis in plan.raw_order)
    if array.shape != expected:
        raise ValueError(
            f"decoded ND2 shape {array.shape} does not match sizes {expected}"
        )
    if len(channel_names) != plan.n_channels:
        raise ValueError(
            f"got {len(channel_names)} channel name(s) for {plan.n_channels} channel(s)"
        )
    if plan.n_z > 1 and not 0 <= z_plane < plan.n_z:
        raise ValueError(f"z_plane {z_plane} out of range for {plan.n_z} plane(s)")

    stacks: dict[int, dict[str, np.ndarray]] = {}
    for p in range(plan.n_positions):
        index: list[Any] = []
        for axis in plan.raw_order:
            size = plan.sizes[axis]
            if axis in ("T", "C", "Y", "X"):
                index.append(slice(None))
            elif axis == "Z":
                index.append(z_plane if size > 1 else 0)
            elif axis == "P":
                index.append(p)
            else:  # pragma: no cover - plan_from_sizes rejects these
                if size > 1:
                    raise ValueError(f"unsupported axis {axis!r} of size {size}")
                index.append(0)
        sub = np.asarray(array[tuple(index)])
        kept = [axis for axis in plan.raw_order if axis in ("T", "C", "Y", "X")]
        sub = np.transpose(sub, [kept.index(axis) for axis in ("T", "C", "Y", "X")])
        stacks[p] = {
            name: np.ascontiguousarray(sub[:, c])
            for c, name in enumerate(channel_names)
        }
    return stacks


def _channel_names_from_metadata(reader: Any) -> list[str] | None:
    """Best-effort channel names from ND2File.metadata.channels."""
    metadata = getattr(reader, "metadata", None)
    channels = getattr(metadata, "channels", None)
    if not channels:
        return None
    names: list[str] = []
    for i, entry in enumerate(channels):
        name = None
        if isinstance(entry, Mapping):
            name = entry.get("name")
        else:
            name = getattr(entry, "name", None)
        if not name:
            channel = getattr(entry, "channel", None)
            name = getattr(channel, "name", None) if channel is not None else None
        names.append(str(name) if name else f"ch{i}")
    return names


def _pixel_microns_from_metadata(reader: Any) -> float | None:
    """Best-effort pixel size (um) from an ND2File."""
    voxel_size = getattr(reader, "voxel_size", None)
    if callable(voxel_size):
        try:
            values = voxel_size()
        except Exception:  # pragma: no cover - reader-specific failures
            values = None
        if values is not None:
            flat = [float(v) for v in np.ravel(values) if v is not None]
            if flat and flat[-1] > 0:
                return flat[-1]
    metadata = getattr(reader, "metadata", None)
    for attr in ("pixel_size", "pixel_microns"):
        value = getattr(metadata, attr, None)
        if value is not None and float(value) > 0:
            return float(value)
    return None


def _open_nd2(path: Path) -> Any:
    """Open an ND2 file with the optional nd2 package."""
    try:
        import nd2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "the 'nd2' package is required for io.import_nd2; install the "
            "'io' extra (pip install nd2)"
        ) from exc
    return nd2.ND2File(str(path))


def read_nd2(
    path: str | Path,
    *,
    opener: Callable[[Path], Any] | None = None,
    z_plane: int = 0,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Decode one ND2 file into per-FOV/channel stacks plus metadata.

    opener is injectable so tests can substitute a fake ND2File without the
    optional dependency.
    """
    src = Path(path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"ND2 not found: {src}")
    open_file = opener or _open_nd2
    reader = open_file(src)
    try:
        sizes = getattr(reader, "sizes", None)
        if not sizes:
            raise ValueError(f"ND2 reader for {src} exposes no 'sizes' mapping")
        plan = plan_from_sizes(sizes)
        array = np.asarray(reader.to_array())
        names = _channel_names_from_metadata(reader)
        channel_names = names or [f"ch{i}" for i in range(plan.n_channels)]
        if len(channel_names) != plan.n_channels:
            channel_names = [f"ch{i}" for i in range(plan.n_channels)]
        pixel_microns = _pixel_microns_from_metadata(reader)
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()
    stacks = stacks_from_array(array, plan, channel_names, z_plane=z_plane)
    return (
        stacks,
        {
            "path": str(src),
            "sizes": dict(plan.sizes),
            "n_positions": plan.n_positions,
            "n_frames": plan.n_frames,
            "n_channels": plan.n_channels,
            "n_z": plan.n_z,
            "shape": plan.shape,
            "channel_names": channel_names,
            "pixel_microns": pixel_microns,
        },
    )


@register(
    "io.import_nd2",
    params_model=ImportNd2Params,
    inputs={},
    outputs={"experiment": "store"},
    shard=None,
    description=(
        "Import one ND2 file into an experiment zarr store; positions map to "
        "FOVs, T to frames, C to channels, and Z must be a single plane."
    ),
)
def run_import_nd2(
    inputs: dict[str, Any], params: ImportNd2Params, ctx: StepContext
) -> dict[str, Any]:
    """Runner of io.import_nd2 (see registry description)."""
    del inputs  # registry-only op
    src = Path(params.path).expanduser()
    stacks, meta = read_nd2(src, z_plane=params.z_plane)
    source_names = list(meta["channel_names"])
    channel_names = params.channels or source_names
    if len(channel_names) != int(meta["n_channels"]):
        raise ValueError(
            f"got {len(channel_names)} channel name(s) for "
            f"{meta['n_channels']} ND2 channel(s)"
        )
    pixel_microns = params.pixel_microns if params.pixel_microns is not None else meta["pixel_microns"]
    if pixel_microns is None or float(pixel_microns) <= 0:
        raise ValueError(
            "ND2 metadata carries no usable pixel size; pass pixel_microns "
            "explicitly (momia3 never assumes a default calibration)"
        )
    store_path = ctx.output_path("experiment")
    store = ExperimentStore.create(
        store_path,
        pixel_microns=float(pixel_microns),
        channels=list(channel_names),
        time_interval_s=params.time_interval_s,
        provenance={
            "op": "io.import_nd2",
            "source": str(src),
            "sizes": meta["sizes"],
            "z_plane": params.z_plane,
        },
    )
    for position in sorted(stacks):
        fov_id = int(params.fov_start) + int(position)
        for new_name, old_name in zip(channel_names, source_names):
            store.write_images(fov_id, new_name, stacks[position][old_name])
    logger.info(
        "io.import_nd2: %s -> %s (positions=%d, frames=%d, channels=%d)",
        src,
        store_path,
        len(stacks),
        meta["n_frames"],
        len(channel_names),
    )
    return {"experiment": store_path}
