"""Pipeline step runners: store-level composite ops and raw-op adapters.

Two families of registry entries live here:

1. **Composite chain steps** (the canonical pipeline of the demo):
   ``preprocess.drift``, ``segment.threshold``, ``measure.cells``,
   ``outline.extract``, ``midline.extract`` and ``track.link``. Each runner
   adapts the pure functions of :mod:`momia3.ops` to the engine protocol
   ``run(inputs, params, ctx) -> outputs`` and implements the sharding of
   ARCHITECTURE.md §3.2 (frame-sharded ops map over ``(fov, frame)`` via
   :meth:`momia3.pipeline.registry.StepContext.map_shards`; fov-sharded ops
   map over FOVs). No ``momia3.ops`` internals are modified — this module
   is the only adapter glue.
2. **Raw-op registrations** under the systematic names
   ``"<module>.<function>"`` for every public function of the seven ops
   modules, so the registry is complete for CLI/GUI/MCP schema export.
   These runners operate on workspace artifacts (``.npy`` arrays, parquet
   tables, JSON), converting containers at the boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field
from scipy import sparse
from skimage import measure as ski_measure

from momia3 import keys as _keys
from momia3 import schema as _schema
from momia3.ops import characterize as characterize_ops
from momia3.ops import measure as measure_ops
from momia3.ops import midline as midline_ops
from momia3.ops import outline as outline_ops
from momia3.ops import preprocess as preprocess_ops
from momia3.ops import segment as segment_ops
from momia3.ops import track as track_ops
from momia3.pipeline.registry import StepContext, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# conversion helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy containers/scalars to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _outlines_to_json(outlines: Mapping[int, np.ndarray]) -> dict[str, Any]:
    """``label -> (N, 2)`` outlines to a JSON-safe dict."""
    return {
        str(label): {
            "x": np.asarray(ring, dtype=np.float64)[:, 0].tolist(),
            "y": np.asarray(ring, dtype=np.float64)[:, 1].tolist(),
        }
        for label, ring in outlines.items()
    }


def _outlines_from_json(data: Mapping[str, Any]) -> dict[int, np.ndarray]:
    """Inverse of :func:`_outlines_to_json`."""
    return {
        int(label): np.column_stack([ring["x"], ring["y"]]).astype(np.float64)
        for label, ring in data.items()
    }


def _midlines_to_json(results: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """:func:`momia3.ops.midline.extract_midlines` output to JSON-safe dict."""
    out: dict[str, Any] = {}
    for label, res in results.items():
        entry = {k: _jsonable(v) for k, v in res.items() if k != "midline"}
        entry["midline"] = [
            {"x": np.asarray(b, dtype=np.float64)[:, 0].tolist(),
             "y": np.asarray(b, dtype=np.float64)[:, 1].tolist()}
            for b in res["midline"]
        ]
        out[str(label)] = entry
    return out


def _midlines_from_json(data: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Inverse of :func:`_midlines_to_json` (branches back to arrays)."""
    out: dict[int, dict[str, Any]] = {}
    for label, res in data.items():
        entry = dict(res)
        entry["midline"] = [
            np.column_stack([b["x"], b["y"]]).astype(np.float64)
            for b in res["midline"]
        ]
        out[int(label)] = entry
    return out


def _patch_grid_to_json(grid: segment_ops.PatchGrid) -> dict[str, Any]:
    """Serialize a :class:`momia3.ops.segment.PatchGrid`."""
    return {
        "image_shape": list(grid.image_shape),
        "patch_size": grid.patch_size,
        "overlap": grid.overlap,
        "stride": grid.stride,
        "padded_shape": list(grid.padded_shape),
        "coords": [list(c) for c in grid.coords],
        "window": grid.window.tolist(),
    }


def _patch_grid_from_json(data: Mapping[str, Any]) -> segment_ops.PatchGrid:
    """Inverse of :func:`_patch_grid_to_json`."""
    return segment_ops.PatchGrid(
        image_shape=(int(data["image_shape"][0]), int(data["image_shape"][1])),
        patch_size=int(data["patch_size"]),
        overlap=int(data["overlap"]),
        stride=int(data["stride"]),
        padded_shape=(int(data["padded_shape"][0]), int(data["padded_shape"][1])),
        coords=tuple((int(y), int(x)) for y, x in data["coords"]),
        window=np.asarray(data["window"], dtype=np.float32),
    )


def _overlap_to_json(matrix: sparse.spmatrix) -> dict[str, Any]:
    """COO serialization of a sparse overlap matrix."""
    coo = sparse.coo_matrix(matrix)
    return {
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": np.asarray(coo.data, dtype=np.int64).tolist(),
        "shape": list(coo.shape),
    }


def _overlap_from_json(data: Mapping[str, Any]) -> sparse.coo_matrix:
    """Inverse of :func:`_overlap_to_json`."""
    return sparse.coo_matrix(
        (
            np.asarray(data["data"], dtype=np.int64),
            (np.asarray(data["row"], dtype=np.int64), np.asarray(data["col"], dtype=np.int64)),
        ),
        shape=tuple(int(v) for v in data["shape"]),
    )


def _normalize_cells_frame_table(table: pa.Table) -> pa.Table:
    """Ensure the extra ``weighted_orientation`` column for concat.

    Empty-frame tables from :func:`momia3.ops.measure.labels_to_cells` lack
    the derived column; append a typed empty one so
    :func:`pyarrow.concat_tables` sees identical schemas.
    """
    if "weighted_orientation" not in table.column_names:
        table = table.append_column(
            "weighted_orientation", pa.array([], type=pa.float32())
        )
    return table


# ---------------------------------------------------------------------------
# chain step: preprocess.drift (shard = fov)
# ---------------------------------------------------------------------------


def _drift_fov_shard(args: tuple[str, str, int, list[str], dict[str, Any]]) -> tuple[int, list[list[float]]]:
    """Drift-correct one FOV's channel stacks and write them to the output store."""
    src_path, dst_path, fov_id, channels, params_dict = args
    src = ExperimentStore.open(src_path, mode="r")
    dst = ExperimentStore.open(dst_path, mode="a")
    stacks = {ch: np.asarray(src.images(fov_id, ch)[:]) for ch in channels}
    corrected, shifts = preprocess_ops.drift_correct(
        stacks, preprocess_ops.DriftParams(**params_dict)
    )
    for channel in channels:
        dst.write_images(fov_id, channel, corrected[channel])
    return fov_id, np.asarray(shifts, dtype=np.float64).tolist()


@register(
    "preprocess.drift",
    params_model=preprocess_ops.DriftParams,
    inputs={"experiment": "store"},
    outputs={"experiment": "store", "shifts": "json"},
    shard="fov",
    description=(
        "Phase-correlation drift correction of every channel of an experiment "
        "store (shifts estimated once on the reference channel per frame pair, "
        "applied to all channels); writes a corrected store plus the per-FOV "
        "cumulative shift series."
    ),
    func=preprocess_ops.drift_correct,
)
def run_drift(
    inputs: dict[str, Any], params: preprocess_ops.DriftParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``preprocess.drift`` (see registry description)."""
    src_path = Path(inputs["experiment"])
    dst_path = ctx.output_path("experiment")
    src = ExperimentStore.open(src_path, mode="r")
    dst = ExperimentStore.create(
        dst_path,
        pixel_microns=src.pixel_microns,
        channels=list(src.channels),
        time_interval_s=src.time_interval_s,
        provenance={"op": "preprocess.drift"},
    )
    del dst  # workers open their own handles
    items = [
        (str(src_path), str(dst_path), fov_id, list(src.channels), params.model_dump(mode="json"))
        for fov_id in src.fov_ids
    ]
    results = ctx.map_shards(_drift_fov_shard, items)
    shifts = {str(fov_id): series for fov_id, series in results}
    return {"experiment": dst_path, "shifts": shifts}


# ---------------------------------------------------------------------------
# chain step: segment.threshold (shard = frame)
# ---------------------------------------------------------------------------


class ThresholdRunParams(segment_ops.ThresholdParams):
    """``segment.threshold`` step parameters (threshold params + channel)."""

    channel: str | None = Field(
        None, description="Channel to segment; None uses the first store channel"
    )


def _normalized_frame(image: np.ndarray) -> np.ndarray:
    """Scale an integer frame to float32 ``[0, 1]`` (momia2 convention).

    The threshold operators (Sauvola in particular) assume normalized
    images; float frames are passed through unchanged.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / np.float32(np.iinfo(image.dtype).max)
    return image.astype(np.float32, copy=False)


def _threshold_frame_shard(args: tuple[str, int, int, str, dict[str, Any]]) -> int:
    """Binarize one frame and write its instance labels into the store."""
    store_path, fov_id, t, channel, params_dict = args
    store = ExperimentStore.open(store_path, mode="a")
    image = _normalized_frame(store.get_frame(fov_id, channel, t))
    mask = segment_ops.threshold_segment(image, segment_ops.ThresholdParams(**params_dict))
    labels = ski_measure.label(mask).astype(np.uint32)
    store.set_labels_frame(fov_id, t, labels)
    return int(labels.max())


@register(
    "segment.threshold",
    params_model=ThresholdRunParams,
    inputs={"experiment": "store"},
    outputs={"experiment": "store"},
    shard="frame",
    description=(
        "Traditional threshold segmentation of every (fov, frame) shard of an "
        "experiment store; instance labels are written into the store's labels "
        "arrays in place."
    ),
    verify=("fov_*/labels",),
    func=segment_ops.threshold_segment,
)
def run_threshold(
    inputs: dict[str, Any], params: ThresholdRunParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``segment.threshold`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="a")
    channel = params.channel or store.channels[0]
    seg_params = segment_ops.ThresholdParams(
        **{k: v for k, v in params.model_dump(mode="json").items() if k != "channel"}
    )
    items: list[tuple[str, int, int, str, dict[str, Any]]] = []
    for fov_id in store.fov_ids:
        shape = store.shape(fov_id, channel)
        # Pre-create the labels stack in the parent so parallel workers only
        # ever overwrite their own t-slice (never race on array creation).
        store.write_labels(fov_id, np.zeros(shape, dtype=np.uint32))
        items += [
            (str(store_path), fov_id, t, channel, seg_params.model_dump(mode="json"))
            for t in range(shape[0])
        ]
    counts = ctx.map_shards(_threshold_frame_shard, items)
    logger.info(
        "segment.threshold: %d shard(s), max %d instance(s)/frame",
        len(counts),
        max(counts, default=0),
    )
    return {"experiment": store_path}


# ---------------------------------------------------------------------------
# chain step: segment.declump_store (shard = frame)
# ---------------------------------------------------------------------------


def _declump_frame_shard(args: tuple[str, int, int, dict[str, Any]]) -> int:
    """Split merged blobs of one frame's labels and write back in place."""
    store_path, fov_id, t, params_dict = args
    store = ExperimentStore.open(store_path, mode="a")
    labels = store.get_labels_frame(fov_id, t)
    if labels.max() == 0:
        return 0
    new_labels = segment_ops.declump(
        labels > 0, params=segment_ops.DeclumpParams(**params_dict)
    )
    store.set_labels_frame(fov_id, t, new_labels.astype(np.uint32))
    return int(new_labels.max())


@register(
    "segment.declump_store",
    params_model=segment_ops.DeclumpParams,
    inputs={"experiment": "store"},
    outputs={"experiment": "store"},
    shard="frame",
    description=(
        "Split merged foreground blobs of every (fov, frame) shard of an "
        "experiment store (EDT + peak_local_max seeds + marker watershed); "
        "labels are rewritten in place."
    ),
    verify=("fov_*/labels",),
    func=segment_ops.declump,
)
def run_declump_store(
    inputs: dict[str, Any], params: segment_ops.DeclumpParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``segment.declump_store`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    items: list[tuple[str, int, int, dict[str, Any]]] = []
    for fov_id in store.fov_ids:
        n_t = int(store.labels(fov_id).shape[0])
        items += [
            (str(store_path), fov_id, t, params.model_dump(mode="json"))
            for t in range(n_t)
        ]
    counts = ctx.map_shards(_declump_frame_shard, items)
    logger.info(
        "segment.declump_store: %d shard(s), max %d instance(s)/frame",
        len(counts),
        max(counts, default=0),
    )
    return {"experiment": store_path}


# ---------------------------------------------------------------------------
# chain step: measure.cells (shard = frame)
# ---------------------------------------------------------------------------


class CellsRunParams(BaseModel):
    """``measure.cells`` step parameters."""

    pixel_microns: float | None = Field(
        None, gt=0, description="Physical pixel size; None reads the store attrs"
    )
    channel: str | None = Field(
        None, description="Intensity channel; None uses the first store channel"
    )
    include_intensity: bool = Field(
        True, description="Add the per-channel intensity cluster columns"
    )


def _cells_frame_shard(
    args: tuple[str, int, int, str, bool, float],
) -> pa.Table:
    """Measure one frame's labels into a cells table (one row per instance)."""
    store_path, fov_id, t, channel, include_intensity, pixel_microns = args
    store = ExperimentStore.open(store_path, mode="r")
    labels = store.get_labels_frame(fov_id, t)
    intensity = None
    if include_intensity:
        intensity = {channel: store.get_frame(fov_id, channel, t)}
    return measure_ops.labels_to_cells(
        labels, fov_id, t, intensity=intensity, pixel_microns=pixel_microns
    )


@register(
    "measure.cells",
    params_model=CellsRunParams,
    inputs={"experiment": "store"},
    outputs={"cells": "table:cells"},
    shard="frame",
    description=(
        "Regionprops + intensity measurement of every (fov, frame) shard into "
        "one canonical cells table (geometry columns left null for the "
        "outline/midline steps)."
    ),
    func=measure_ops.labels_to_cells,
)
def run_cells(
    inputs: dict[str, Any], params: CellsRunParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``measure.cells`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    channel = params.channel or store.channels[0]
    pixel_microns = params.pixel_microns or store.pixel_microns
    items = [
        (str(store_path), fov_id, t, channel, params.include_intensity, pixel_microns)
        for fov_id in store.fov_ids
        for t in range(store.n_frames(fov_id))
    ]
    tables = [
        _normalize_cells_frame_table(t) for t in ctx.map_shards(_cells_frame_shard, items)
    ]
    channels = (channel,) if params.include_intensity else ()
    if tables:
        table = pa.concat_tables(tables, promote_options="none")
    else:
        table = _schema.empty_cells_table(channels)
    return {"cells": table}


# ---------------------------------------------------------------------------
# chain step: outline.extract (shard = frame)
# ---------------------------------------------------------------------------


class OutlineParams(BaseModel):
    """Sub-pixel outline extraction parameters (outline.extract step and the
    raw outline.extract_outlines op).

    Defaults mirror momia3.ops.outline._DEFAULT_PARAMS: pixel-equidistant
    sampling plus one round of edge+intensity subpixel refinement.
    """

    sigma: float = Field(1.0, ge=0, description="Mask Gaussian smoothing sigma (px)")
    level: float = Field(0.5, gt=0, lt=1, description="Marching-squares level")
    simplify: bool = Field(False, description="RDP polygon simplification")
    tolerance: float = Field(0.75, ge=0, description="RDP tolerance (px)")
    smooth: float = Field(1.0, ge=0, description="Closed B-spline smoothing factor")
    n_points: int | None = Field(
        None,
        ge=4,
        description="Uniform resample count; None = perimeter / downsample_factor "
        "(pixel-equidistant, the optimized default)",
    )
    downsample_factor: float = Field(
        2.0, gt=0, description="Target pixel spacing of the resampled outline (px/point)"
    )
    smooth_scale: float = Field(
        0.03, ge=0, description="B-spline stiffness as a fraction of the point count"
    )
    active_contour: bool = Field(
        False, description="Refine with an intensity-edge snake (needs channel)"
    )
    alpha: float = Field(0.015, ge=0, description="Snake length energy weight")
    beta: float = Field(10.0, ge=0, description="Snake smoothness energy weight")
    gamma: float = Field(0.001, gt=0, description="Snake explicit time step")
    pad: int = Field(2, ge=0, description="Bounding-box padding (px)")
    constriction_iters: int = Field(
        0, ge=0, description="Deprecated mask-target constriction rounds (0 = off)"
    )
    constriction_strength: float = Field(0.6, ge=0, le=1)
    constriction_smooth_scale: float = Field(0.01, ge=0)
    constriction_image_iters: int = Field(
        0, ge=0, description="Intensity-guided constriction rounds (0 = off)"
    )
    constriction_image_max_shift: float = Field(3.0, gt=0, description="Inward cap (px/round)")
    constriction_image_strength: float = Field(0.8, ge=0, le=1)
    constriction_image_threshold_frac: float = Field(0.55, gt=0, lt=1)
    refine_outline_iters: int = Field(
        1, ge=0, description="Combined edge+intensity refinement rounds (0 = off)"
    )
    refine_outline_mode: Literal["edge", "intensity", "edge+intensity"] = Field(
        "edge+intensity", description="Which refinement kernels each round applies"
    )
    refine_edge_step: float = Field(0.5, gt=0, description="Normal sampling step (px)")
    refine_edge_cutoff: float = Field(0.2, ge=0, description="Weak-edge cutoff")
    refine_edge_max_shift: float = Field(1.5, gt=0, description="Move cap (px/round)")
    refine_edge_max_round_shift: float = Field(
        1.5, gt=0, description="Revert points moving farther than this per round (px)"
    )
    refine_edge_pre_expand: float = Field(
        0.3, ge=0, description="Outward pre-expansion before the first round (px)"
    )
    advanced_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch: raw momia3.ops.outline parameter overrides "
        "(keys not listed above, e.g. future tuning knobs)",
    )


class OutlineRunParams(OutlineParams):
    """outline.extract step parameters (adds the intensity channel)."""

    channel: str | None = Field(
        None,
        description="Intensity channel for edge/intensity refinement; "
        "None uses the first store channel",
    )


def _merged_params(params: BaseModel, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validated params dict for an ops function, with advanced_params merged.

    Keys in exclude are dropped (e.g. the store channel or pixel_microns,
    which are passed separately); advanced_params entries override the typed
    fields, giving access to the full ops tuning surface without duplicating
    every knob in the pydantic schema.
    """
    data = params.model_dump(mode="json")
    advanced = dict(data.pop("advanced_params", {}) or {})
    for key in exclude:
        data.pop(key, None)
    data.update(advanced)
    return data


def _outline_frame_shard(
    args: tuple[str, int, int, str | None, dict[str, Any]],
) -> tuple[int, int, dict[int, np.ndarray]]:
    """Extract sub-pixel outlines of one frame's instances."""
    store_path, fov_id, t, channel, params_dict = args
    store = ExperimentStore.open(store_path, mode="r")
    labels = store.get_labels_frame(fov_id, t)
    intensity = None
    if channel is not None and params_dict.get("active_contour"):
        intensity = store.get_frame(fov_id, channel, t)
    return fov_id, t, outline_ops.extract_outlines(labels, intensity, params_dict)


@register(
    "outline.extract",
    params_model=OutlineRunParams,
    inputs={"experiment": "store", "cells": "table:cells"},
    outputs={"cells": "table:cells"},
    shard="frame",
    description=(
        "Sub-pixel outline extraction of every (fov, frame) shard; outlines "
        "and curvature scalars are filled into the cells table (keyed by "
        "cell_key)."
    ),
    func=outline_ops.extract_outlines,
)
def run_outline_extract(
    inputs: dict[str, Any], params: OutlineRunParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``outline.extract`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    params_dict = _merged_params(params, exclude=("channel",))
    items = [
        (str(store_path), fov_id, t, params.channel, params_dict)
        for fov_id in store.fov_ids
        for t in range(store.n_frames(fov_id))
    ]
    by_cell_key: dict[int, np.ndarray] = {}
    for fov_id, t, outlines in ctx.map_shards(_outline_frame_shard, items):
        for label, ring in outlines.items():
            by_cell_key[_keys.encode_key(fov_id, t, label)] = ring
    table = outline_ops.fill_cells_geometry(inputs["cells"], by_cell_key)
    logger.info("outline.extract: %d outline(s) filled", len(by_cell_key))
    return {"cells": table}


# ---------------------------------------------------------------------------
# chain step: midline.extract (shard = frame)
# ---------------------------------------------------------------------------


class MidlineParams(BaseModel):
    """Midline extraction parameters (midline.extract step and the raw
    midline.extract_midlines op).

    The optimized route is midline_method="equidistant" (outline-SDF or
    mask-skeleton contour-cut midline with guard + degenerate gate); the
    default stays "spline" for backwards compatibility. Every key of
    momia3.ops.midline._DEFAULT_PARAMS can be overridden through
    advanced_params.
    """

    method: Literal["zhang", "lee"] = Field("zhang", description="Skeletonization")
    min_branch_length_um: float = Field(
        0.2, ge=0, description="Spur pruning threshold (um)"
    )
    n_points: int = Field(32, ge=2, description="Resampled main-branch size")
    smooth: float = Field(1.0, ge=0, description="Open B-spline smoothing factor")
    refine_iterations: int = Field(
        10, ge=0, description="Legacy perpendicular refinement iterations (unused)"
    )
    refine_tolerance: float = Field(
        0.1, gt=0, description="Legacy refinement convergence tolerance (px, unused)"
    )
    refine_anchor: int = Field(
        3, ge=0, description="Legacy end samples kept fixed (unused)"
    )
    refine_max_step: float = Field(
        1.5, gt=0, description="Legacy per-iteration move cap (px, unused)"
    )
    pole_fit_points: int = Field(5, ge=2, description="End-tangent fit window")
    knn: int = Field(24, ge=1, description="Outline segment candidates per query")
    pad: int = Field(2, ge=0, description="Bounding-box padding (px)")
    pixel_microns: float | None = Field(
        None, gt=0, description="Physical pixel size; None reads the store attrs"
    )
    midline_method: Literal["spline", "energy", "equidistant"] = Field(
        "spline",
        description="Algorithm route: legacy spline, experimental energy, or the "
        "optimized equidistant contour-cut method",
    )
    n_points_auto: bool = Field(
        False, description="Spline route: n = arc_length / n_points_target_spacing_px"
    )
    n_points_target_spacing_px: float = Field(2.0, gt=0)
    pole_endpoint: Literal["extended", "initial"] = Field(
        "extended", description="Pole handling: legacy extended or outline pole pair"
    )
    degenerate_ar: float = Field(1.4, gt=0, description="Aspect-ratio gate for near-round cells")
    degenerate_refine: bool = Field(
        True, description="One chord-midpoint equidistant pass on the degenerate axis"
    )
    degenerate_refine_iters: int = Field(1, ge=0)
    degenerate_refine_points: int = Field(12, ge=6)
    degenerate_refine_anchor: int = Field(2, ge=0)
    degenerate_refine_damp: float = Field(0.6, ge=0, le=1)
    degenerate_refine_smooth: float = Field(0.03, ge=0)
    degenerate_solidity_min: float = Field(0.85, gt=0, le=1)
    degenerate_simple_solidity: float = Field(0.88, gt=0, le=1)
    degenerate_simple_ellipse_dev: float = Field(0.12, ge=0)
    degenerate_simple_sagitta: float = Field(0.08, ge=0)
    equidistant_guard: bool = Field(
        True, description="Guarded equidistant: reject candidates that leave the chord/outline"
    )
    equidistant_skeleton_from_outline: bool = Field(
        True, description="Build the SDF skeleton from the smooth outline geometry"
    )
    equidistant_sdf_max_dim: int = Field(
        400, ge=8, description="Use the mask-skeleton route above this bbox size (px)"
    )
    equidistant_sdf_max_pixels: float = Field(
        4_000_000.0, gt=0, description="Supersampled SDF grid budget"
    )
    equidistant_n_points_auto: bool = Field(
        True, description="n = arc_length / equidistant_n_points_spacing_px"
    )
    equidistant_n_points_spacing_px: float = Field(2.0, gt=0)
    equidistant_inside_clamp: bool = Field(
        True, description="Clamp interior samples inside the mask on long bends"
    )
    equidistant_branch_min_px: float = Field(3.0, ge=0)
    equidistant_branch_min_um: float = Field(0.2, ge=0)
    equidistant_fill_holes: bool = Field(True)
    equidistant_opening_radius: float = Field(1.0, ge=0)
    equidistant_outline_from_clean_mask: bool = Field(True)
    equidistant_main_smooth_factor: float = Field(0.02, ge=0)
    equidistant_main_smooth_max: float = Field(60.0, ge=0)
    equidistant_guard_chord_solidity: float = Field(0.85, gt=0, le=1)
    equidistant_guard_chord_dev_frac: float = Field(0.30, gt=0)
    equidistant_guard_chord_dev_px: float = Field(1.5, gt=0)
    advanced_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Escape hatch: raw momia3.ops.midline parameter overrides for "
        "keys not listed above (e.g. equidistant axis sampling, Taubin smoothing)",
    )


def _midline_frame_shard(
    args: tuple[str, int, int, dict[str, Any], dict[str, Any], float],
) -> tuple[int, int, dict[int, dict[str, Any]]]:
    """Extract midlines/poles/width profiles of one frame's instances."""
    store_path, fov_id, t, outlines_json, params_dict, pixel_microns = args
    store = ExperimentStore.open(store_path, mode="r")
    labels = store.get_labels_frame(fov_id, t)
    outlines = _outlines_from_json(outlines_json)
    results = midline_ops.extract_midlines(
        labels, outlines or None, params_dict, pixel_microns=pixel_microns
    )
    return fov_id, t, results


@register(
    "midline.extract",
    params_model=MidlineParams,
    inputs={"experiment": "store", "cells": "table:cells"},
    outputs={"cells": "table:cells"},
    shard="frame",
    description=(
        "Midline/pole/width-profile extraction of every (fov, frame) shard, "
        "reusing the outlines already stored in the cells table (no recomputation); "
        "midline geometry and scalars are filled into the cells table."
    ),
    func=midline_ops.extract_midlines,
)
def run_midline_extract(
    inputs: dict[str, Any], params: MidlineParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``midline.extract`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    pixel_microns = params.pixel_microns or store.pixel_microns
    params_dict = _merged_params(params, exclude=("pixel_microns",))
    cells = inputs["cells"]
    if not isinstance(cells, pa.Table):
        cells = cells.to_arrow()
    # Outlines already live in the cells table (outline.extract); group them
    # per (fov, frame) so the midline op never recomputes them (§3.2).
    outlines_by_frame: dict[tuple[int, int], dict[str, Any]] = {}
    for row in cells.select(["fov_id", "frame", "label", "outline"]).to_pylist():
        ring = row["outline"]
        if ring is None:
            continue
        outlines_by_frame.setdefault((int(row["fov_id"]), int(row["frame"])), {})[
            str(int(row["label"]))
        ] = ring
    items = [
        (
            str(store_path),
            fov_id,
            t,
            outlines_by_frame.get((fov_id, t), {}),
            params_dict,
            pixel_microns,
        )
        for fov_id in store.fov_ids
        for t in range(store.n_frames(fov_id))
    ]
    by_cell_key: dict[int, dict[str, Any]] = {}
    for fov_id, t, results in ctx.map_shards(_midline_frame_shard, items):
        for label, res in results.items():
            by_cell_key[_keys.encode_key(fov_id, t, label)] = res
    table = midline_ops.fill_cells_midline(cells, by_cell_key)
    logger.info("midline.extract: %d midline(s) filled", len(by_cell_key))
    return {"cells": table}


# ---------------------------------------------------------------------------
# chain step: track.link (shard = fov)
# ---------------------------------------------------------------------------


@register(
    "track.link",
    params_model=track_ops.TrackParams,
    inputs={"experiment": "store", "cells": "table:cells"},
    outputs={"edges": "table:edges", "tracks": "table:tracks"},
    shard="fov",
    description=(
        "LAP linking + gap closing of the labels stacks of an experiment "
        "store into canonical edges and tracks tables (per-FOV linking, "
        "Kalman-gated)."
    ),
    func=track_ops.link_timelapse,
)
def run_track_link(
    inputs: dict[str, Any], params: track_ops.TrackParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``track.link`` (see registry description)."""
    del ctx  # linking is serial in time per FOV; the op maps FOVs internally
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    stacks = {fov_id: np.asarray(store.labels(fov_id)[:]) for fov_id in store.fov_ids}
    edges, tracks = track_ops.link_timelapse(stacks, inputs["cells"], params)
    return {"edges": edges, "tracks": tracks}


# ---------------------------------------------------------------------------
# raw-op registrations ("<module>.<function>") — preprocess
# ---------------------------------------------------------------------------


@register(
    "preprocess.illumination_correct",
    params_model=preprocess_ops.IlluminationParams,
    inputs={"stack": "array"},
    outputs={"stack": "array"},
    shard="frame",
    description="Simplified BaSiC flatfield/darkfield correction of a (t, y, x) stack.",
    func=preprocess_ops.illumination_correct,
)
def _raw_illumination_correct(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"stack": preprocess_ops.illumination_correct(inputs["stack"], params)}


@register(
    "preprocess.rolling_ball_background_fft",
    params_model=preprocess_ops.RollingBallParams,
    inputs={"image": "array"},
    outputs={"image": "array"},
    shard="frame",
    description="FFT rolling-ball background subtraction of a 2D image or (t, y, x) stack.",
    func=preprocess_ops.rolling_ball_background_fft,
)
def _raw_rolling_ball(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"image": preprocess_ops.rolling_ball_background_fft(inputs["image"], params)}


class ShrinkLocalMinParams(BaseModel):
    """Parameters of the raw ``preprocess.shrink_img_local_min`` op."""

    shrink_factor: int = Field(4, ge=1, description="Block side (px)")


@register(
    "preprocess.shrink_img_local_min",
    params_model=ShrinkLocalMinParams,
    inputs={"image": "array"},
    outputs={"image": "array"},
    shard="frame",
    description="Block-minimum downsample (momia2 rolling-ball amortization primitive).",
    func=preprocess_ops.shrink_img_local_min,
)
def _raw_shrink_img_local_min(inputs: dict[str, Any], params: ShrinkLocalMinParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {
        "image": preprocess_ops.shrink_img_local_min(
            inputs["image"], params.shrink_factor
        )
    }


@register(
    "preprocess.drift_correct",
    params_model=preprocess_ops.DriftParams,
    inputs={"stack": "array"},
    outputs={"stack": "array", "shifts": "array"},
    shard=None,
    description="Phase-correlation drift correction of a (t, y, x) stack (single channel).",
    func=preprocess_ops.drift_correct,
)
def _raw_drift_correct(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    corrected, shifts = preprocess_ops.drift_correct(inputs["stack"], params)
    return {"stack": corrected, "shifts": shifts}


@register(
    "preprocess.bleach_correct",
    params_model=preprocess_ops.BleachParams,
    inputs={"stack": "array"},
    outputs={"stack": "array"},
    shard=None,
    description="Exponential bleach correction of a (t, y, x) stack.",
    func=preprocess_ops.bleach_correct,
)
def _raw_bleach_correct(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"stack": preprocess_ops.bleach_correct(inputs["stack"], params)}


@register(
    "preprocess.normalize",
    params_model=preprocess_ops.NormalizeParams,
    inputs={"image": "array"},
    outputs={"image": "array"},
    shard="frame",
    description="Percentile/gamma normalization with an explicit output dtype.",
    func=preprocess_ops.normalize,
)
def _raw_normalize(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"image": preprocess_ops.normalize(inputs["image"], params)}


@register(
    "preprocess.background_correct",
    params_model=preprocess_ops.BackgroundParams,
    inputs={"image": "array"},
    outputs={"image": "array"},
    shard="frame",
    description="Background/illumination suppression (rolling-ball, black top-hat, "
    "block percentile or block median; optional mean restoration and light blur).",
    func=preprocess_ops.background_correct,
)
def _raw_background_correct(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"image": preprocess_ops.background_correct(inputs["image"], params)}


@register(
    "preprocess.denoise",
    params_model=preprocess_ops.DenoiseParams,
    inputs={"image": "array"},
    outputs={"image": "array"},
    shard="frame",
    description="Denoise a 2D image (NLM / gaussian / bilateral).",
    func=preprocess_ops.denoise,
)
def _raw_denoise(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"image": preprocess_ops.denoise(inputs["image"], params)}


# ---------------------------------------------------------------------------
# raw-op registrations — segment
# ---------------------------------------------------------------------------


@register(
    "segment.threshold_segment",
    params_model=segment_ops.ThresholdParams,
    inputs={"image": "array"},
    outputs={"mask": "array"},
    shard="frame",
    description="Binarize a 2D image into a boolean foreground mask.",
    func=segment_ops.threshold_segment,
)
def _raw_threshold_segment(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"mask": segment_ops.threshold_segment(inputs["image"], params)}


@register(
    "segment.prob_to_labels",
    params_model=segment_ops.ProbToLabelsParams,
    inputs={"prob": "array"},
    outputs={"labels": "array"},
    shard="frame",
    description="Instance labels from a (c, y, x) 3-class probability map.",
    func=segment_ops.prob_to_labels,
)
def _raw_prob_to_labels(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"labels": segment_ops.prob_to_labels(inputs["prob"], params)}


@register(
    "segment.declump",
    params_model=segment_ops.DeclumpParams,
    inputs={"mask": "array", "dist": "array?"},
    outputs={"labels": "array"},
    shard="frame",
    description=(
        "Split merged foreground blobs into per-cell instances (EDT + "
        "peak_local_max seeds + marker watershed)."
    ),
    func=segment_ops.declump,
)
def _raw_declump(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"labels": segment_ops.declump(inputs["mask"], inputs.get("dist"), params)}


class ComputeDistParams(BaseModel):
    """Parameters of the raw ``segment.compute_dist`` op."""

    max_value: float = Field(7.0, gt=0, description="Upper bound of the normalized range")


@register(
    "segment.compute_dist",
    params_model=ComputeDistParams,
    inputs={"labels": "array"},
    outputs={"dist": "array"},
    shard="frame",
    description="Normalized per-instance Euclidean distance transform of a label image.",
    func=segment_ops.compute_dist,
)
def _raw_compute_dist(inputs: dict[str, Any], params: ComputeDistParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"dist": segment_ops.compute_dist(inputs["labels"], max_value=params.max_value)}


class PatchifyParams(BaseModel):
    """Parameters of the raw ``segment.patchify`` op."""

    patch_size: int = Field(256, ge=1, description="Square patch side (px)")
    overlap: int = Field(64, ge=0, description="Overlap between adjacent patches (px)")


@register(
    "segment.patchify",
    params_model=PatchifyParams,
    inputs={"image": "array"},
    outputs={"patches": "array", "grid": "json"},
    shard=None,
    description="Split an image into overlapping patches (with stitching geometry).",
    func=segment_ops.patchify,
)
def _raw_patchify(inputs: dict[str, Any], params: PatchifyParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    patches, grid = segment_ops.patchify(inputs["image"], params.patch_size, params.overlap)
    return {"patches": patches, "grid": _patch_grid_to_json(grid)}


class UnpatchifyParams(BaseModel):
    """Parameters of the raw ``segment.unpatchify`` op."""

    mode: Literal["weighted", "mean", "max", "min"] = Field(
        "weighted", description="Patch merge mode"
    )


@register(
    "segment.unpatchify",
    params_model=UnpatchifyParams,
    inputs={"patches": "array", "grid": "json"},
    outputs={"image": "array"},
    shard=None,
    description="Stitch patches back into an image.",
    func=segment_ops.unpatchify,
)
def _raw_unpatchify(inputs: dict[str, Any], params: UnpatchifyParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    grid = _patch_grid_from_json(inputs["grid"])
    return {"image": segment_ops.unpatchify(inputs["patches"], grid, mode=params.mode)}


class GridParams(BaseModel):
    """Parameters of the raw ``segment.build_patch_grid`` op."""

    image_shape: tuple[int, int] = Field(description="(h, w) of the unpadded image")
    patch_size: int = Field(256, ge=1, description="Square patch side (px)")
    overlap: int = Field(64, ge=0, description="Overlap between adjacent patches (px)")


@register(
    "segment.build_patch_grid",
    params_model=GridParams,
    inputs={},
    outputs={"grid": "json"},
    shard=None,
    description="Compute the deterministic patch-stitching grid of an image shape.",
    func=segment_ops.build_patch_grid,
)
def _raw_build_patch_grid(inputs: dict[str, Any], params: GridParams, ctx: StepContext) -> dict[str, Any]:
    del inputs, ctx
    grid = segment_ops.build_patch_grid(tuple(params.image_shape), params.patch_size, params.overlap)
    return {"grid": _patch_grid_to_json(grid)}


class NameParams(BaseModel):
    """Single-name parameter (segmenter registry lookups)."""

    name: str = Field(description="Registered segmenter name")


@register(
    "segment.register_segmenter",
    params_model=NameParams,
    inputs={},
    outputs={"status": "json"},
    shard=None,
    description=(
        "Register an AI segmentation backend by name. Not executable through a "
        "JSON pipeline (callables cannot cross the JSON boundary); registered "
        "for schema completeness."
    ),
    func=segment_ops.register_segmenter,
)
def _raw_register_segmenter(inputs: dict[str, Any], params: NameParams, ctx: StepContext) -> dict[str, Any]:
    del inputs, ctx
    raise RuntimeError(
        "segment.register_segmenter cannot run inside a JSON pipeline: AI "
        "segmenter callables must be registered from Python via "
        "momia3.ops.register_segmenter"
    )


@register(
    "segment.get_segmenter",
    params_model=NameParams,
    inputs={},
    outputs={"info": "json"},
    shard=None,
    description="Look up a registered AI segmentation backend by name.",
    func=segment_ops.get_segmenter,
)
def _raw_get_segmenter(inputs: dict[str, Any], params: NameParams, ctx: StepContext) -> dict[str, Any]:
    del inputs, ctx
    registered = params.name in segment_ops.SEGMENTERS
    info: dict[str, Any] = {"name": params.name, "registered": registered, "type": None}
    if registered:
        info["type"] = type(segment_ops.SEGMENTERS[params.name]).__name__
    return {"info": info}


# ---------------------------------------------------------------------------
# raw-op registrations — measure
# ---------------------------------------------------------------------------


class CellsRawParams(BaseModel):
    """Parameters of the raw ``measure.labels_to_cells`` op."""

    fov_id: int = Field(0, ge=0, description="FOV id encoded into cell_key")
    frame: int = Field(0, ge=0, description="Frame index encoded into cell_key")
    pixel_microns: float = Field(1.0, gt=0, description="Physical pixel size (um)")
    channel: str = Field("phase", description="Intensity channel name")
    include_intensity: bool = Field(
        True, description="Add the per-channel intensity cluster columns"
    )


@register(
    "measure.labels_to_cells",
    params_model=CellsRawParams,
    inputs={"labels": "array", "image": "array?"},
    outputs={"cells": "table:cells"},
    shard="frame",
    description="Measure one frame's instance labels into a canonical cells table.",
    func=measure_ops.labels_to_cells,
)
def _raw_labels_to_cells(inputs: dict[str, Any], params: CellsRawParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    intensity = None
    if params.include_intensity and inputs.get("image") is not None:
        intensity = {params.channel: inputs["image"]}
    table = measure_ops.labels_to_cells(
        inputs["labels"],
        params.fov_id,
        params.frame,
        intensity=intensity,
        pixel_microns=params.pixel_microns,
    )
    return {"cells": table}


class PrefixParams(BaseModel):
    """Parameters of the raw ``measure.intensity_stats`` op."""

    prefix: str = Field("ch0", description="Channel name prefix of the output keys")


@register(
    "measure.intensity_stats",
    params_model=PrefixParams,
    inputs={"labels": "array", "image": "array"},
    outputs={"stats": "json"},
    shard="frame",
    description="Vectorized per-cell intensity statistics of one channel.",
    func=measure_ops.intensity_stats,
)
def _raw_intensity_stats(inputs: dict[str, Any], params: PrefixParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    stats = measure_ops.intensity_stats(inputs["labels"], inputs["image"], params.prefix)
    return {"stats": _jsonable(stats)}


class HaralickParams(BaseModel):
    """Parameters of the raw ``measure.texture_haralick`` op."""

    levels: int = Field(64, ge=2, description="Gray levels for quantization")
    distances: tuple[int, ...] = Field((1,), description="GLCM distances (px)")
    angles: tuple[float, ...] = Field(
        (0.0, float(np.pi / 4), float(np.pi / 2), float(3 * np.pi / 4)),
        description="GLCM angles (rad)",
    )


@register(
    "measure.texture_haralick",
    params_model=HaralickParams,
    inputs={"labels": "array", "image": "array"},
    outputs={"features": "json"},
    shard="frame",
    description="Per-cell Haralick texture features (CellProfiler analogue).",
    func=measure_ops.texture_haralick,
)
def _raw_texture_haralick(inputs: dict[str, Any], params: HaralickParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    features = measure_ops.texture_haralick(
        inputs["labels"],
        inputs["image"],
        levels=params.levels,
        distances=params.distances,
        angles=params.angles,
    )
    return {"features": _jsonable(features)}


class LbpParams(BaseModel):
    """Parameters of the raw ``measure.texture_lbp`` op."""

    P: int = Field(8, ge=1, description="Number of circularly symmetric neighbors")
    R: float = Field(1.0, gt=0, description="Radius of the sampling circle (px)")


@register(
    "measure.texture_lbp",
    params_model=LbpParams,
    inputs={"labels": "array", "image": "array"},
    outputs={"features": "json"},
    shard="frame",
    description="Per-cell uniform-LBP histogram features.",
    func=measure_ops.texture_lbp,
)
def _raw_texture_lbp(inputs: dict[str, Any], params: LbpParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    features = measure_ops.texture_lbp(inputs["labels"], inputs["image"], P=params.P, R=params.R)
    return {"features": _jsonable(features)}


class EmptyParams(BaseModel):
    """Parameterless operation."""


@register(
    "measure.add_features",
    params_model=EmptyParams,
    inputs={"cells": "table", "features": "json"},
    outputs={"cells": "table"},
    shard=None,
    description="Append per-cell feature columns to a cells table.",
    func=measure_ops.add_features,
)
def _raw_add_features(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    return {"cells": measure_ops.add_features(inputs["cells"], dict(inputs["features"]))}


# ---------------------------------------------------------------------------
# raw-op registrations — outline / midline
# ---------------------------------------------------------------------------


@register(
    "outline.extract_outlines",
    params_model=OutlineParams,
    inputs={"labels": "array", "intensity": "array?"},
    outputs={"outlines": "json"},
    shard="frame",
    description="Sub-pixel closed outlines of every labelled instance of one frame.",
    func=outline_ops.extract_outlines,
)
def _raw_extract_outlines(inputs: dict[str, Any], params: OutlineParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    outlines = outline_ops.extract_outlines(
        inputs["labels"], inputs.get("intensity"), _merged_params(params)
    )
    return {"outlines": _outlines_to_json(outlines)}


class CurvatureParams(BaseModel):
    """Parameters of the raw ``outline.curvature_features`` op."""

    window: int = Field(3, ge=1, description="Bend-angle half-window (samples)")


@register(
    "outline.curvature_features",
    params_model=CurvatureParams,
    inputs={"outline": "json"},
    outputs={"curvature": "json"},
    shard=None,
    description="Windowed bend-angle statistics of one closed outline.",
    func=outline_ops.curvature_features,
)
def _raw_curvature_features(inputs: dict[str, Any], params: CurvatureParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    ring = np.column_stack([inputs["outline"]["x"], inputs["outline"]["y"]])
    return {"curvature": outline_ops.curvature_features(ring, window=params.window)}


class EfdParams(BaseModel):
    """Parameters of the raw ``outline.elliptic_fourier`` op."""

    n_harmonics: int = Field(10, ge=1, description="Number of harmonics")


@register(
    "outline.ellipse_deviation",
    params_model=EmptyParams,
    inputs={"outline": "json"},
    outputs={"value": "json"},
    shard=None,
    description="Mean relative deviation of one outline from its best-fit ellipse.",
    func=outline_ops.ellipse_deviation,
)
def _raw_ellipse_deviation(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    ring = np.column_stack([inputs["outline"]["x"], inputs["outline"]["y"]])
    return {"value": float(outline_ops.ellipse_deviation(ring))}


@register(
    "outline.chord_sagitta_ratio",
    params_model=EmptyParams,
    inputs={"outline": "json"},
    outputs={"value": "json"},
    shard=None,
    description="Max chord-midpoint sagitta of one outline, normalized by width.",
    func=outline_ops.chord_sagitta_ratio,
)
def _raw_chord_sagitta_ratio(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    ring = np.column_stack([inputs["outline"]["x"], inputs["outline"]["y"]])
    return {"value": float(outline_ops.chord_sagitta_ratio(ring))}


@register(
    "outline.shape_descriptors",
    params_model=EmptyParams,
    inputs={"outline": "json"},
    outputs={"shape": "json"},
    shard=None,
    description="Simple-shape descriptors (ellipse_dev, sagitta_ratio) of one outline.",
    func=outline_ops.shape_descriptors,
)
def _raw_shape_descriptors(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    ring = np.column_stack([inputs["outline"]["x"], inputs["outline"]["y"]])
    return {"shape": outline_ops.shape_descriptors(ring)}


@register(
    "outline.elliptic_fourier",
    params_model=EfdParams,
    inputs={"outline": "json"},
    outputs={"coefficients": "array"},
    shard=None,
    description="Kuhl-Giardina-normalized elliptic Fourier descriptors of an outline.",
    func=outline_ops.elliptic_fourier,
)
def _raw_elliptic_fourier(inputs: dict[str, Any], params: EfdParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    ring = np.column_stack([inputs["outline"]["x"], inputs["outline"]["y"]])
    return {"coefficients": outline_ops.elliptic_fourier(ring, n_harmonics=params.n_harmonics)}


@register(
    "outline.fill_cells_geometry",
    params_model=EmptyParams,
    inputs={"cells": "table", "outlines": "json"},
    outputs={"cells": "table"},
    shard=None,
    description="Write outline geometry and curvature scalars into a cells table.",
    func=outline_ops.fill_cells_geometry,
)
def _raw_fill_cells_geometry(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    outlines = _outlines_from_json(inputs["outlines"])
    return {"cells": outline_ops.fill_cells_geometry(inputs["cells"], outlines)}


@register(
    "midline.extract_midlines",
    params_model=MidlineParams,
    inputs={"labels": "array", "outlines": "json?"},
    outputs={"midlines": "json"},
    shard="frame",
    description="Midlines, poles and width profiles of every labelled instance of one frame.",
    func=midline_ops.extract_midlines,
)
def _raw_extract_midlines(inputs: dict[str, Any], params: MidlineParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    outlines = _outlines_from_json(inputs["outlines"]) if inputs.get("outlines") else None
    params_dict = _merged_params(params, exclude=("pixel_microns",))
    pixel_microns = params.pixel_microns if params.pixel_microns is not None else 0.065
    results = midline_ops.extract_midlines(
        inputs["labels"], outlines, params_dict, pixel_microns=pixel_microns
    )
    return {"midlines": _midlines_to_json(results)}


@register(
    "midline.fill_cells_midline",
    params_model=EmptyParams,
    inputs={"cells": "table", "midlines": "json"},
    outputs={"cells": "table"},
    shard=None,
    description="Write midline geometry and scalars into a cells table.",
    func=midline_ops.fill_cells_midline,
)
def _raw_fill_cells_midline(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    results = _midlines_from_json(inputs["midlines"])
    return {"cells": midline_ops.fill_cells_midline(inputs["cells"], results)}


# ---------------------------------------------------------------------------
# raw-op registrations — characterize
# ---------------------------------------------------------------------------


class DemographicParams(BaseModel):
    """Parameters of the raw ``characterize.demographic_projection`` op."""

    n_l: int = Field(64, ge=1, description="Samples along the midline")
    n_d: int = Field(16, ge=1, description="Samples across the cell width")


@register(
    "characterize.demographic_projection",
    params_model=DemographicParams,
    inputs={"cells": "table:cells", "image": "array", "labels": "array"},
    outputs={"projections": "json"},
    shard="frame",
    description="Resample channel intensities into normalized per-cell (l, d) coordinates.",
    func=characterize_ops.demographic_projection,
)
def _raw_demographic_projection(inputs: dict[str, Any], params: DemographicParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    projections = characterize_ops.demographic_projection(
        inputs["cells"], inputs["image"], inputs["labels"], n_l=params.n_l, n_d=params.n_d
    )
    return {"projections": _jsonable(projections)}


@register(
    "characterize.growth_stats",
    params_model=EmptyParams,
    inputs={"tracks": "table:tracks", "cells": "table:cells"},
    outputs={"growth": "json"},
    shard=None,
    description="Per-track growth dynamics plus the population adder/sizer summary.",
    func=characterize_ops.growth_stats,
)
def _raw_growth_stats(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    return {"growth": _jsonable(characterize_ops.growth_stats(inputs["tracks"], inputs["cells"]))}


class SeptumParams(BaseModel):
    """Parameters of the raw ``characterize.detect_septum`` op."""

    min_prominence: float = Field(
        0.05, gt=0, description="Minimum prominence of the width dip (same unit as the profile)"
    )


@register(
    "characterize.detect_septum",
    params_model=SeptumParams,
    inputs={"width_profile": "json"},
    outputs={"septum_pos": "json"},
    shard=None,
    description="Locate a septum as the deepest width dip along the midline.",
    func=characterize_ops.detect_septum,
)
def _raw_detect_septum(inputs: dict[str, Any], params: SeptumParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    pos = characterize_ops.detect_septum(inputs["width_profile"], params.min_prominence)
    return {"septum_pos": {"septum_pos": pos}}


@register(
    "characterize.orientation_coherence",
    params_model=EmptyParams,
    inputs={"labels": "array", "cells": "table:cells"},
    outputs={"coherence": "json"},
    shard="frame",
    description="Local nematic order parameter of cell major axes of one frame.",
    func=characterize_ops.orientation_coherence,
)
def _raw_orientation_coherence(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    value = characterize_ops.orientation_coherence(inputs["labels"], inputs["cells"])
    return {"coherence": {"coherence": float(value)}}


@register(
    "characterize.find_puncta",
    params_model=characterize_ops.PunctaParams,
    inputs={"image": "array", "mask": "array?"},
    outputs={"puncta": "array"},
    shard="frame",
    description="Subpixel point-source (FISH puncta) detection in one channel.",
    func=characterize_ops.find_puncta,
)
def _raw_find_puncta(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"puncta": characterize_ops.find_puncta(inputs["image"], inputs.get("mask"), params)}


@register(
    "characterize.assign_pole_age",
    params_model=EmptyParams,
    inputs={"cells": "table:cells", "edges": "table:edges"},
    outputs={"cells": "table"},
    shard=None,
    description=(
        "Pole-age inheritance across continue/division lineage edges; appends "
        "pole1_age/pole2_age int columns (-1 = unknown) to the cells table."
    ),
    func=characterize_ops.assign_pole_age,
)
def _raw_assign_pole_age(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    return {"cells": characterize_ops.assign_pole_age(inputs["cells"], inputs["edges"])}


@register(
    "characterize.track_demographic_mean",
    params_model=EmptyParams,
    inputs={"projections": "json", "cells": "table:cells"},
    outputs={"means": "table"},
    shard=None,
    description=(
        "Per-track mean of demographic_projection maps; rows of track_id, "
        "n_cells and the flattened mean map."
    ),
    func=characterize_ops.track_demographic_mean,
)
def _raw_track_demographic_mean(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    projections = {
        int(cell_key): np.asarray(proj, dtype=np.float32)
        for cell_key, proj in inputs["projections"].items()
    }
    return {"means": characterize_ops.track_demographic_mean(projections, inputs["cells"])}


class ShapeSpaceParams(BaseModel):
    """Parameters of the raw ``characterize.efd_shape_space`` op."""

    n_harmonics: int = Field(10, ge=1, description="EFD harmonics per outline")
    n_components: int = Field(3, ge=1, description="PCA components to keep")


@register(
    "characterize.efd_shape_space",
    params_model=ShapeSpaceParams,
    inputs={"cells": "table:cells"},
    outputs={"cells": "table", "pca": "json"},
    shard=None,
    description=(
        "PCA shape space over elliptic Fourier descriptors of the per-cell "
        "outlines; appends efd_pc1..n float columns (null without outlines)."
    ),
    func=characterize_ops.efd_shape_space,
)
def _raw_efd_shape_space(inputs: dict[str, Any], params: ShapeSpaceParams, ctx: StepContext) -> dict[str, Any]:
    del ctx
    table, info = characterize_ops.efd_shape_space(
        inputs["cells"],
        n_harmonics=params.n_harmonics,
        n_components=params.n_components,
    )
    return {"cells": table, "pca": _jsonable(info)}


@register(
    "characterize.contact_metrics",
    params_model=EmptyParams,
    inputs={"labels": "array", "cells": "table:cells"},
    outputs={"cells": "table"},
    shard="frame",
    description=(
        "Shared-boundary contact fraction and shared-boundary-weighted "
        "neighbour alignment of one frame; appends contact_fraction and "
        "mean_neighbor_alignment float columns."
    ),
    func=characterize_ops.contact_metrics,
)
def _raw_contact_metrics(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    return {"cells": characterize_ops.contact_metrics(inputs["labels"], inputs["cells"])}


# ---------------------------------------------------------------------------
# raw-op registrations — track
# ---------------------------------------------------------------------------


@register(
    "track.overlap_matrix",
    params_model=EmptyParams,
    inputs={"labels_t": "array", "labels_t1": "array"},
    outputs={"overlap": "json"},
    shard=None,
    description="Shared-pixel counts between two instance-label frames (COO JSON).",
    func=track_ops.overlap_matrix,
)
def _raw_overlap_matrix(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    return {"overlap": _overlap_to_json(track_ops.overlap_matrix(inputs["labels_t"], inputs["labels_t1"]))}


@register(
    "track.link_frames",
    params_model=track_ops.LinkParams,
    inputs={"cells_t": "json", "cells_t1": "json", "overlap": "json"},
    outputs={"edges": "json"},
    shard=None,
    description="Link one frame pair into lineage edge records (LAP + division detection).",
    func=track_ops.link_frames,
)
def _raw_link_frames(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    overlap = _overlap_from_json(inputs["overlap"])
    edges = track_ops.link_frames(inputs["cells_t"], inputs["cells_t1"], overlap, params)
    return {"edges": _jsonable(edges)}


@register(
    "track.link_timelapse",
    params_model=track_ops.TrackParams,
    inputs={"labels": "array", "cells": "table:cells?"},
    outputs={"edges": "table:edges", "tracks": "table:tracks"},
    shard="fov",
    description="Link a (t, y, x) labels stack into edges and tracks tables.",
    func=track_ops.link_timelapse,
)
def _raw_link_timelapse(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    edges, tracks = track_ops.link_timelapse(inputs["labels"], inputs.get("cells"), params)
    return {"edges": edges, "tracks": tracks}


@register(
    "track.refine_lineage",
    params_model=track_ops.RefineParams,
    inputs={"labels": "array", "edges": "table:edges"},
    outputs={"edges": "table:edges"},
    shard="fov",
    description="Correct spurious convergence events by re-splitting suspect cells.",
    func=track_ops.refine_lineage,
)
def _raw_refine_lineage(inputs: dict[str, Any], params: BaseModel, ctx: StepContext) -> dict[str, Any]:
    del ctx
    return {"edges": track_ops.refine_lineage(inputs["labels"], inputs["edges"], None, params)}


@register(
    "track.ctc_metrics",
    params_model=EmptyParams,
    inputs={
        "pred_edges": "json",
        "pred_labels": "array",
        "truth_edges": "json",
        "truth_labels": "array",
    },
    outputs={"metrics": "json"},
    shard=None,
    description="Lightweight Cell-Tracking-Challenge-style benchmark scores.",
    func=track_ops.ctc_metrics,
)
def _raw_ctc_metrics(inputs: dict[str, Any], params: EmptyParams, ctx: StepContext) -> dict[str, Any]:
    del params, ctx
    metrics = track_ops.ctc_metrics(
        inputs["pred_edges"],
        inputs["pred_labels"],
        inputs["truth_edges"],
        inputs["truth_labels"],
    )
    return {"metrics": _jsonable(metrics)}
