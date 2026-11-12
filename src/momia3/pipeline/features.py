"""Aggregated cell-feature pipeline step (ARCHITECTURE.md §4.3–§4.6).

Registers ``measure.cell_features``: one frame-sharded step that computes
every requested feature group through
:func:`momia3.ops.features.cell_features` and appends the resulting
columns to the canonical cells table, keyed by ``cell_key``. Computation
is per frame so the shared :class:`~momia3.ops._batch.FrameBatch` and
:class:`~momia3.ops.rod_frame.RodFrame` artifacts are built once and
reused across groups.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

from momia3 import keys as _keys
from momia3 import schema as _schema
from momia3.ops import features as _features
from momia3.pipeline.registry import StepContext, register
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = ["CellFeaturesParams", "run_cell_features"]

FeatureGroup = Literal["shape", "geometry", "intensity", "texture", "coloc", "puncta"]


class CellFeaturesParams(BaseModel):
    """Parameters of ``measure.cell_features``."""

    groups: list[FeatureGroup] = Field(
        default_factory=lambda: list(_features.FEATURE_GROUPS),
        description="Feature groups to compute",
    )
    channels: list[str] | None = Field(
        None, description="Channels to measure; None uses every store channel"
    )
    pixel_microns: float | None = Field(
        None, gt=0, description="Physical pixel size; None reads the store attrs"
    )
    n_midline_points: int = Field(
        32, ge=4, description="Rod-frame resampling count"
    )
    texture_levels: int = Field(64, ge=2, description="GLCM quantization levels")


def _frame_rows(table: pa.Table, fov_id: int, frame: int) -> tuple[pa.Table, np.ndarray]:
    """Rows of ``table`` for one (fov, frame), plus their indices."""
    # pyarrow's ``to_numpy`` imports pandas lazily; ``to_pylist`` on these
    # non-nullable integer columns avoids that one-off cost.
    fov = np.asarray(table.column("fov_id").to_pylist(), dtype=np.int64)
    t = np.asarray(table.column("frame").to_pylist(), dtype=np.int64)
    mask = (fov == fov_id) & (t == frame)
    rows = np.flatnonzero(mask)
    return table.take(pa.array(rows)), rows


def _upsert_columns(
    table: pa.Table, columns: dict[str, np.ndarray]
) -> pa.Table:
    """Replace or append float32 feature columns."""
    for name, values in columns.items():
        arr = pa.array(np.asarray(values, dtype=np.float32), type=pa.float32())
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, arr)
        else:
            table = table.append_column(name, arr)
    return table


@register(
    "measure.cell_features",
    params_model=CellFeaturesParams,
    inputs={"experiment": "store", "cells": "table:cells"},
    outputs={"cells": "table:cells"},
    shard="frame",
    description=(
        "Aggregated per-cell features (shape/geometry/intensity/texture/"
        "coloc/puncta) appended to the cells table; one FrameBatch and "
        "RodFrame per frame, shared across all groups."
    ),
    func=_features.cell_features,
)
def run_cell_features(
    inputs: dict[str, Any], params: CellFeaturesParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``measure.cell_features`` (see registry description)."""
    store_path = Path(inputs["experiment"])
    store = ExperimentStore.open(store_path, mode="r")
    channels = list(params.channels or store.channels)
    pixel_microns = params.pixel_microns or store.pixel_microns
    cells = inputs["cells"]
    if not isinstance(cells, pa.Table):
        cells = pa.Table.from_pylist(list(cells))
    table = cells
    row_of_key = {
        int(k): i
        for i, k in enumerate(table.column("cell_key").to_pylist())
    }
    groups = tuple(params.groups)
    for fov_id in store.fov_ids:
        for t in range(store.n_frames(fov_id)):
            labels = store.get_labels_frame(fov_id, t)
            intensity = {
                name: store.get_frame(fov_id, name, t) for name in channels
            }
            sub, rows = _frame_rows(table, fov_id, t)
            frame_labels = np.unique(labels)
            frame_labels = frame_labels[frame_labels > 0]
            if frame_labels.size == 0:
                continue
            computed = _features.cell_features(
                labels,
                intensity,
                sub,
                groups=groups,
                pixel_microns=float(pixel_microns),
                n_midline_points=int(params.n_midline_points),
                texture_levels=int(params.texture_levels),
            )
            keys = _keys.encode_keys(
                int(fov_id), int(t), frame_labels.astype(np.int64)
            )
            target_rows = np.array([row_of_key[int(k)] for k in keys])
            columns = {
                name: _distribute(values, target_rows, table.num_rows)
                for name, values in computed.items()
            }
            table = _upsert_columns(table, columns)
    logger.info(
        "measure.cell_features: %d rows, %d groups", table.num_rows, len(groups)
    )
    return {"cells": _schema.validate_cells(table)}


def _distribute(
    values: np.ndarray, target_rows: np.ndarray, n_rows: int
) -> np.ndarray:
    """Place per-frame values into a full-length NaN-filled column."""
    out = np.full(n_rows, np.nan, dtype=np.float32)
    out[target_rows] = np.asarray(values, dtype=np.float32)
    return out
