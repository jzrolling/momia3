"""Shared fixtures for the momia3 test suite."""

from __future__ import annotations

import pyarrow as pa
import pytest

from momia3 import schema
from momia3.keys import encode_key


def build_cells_table(n_rows: int = 2, channels: tuple[str, ...] = ()) -> pa.Table:
    """Build a small cells table with nested geometry, conforming to schema.

    Row 0 carries full geometry; row 1 has null outline/midline/width
    profile to exercise nullability.
    """
    sch = schema.cells_schema(channels)
    full_outline = {"x": [0.0, 10.0, 10.0, 0.0], "y": [0.0, 0.0, 20.0, 20.0]}
    full_midline = [{"x": [0.0, 10.0], "y": [10.0, 10.0]}]
    values: dict[str, list] = {
        "cell_key": [encode_key(0, 0, i + 1) for i in range(n_rows)],
        "fov_id": [0] * n_rows,
        "frame": [0] * n_rows,
        "label": [i + 1 for i in range(n_rows)],
        "track_id": [None] * n_rows,
        "parent_track": [None] * n_rows,
        "touching_edge": [False] * n_rows,
        "include": [True] * n_rows,
        "bbox": [{"x0": 0, "y0": 0, "x1": 10, "y1": 20}] * n_rows,
        "centroid": [{"x": 5.0, "y": 10.0}] * n_rows,
        "outline": [full_outline] + [None] * (n_rows - 1),
        "midline": [full_midline] + [None] * (n_rows - 1),
        "width_profile": [[0.8, 0.8]] + [None] * (n_rows - 1),
        "coords": [{"x": [1, 2], "y": [3, 4]}] * n_rows,
        "embedding": [None] * n_rows,
        "annotation": [None] * n_rows,
    }
    for name in schema.SCALAR_MORPHOLOGY_COLUMNS:
        values[name] = [1.0] * n_rows
    arrays = [
        pa.array(values.get(field.name, [None] * n_rows), type=field.type)
        for field in sch
    ]
    return pa.Table.from_arrays(arrays, schema=sch)


@pytest.fixture
def cells_table() -> pa.Table:
    """Two-row canonical cells table (row 1 has null geometry)."""
    return build_cells_table()
