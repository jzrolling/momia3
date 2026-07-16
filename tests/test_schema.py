"""Tests for momia3.schema — Arrow schema definitions and validation."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import pytest

from momia3 import schema
from momia3.storage import tables


def test_empty_cells_columns() -> None:
    table = schema.empty_cells_table()
    assert table.num_rows == 0
    for name in (
        schema.IDENTITY_COLUMNS
        + schema.GEOMETRY_COLUMNS
        + schema.SCALAR_MORPHOLOGY_COLUMNS
        + ("embedding", "annotation")
    ):
        assert name in table.column_names, name
    assert table.schema.field("outline").type == schema.OUTLINE_T
    assert table.schema.field("midline").type == schema.MIDLINE_T
    assert table.schema.field("bbox").type == schema.BBOX_T
    assert table.schema.field("cell_key").type == pa.uint64()
    assert table.schema.field("track_id").type == pa.uint32()


def test_scalar_column_metadata() -> None:
    sch = schema.cells_schema()
    length = sch.field("length_um")
    assert length.metadata[b"unit"] == b"um"
    assert length.metadata[b"semantic"] == b"morphology.length"
    area = sch.field("area")
    assert area.metadata[b"unit"] == b"px^2"
    pole = sch.field("pole1_x")
    assert pole.metadata[b"unit"] == b"px"
    assert sch.metadata[b"schema_version"] == schema.SCHEMA_VERSION.encode()


def test_cells_with_channels() -> None:
    table = schema.empty_cells_table(channels=["gfp", "rfp"])
    for ch in ("gfp", "rfp"):
        for stat in schema.INTENSITY_STATS:
            assert f"{ch}_{stat}" in table.column_names
        assert f"puncta_{ch}" in table.column_names
    gfp_mean = table.schema.field("gfp_mean")
    assert gfp_mean.metadata[b"channel"] == b"gfp"


def test_empty_edges_tracks() -> None:
    edges = schema.empty_edges_table()
    assert edges.num_rows == 0
    assert edges.schema.field("parent_key").type == pa.uint64()
    assert edges.schema.field("event").type == schema.EVENT_T
    assert edges.schema.field("confidence").type == pa.float32()
    tracks = schema.empty_tracks_table()
    assert tracks.num_rows == 0
    assert tracks.schema.field("fate").type == schema.FATE_T
    assert tracks.schema.field("division_frame").type == pa.uint32()


def test_validate_missing_column() -> None:
    bad = pa.table({"cell_key": pa.array([1], type=pa.uint64())})
    with pytest.raises(schema.SchemaError, match="missing required columns"):
        schema.validate_cells(bad)
    with pytest.raises(schema.SchemaError):
        schema.validate_edges(bad)


def test_validate_wrong_type(cells_table) -> None:
    idx = cells_table.schema.get_field_index("cell_key")
    bad = cells_table.set_column(
        idx,
        "cell_key",
        pa.chunked_array([["not-a-key"] * cells_table.num_rows]),
    )
    with pytest.raises(schema.SchemaError):
        schema.validate_cells(bad)


def test_validate_type_error() -> None:
    with pytest.raises(TypeError):
        schema.validate_cells("not a table")


def test_cells_parquet_roundtrip(tmp_path, cells_table) -> None:
    validated = schema.validate_cells(cells_table)
    assert validated.equals(cells_table, check_metadata=True)

    path = tmp_path / "cells.parquet"
    tables.write_cells(pl.from_arrow(cells_table), path)

    stored = pq.read_table(path)
    assert stored.schema.field("length_um").metadata[b"unit"] == b"um"
    assert stored.schema.field("outline").type == schema.OUTLINE_T

    back = tables.read_cells(path)
    assert back.height == cells_table.num_rows
    rt = schema.validate_cells(back.to_arrow())
    assert rt.equals(cells_table, check_metadata=True)


def test_edges_parquet_roundtrip(tmp_path) -> None:
    df = pl.DataFrame(
        {
            "parent_key": pl.Series([0, 10, 10], dtype=pl.UInt64),
            "child_key": pl.Series([10, 11, 12], dtype=pl.UInt64),
            "event": pl.Series(
                ["appear", "division", "division"], dtype=pl.Categorical
            ),
            "confidence": pl.Series([1.0, 0.9, 0.8], dtype=pl.Float32),
        }
    )
    path = tmp_path / "edges.parquet"
    tables.write_edges(df, path)

    stored = pq.read_table(path)
    assert pa.types.is_dictionary(stored.schema.field("event").type)

    back = tables.read_edges(path)
    assert back.sort("child_key").rows() == df.sort("child_key").rows()


def test_tracks_parquet_roundtrip(tmp_path) -> None:
    df = pl.DataFrame(
        {
            "track_id": pl.Series([1, 2], dtype=pl.UInt32),
            "fov_id": pl.Series([0, 0], dtype=pl.UInt32),
            "birth_frame": pl.Series([0, 5], dtype=pl.UInt32),
            "last_frame": pl.Series([9, 9], dtype=pl.UInt32),
            "division_frame": pl.Series([9, None], dtype=pl.UInt32),
            "parent_track": pl.Series([None, 1], dtype=pl.UInt32),
            "n_frames": pl.Series([10, 5], dtype=pl.UInt32),
            "fate": pl.Series(["divide", "edge"], dtype=pl.Categorical),
        }
    )
    path = tmp_path / "tracks.parquet"
    tables.write_tracks(df, path)
    back = tables.read_tracks(path)
    assert back.sort("track_id").rows() == df.sort("track_id").rows()
