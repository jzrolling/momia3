"""Tests for momia3.storage.tables — parquet IO helpers, partition, merge."""

from __future__ import annotations

import polars as pl

from momia3.storage import tables


def test_partition_by_fov() -> None:
    df = pl.DataFrame(
        {
            "cell_key": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "fov_id": pl.Series([1, 0, 1, 0], dtype=pl.UInt32),
            "length_um": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
        }
    )
    parts = list(tables.partition_by_fov(df))
    assert [fov for fov, _ in parts] == [0, 1]
    assert parts[0][1].height == 2
    assert parts[1][1].height == 2
    assert parts[0][1].get_column("cell_key").to_list() == [2, 4]


def test_partition_by_fov_missing_column() -> None:
    import pytest

    from momia3 import schema

    with pytest.raises(schema.SchemaError):
        list(tables.partition_by_fov(pl.DataFrame({"x": [1]})))


def test_merge_frames_keep_last() -> None:
    a = pl.DataFrame(
        {
            "cell_key": pl.Series([1, 2], dtype=pl.UInt64),
            "length_um": pl.Series([1.0, 1.0], dtype=pl.Float32),
        }
    )
    b = pl.DataFrame(
        {
            "cell_key": pl.Series([2, 3], dtype=pl.UInt64),
            "length_um": pl.Series([2.5, 3.5], dtype=pl.Float32),
        }
    )
    merged = tables.merge_frames(a, b, subset=("cell_key",))
    assert merged.height == 3
    assert merged.get_column("cell_key").to_list() == [1, 2, 3]
    assert merged.get_column("length_um").to_list() == [1.0, 2.5, 3.5]


def test_append_table_creates_then_merges(tmp_path, cells_table) -> None:
    df = pl.from_arrow(cells_table)
    path = tmp_path / "cells.parquet"
    tables.append_table(df, path, "cells")
    first = tables.read_cells(path)
    assert first.height == cells_table.num_rows
    # Appending the same rows again must deduplicate on cell_key.
    tables.append_table(df, path, "cells")
    second = tables.read_cells(path)
    assert second.height == cells_table.num_rows


def test_write_unknown_kind(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown table kind"):
        tables.write_table(pl.DataFrame({"a": [1]}), tmp_path / "x.parquet", "bogus")
