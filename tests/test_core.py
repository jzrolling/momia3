"""Tests for momia3.core — Field, CellSet, lineage views."""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from momia3 import schema
from momia3.core import CellSet, Field, edges_to_nx, nx_to_edges
from momia3.storage.zarr_store import ExperimentStore


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


@pytest.fixture
def field(tmp_path) -> Field:
    store = ExperimentStore.create(
        tmp_path / "exp.zarr", pixel_microns=0.065, channels=["phase"]
    )
    rng = np.random.default_rng(0)
    store.write_images(3, "phase", rng.integers(0, 1000, (3, 64, 64), dtype=np.uint16))
    store.write_labels(3, np.zeros((3, 64, 64), dtype=np.uint32))
    return Field(store=store, fov_id=3)


def test_field_properties(field: Field) -> None:
    assert field.channels == ("phase",)
    assert field.shape == (3, 64, 64)
    assert field.n_frames == 3
    assert field.pixel_microns == 0.065


def test_field_frame(field: Field) -> None:
    frame = field.frame("phase", 1)
    assert frame.shape == (64, 64)
    assert frame.dtype == np.uint16
    np.testing.assert_array_equal(
        frame, np.asarray(field.store.images(3, "phase")[1])
    )
    with pytest.raises(KeyError):
        field.frame("gfp", 0)


def test_field_labels_roundtrip(field: Field) -> None:
    np.testing.assert_array_equal(field.labels(0), np.zeros((64, 64), dtype=np.uint32))
    lab = np.ones((64, 64), dtype=np.uint32) * 7
    field.set_labels(1, lab)
    np.testing.assert_array_equal(field.labels(1), lab)


def test_field_immutable(field: Field) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        field.fov_id = 9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CellSet
# ---------------------------------------------------------------------------


def test_cellset_len_and_scalars(cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    assert len(cs) == cells_table.num_rows
    scalars = cs.scalar_features()
    for geom in schema.GEOMETRY_COLUMNS:
        assert geom not in scalars.columns
    assert "length_um" in scalars.columns
    assert "cell_key" in scalars.columns


def test_cellset_get_geometry(cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    key0 = int(cells_table.column("cell_key")[0])
    geo = cs.get_geometry(key0)
    assert geo.outline is not None
    assert geo.outline.shape == (4, 2)
    np.testing.assert_allclose(geo.outline[:, 0], [0.0, 10.0, 10.0, 0.0])
    assert geo.midline is not None and len(geo.midline) == 1
    assert geo.midline[0].shape == (2, 2)
    np.testing.assert_allclose(geo.width_profile, [0.8, 0.8])
    assert geo.bbox == (0, 0, 10, 20)
    assert geo.centroid == (5.0, 10.0)


def test_cellset_get_geometry_null(cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    key1 = int(cells_table.column("cell_key")[1])
    geo = cs.get_geometry(key1)
    assert geo.outline is None
    assert geo.midline is None
    assert geo.width_profile is None
    assert geo.bbox == (0, 0, 10, 20)


def test_cellset_get_geometry_missing(cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    with pytest.raises(KeyError):
        cs.get_geometry(999)


def test_cellset_filter(cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    assert len(cs.filter(pl.col("label") == 2)) == 1
    assert len(cs.filter(fov_id=0)) == 2
    assert len(cs.filter(pl.col("length_um") > 0.5, fov_id=0)) == 2
    with pytest.raises(KeyError):
        cs.filter(no_such_column=1)


def test_cellset_parquet_roundtrip(tmp_path, cells_table) -> None:
    cs = CellSet(pl.from_arrow(cells_table))
    path = tmp_path / "cells.parquet"
    cs.to_parquet(path)
    back = CellSet.from_parquet(path)
    assert len(back) == len(cs)
    assert back.df.to_arrow().equals(cs.df.to_arrow(), check_metadata=False)


def test_cellset_rejects_bad_schema() -> None:
    bad = pl.DataFrame({"cell_key": pl.Series([1], dtype=pl.UInt64)})
    with pytest.raises(schema.SchemaError):
        CellSet(bad)


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def _edges_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "parent_key": pl.Series([0, 10, 10, 11], dtype=pl.UInt64),
            "child_key": pl.Series([10, 11, 12, 13], dtype=pl.UInt64),
            "event": pl.Series(
                ["appear", "division", "division", "continue"], dtype=pl.Categorical
            ),
            "confidence": pl.Series([1.0, 0.9, 0.8, 1.0], dtype=pl.Float32),
        }
    )


def test_edges_to_nx() -> None:
    graph = edges_to_nx(_edges_df())
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 4
    assert graph[10][11]["event"] == "division"
    assert graph[10][11]["confidence"] == pytest.approx(0.9)
    assert graph[0][10]["event"] == "appear"
    assert set(graph.successors(10)) == {11, 12}


def test_nx_to_edges_roundtrip() -> None:
    df = _edges_df().sort("parent_key", "child_key")
    graph = edges_to_nx(df)
    back = nx_to_edges(graph).sort("parent_key", "child_key")
    assert back.get_column("parent_key").to_list() == df.get_column(
        "parent_key"
    ).to_list()
    assert back.get_column("child_key").to_list() == df.get_column(
        "child_key"
    ).to_list()
    assert back.get_column("event").to_list() == df.get_column("event").to_list()
    assert back.get_column("confidence").to_list() == pytest.approx(
        df.get_column("confidence").to_list()
    )


def test_edges_to_nx_missing_column() -> None:
    with pytest.raises(schema.SchemaError):
        edges_to_nx(pl.DataFrame({"parent_key": [1]}))
