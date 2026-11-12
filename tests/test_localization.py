"""Tests for the localization / distribution operator family."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from momia3 import schema
from momia3.keys import encode_key
from momia3.ops import localization as loc


def _capsule_table() -> pa.Table:
    outline = {"x": [5.0, 25.0, 25.0, 5.0, 5.0], "y": [4.0, 4.0, 12.0, 12.0, 4.0]}
    midline = [{"x": [5.0, 15.0, 25.0], "y": [8.0, 8.0, 8.0]}]
    return pa.table(
        {
            "cell_key": pa.array([encode_key(0, 0, 1)], type=pa.uint64()),
            "frame": pa.array([0], type=pa.uint32()),
            "label": pa.array([1], type=pa.uint32()),
            "midline": pa.array([midline], type=schema.MIDLINE_T),
            "outline": pa.array([outline], type=schema.OUTLINE_T),
            "width_profile": pa.array([[8.0, 8.0]], type=pa.list_(pa.float32())),
        }
    )


def _dummy_labels() -> np.ndarray:
    labels = np.zeros((16, 32), dtype=np.uint32)
    labels[4:12, 5:25] = 1
    return labels


def test_rod_coordinates_and_profile() -> None:
    labels = _dummy_labels()
    image = np.broadcast_to(np.arange(32, dtype=np.float64)[None, :], labels.shape).copy()
    table = _capsule_table()
    coords = loc.rod_coordinates(table, labels, n_points=24, pixel_microns=1.0)
    key = encode_key(0, 0, 1)
    assert set(coords) == {key}
    rec = coords[key]
    assert rec["longitudinal"].shape == rec["radial"].shape
    assert np.nanmin(rec["longitudinal"]) >= 0.0
    assert np.nanmax(rec["longitudinal"]) <= 1.0
    assert np.isfinite(rec["radial"]).all()

    profile = loc.rod_profile(table, image, labels, n_bins=16, mode="mean", pixel_microns=1.0)
    assert set(profile) == {key}
    assert profile[key].shape == (16,)
    assert np.isfinite(profile[key]).all()
    assert profile[key][-1] > profile[key][0]


def test_radial_and_compartment_stats() -> None:
    labels = _dummy_labels()
    image = np.zeros_like(labels, dtype=np.float64)
    image[4:12, 5:25] = 7.0
    image[4:7, 5:25] = 11.0
    radial = loc.radial_intensity_distribution(labels, image, n_bins=4)
    assert list(radial) == [1]
    assert radial[1].shape == (4,)
    assert np.isfinite(radial[1]).all()

    comp = np.zeros_like(labels, dtype=bool)
    comp[4:7, 5:25] = True
    stats = loc.compartment_stats(labels, image, comp, "pole")
    assert stats["pole_mean"][0] == pytest.approx(11.0)
    assert stats["pole_area"][0] == 3 * 20
    assert stats["pole_fraction"][0] == pytest.approx((3 * 20) / (8 * 20))


