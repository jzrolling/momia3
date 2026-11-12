"""Tests for per-cell colocalization."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops import measure


def _labels() -> np.ndarray:
    labels = np.zeros((24, 24), dtype=np.uint32)
    labels[3:11, 3:11] = 1
    labels[13:21, 13:21] = 2
    return labels


def test_colocalization_correlated_and_anticorrelated() -> None:
    labels = _labels()
    rng = np.random.default_rng(4)
    a = rng.normal(100, 20, labels.shape)
    b = 2.0 * a + rng.normal(0, 1, labels.shape)
    out = measure.colocalization(labels, a, b, prefix_a="phase", prefix_b="gfp")
    assert out["phase_gfp_pearson"].shape == (2,)
    assert np.all(out["phase_gfp_pearson"] > 0.99)
    assert np.all(out["phase_gfp_spearman"] > 0.99)
    assert np.all(out["phase_gfp_overlap"] > 0.99)

    anti = measure.colocalization(labels, a, -b, prefix_a="phase", prefix_b="gfp")
    assert np.all(anti["phase_gfp_pearson"] < -0.99)


def test_colocalization_manders_thresholds() -> None:
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[2:6, 2:6] = 1
    a = np.zeros((10, 10), dtype=np.float64)
    b = np.zeros((10, 10), dtype=np.float64)
    a[2:6, 2:6] = 10.0
    b[2:6, 2:6] = 10.0
    out = measure.colocalization(
        labels, a, b, prefix_a="a", prefix_b="b",
        threshold_a=5.0, threshold_b=5.0,
    )
    assert np.allclose(out["a_b_manders_a"], 1.0)
    assert np.allclose(out["a_b_manders_b"], 1.0)


def test_colocalization_degenerate_and_validation() -> None:
    labels = np.zeros((8, 8), dtype=np.uint32)
    labels[1:4, 1:4] = 1
    constant = np.ones((8, 8), dtype=np.float64)
    out = measure.colocalization(labels, constant, constant)
    assert np.isnan(out["A_B_pearson"][0])
    assert np.isnan(out["A_B_spearman"][0])

    empty = measure.colocalization(np.zeros((8, 8), dtype=np.uint32), constant, constant)
    assert empty["A_B_pearson"].shape == (0,)
    with pytest.raises(ValueError, match="share shape"):
        measure.colocalization(labels, constant, np.ones((8, 9)))


def test_colocalization_shared_batch_is_equivalent() -> None:
    from momia3.ops._batch import FrameBatch

    labels = _labels()
    rng = np.random.default_rng(5)
    a = rng.normal(100, 20, labels.shape)
    b = rng.normal(50, 10, labels.shape)
    expected = measure.colocalization(labels, a, b, "phase", "gfp")
    shared = measure.colocalization(
        labels, a, b, "phase", "gfp", batch=FrameBatch(labels)
    )
    assert set(shared) == set(expected)
    for key in expected:
        assert np.array_equal(expected[key], shared[key], equal_nan=True)
    with pytest.raises(ValueError, match="batch"):
        measure.colocalization(labels[:8], a[:8], b[:8], batch=FrameBatch(labels))


def test_colocalization_append_runner(tmp_path) -> None:
    from momia3.pipeline import steps
    from momia3.storage import ExperimentStore

    store = ExperimentStore.create(
        tmp_path / "experiment.zarr",
        pixel_microns=0.065,
        channels=["phase", "gfp"],
    )
    labels = np.zeros((1, 24, 24), dtype=np.uint32)
    labels[0, 4:12, 4:12] = 1
    a = np.zeros((1, 24, 24), dtype=np.uint16)
    b = np.zeros((1, 24, 24), dtype=np.uint16)
    a[0, 4:12, 4:12] = 100
    b[0, 4:12, 4:12] = 200
    store.write_images(0, "phase", a)
    store.write_images(0, "gfp", b)
    store.write_labels(0, labels)
    cells = measure.labels_to_cells(labels[0], 0, 0, pixel_microns=0.065)
    out = steps.run_colocalization_append(
        {"experiment": store.path, "cells": cells},
        steps.ColocalizationAppendParams(channel_a="phase", channel_b="gfp"),
        None,
    )
    cols = out["cells"].column_names
    assert "phase_gfp_pearson" in cols
    assert "phase_gfp_spearman" in cols
    assert "phase_gfp_manders_a" in cols
