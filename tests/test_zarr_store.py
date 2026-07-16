"""Tests for momia3.storage.zarr_store — experiment OME-Zarr store."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from momia3.storage.zarr_store import ExperimentStore


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "experiment.zarr"


@pytest.fixture
def store(store_path):
    return ExperimentStore.create(
        store_path,
        pixel_microns=0.065,
        channels=["phase"],
        time_interval_s=60.0,
        provenance={"source": "pytest"},
    )


def _stack(t: int = 4, y: int = 96, x: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 4095, size=(t, y, x), dtype=np.uint16)


def test_attrs(store) -> None:
    assert store.pixel_microns == 0.065
    assert store.channels == ("phase",)
    assert store.time_interval_s == 60.0
    assert store.attrs["provenance"] == {"source": "pytest"}


def test_images_roundtrip(store) -> None:
    stack = _stack()
    arr = store.write_images(0, "phase", stack)
    assert isinstance(arr, zarr.Array)
    assert arr.shape == stack.shape
    assert arr.chunks == (1, 96, 128)  # (1, min(512, y), min(512, x))

    frame = store.get_frame(0, "phase", 2)
    assert frame.shape == (96, 128)
    assert frame.dtype == np.uint16
    np.testing.assert_array_equal(frame, stack[2])
    np.testing.assert_array_equal(np.asarray(store.images(0, "phase")[...]), stack)


def test_lazy_handle(store) -> None:
    store.write_images(0, "phase", _stack())
    handle = store.images(0, "phase")
    assert isinstance(handle, zarr.Array)  # lazy: no data loaded
    assert store.shape(0) == (4, 96, 128)
    assert store.n_frames(0) == 4


def test_channel_auto_registration(store) -> None:
    store.write_images(0, "gfp", _stack())
    assert "gfp" in store.channels


def test_labels_roundtrip(store) -> None:
    labels = np.arange(4 * 96 * 128, dtype=np.uint32).reshape(4, 96, 128)
    store.write_labels(0, labels)
    assert store.labels(0).dtype == np.uint32
    np.testing.assert_array_equal(store.get_labels_frame(0, 3), labels[3])


def test_set_labels_frame_creates_and_resizes(store) -> None:
    lab = np.ones((96, 128), dtype=np.uint32)
    store.set_labels_frame(0, 0, lab)
    assert store.labels(0).shape == (1, 96, 128)
    store.set_labels_frame(0, 3, lab * 2)
    assert store.labels(0).shape == (4, 96, 128)
    np.testing.assert_array_equal(store.get_labels_frame(0, 3), lab * 2)
    np.testing.assert_array_equal(
        store.get_labels_frame(0, 1), np.zeros((96, 128), dtype=np.uint32)
    )


def test_prob_roundtrip(store) -> None:
    rng = np.random.default_rng(1)
    prob = rng.random((4, 2, 96, 128), dtype=np.float32)
    store.write_prob(0, prob)
    arr = store.prob(0)
    assert arr.shape == (4, 2, 96, 128)
    assert arr.dtype == np.float16
    np.testing.assert_allclose(np.asarray(arr[0]), prob[0].astype(np.float16))


def test_reopen(store_path, store) -> None:
    stack = _stack()
    store.write_images(1, "phase", stack)
    reopened = ExperimentStore.open(store_path, mode="r")
    assert reopened.fov_ids == (1,)
    assert reopened.pixel_microns == 0.065
    np.testing.assert_array_equal(reopened.get_frame(1, "phase", 0), stack[0])


def test_errors(store) -> None:
    store.write_images(0, "phase", _stack())
    with pytest.raises(IndexError):
        store.get_frame(0, "phase", 99)
    with pytest.raises(KeyError):
        store.get_frame(0, "missing-channel", 0)
    with pytest.raises(KeyError):
        store.get_frame(7, "phase", 0)
    with pytest.raises(KeyError):
        store.labels(0)
    with pytest.raises(ValueError):
        store.write_images(0, "phase", np.zeros((96, 128), dtype=np.uint16))
