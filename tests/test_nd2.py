"""Tests for the ported ND2 importer (momia2 momia_IO/_io.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from momia3.pipeline import io_nd2
from momia3.pipeline.registry import StepContext
from momia3.storage.zarr_store import ExperimentStore


def _fake_array(sizes: dict[str, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = tuple(sizes.values())
    return rng.random(shape).astype(np.float32)


class _Channel:
    def __init__(self, name: str):
        self.name = name


class _Metadata:
    def __init__(self, names):
        self.channels = [_Channel(n) for n in names]


class _FakeNd2:
    """Minimal stand-in for nd2.ND2File (sizes/to_array/metadata/voxel_size)."""

    def __init__(self, array, sizes, names=("phase", "gfp"), voxel=(1.0, 0.1, 0.1)):
        self._array = np.asarray(array)
        self.sizes = dict(sizes)
        self.metadata = _Metadata(names)
        self._voxel = voxel
        self.closed = False

    def to_array(self):
        return self._array

    def voxel_size(self):
        return self._voxel

    def close(self):
        self.closed = True


def test_plan_from_sizes_positions_frames_channels():
    plan = io_nd2.plan_from_sizes(
        {"P": 3, "T": 5, "Z": 1, "C": 2, "Y": 64, "X": 48}
    )
    assert plan.fov_count == 3
    assert plan.n_frames == 5
    assert plan.n_channels == 2
    assert plan.n_z == 1
    assert plan.shape == (64, 48)


def test_plan_from_sizes_requires_spatial_axes():
    with pytest.raises(ValueError, match="'Y'"):
        io_nd2.plan_from_sizes({"X": 10, "T": 2})


def test_plan_from_sizes_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unsupported ND2 axes"):
        io_nd2.plan_from_sizes({"Y": 8, "X": 8, "S": 3})


def test_stacks_from_array_positions_channels():
    sizes = {"P": 2, "T": 3, "C": 2, "Y": 8, "X": 6}
    array = _fake_array(sizes, seed=1)
    plan = io_nd2.plan_from_sizes(sizes)
    stacks = io_nd2.stacks_from_array(array, plan, ["phase", "gfp"])
    assert sorted(stacks) == [0, 1]
    for pos in (0, 1):
        assert set(stacks[pos]) == {"phase", "gfp"}
        assert stacks[pos]["phase"].shape == (3, 8, 6)
        np.testing.assert_array_equal(stacks[pos]["phase"], array[pos, :, 0])
        np.testing.assert_array_equal(stacks[pos]["gfp"], array[pos, :, 1])


def test_stacks_from_array_z_plane_selection():
    sizes = {"T": 2, "Z": 3, "C": 1, "Y": 4, "X": 4}
    array = _fake_array(sizes, seed=2)
    plan = io_nd2.plan_from_sizes(sizes)
    stacks = io_nd2.stacks_from_array(array, plan, ["phase"], z_plane=2)
    np.testing.assert_array_equal(stacks[0]["phase"], array[:, 2, 0])
    with pytest.raises(ValueError, match="out of range"):
        io_nd2.stacks_from_array(array, plan, ["phase"], z_plane=3)


def test_stacks_from_array_rejects_shape_mismatch():
    plan = io_nd2.plan_from_sizes({"T": 2, "C": 1, "Y": 4, "X": 4})
    with pytest.raises(ValueError, match="does not match sizes"):
        io_nd2.stacks_from_array(np.zeros((2, 1, 5, 5), np.float32), plan, ["ch0"])


def test_stacks_from_array_rejects_channel_count():
    sizes = {"T": 1, "C": 2, "Y": 4, "X": 4}
    plan = io_nd2.plan_from_sizes(sizes)
    with pytest.raises(ValueError, match="channel name"):
        io_nd2.stacks_from_array(_fake_array(sizes), plan, ["only"])


def test_read_nd2_with_injected_opener(tmp_path):
    sizes = {"P": 2, "T": 2, "C": 2, "Y": 8, "X": 6}
    array = _fake_array(sizes, seed=3)
    reader = _FakeNd2(array, sizes)
    src = tmp_path / "sample.nd2"
    src.write_bytes(b"")  # only existence is checked; the opener is injected
    stacks, meta = io_nd2.read_nd2(src, opener=lambda _path: reader)
    assert reader.closed
    assert meta["channel_names"] == ["phase", "gfp"]
    assert meta["n_positions"] == 2
    assert meta["pixel_microns"] == pytest.approx(0.1)
    assert sorted(stacks) == [0, 1]


def test_read_nd2_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_nd2.read_nd2(tmp_path / "absent.nd2")


def test_open_nd2_without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "nd2", None)
    with pytest.raises(ImportError, match="'nd2' package is required"):
        io_nd2._open_nd2(Path("x.nd2"))


def _run_import(tmp_path, array, sizes, **param_kwargs):
    src = tmp_path / "sample.nd2"
    src.write_bytes(b"")
    fake = _FakeNd2(array, sizes)
    io_nd2._open_nd2 = lambda _path: fake  # type: ignore[assignment]
    store_path = tmp_path / "experiment.zarr"
    ctx = StepContext(step_id="import", workspace=tmp_path, outputs={"experiment": store_path})
    params = io_nd2.ImportNd2Params(path=str(src), **param_kwargs)
    return io_nd2.run_import_nd2({}, params, ctx), store_path


def test_run_import_nd2_multiposition_store(tmp_path, monkeypatch):
    sizes = {"P": 2, "T": 3, "C": 2, "Y": 8, "X": 6}
    array = _fake_array(sizes, seed=4)
    result, store_path = _run_import(tmp_path, array, sizes, pixel_microns=0.065)
    assert result["experiment"] == store_path
    store = ExperimentStore.open(store_path, mode="r")
    assert store.pixel_microns == pytest.approx(0.065)
    assert list(store.channels) == ["phase", "gfp"]
    assert store.fov_ids == (0, 1)
    assert store.shape(1, "gfp") == (3, 8, 6)
    np.testing.assert_allclose(
        np.asarray(store.images(1, "phase")[:]), array[1, :, 0]
    )


def test_run_import_nd2_fov_start_and_override_names(tmp_path):
    sizes = {"P": 1, "T": 2, "C": 2, "Y": 4, "X": 4}
    array = _fake_array(sizes, seed=5)
    _result, store_path = _run_import(
        tmp_path,
        array,
        sizes,
        pixel_microns=0.05,
        channels=["bf", "fl"],
        fov_start=7,
    )
    store = ExperimentStore.open(store_path, mode="r")
    assert store.fov_ids == (7,)
    assert list(store.channels) == ["bf", "fl"]


def test_run_import_nd2_requires_calibration(tmp_path):
    sizes = {"T": 2, "C": 1, "Y": 4, "X": 4}
    array = _fake_array(sizes)
    fake = _FakeNd2(array, sizes, voxel=None)
    src = tmp_path / "sample.nd2"
    src.write_bytes(b"")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(io_nd2, "_open_nd2", lambda _path: fake)
    try:
        store_path = tmp_path / "experiment.zarr"
        ctx = StepContext(step_id="import", workspace=tmp_path, outputs={"experiment": store_path})
        params = io_nd2.ImportNd2Params(path=str(src))
        with pytest.raises(ValueError, match="no usable pixel size"):
            io_nd2.run_import_nd2({}, params, ctx)
    finally:
        monkeypatch.undo()
