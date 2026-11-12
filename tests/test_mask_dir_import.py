"""Tests for the pre-segmented mask-directory importer (P0 mask ingress)."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from momia3.pipeline import io_ops
from momia3.storage.zarr_store import ExperimentStore


def _store(tmp_path, n_frames: int = 3, shape: tuple[int, int] = (24, 32)):
    store = ExperimentStore.create(
        tmp_path / "experiment.zarr",
        pixel_microns=0.065,
        channels=["phase"],
    )
    store.write_images(0, "phase", np.zeros((n_frames, *shape), dtype=np.uint16))
    return store


def _mask(shape: tuple[int, int], y0: int, x0: int, label: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint32)
    mask[y0:y0 + 6, x0:x0 + 8] = label
    return mask


def test_import_mask_dir_natural_sort_and_values(tmp_path) -> None:
    shape = (24, 32)
    store = _store(tmp_path, n_frames=3, shape=shape)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    expected = []
    for frame, (y0, x0) in enumerate([(2, 2), (8, 8), (14, 16)]):
        mask = _mask(shape, y0, x0, label=frame + 1)
        expected.append(mask)
        # Deliberately non-lexicographic names: natural sort must give 1,2,10.
        name = ["mask_1.tif", "mask_2.tif", "mask_10.tif"][frame]
        tifffile.imwrite(mask_dir / name, mask)

    out = io_ops.run_import_mask_dir(
        {"experiment": store.path},
        io_ops.ImportMaskDirParams(path=str(mask_dir)),
        None,
    )
    assert out["experiment"] == store.path
    reopened = ExperimentStore.open(store.path, mode="r")
    stack = np.stack([reopened.get_labels_frame(0, t) for t in range(3)])
    np.testing.assert_array_equal(stack, np.stack(expected))
    assert reopened.get_labels_frame(0, 0).dtype == np.uint32


def test_import_mask_dir_accepts_npy_and_singleton_axis(tmp_path) -> None:
    shape = (16, 20)
    store = _store(tmp_path, n_frames=2, shape=shape)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(mask_dir / "a.tif", _mask(shape, 2, 2, 1)[None])
    np.save(mask_dir / "b.npy", _mask(shape, 6, 8, 2))

    io_ops.run_import_mask_dir(
        {"experiment": store.path},
        io_ops.ImportMaskDirParams(path=str(mask_dir), pattern="*.*"),
        None,
    )
    reopened = ExperimentStore.open(store.path, mode="r")
    assert int(reopened.get_labels_frame(0, 0).max()) == 1
    assert int(reopened.get_labels_frame(0, 1).max()) == 2


def test_import_mask_dir_rejects_count_and_shape_mismatch(tmp_path) -> None:
    shape = (24, 32)
    store = _store(tmp_path, n_frames=3, shape=shape)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(mask_dir / "only.tif", _mask(shape, 2, 2))
    with pytest.raises(ValueError, match="one mask per frame"):
        io_ops.run_import_mask_dir(
            {"experiment": store.path},
            io_ops.ImportMaskDirParams(path=str(mask_dir)),
            None,
        )

    store.write_images(0, "phase", np.zeros((1, *shape), dtype=np.uint16))
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    tifffile.imwrite(bad_dir / "m.tif", np.zeros((8, 8), dtype=np.uint32))
    with pytest.raises(ValueError, match="does not match image shape"):
        io_ops.run_import_mask_dir(
            {"experiment": store.path},
            io_ops.ImportMaskDirParams(path=str(bad_dir)),
            None,
        )


def test_import_mask_dir_missing_directory(tmp_path) -> None:
    store = _store(tmp_path, n_frames=1)
    with pytest.raises(FileNotFoundError, match="mask directory"):
        io_ops.run_import_mask_dir(
            {"experiment": store.path},
            io_ops.ImportMaskDirParams(path=str(tmp_path / "nope")),
            None,
        )



def test_import_mask_dir_multi_fov_regex(tmp_path) -> None:
    import re

    shape = (16, 20)
    store = ExperimentStore.create(
        tmp_path / "experiment.zarr",
        pixel_microns=0.065,
        channels=["phase"],
    )
    store.write_images(0, "phase", np.zeros((2, *shape), dtype=np.uint16))
    store.write_images(1, "phase", np.zeros((2, *shape), dtype=np.uint16))
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    expected = {}
    for fov in (0, 1):
        expected[fov] = []
        for frame in (0, 1):
            mask = _mask(shape, 2 + frame * 4, 2 + fov * 4, label=fov + 1)
            expected[fov].append(mask)
            tifffile.imwrite(mask_dir / f"fov{fov}_t{frame}.tif", mask)

    io_ops.run_import_mask_dir(
        {"experiment": store.path},
        io_ops.ImportMaskDirParams(
            path=str(mask_dir),
            pattern="*.tif",
            filename_regex=r"fov(?P<fov>\d+)_t(?P<frame>\d+)\.tif",
        ),
        None,
    )
    reopened = ExperimentStore.open(store.path, mode="r")
    for fov in (0, 1):
        stack = np.stack([reopened.get_labels_frame(fov, t) for t in (0, 1)])
        np.testing.assert_array_equal(stack, np.stack(expected[fov]))


def test_import_mask_dir_multi_fov_missing_frame(tmp_path) -> None:
    shape = (16, 20)
    store = ExperimentStore.create(
        tmp_path / "experiment.zarr",
        pixel_microns=0.065,
        channels=["phase"],
    )
    store.write_images(0, "phase", np.zeros((2, *shape), dtype=np.uint16))
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    tifffile.imwrite(mask_dir / "fov0_t0.tif", _mask(shape, 2, 2))
    with pytest.raises(ValueError, match="do not cover"):
        io_ops.run_import_mask_dir(
            {"experiment": store.path},
            io_ops.ImportMaskDirParams(
                path=str(mask_dir),
                pattern="*.tif",
                filename_regex=r"fov(?P<fov>\d+)_t(?P<frame>\d+)\.tif",
            ),
            None,
        )
