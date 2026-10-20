"""Tests for the preview sandbox, channel roles and io.import_labels.

Covers the GUI test-tune-commit loop building blocks:
- ``ExperimentStore`` channel-role metadata
- ``io.import_labels`` (pre-segmented mask import)
- ``momia3.gui.preview`` single-step sandbox preview
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

import momia3.ops  # noqa: F401  (registers ops)
import momia3.pipeline.io_ops as io_ops
from momia3.gui.preview import build_sandbox_store, preview_step
from momia3.pipeline.registry import StepContext
from momia3.storage.zarr_store import CHANNEL_ROLES, ExperimentStore


def _make_store(tmp_path: Path, with_labels: bool = False) -> ExperimentStore:
    """One-frame store with two dark rods on a bright background."""
    image = np.full((64, 64), 1000, dtype=np.uint16)
    labels = np.zeros((64, 64), dtype=np.uint32)
    image[10:20, 10:16] = 200  # rod 1 (10 x 6 px)
    labels[10:20, 10:16] = 1
    image[40:52, 30:38] = 200  # rod 2 (12 x 8 px)
    labels[40:52, 30:38] = 2
    store = ExperimentStore.create(
        tmp_path / "experiment.zarr", pixel_microns=0.05, channels=["phase"]
    )
    store.write_images(0, "phase", image[np.newaxis])
    if with_labels:
        store.write_labels(0, labels[np.newaxis])
    return store


# ---------------------------------------------------------------------------
# channel roles
# ---------------------------------------------------------------------------


class TestChannelRoles:
    def test_set_get_clear(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.channel_roles == {}
        store.set_channel_role("phase", "brightfield")
        assert store.channel_roles == {"phase": "brightfield"}
        assert store.channels_by_role("brightfield") == ("phase",)
        assert store.channels_by_role("fluorescence") == ()
        store.set_channel_role("phase", None)
        assert store.channel_roles == {}

    def test_invalid_role_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="invalid role"):
            store.set_channel_role("phase", "nucleus")

    def test_unknown_channel_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(KeyError):
            store.set_channel_role("gfp", "fluorescence")

    def test_roles_roundtrip_reopen(self, tmp_path):
        store = _make_store(tmp_path)
        store.set_channel_role("phase", "brightfield")
        reopened = ExperimentStore.open(store.path, mode="r")
        assert reopened.channel_roles == {"phase": "brightfield"}

    def test_all_roles_accepted(self, tmp_path):
        store = _make_store(tmp_path)
        for role in CHANNEL_ROLES:
            store.set_channel_role("phase", role)
            assert store.channel_roles["phase"] == role


# ---------------------------------------------------------------------------
# io.import_labels
# ---------------------------------------------------------------------------


class TestImportLabels:
    def _run(self, store: ExperimentStore, mask_path: Path) -> None:
        params = io_ops.ImportLabelsParams(path=str(mask_path), fov_id=0)
        ctx = StepContext(
            step_id="import_labels",
            workspace=store.path.parent,
            outputs={"experiment": store.path},
        )
        io_ops.run_import_labels({"experiment": store.path}, params, ctx)

    def test_import_2d_mask(self, tmp_path):
        store = _make_store(tmp_path)
        mask = np.zeros((64, 64), dtype=np.uint16)
        mask[5:9, 5:9] = 7
        mask_path = tmp_path / "mask.tif"
        tifffile.imwrite(str(mask_path), mask)
        self._run(store, mask_path)
        out = np.asarray(store.labels(0))
        assert out.shape == (1, 64, 64)
        assert out.dtype == np.uint32
        assert int(out.max()) == 7

    def test_import_stack(self, tmp_path):
        store = _make_store(tmp_path)
        mask = np.zeros((1, 64, 64), dtype=np.int32)
        mask[0, 5:9, 5:9] = 3
        mask_path = tmp_path / "mask_stack.tif"
        tifffile.imwrite(str(mask_path), mask)
        self._run(store, mask_path)
        assert int(np.asarray(store.labels(0)).max()) == 3

    def test_shape_mismatch_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        mask = np.zeros((32, 32), dtype=np.uint16)
        mask_path = tmp_path / "bad.tif"
        tifffile.imwrite(str(mask_path), mask)
        with pytest.raises(ValueError, match="does not match"):
            self._run(store, mask_path)

    def test_negative_values_rejected(self, tmp_path):
        store = _make_store(tmp_path)
        mask = np.full((64, 64), -1, dtype=np.int32)
        mask_path = tmp_path / "neg.tif"
        tifffile.imwrite(str(mask_path), mask)
        with pytest.raises(ValueError, match="negative"):
            self._run(store, mask_path)

    def test_missing_file(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            self._run(store, tmp_path / "nope.tif")


# ---------------------------------------------------------------------------
# preview sandbox
# ---------------------------------------------------------------------------


class TestPreviewSandbox:
    def test_sandbox_store_copies_one_frame(self, tmp_path):
        store = _make_store(tmp_path, with_labels=True)
        store.set_channel_role("phase", "brightfield")
        sandbox_path = build_sandbox_store(store, 0, 0, tmp_path / "sb")
        sandbox = ExperimentStore.open(sandbox_path, mode="r")
        assert sandbox.channels == ("phase",)
        assert sandbox.shape(0, "phase") == (1, 64, 64)
        assert np.asarray(sandbox.labels(0)).shape == (1, 64, 64)
        assert sandbox.pixel_microns == pytest.approx(0.05)
        assert sandbox.channel_roles == {"phase": "brightfield"}

    def test_preview_segmentation_does_not_touch_source(self, tmp_path):
        store = _make_store(tmp_path, with_labels=False)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step(
                "segment.threshold",
                {"channel": "phase", "min_size": 10},
                store,
                0,
                0,
                Path(td),
            )
            assert result.sandbox_store is not None
            sandbox = ExperimentStore.open(result.sandbox_store, mode="r")
            labels = np.asarray(sandbox.labels(0))
            assert labels.max() >= 2  # both rods found in the sandbox
        # source store still has no labels
        with pytest.raises(KeyError):
            store.labels(0)

    def test_preview_measure_cells_returns_table(self, tmp_path):
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step("measure.cells", {}, store, 0, 0, Path(td))
            df = result.table("cells")
            assert df.height == 2
            assert "label" in df.columns

    def test_preview_outline_extract_implicit_cells(self, tmp_path):
        """table:cells input is auto-produced by an implicit measure.cells."""
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step(
                "outline.extract", {"n_points": 32}, store, 0, 0, Path(td)
            )
            df = result.table("cells")
            assert df.height == 2
            assert "outline" in df.columns
            assert df["outline"][0] is not None

    def test_preview_midline_extract_implicit_cells(self, tmp_path):
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step(
                "midline.extract", {"n_points": 16}, store, 0, 0, Path(td)
            )
            df = result.table("cells")
            assert "midline" in df.columns

    def test_preview_unsupported_input_kind_rejected(self, tmp_path):
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="preview does not support"):
                preview_step(
                    "track.link_frames", {}, store, 0, 0, Path(td)
                )

    def test_preview_array_channel_input(self, tmp_path):
        """array 'image' inputs are fed from the selected channel frame."""
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step("preprocess.denoise", {}, store, 0, 0, Path(td))
            assert result.outputs["image"].shape == (64, 64)
            assert result.outputs["image"].dtype == np.float32

    def test_preview_array_labels_and_channel_input(self, tmp_path):
        """array 'labels'/'intensity' inputs map to labels + channel frames."""
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            result = preview_step(
                "outline.extract_outlines", {}, store, 0, 0, Path(td)
            )
            outlines = result.outputs["outlines"]
            assert len(outlines) >= 2  # one closed outline per rod

    def test_preview_intermediate_array_rejected(self, tmp_path):
        """Arrays that are not store frames (dist/prob/patches) are refused."""
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            # mask comes from labels; dist is an intermediate array.
            with pytest.raises(ValueError, match="cannot produce the array input"):
                preview_step("segment.declump", {}, store, 0, 0, Path(td))

    def test_preview_param_validation_applies(self, tmp_path):
        store = _make_store(tmp_path, with_labels=True)
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(Exception):
                preview_step(
                    "segment.threshold",
                    {"min_size": -5},  # invalid per the params model
                    store,
                    0,
                    0,
                    Path(td),
                )
