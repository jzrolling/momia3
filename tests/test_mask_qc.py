"""Tests for mask-ingress QC."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops import segment
from momia3.storage import ExperimentStore


def _clean_stack() -> np.ndarray:
    labels = np.zeros((2, 32, 32), dtype=np.uint32)
    labels[:, 5:12, 5:12] = 1
    labels[:, 18:26, 18:28] = 2
    return labels


def test_mask_qc_clean_stack() -> None:
    report = segment.mask_qc(_clean_stack(), min_area=10)
    assert report["ok"] is True
    assert report["n_frames"] == 2
    assert report["total_cells"] == 4
    for frame in report["frames"]:
        assert frame["n_cells"] == 2
        assert frame["empty"] is False
        assert frame["missing_label_count"] == 0
        assert frame["border_object_count"] == 0
        assert frame["area_min"] == 49


def test_mask_qc_reports_defects() -> None:
    labels = np.zeros((2, 24, 24), dtype=np.uint32)
    labels[0, 5:12, 5:12] = 1
    labels[1, 2:5, 2:5] = 3       # label 2 missing
    labels[1, 0:2, 10:14] = 4     # border touching
    labels[1, 10, 10] = 5         # small object
    report = segment.mask_qc(labels, min_area=10, max_area=1000, border_width=1)
    assert report["ok"] is False
    issues = report["issues"]
    assert any("empty_frame" in item for item in issues) is False  # both frames non-empty
    assert any("non_contiguous_label_ids" in item for item in issues)
    assert any("small_objects" in item for item in issues)
    assert any("border_touching_objects" in item for item in issues)
    frame1 = report["frames"][1]
    assert frame1["missing_label_ids"] == [1, 2]
    assert 4 in frame1["border_label_ids"]
    assert 5 in frame1["small_label_ids"]


def test_mask_qc_empty_frame_and_validation() -> None:
    labels = np.zeros((2, 16, 16), dtype=np.uint32)
    labels[0, 2:6, 2:6] = 1
    report = segment.mask_qc(labels)
    assert report["frames"][1]["empty"] is True
    assert any("empty_frame" in item for item in report["issues"])

    with pytest.raises(ValueError, match="2-D or 3-D"):
        segment.mask_qc(np.zeros((2, 2, 8, 8), dtype=np.uint32))
    with pytest.raises(ValueError, match="integer"):
        segment.mask_qc(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="max_area"):
        segment.mask_qc(labels, min_area=10, max_area=5)


def test_mask_qc_store_runner(tmp_path) -> None:
    store = ExperimentStore.create(
        tmp_path / "experiment.zarr",
        pixel_microns=0.065,
        channels=["phase"],
    )
    store.write_images(0, "phase", np.zeros((2, 24, 24), dtype=np.uint16))
    store.write_labels(0, _clean_stack()[:, :24, :24])
    from momia3.pipeline import steps

    out = steps.run_mask_qc_store(
        {"experiment": store.path},
        steps.MaskQCParams(min_area=10),
        None,
    )
    assert "0" in out["report"]
    assert out["report"]["0"]["n_frames"] == 2
