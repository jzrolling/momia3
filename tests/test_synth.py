"""Tests for tests/synth.py — synthetic rod-bacteria timelapse generator."""

from __future__ import annotations

import numpy as np

from momia3.keys import decode_key
from synth import generate_timelapse

PARAMS = dict(
    n_frames=10,
    shape=(160, 160),
    n_initial=4,
    pixel_microns=0.13,
    growth_rate_um_per_frame=0.1,
    division_length_um=2.0,
    drift_px_per_frame=0.3,
    noise_sigma=0.02,
    seed=7,
)


def test_shapes_and_dtypes() -> None:
    out = generate_timelapse(**PARAMS)
    assert out["images"].shape == (10, 160, 160)
    assert out["images"].dtype == np.float32
    assert out["labels"].shape == (10, 160, 160)
    assert out["labels"].dtype == np.uint32
    assert float(out["images"].min()) >= 0.0
    assert float(out["images"].max()) <= 1.0
    assert isinstance(out["edges"], list)
    assert isinstance(out["truth"], list)
    assert out["meta"]["seed"] == 7


def test_determinism() -> None:
    a = generate_timelapse(**PARAMS)
    b = generate_timelapse(**PARAMS)
    np.testing.assert_array_equal(a["images"], b["images"])
    np.testing.assert_array_equal(a["labels"], b["labels"])
    assert a["edges"] == b["edges"]
    assert a["truth"] == b["truth"]

    c = generate_timelapse(**{**PARAMS, "seed": 8})
    assert not np.array_equal(a["images"], c["images"])


def test_labels_match_image_foreground() -> None:
    out = generate_timelapse(**PARAMS)
    images, labels = out["images"], out["labels"]
    foreground = labels > 0
    fraction = float(foreground.mean())
    assert 0.001 < fraction < 0.5
    # Rods are dark on bright background.
    assert float(images[foreground].mean()) < float(images[~foreground].mean())
    # Thresholded dark pixels should largely coincide with the label mask.
    dark = images < 0.65
    intersection = int((dark & foreground).sum())
    dice = 2.0 * intersection / (int(dark.sum()) + int(foreground.sum()))
    assert dice > 0.8


def test_edges_reference_existing_cells() -> None:
    out = generate_timelapse(**PARAMS)
    labels = out["labels"]
    events = [event for _, _, event in out["edges"]]
    assert "division" in events, "expected at least one division in 10 frames"
    assert "continue" in events
    assert events.count("appear") == PARAMS["n_initial"]

    for parent_key, child_key, event in out["edges"]:
        p_fov, p_frame, p_label = decode_key(parent_key)
        c_fov, c_frame, c_label = decode_key(child_key)
        assert p_fov == 0 and c_fov == 0
        assert event in ("appear", "continue", "division")
        if event == "appear":
            assert parent_key == 0
            assert c_frame == 0
        else:
            assert p_frame == c_frame - 1
            assert p_label > 0
            assert bool((labels[p_frame] == p_label).any()), (
                f"parent label {p_label} missing at frame {p_frame}"
            )
        assert c_label > 0
        assert bool((labels[c_frame] == c_label).any()), (
            f"child label {c_label} missing at frame {c_frame}"
        )


def test_division_conserves_cell_count() -> None:
    out = generate_timelapse(**PARAMS)
    n_divisions = sum(1 for _, _, e in out["edges"] if e == "division") // 2
    final_labels = np.unique(out["labels"][-1])
    final_labels = final_labels[final_labels > 0]
    assert len(final_labels) == PARAMS["n_initial"] + n_divisions


def test_truth_consistency() -> None:
    out = generate_timelapse(**PARAMS)
    labels = out["labels"]
    assert len(out["truth"]) > 0
    for rec in out["truth"]:
        fov, frame, label = decode_key(rec["cell_key"])
        assert fov == 0
        assert frame == rec["frame"]
        assert label == rec["label"]
        assert rec["length_um"] > 0.0
        assert rec["width_um"] > 0.0
        assert bool((labels[frame] == label).any())


def test_vsnap_smoke() -> None:
    out = generate_timelapse(**{**PARAMS, "vsnap": True, "seed": 3})
    assert out["images"].shape == (10, 160, 160)
    assert any(event == "division" for _, _, event in out["edges"])
