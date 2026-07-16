"""Tests for momia3.ops.segment (§4.2)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from momia3.ops import segment as seg
from tests.synth import generate_timelapse


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def _clean_frame(seed: int = 3, shape: tuple[int, int] = (160, 160)) -> dict:
    return generate_timelapse(
        n_frames=2,
        shape=shape,
        n_initial=4,
        drift_px_per_frame=0.0,
        noise_sigma=0.005,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# threshold_segment
# ---------------------------------------------------------------------------


# Per-method parameters tuned for the synthetic phase-contrast frames
# (momia2 likewise used per-method config entries).
_METHOD_PARAMS: dict[str, dict] = {
    "isodata": {},
    "sauvola": {"sauvola_window": 25, "sauvola_k": 0.1},
    "local": {"block_size": 41, "local_offset": 0.02},
    "composite": {},
}


@pytest.mark.parametrize("method", ["isodata", "sauvola", "local", "composite"])
def test_threshold_segment_foreground_iou(method: str) -> None:
    data = _clean_frame()
    image = data["images"][0]
    gt = data["labels"][0] > 0
    mask = seg.threshold_segment(
        image, seg.ThresholdParams(method=method, **_METHOD_PARAMS[method])
    )
    assert mask.dtype == bool
    assert mask.shape == image.shape
    assert _iou(mask, gt) >= 0.85, method


def test_threshold_segment_composite_overlap_filter() -> None:
    """Components with low global-mask overlap are removed (momia2 port)."""
    data = _clean_frame(seed=11)
    image = data["images"][0]
    gt = data["labels"][0] > 0
    mask = seg.threshold_segment(
        image,
        seg.ThresholdParams(method="composite", overlap_threshold=0.5, dilation=True),
    )
    assert _iou(mask, gt) >= 0.85
    # No tiny specks survive the composite pipeline.
    n_small = int(
        (np.bincount(seg.measure.label(mask).ravel())[1:] < 10).sum()
    )
    assert n_small == 0


def test_threshold_segment_bright_foreground() -> None:
    rng = np.random.default_rng(5)
    image = rng.normal(0.2, 0.01, (96, 96)).astype(np.float32)
    image[30:60, 30:60] = 0.8
    mask = seg.threshold_segment(
        image, seg.ThresholdParams(method="isodata", dark_foreground=False)
    )
    assert mask[45, 45]
    assert not mask[5, 5]


# ---------------------------------------------------------------------------
# prob_to_labels
# ---------------------------------------------------------------------------


def _synthetic_prob(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """3-class prob map (bg, edge, interior) from ground-truth labels."""
    rng = np.random.default_rng(seed)
    fg = labels > 0
    seeds = np.zeros_like(fg)
    for label_id in range(1, int(labels.max()) + 1):
        cell = labels == label_id
        seeds |= ndimage.binary_erosion(cell, iterations=2)
    boundary = ndimage.binary_dilation(fg, iterations=1) ^ ndimage.binary_erosion(
        fg, iterations=1
    )
    prob = np.zeros((3,) + labels.shape, dtype=np.float32)
    prob[0] = np.where(fg, 0.10, 0.90)  # background prob: low inside cells
    prob[1] = np.where(boundary, 0.85, 0.05)  # edge prob
    prob[2] = np.where(seeds, 0.90, 0.05)  # interior/seed prob
    prob += rng.normal(0.0, 0.02, prob.shape).astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def test_prob_to_labels_recovers_instances() -> None:
    # Seed 8: cells are pairwise non-overlapping (one touching pair), so the
    # eroded per-cell cores are single components and recovery is unambiguous.
    data = _clean_frame(seed=8)
    gt = data["labels"][0]
    prob = _synthetic_prob(gt)

    labels = seg.prob_to_labels(
        prob,
        seg.ProbToLabelsParams(
            fg_threshold=0.4,
            fg_erosion_radius=0,
            seed_thresholds=(0.5, 0.3),
            min_seed_size=5,
            seed_opening_radius=0,
            min_cell_size=20,
        ),
    )

    assert labels.dtype == np.uint32
    n_gt = int(gt.max())
    n_out = int(labels.max())
    assert n_out == n_gt
    # Greedy 1:1 matching by IoU.
    ious = []
    for label_id in range(1, n_gt + 1):
        cell = gt == label_id
        best = max(
            (_iou(cell, labels == out_id) for out_id in range(1, n_out + 1)),
            default=0.0,
        )
        ious.append(best)
    assert float(np.mean(ious)) >= 0.9
    assert min(ious) >= 0.8


def test_prob_to_labels_empty_when_no_seeds() -> None:
    prob = np.zeros((3, 32, 32), dtype=np.float32)
    prob[0] = 0.9  # all background
    labels = seg.prob_to_labels(prob)
    assert labels.max() == 0


# ---------------------------------------------------------------------------
# compute_dist
# ---------------------------------------------------------------------------


def test_compute_dist_normalized_per_instance() -> None:
    labels = np.zeros((40, 40), dtype=np.uint32)
    labels[5:15, 5:15] = 1
    labels[25:35, 25:38] = 2
    dist = seg.compute_dist(labels, max_value=7.0)
    assert dist.dtype == np.float32
    assert dist.max() == pytest.approx(7.0)
    assert dist[0, 0] == 0.0
    # Peak of each instance is its interior center; border pixels ~ small.
    assert dist[5, 10] < dist[10, 10]
    # Distance does not bleed across instances.
    assert dist[24, 31] == 0.0


# ---------------------------------------------------------------------------
# declump
# ---------------------------------------------------------------------------


def _capsule(shape: tuple[int, int], cy: float, cx: float, ang: float,
             length: float, half_width: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    wy, wx = np.cos(ang), np.sin(ang)
    along = (yy - cy) * wy + (xx - cx) * wx
    perp = -(yy - cy) * wx + (xx - cx) * wy
    return (np.abs(along) < length / 2) & (np.abs(perp) <= half_width)


def _crowded_field(seed: int, shape: int = 210, n_anchors: int = 9) -> np.ndarray:
    """Crowded-field ground truth: clusters of 1-4 touching rods.

    Junctions are tip-to-tip contacts with 0.5-1.5 px shaft overlap and a
    29-52 deg kink — the physical signature of post-division cells that
    distance-based declumping exploits (a neck). Plain connected-component
    labeling merges each cluster into one blob.
    """
    rng = np.random.default_rng(seed)
    gt = np.zeros((shape, shape), np.uint32)
    grid = int(np.ceil(np.sqrt(n_anchors)))
    step = shape / (grid + 0.4)
    lid = 0
    for i in range(n_anchors):
        gy, gx = divmod(i, grid)
        cy = step * (gy + 0.75) + rng.uniform(-7, 7)
        cx = step * (gx + 0.75) + rng.uniform(-7, 7)
        ang = rng.uniform(0, np.pi)
        n = int(rng.choice([1, 2, 2, 3, 3, 4]))
        prev_len = 0.0
        for k in range(n):
            length = rng.uniform(20, 27)
            hw = rng.uniform(3.2, 4.0)
            if k > 0:
                ang = ang + rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 0.9)
                dist = (prev_len + length) / 2 - rng.uniform(0.5, 1.5)
                cy = cy + np.cos(ang) * dist + rng.uniform(-0.8, 0.8)
                cx = cx + np.sin(ang) * dist + rng.uniform(-0.8, 0.8)
            lid += 1
            m = _capsule(gt.shape, cy, cx, ang, length, hw)
            if m.any():
                gt[m] = lid
            prev_len = length
    return gt


def _match_recall(gt: np.ndarray, out: np.ndarray, thresh: float = 0.5) -> float:
    """Greedy 1:1 IoU matching; fraction of GT cells matched at >= thresh."""
    n_gt, n_out = int(gt.max()), int(out.max())
    pairs = []
    for g in range(1, n_gt + 1):
        cell = gt == g
        for o in range(1, n_out + 1):
            v = _iou(cell, out == o)
            if v > 0:
                pairs.append((v, g, o))
    pairs.sort(reverse=True)
    used_g: set[int] = set()
    used_o: set[int] = set()
    matched = 0
    for v, g, o in pairs:
        if v < thresh:
            break
        if g not in used_g and o not in used_o:
            used_g.add(g)
            used_o.add(o)
            matched += 1
    return matched / n_gt


_DECLUMP_SEEDS = (3, 5, 7, 11, 13, 17, 19, 23)


def test_declump_crowded_field_recall() -> None:
    """Crowded-field contract: recall <0.6 with plain labeling -> >=0.9.

    Aggregate over an 8-seed suite of touching-cluster fields. Per-seed
    guard >=0.8; the regime boundary (collinear contacts without a neck,
    fat multi-way junctions) is documented in benchmarks/robustness_report.md.
    """
    params = seg.DeclumpParams(
        min_distance=5, threshold_abs=2.0, smooth_sigma=1.5,
        h_maxima=0.5, min_cell_size=20,
    )
    plain_recalls, declump_recalls = [], []
    for seed in _DECLUMP_SEEDS:
        gt = _crowded_field(seed)
        plain = seg.measure.label(gt > 0)
        out = seg.declump(gt > 0, params=params)
        assert out.dtype == np.uint32
        plain_recalls.append(_match_recall(gt, plain))
        declump_recalls.append(_match_recall(gt, out))
    assert float(np.mean(plain_recalls)) < 0.6
    assert float(np.mean(declump_recalls)) >= 0.9
    assert min(declump_recalls) >= 0.8


def test_declump_empty_mask() -> None:
    out = seg.declump(np.zeros((48, 48), dtype=bool))
    assert out.dtype == np.uint32
    assert out.shape == (48, 48)
    assert out.max() == 0


def test_declump_label_input_and_component_fallback() -> None:
    # Two well-separated capsules as a label image: declumping must not
    # merge, lose, or over-split them.
    labels_in = np.zeros((80, 80), dtype=np.uint32)
    labels_in[_capsule(labels_in.shape, 25, 25, 0.3, 24, 3.6)] = 7
    labels_in[_capsule(labels_in.shape, 55, 55, 1.2, 24, 3.6)] = 9

    out = seg.declump(labels_in)
    assert out.dtype == np.uint32
    assert int(out.max()) == 2
    for src in (7, 9):
        cell = labels_in == src
        best = max(_iou(cell, out == o) for o in (1, 2))
        assert best >= 0.9

    # Absurdly high seed threshold: no peaks survive, so the op falls back
    # to connected components instead of dropping cells.
    merged = np.zeros((60, 60), dtype=bool)
    merged[_capsule(merged.shape, 20, 20, 0.0, 20, 3.5)] = True
    merged[_capsule(merged.shape, 32, 20, 0.9, 20, 3.5)] = True
    fallback = seg.declump(merged, params=seg.DeclumpParams(threshold_abs=1e6))
    assert int(fallback.max()) == 1  # one connected blob, kept whole
    assert fallback[merged].min() == 1


def test_declump_accepts_precomputed_dist() -> None:
    gt = _crowded_field(11)
    # Provided distance maps are used as-is (thresholds in the map's own
    # units): a 0.9x scaled EDT must split the clusters just as well.
    dist = ndimage.distance_transform_edt(gt > 0) * 0.9
    out = seg.declump(gt > 0, dist, params=seg.DeclumpParams(
        min_distance=5, threshold_abs=1.8, smooth_sigma=1.5,
        h_maxima=0.45, min_cell_size=20,
    ))
    assert _match_recall(gt, out) >= 0.85
    with pytest.raises(ValueError, match="does not match"):
        seg.declump(gt > 0, dist[:-2, :-2])


# ---------------------------------------------------------------------------
# patchify / unpatchify
# ---------------------------------------------------------------------------


def test_patchify_unpatchify_roundtrip() -> None:
    rng = np.random.default_rng(1)
    image = rng.random((100, 120)).astype(np.float32)

    patches, grid = seg.patchify(image, patch_size=32, overlap=8)
    assert patches.ndim == 3
    assert patches.shape[1:] == (32, 32)
    assert len(grid.coords) == patches.shape[0]

    out_weighted = seg.unpatchify(patches, grid, mode="weighted")
    np.testing.assert_allclose(out_weighted, image, atol=1e-5)
    out_mean = seg.unpatchify(patches, grid, mode="mean")
    np.testing.assert_allclose(out_mean, image, atol=1e-5)

    # Non-overlapping grid: exact identity in mean mode.
    patches0, grid0 = seg.patchify(image, patch_size=32, overlap=0)
    out0 = seg.unpatchify(patches0, grid0, mode="mean")
    np.testing.assert_array_equal(out0, image)


def test_patchify_multichannel_and_modes() -> None:
    rng = np.random.default_rng(2)
    image = rng.random((64, 80, 3)).astype(np.float32)
    patches, grid = seg.patchify(image, patch_size=24, overlap=8)
    assert patches.shape[-1] == 3
    out = seg.unpatchify(patches, grid, mode="weighted")
    assert out.shape == image.shape
    np.testing.assert_allclose(out, image, atol=1e-5)
    out_max = seg.unpatchify(patches, grid, mode="max")
    assert out_max.shape == image.shape
    np.testing.assert_allclose(out_max, image, atol=1e-6)


def test_patchify_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="overlap"):
        seg.patchify(np.zeros((32, 32)), patch_size=16, overlap=16)
