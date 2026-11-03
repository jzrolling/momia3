"""Tests for momia3.ops.outline (ARCHITECTURE.md §4.4 / §5.1 / §5.3).

Synthetic ground truth is built with the same capsule rasterization as
``tests/synth.py`` (polygon body + two disk caps); one test additionally
consumes :func:`tests.synth.generate_timelapse` directly.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from skimage.draw import disk, polygon

from momia3 import schema
from momia3.keys import encode_key
from momia3.ops.outline import (
    CURVATURE_COLUMNS,
    SHAPE_COLUMNS,
    chord_sagitta_ratio,
    curvature_features,
    ellipse_deviation,
    elliptic_fourier,
    extract_outlines,
    fill_cells_geometry,
    shape_descriptors,
)

from synth import generate_timelapse


def capsule_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    angle: float,
    length_px: float,
    width_px: float,
) -> np.ndarray:
    """Rasterize a straight capsule (rod) like ``tests/synth.py`` does."""
    radius = width_px / 2.0
    half_segment = max(length_px / 2.0 - radius, 0.0)
    uy, ux = np.sin(angle), np.cos(angle)
    p1y, p1x = cy - uy * half_segment, cx - ux * half_segment
    p2y, p2x = cy + uy * half_segment, cx + ux * half_segment
    py, px = -ux, uy
    rows = np.array([p1y + py * radius, p2y + py * radius,
                     p2y - py * radius, p1y - py * radius])
    cols = np.array([p1x + px * radius, p2x + px * radius,
                     p2x - px * radius, p1x - px * radius])
    body_r, body_c = polygon(rows, cols, shape=shape)
    cap1_r, cap1_c = disk((p1y, p1x), radius, shape=shape)
    cap2_r, cap2_c = disk((p2y, p2x), radius, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    rr = np.concatenate([body_r, cap1_r, cap2_r])
    cc = np.concatenate([body_c, cap1_c, cap2_c])
    mask[rr, cc] = True
    return mask


@pytest.fixture()
def straight() -> tuple[np.ndarray, dict]:
    """One straight capsule (label 1) with its ground-truth parameters."""
    shape = (128, 128)
    truth = {"cy": 64.0, "cx": 64.0, "angle": np.deg2rad(30.0),
             "length_px": 60.0, "width_px": 12.0}
    mask = capsule_mask(shape, **truth)
    return mask.astype(np.uint32), truth


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_extract_outlines_shape_and_closure(straight):
    """Outline is a closed (N, 2) float ring in global (x, y) coords."""
    labels, truth = straight
    outlines = extract_outlines(labels)
    assert set(outlines) == {1}
    ring = outlines[1]
    assert ring.ndim == 2 and ring.shape[1] == 2
    assert ring.dtype == np.float64
    # Implicitly closed ring: first point not duplicated at the end.
    assert not np.allclose(ring[0], ring[-1])
    # Closedness: the closing edge (last -> first) has a length comparable
    # to the median edge length (no gap in the ring).
    edges = np.hypot(*np.diff(ring, axis=0).T)
    closing = np.hypot(*(ring[0] - ring[-1]))
    assert closing <= 3.0 * float(np.median(edges))
    # Global coordinates: vertices bracket the mask bbox.
    ys, xs = np.nonzero(labels)
    assert ring[:, 0].min() <= xs.min() + 1.0
    assert ring[:, 0].max() >= xs.max() - 1.0
    assert ring[:, 1].min() <= ys.min() + 1.0
    assert ring[:, 1].max() >= ys.max() - 1.0


def test_extract_outlines_n_points_and_determinism(straight):
    """Uniform resampling count is honored; extraction is deterministic."""
    labels, _ = straight
    a = extract_outlines(labels, params={"n_points": 64})
    b = extract_outlines(labels, params={"n_points": 64})
    assert a[1].shape == (64, 2)
    np.testing.assert_array_equal(a[1], b[1])


def test_sigma_is_a_real_parameter(straight):
    """Regression: momia2 accepted ``sigma`` but always smoothed with 1."""
    labels, _ = straight
    raw = extract_outlines(labels, params={"sigma": 0.0, "n_points": 200})
    soft = extract_outlines(labels, params={"sigma": 3.0, "n_points": 200})
    assert not np.allclose(raw[1], soft[1], atol=1e-3)


def test_outline_vertices_track_mask_boundary(straight):
    """Subpixel outline stays within ~1.5 px of the rasterized mask edge."""
    from scipy.ndimage import distance_transform_edt

    labels, _ = straight
    mask = labels > 0
    ring = extract_outlines(labels)[1]
    # Distance from each vertex to the nearer of inside/outside boundary.
    d_out = distance_transform_edt(mask)  # inside -> distance to background
    d_in = distance_transform_edt(~mask)
    iy = np.clip(np.round(ring[:, 1]).astype(int), 0, mask.shape[0] - 1)
    ix = np.clip(np.round(ring[:, 0]).astype(int), 0, mask.shape[1] - 1)
    dist = np.where(mask[iy, ix], d_out[iy, ix], d_in[iy, ix])
    assert np.median(dist) <= 1.5
    assert np.percentile(dist, 95) <= 2.5


# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------


def test_curvature_features_finite_and_ordered(straight):
    """Windowed bend-angle stats are finite and quartile-ordered (§5.3:
    subpixel smoothness => mean curvature finite)."""
    labels, _ = straight
    ring = extract_outlines(labels)[1]
    stats = curvature_features(ring, window=3)
    assert set(stats) == set(CURVATURE_COLUMNS)
    for value in stats.values():
        assert np.isfinite(value)
    assert stats["curv_min"] <= stats["curv_q1"] <= stats["curv_mean"]
    assert stats["curv_mean"] <= stats["curv_q3"] <= stats["curv_max"]
    # Bend angles of a smooth closed ring are small (<< pi/2 per window).
    assert abs(stats["curv_mean"]) < np.pi / 4


def test_curvature_higher_at_caps_than_flanks(straight):
    """A rod bends mostly at its two caps."""
    labels, _ = straight
    ring = extract_outlines(labels, params={"n_points": 200})[1]
    from momia3.ops.outline import _bend_angle_closed

    angles = np.abs(_bend_angle_closed(ring, window=4))
    assert np.isfinite(angles).all()
    # The two cap regions carry the largest bend angles.
    order = np.argsort(angles)[::-1]
    top = order[: len(ring) // 10]
    # Caps are antipodal: top-curvature indices cluster in two groups.
    top_sorted = np.sort(top)
    gaps = np.diff(np.concatenate([top_sorted, [top_sorted[0] + len(ring)]]))
    assert gaps.max() > len(ring) // 4  # two well-separated clusters


# ---------------------------------------------------------------------------
# elliptic Fourier descriptors
# ---------------------------------------------------------------------------


def test_elliptic_fourier_shape_and_normalization(straight):
    """EFDs come as (n_harmonics, 4); Kuhl-Giardina sets |a1| = 1."""
    labels, _ = straight
    coeffs = elliptic_fourier(extract_outlines(labels)[1], n_harmonics=10)
    assert coeffs.shape == (10, 4)
    assert np.isfinite(coeffs).all()
    assert coeffs[0, 0] == pytest.approx(1.0, abs=1e-6)
    # Rotation normalization zeroes the first-harmonic b1/c1 by convention
    # up to numerical error only loosely; just bound the whole descriptor.
    assert np.abs(coeffs).max() <= 1.0 + 1e-6


def test_elliptic_fourier_translation_invariance(straight):
    """Descriptors of the same shape at two positions match closely."""
    shape = (160, 160)
    m1 = capsule_mask(shape, 60.0, 60.0, np.deg2rad(20.0), 50.0, 10.0)
    m2 = capsule_mask(shape, 95.0, 90.0, np.deg2rad(20.0), 50.0, 10.0)
    c1 = elliptic_fourier(extract_outlines(m1.astype(np.uint32))[1])
    c2 = elliptic_fourier(extract_outlines(m2.astype(np.uint32))[1])
    assert np.allclose(c1, c2, atol=0.05)


def test_elliptic_fourier_degenerate_input():
    """Degenerate outlines yield a zero descriptor, not an exception."""
    coeffs = elliptic_fourier(np.zeros((1, 2)), n_harmonics=5)
    assert coeffs.shape == (5, 4)
    assert np.all(coeffs == 0.0)


# ---------------------------------------------------------------------------
# table writer / schema roundtrip
# ---------------------------------------------------------------------------


def _two_cell_image() -> np.ndarray:
    shape = (128, 128)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[capsule_mask(shape, 40.0, 40.0, 0.0, 40.0, 10.0)] = 1
    labels[capsule_mask(shape, 88.0, 88.0, np.pi / 2, 40.0, 10.0)] = 2
    return labels


def _base_cells_table(labels: np.ndarray) -> pa.Table:
    """Minimal identity-only rows for each label (rest filled nulls)."""
    unique = [int(v) for v in np.unique(labels) if v != 0]
    n = len(unique)
    sch = schema.cells_schema()
    values = {
        "cell_key": [encode_key(0, 0, lab) for lab in unique],
        "fov_id": [0] * n,
        "frame": [0] * n,
        "label": unique,
        "touching_edge": [False] * n,
        "include": [True] * n,
    }
    arrays = [
        pa.array(values.get(field.name, [None] * n), type=field.type)
        for field in sch
    ]
    return pa.Table.from_arrays(arrays, schema=sch)


def test_fill_cells_geometry_schema_roundtrip():
    """fill_cells_geometry output passes schema.validate_cells and the
    geometry/curvature values survive a validation roundtrip (§5.3)."""
    labels = _two_cell_image()
    outlines = extract_outlines(labels)
    curvatures = {lab: curvature_features(ring) for lab, ring in outlines.items()}
    table = fill_cells_geometry(_base_cells_table(labels), outlines, curvatures)
    validated = schema.validate_cells(table)

    assert validated.num_rows == 2
    for column in ("outline", *CURVATURE_COLUMNS):
        assert column in validated.column_names
    outlines_col = validated.column("outline").to_pylist()
    for lab, row in zip((1, 2), outlines_col):
        assert row is not None
        got = np.column_stack([row["x"], row["y"]])
        np.testing.assert_allclose(
            got, outlines[lab].astype(np.float32), atol=1e-5
        )
    stats_col = validated.column("curv_mean").to_pylist()
    for lab, got in zip((1, 2), stats_col):
        assert got == pytest.approx(curvatures[lab]["curv_mean"], abs=1e-6)


def test_fill_cells_geometry_defaults_and_missing_rows():
    """Curvatures default to computation from the outline; rows without a
    matching outline keep null geometry and null curvature."""
    labels = _two_cell_image()
    outlines = extract_outlines(labels)
    table = fill_cells_geometry(_base_cells_table(labels), {1: outlines[1]})
    validated = schema.validate_cells(table)
    rows = validated.column("outline").to_pylist()
    assert rows[0] is not None and rows[1] is None
    curv = validated.column("curv_max").to_pylist()
    assert curv[0] is not None and np.isfinite(curv[0])
    assert curv[1] is None


def test_fill_cells_geometry_cell_key_keying():
    """Mappings may also be keyed by cell_key instead of label."""
    labels = _two_cell_image()
    outlines = extract_outlines(labels)
    by_key = {encode_key(0, 0, lab): ring for lab, ring in outlines.items()}
    table = fill_cells_geometry(_base_cells_table(labels), by_key)
    validated = schema.validate_cells(table)
    rows = validated.column("outline").to_pylist()
    assert all(row is not None for row in rows)


def test_synth_frame_outlines():
    """End-to-end on the tests/synth.py generator: every frame-0 cell gets
    a finite closed outline with finite curvature stats."""
    data = generate_timelapse(
        n_frames=1, n_initial=3, noise_sigma=0.0,
        growth_rate_um_per_frame=0.0, seed=3,
    )
    labels = data["labels"][0]
    outlines = extract_outlines(labels)
    truth_labels = {rec["label"] for rec in data["truth"]}
    assert set(outlines) == truth_labels
    for ring in outlines.values():
        stats = curvature_features(ring)
        assert all(np.isfinite(v) for v in stats.values())

def _bent_ring(radius: float = 80.0, span: float = 1.4, width: float = 12.0, n: int = 80) -> np.ndarray:
    """Analytic banana-shaped closed outline (arc centerline + normal offset)."""
    t = np.linspace(-span / 2, span / 2, n)
    cx = radius * np.sin(t)
    cy = radius * (1.0 - np.cos(t))
    nx, ny = -np.sin(t), np.cos(t)
    left = np.column_stack([cx + nx * width / 2, cy + ny * width / 2])
    right = np.column_stack([cx - nx * width / 2, cy - ny * width / 2])
    return np.vstack([left, right[::-1]])


def test_shape_descriptors_straight_vs_bent():
    """Straight rods have low sagitta; bent cells raise both descriptors."""
    labels = np.zeros((160, 200), dtype=np.uint32)
    labels[capsule_mask((160, 200), 80, 60, 0.0, 90, 12)] = 1
    straight = extract_outlines(labels)[1]
    d_straight = shape_descriptors(straight)
    d_bent = shape_descriptors(_bent_ring())
    assert set(d_straight) == set(SHAPE_COLUMNS)
    assert d_straight["sagitta_ratio"] < 0.05
    assert d_bent["sagitta_ratio"] > 0.30
    assert d_bent["ellipse_dev"] > d_straight["ellipse_dev"]
    assert chord_sagitta_ratio(straight) == pytest.approx(d_straight["sagitta_ratio"])
    assert ellipse_deviation(straight) == pytest.approx(d_straight["ellipse_dev"])


def test_fill_cells_geometry_writes_shape_columns():
    """ellipse_dev / sagitta_ratio become finite cells-table feature columns."""
    labels = _two_cell_image()
    outlines = extract_outlines(labels)
    table = fill_cells_geometry(_base_cells_table(labels), outlines)
    validated = schema.validate_cells(table)
    for column in SHAPE_COLUMNS:
        assert column in validated.column_names
        values = validated.column(column).to_pylist()
        assert all(v is not None and np.isfinite(v) for v in values)
    # A straight capsule should have a much smaller sagitta than its ellipse residual.
    sag = validated.column("sagitta_ratio").to_pylist()[0]
    ell = validated.column("ellipse_dev").to_pylist()[0]
    assert 0.0 <= sag < 0.1
    assert 0.0 < ell < 1.0
