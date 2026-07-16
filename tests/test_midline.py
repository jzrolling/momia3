"""Tests for momia3.ops.midline (ARCHITECTURE.md §4.5 / §5.2 / §5.3).

Acceptance coverage (§5.3):
- straight synthetic capsule: midline length within 2 % of ground truth,
  width_median within 1 px equivalent;
- >= 99 % of midline points inside the mask (straight and bent);
- poles within 2 px of the true capsule tips on straight and bent shapes.

Note on the inside-mask check: the two endpoints of the main branch are
the subpixel poles, which sit on the extracted outline — up to ~1 px
outside the rasterized mask by construction (the rasterized edge is a
half-quantization inward of the true capsule). The check therefore uses
a 1 px distance-transform tolerance and additionally asserts that all
interior samples are strictly inside.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from scipy.ndimage import distance_transform_edt
from skimage.draw import disk, polygon

from momia3 import schema
from momia3.keys import encode_key
from momia3.ops.midline import (
    MIDLINE_SCALAR_COLUMNS,
    extract_midlines,
    fill_cells_midline,
)
from momia3.ops.outline import extract_outlines

from synth import generate_timelapse

PIXEL_MICRONS = 0.065


# ---------------------------------------------------------------------------
# synthetic shapes
# ---------------------------------------------------------------------------


def capsule_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    angle: float,
    length_px: float,
    width_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Straight capsule; returns (mask, tip1_xy, tip2_xy)."""
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
    mask[np.concatenate([body_r, cap1_r, cap2_r]),
         np.concatenate([body_c, cap1_c, cap2_c])] = True
    tip1 = np.array([cx - ux * length_px / 2.0, cy - uy * length_px / 2.0])
    tip2 = np.array([cx + ux * length_px / 2.0, cy + uy * length_px / 2.0])
    return mask, tip1, tip2


def bent_capsule_mask(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    radius: float,
    theta_deg: float,
    width_px: float,
    n_disk: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bent capsule: centerline is a circular arc of ``radius`` around
    ``(cy, cx)`` spanning ``theta_deg`` degrees, round caps at both ends.

    Returns (mask, tip1_xy, tip2_xy, true_midline_length_px). The arc is
    parameterized by angle a in [-theta/2, theta/2]; the outward tip
    direction at an end is the arc tangent pointing AWAY from the bend.
    """
    r = width_px / 2.0
    theta = np.deg2rad(theta_deg)
    mask = np.zeros(shape, dtype=bool)
    for a in np.linspace(-theta / 2.0, theta / 2.0, n_disk):
        rr, cc = disk((cy + radius * np.sin(a), cx + radius * np.cos(a)),
                      r, shape=shape)
        mask[rr, cc] = True
    tips = []
    for a, sign in ((-theta / 2.0, -1.0), (theta / 2.0, 1.0)):
        # Arc tangent along increasing a is (cos a, -sin a) in (dy, dx);
        # the outward tip direction flips sign at the lower end.
        ty, tx = sign * np.cos(a), -sign * np.sin(a)
        cap_y, cap_x = cy + radius * np.sin(a), cx + radius * np.cos(a)
        tips.append(np.array([cap_x + tx * r, cap_y + ty * r]))
    true_length = radius * theta + 2.0 * r
    return mask, tips[0], tips[1], true_length


def y_shape_mask(shape: tuple[int, int], width_px: float = 12.0) -> np.ndarray:
    """Three-armed (branched) cell-like mask."""
    r = width_px / 2.0
    mask = np.zeros(shape, dtype=bool)
    arms = [
        zip(80 + np.linspace(0, 40, 60), np.full(60, 80.0)),
        zip(80 - np.linspace(0, 30, 50) * np.sin(np.deg2rad(40)),
            80 + np.linspace(0, 30, 50) * np.cos(np.deg2rad(40))),
        zip(80 - np.linspace(0, 30, 50) * np.sin(np.deg2rad(40)),
            80 - np.linspace(0, 30, 50) * np.cos(np.deg2rad(40))),
    ]
    for arm in arms:
        for y, x in arm:
            rr, cc = disk((y, x), r, shape=shape)
            mask[rr, cc] = True
    return mask


def _pole_error(pole1: np.ndarray, pole2: np.ndarray,
                tip1: np.ndarray, tip2: np.ndarray) -> tuple[float, float]:
    """Per-tip distance to the nearer extracted pole (order-free)."""
    return (
        min(np.hypot(*(pole1 - tip1)), np.hypot(*(pole2 - tip1))),
        min(np.hypot(*(pole1 - tip2)), np.hypot(*(pole2 - tip2))),
    )


def _inside_fraction(points: np.ndarray, mask: np.ndarray,
                     tol_px: float = 1.0) -> float:
    """Fraction of points within ``tol_px`` of the mask (distance transform)."""
    dist = distance_transform_edt(~mask)
    iy = np.clip(np.round(points[:, 1]).astype(int), 0, mask.shape[0] - 1)
    ix = np.clip(np.round(points[:, 0]).astype(int), 0, mask.shape[1] - 1)
    return float(np.mean(dist[iy, ix] <= tol_px))


@pytest.fixture()
def straight():
    shape = (128, 128)
    mask, tip1, tip2 = capsule_mask(
        shape, 64.0, 64.0, np.deg2rad(30.0), 60.0, 12.0
    )
    labels = mask.astype(np.uint32)
    outlines = extract_outlines(labels)
    results = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
    return {"mask": mask, "labels": labels, "tip1": tip1, "tip2": tip2,
            "length_px": 60.0, "width_px": 12.0, "res": results[1]}


@pytest.fixture()
def bent():
    shape = (160, 160)
    mask, tip1, tip2, length_px = bent_capsule_mask(
        shape, 40.0, 40.0, 40.0, 100.0, 12.0
    )
    labels = mask.astype(np.uint32)
    outlines = extract_outlines(labels)
    results = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
    return {"mask": mask, "labels": labels, "tip1": tip1, "tip2": tip2,
            "length_px": length_px, "width_px": 12.0, "res": results[1]}


# ---------------------------------------------------------------------------
# §5.3 acceptance: straight capsule
# ---------------------------------------------------------------------------


def test_straight_length_within_two_percent(straight):
    got_px = straight["res"]["length_um"] / PIXEL_MICRONS
    assert got_px == pytest.approx(straight["length_px"], rel=0.02)


def test_straight_width_median_within_one_pixel(straight):
    got_px = straight["res"]["width_median_um"] / PIXEL_MICRONS
    assert got_px == pytest.approx(straight["width_px"], abs=1.0)


def test_straight_midline_inside_mask(straight):
    mid = straight["res"]["midline"][0]
    assert _inside_fraction(mid, straight["mask"], tol_px=1.0) >= 0.99
    # Interior samples are strictly inside the rasterized mask.
    inner = mid[1:-1]
    assert _inside_fraction(inner, straight["mask"], tol_px=0.0) == 1.0


def test_straight_poles_within_two_px(straight):
    d1, d2 = _pole_error(
        straight["res"]["pole1"], straight["res"]["pole2"],
        straight["tip1"], straight["tip2"],
    )
    assert d1 < 2.0 and d2 < 2.0


def test_straight_not_branched_low_sinuosity(straight):
    assert straight["res"]["branched"] == 0.0
    assert len(straight["res"]["midline"]) == 1
    assert straight["res"]["sinuosity"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0])
def test_straight_orientation_robustness(angle_deg):
    """Acceptance metrics hold at several orientations."""
    shape = (128, 128)
    mask, tip1, tip2 = capsule_mask(
        shape, 64.0, 64.0, np.deg2rad(angle_deg), 56.0, 12.0
    )
    labels = mask.astype(np.uint32)
    res = extract_midlines(labels, extract_outlines(labels),
                           pixel_microns=PIXEL_MICRONS)[1]
    assert res["length_um"] / PIXEL_MICRONS == pytest.approx(56.0, rel=0.02)
    assert res["width_median_um"] / PIXEL_MICRONS == pytest.approx(12.0, abs=1.0)
    d1, d2 = _pole_error(res["pole1"], res["pole2"], tip1, tip2)
    assert d1 < 2.0 and d2 < 2.0
    assert _inside_fraction(res["midline"][0], mask, tol_px=1.0) >= 0.99


# ---------------------------------------------------------------------------
# §5.3 acceptance: bent capsule
# ---------------------------------------------------------------------------


def test_bent_length_within_two_percent(bent):
    got_px = bent["res"]["length_um"] / PIXEL_MICRONS
    assert got_px == pytest.approx(bent["length_px"], rel=0.02)


def test_bent_midline_inside_mask(bent):
    mid = bent["res"]["midline"][0]
    assert _inside_fraction(mid, bent["mask"], tol_px=1.0) >= 0.99
    assert _inside_fraction(mid[1:-1], bent["mask"], tol_px=0.0) == 1.0


def test_bent_poles_within_two_px(bent):
    d1, d2 = _pole_error(
        bent["res"]["pole1"], bent["res"]["pole2"],
        bent["tip1"], bent["tip2"],
    )
    assert d1 < 2.0 and d2 < 2.0


def test_bent_sinuosity_above_one(bent):
    assert bent["res"]["sinuosity"] > 1.05
    assert bent["res"]["branched"] == 0.0


def test_bent_width_median_within_one_pixel(bent):
    got_px = bent["res"]["width_median_um"] / PIXEL_MICRONS
    assert got_px == pytest.approx(bent["width_px"], abs=1.0)


# ---------------------------------------------------------------------------
# topology / parameters
# ---------------------------------------------------------------------------


def test_branched_shape_detected():
    mask = y_shape_mask((160, 160))
    labels = mask.astype(np.uint32)
    res = extract_midlines(labels, extract_outlines(labels),
                           pixel_microns=PIXEL_MICRONS)[1]
    assert res["branched"] == 1.0
    assert len(res["midline"]) >= 2
    # The main branch stays inside the mask.
    assert _inside_fraction(res["midline"][0], mask, tol_px=1.0) >= 0.99


def test_lee_skeleton_method(straight):
    """``method='lee'`` is honored and still meets length tolerance."""
    res = extract_midlines(
        straight["labels"], extract_outlines(straight["labels"]),
        params={"method": "lee"}, pixel_microns=PIXEL_MICRONS,
    )[1]
    assert res["length_um"] / PIXEL_MICRONS == pytest.approx(
        straight["length_px"], rel=0.02
    )


def test_determinism(straight):
    a = extract_midlines(straight["labels"], pixel_microns=PIXEL_MICRONS)[1]
    b = extract_midlines(straight["labels"], pixel_microns=PIXEL_MICRONS)[1]
    assert a["length_um"] == b["length_um"]
    np.testing.assert_array_equal(a["midline"][0], b["midline"][0])


def test_width_profile_vector(straight):
    """width_profile has one µm value per resampled midline point."""
    res = straight["res"]
    profile = res["width_profile"]
    assert isinstance(profile, np.ndarray)
    assert profile.shape == (len(res["midline"][0]),)
    assert np.all(profile > 0)
    # Scalar summaries are consistent with the profile.
    assert res["width_median_um"] == pytest.approx(float(np.median(profile)))
    assert res["width_max_um"] == pytest.approx(float(np.max(profile)))
    assert res["width_q1_um"] <= res["width_median_um"] <= res["width_q3_um"]


# ---------------------------------------------------------------------------
# table writer / schema roundtrip
# ---------------------------------------------------------------------------


def _base_cells_table(labels: np.ndarray) -> pa.Table:
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


def test_fill_cells_midline_schema_roundtrip(straight):
    """fill_cells_midline output passes schema.validate_cells; geometry and
    scalars survive the roundtrip."""
    labels = straight["labels"]
    results = extract_midlines(labels, pixel_microns=PIXEL_MICRONS)
    table = fill_cells_midline(_base_cells_table(labels), results)
    validated = schema.validate_cells(table)

    assert validated.num_rows == 1
    for column in ("midline", "width_profile", *MIDLINE_SCALAR_COLUMNS):
        assert column in validated.column_names
    row = validated.column("midline").to_pylist()[0]
    assert row is not None and len(row) == 1
    got = np.column_stack([row[0]["x"], row[0]["y"]])
    np.testing.assert_allclose(
        got, results[1]["midline"][0].astype(np.float32), atol=1e-5
    )
    scalars = validated.column("length_um").to_pylist()
    assert scalars[0] == pytest.approx(results[1]["length_um"], abs=1e-5)
    poles = validated.column("pole1_x").to_pylist()[0]
    assert poles == pytest.approx(results[1]["pole1_x"], abs=1e-5)


def test_fill_cells_midline_missing_rows():
    """Rows without results keep null geometry/scalars."""
    shape = (128, 128)
    mask1, _, _ = capsule_mask(shape, 40.0, 40.0, 0.0, 40.0, 10.0)
    mask2, _, _ = capsule_mask(shape, 88.0, 88.0, np.pi / 2, 40.0, 10.0)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[mask1] = 1
    labels[mask2] = 2
    results = extract_midlines((mask1 * 1).astype(np.uint32),
                               pixel_microns=PIXEL_MICRONS)
    table = fill_cells_midline(_base_cells_table(labels), results)
    validated = schema.validate_cells(table)
    rows = validated.column("midline").to_pylist()
    assert rows[0] is not None and rows[1] is None
    length = validated.column("length_um").to_pylist()
    assert length[0] is not None and length[1] is None


# ---------------------------------------------------------------------------
# end-to-end on the synth generator
# ---------------------------------------------------------------------------


def test_synth_frame_acceptance():
    """Every frame-0 cell of the synth generator meets §5.3 tolerances."""
    data = generate_timelapse(
        n_frames=1, n_initial=3, noise_sigma=0.0,
        growth_rate_um_per_frame=0.0, seed=5,
    )
    labels = data["labels"][0]
    outlines = extract_outlines(labels)
    results = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
    assert set(results) == {rec["label"] for rec in data["truth"]}
    for rec in data["truth"]:
        res = results[rec["label"]]
        assert res["length_um"] == pytest.approx(rec["length_um"], rel=0.02)
        assert res["width_median_um"] == pytest.approx(
            rec["width_um"], abs=PIXEL_MICRONS
        )
        uy, ux = np.sin(rec["angle_rad"]), np.cos(rec["angle_rad"])
        half_px = rec["length_um"] / PIXEL_MICRONS / 2.0
        tip1 = np.array([rec["center_x"] - ux * half_px,
                         rec["center_y"] - uy * half_px])
        tip2 = np.array([rec["center_x"] + ux * half_px,
                         rec["center_y"] + uy * half_px])
        d1, d2 = _pole_error(res["pole1"], res["pole2"], tip1, tip2)
        assert d1 < 2.0 and d2 < 2.0
        assert _inside_fraction(res["midline"][0], labels == rec["label"],
                                tol_px=1.0) >= 0.99
