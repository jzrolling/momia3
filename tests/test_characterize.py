"""Tests for momia3.ops.characterize (§4.6)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from skimage.draw import disk, polygon

from momia3 import schema
from momia3.keys import encode_key
from momia3.ops import characterize as ch
from tests.synth import generate_timelapse


# ---------------------------------------------------------------------------
# inline geometry fixtures (midline/outline structs built by hand — the
# midline/outline ops are owned by another agent and are NOT imported here)
# ---------------------------------------------------------------------------


def _capsule_geometry(
    cy: float,
    cx: float,
    angle: float,
    length: float,
    width: float,
    n_cap: int = 16,
) -> tuple[dict, dict]:
    """Straight capsule midline + closed outline structs.

    Parameters use (y, x) center, ``angle`` measured from the +x axis,
    total tip-to-tip ``length`` and ``width`` in pixels. Returns
    ``(midline, outline)`` in the schema's nested-struct layout.
    """
    radius = width / 2.0
    half = max(length / 2.0 - radius, 0.0)
    ux, uy = np.cos(angle), np.sin(angle)  # unit long axis (x, y)
    p1x, p1y = cx - ux * half, cy - uy * half
    p2x, p2y = cx + ux * half, cy + uy * half
    midline = [{"x": [p1x, p2x], "y": [p1y, p2y]}]
    # Stadium outline: cap around p2 then cap around p1, closed ring.
    nx_, ny_ = -uy, ux  # left normal
    th2 = np.linspace(-np.pi / 2, np.pi / 2, n_cap, endpoint=False)
    th1 = np.linspace(np.pi / 2, 3 * np.pi / 2, n_cap, endpoint=False)
    ox = np.concatenate(
        [p2x + radius * (ux * np.cos(th2) + nx_ * np.sin(th2)),
         p1x + radius * (ux * np.cos(th1) + nx_ * np.sin(th1))]
    )
    oy = np.concatenate(
        [p2y + radius * (uy * np.cos(th2) + ny_ * np.sin(th2)),
         p1y + radius * (uy * np.cos(th1) + ny_ * np.sin(th1))]
    )
    outline = {
        "x": np.concatenate([ox, ox[:1]]).tolist(),
        "y": np.concatenate([oy, oy[:1]]).tolist(),
    }
    return midline, outline


def _capsule_labels(
    shape: tuple[int, int],
    cy: float,
    cx: float,
    angle: float,
    length: float,
    width: float,
    label: int = 1,
) -> np.ndarray:
    """Rasterize one capsule into a label image."""
    labels = np.zeros(shape, dtype=np.uint32)
    radius = width / 2.0
    half = max(length / 2.0 - radius, 0.0)
    ux, uy = np.cos(angle), np.sin(angle)
    p1y, p1x = cy - uy * half, cx - ux * half
    p2y, p2x = cy + uy * half, cx + ux * half
    py, px = -ux, uy
    rr = np.array([p1y + py * radius, p2y + py * radius,
                   p2y - py * radius, p1y - py * radius])
    cc = np.array([p1x + px * radius, p2x + px * radius,
                   p2x - px * radius, p1x - px * radius])
    body_r, body_c = polygon(rr, cc, shape=shape)
    c1r, c1c = disk((p1y, p1x), radius, shape=shape)
    c2r, c2c = disk((p2y, p2x), radius, shape=shape)
    labels[np.concatenate([body_r, c1r, c2r]), np.concatenate([body_c, c1c, c2c])] = label
    return labels


def _geometry_cells_table(rows: list[dict]) -> pa.Table:
    """Minimal cells table with the geometry columns projection reads."""
    return pa.table(
        {
            "cell_key": pa.array([r["cell_key"] for r in rows], type=pa.uint64()),
            "frame": pa.array([r.get("frame", 0) for r in rows], type=pa.uint32()),
            "label": pa.array([r["label"] for r in rows], type=pa.uint32()),
            "midline": pa.array([r["midline"] for r in rows], type=schema.MIDLINE_T),
            "outline": pa.array([r["outline"] for r in rows], type=schema.OUTLINE_T),
        }
    )


# ---------------------------------------------------------------------------
# demographic_projection
# ---------------------------------------------------------------------------


def test_demographic_projection_shapes_and_gradient() -> None:
    shape = (64, 64)
    midline, outline = _capsule_geometry(32.0, 32.0, angle=0.0, length=30.0, width=10.0)
    labels = _capsule_labels(shape, 32.0, 32.0, 0.0, 30.0, 10.0, label=1)
    key = encode_key(0, 0, 1)
    table = _geometry_cells_table(
        [{"cell_key": key, "label": 1, "midline": midline, "outline": outline}]
    )
    # Intensity increases linearly along x (the midline direction).
    image = np.broadcast_to(np.arange(shape[1], dtype=np.float64), shape).copy()

    n_l, n_d = 32, 8
    out = ch.demographic_projection(table, image, labels, n_l=n_l, n_d=n_d)

    assert set(out) == {key}
    proj = out[key]
    assert proj.shape == (n_l, n_d)
    assert proj.dtype == np.float32
    assert np.isfinite(proj).all()
    # Monotonic along l, flat along d (interior rows).
    along_l = proj[4:-4].mean(axis=1)
    assert np.all(np.diff(along_l) > 0)
    across_d = proj[4:-4].std(axis=1).mean()
    assert across_d < 1.0


def test_demographic_projection_skips_cells_without_geometry() -> None:
    shape = (32, 32)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[8:16, 8:16] = 1
    labels[20:28, 20:28] = 2
    image = np.ones(shape, dtype=np.float64)
    rows = [
        {  # no midline -> skipped
            "cell_key": encode_key(0, 0, 1),
            "label": 1,
            "midline": None,
            "outline": {"x": [8, 16, 16, 8], "y": [8, 8, 16, 16]},
        },
        {  # label absent from the label image -> skipped
            "cell_key": encode_key(0, 0, 3),
            "label": 3,
            "midline": [{"x": [4.0, 8.0], "y": [4.0, 8.0]}],
            "outline": {"x": [2, 10, 10, 2], "y": [2, 2, 10, 10]},
        },
    ]
    out = ch.demographic_projection(_geometry_cells_table(rows), image, labels)
    assert out == {}


# ---------------------------------------------------------------------------
# growth_stats
# ---------------------------------------------------------------------------


def _growth_fixture() -> tuple[pa.Table, pa.Table, dict]:
    """Four dividing tracks with exact exponential growth.

    Track ``k`` is born at frame 0 with length ``b_k`` and elongates as
    ``b_k * exp(rate * f)``; all tracks divide at frame 4. Added length
    is an exact linear function of birth length, so both adder/sizer
    correlations are +1.
    """
    rate = 0.1
    births = np.array([1.8, 2.0, 2.2, 2.4])
    cell_rows: dict[str, list] = {"track_id": [], "frame": [], "length_um": []}
    track_rows: list[dict] = []
    for k, b in enumerate(births, start=1):
        for f in range(5):
            cell_rows["track_id"].append(k)
            cell_rows["frame"].append(f)
            cell_rows["length_um"].append(float(b * np.exp(rate * f)))
        track_rows.append(
            {
                "track_id": k,
                "fov_id": 0,
                "birth_frame": 0,
                "last_frame": 4,
                "division_frame": 4,
                "parent_track": None,
                "n_frames": 5,
                "fate": "divide",
            }
        )
    cells = pa.table(
        {
            "track_id": pa.array(cell_rows["track_id"], type=pa.uint32()),
            "frame": pa.array(cell_rows["frame"], type=pa.uint32()),
            "length_um": pa.array(cell_rows["length_um"], type=pa.float32()),
        }
    )
    tracks = schema.validate_tracks(
        pa.table(
            {
                "track_id": pa.array([r["track_id"] for r in track_rows], type=pa.uint32()),
                "fov_id": pa.array([0] * 4, type=pa.uint32()),
                "birth_frame": pa.array([0] * 4, type=pa.uint32()),
                "last_frame": pa.array([4] * 4, type=pa.uint32()),
                "division_frame": pa.array([4] * 4, type=pa.uint32()),
                "parent_track": pa.array([None] * 4, type=pa.uint32()),
                "n_frames": pa.array([5] * 4, type=pa.uint32()),
                "fate": pa.array(["divide"] * 4, type=pa.string()),
            }
        )
    )
    return tracks, cells, {"rate": rate, "births": births}


def test_growth_stats_rates_and_adder() -> None:
    tracks, cells, truth = _growth_fixture()
    result = ch.growth_stats(tracks, cells)

    per_track = {r["track_id"]: r for r in result["per_track"]}
    assert set(per_track) == {1, 2, 3, 4}
    for k, rec in per_track.items():
        assert rec["elongation_rate"] == pytest.approx(truth["rate"], abs=1e-4)
        assert rec["interdivision_frames"] == pytest.approx(4.0)
        b = truth["births"][k - 1]
        assert rec["birth_length_um"] == pytest.approx(b, rel=1e-3)
        assert rec["division_length_um"] == pytest.approx(
            b * np.exp(4 * truth["rate"]), rel=1e-3
        )
        assert rec["added_length_um"] == pytest.approx(
            rec["division_length_um"] - rec["birth_length_um"], rel=1e-3
        )

    summary = result["summary"]
    assert summary["n_tracks"] == 4
    assert summary["n_dividing"] == 4
    assert summary["mean_elongation_rate"] == pytest.approx(truth["rate"], abs=1e-4)
    # Added length is exactly linear in birth length; division length too.
    assert summary["corr_birth_added"] == pytest.approx(1.0, abs=1e-6)
    assert summary["corr_birth_division"] == pytest.approx(1.0, abs=1e-6)


def test_growth_stats_on_synth_truth() -> None:
    data = generate_timelapse(
        n_frames=18, shape=(256, 256), n_initial=4, drift_px_per_frame=0.0,
        noise_sigma=0.02, seed=0, division_length_um=3.0,
        growth_rate_um_per_frame=0.12,
    )
    truth = data["truth"]
    cells = pa.table(
        {
            "track_id": pa.array([r["track_id"] for r in truth], type=pa.uint32()),
            "frame": pa.array([r["frame"] for r in truth], type=pa.uint32()),
            "length_um": pa.array([r["length_um"] for r in truth], type=pa.float32()),
        }
    )
    # Tracks table from truth: division when the track heads a division edge.
    div_tracks = {
        r["track_id"]
        for r in truth
        if any(
            p == r["cell_key"] and e == "division" for p, _, e in data["edges"]
        )
    }
    track_ids = sorted({r["track_id"] for r in truth})
    tracks = schema.validate_tracks(
        pa.table(
            {
                "track_id": pa.array(track_ids, type=pa.uint32()),
                "fov_id": pa.array([0] * len(track_ids), type=pa.uint32()),
                "birth_frame": pa.array(
                    [min(r["frame"] for r in truth if r["track_id"] == t) for t in track_ids],
                    type=pa.uint32(),
                ),
                "last_frame": pa.array(
                    [max(r["frame"] for r in truth if r["track_id"] == t) for t in track_ids],
                    type=pa.uint32(),
                ),
                "division_frame": pa.array(
                    [
                        max(r["frame"] for r in truth if r["track_id"] == t)
                        if t in div_tracks
                        else None
                        for t in track_ids
                    ],
                    type=pa.uint32(),
                ),
                "parent_track": pa.array([None] * len(track_ids), type=pa.uint32()),
                "n_frames": pa.array(
                    [len({r["frame"] for r in truth if r["track_id"] == t}) for t in track_ids],
                    type=pa.uint32(),
                ),
                "fate": pa.array(
                    ["divide" if t in div_tracks else "incomplete" for t in track_ids],
                    type=pa.string(),
                ),
            }
        )
    )
    result = ch.growth_stats(tracks, cells)
    summary = result["summary"]
    # Linear elongation of 0.12 µm/frame at ~2-3 µm -> log rate ~0.04-0.06.
    assert 0.02 < summary["mean_elongation_rate"] < 0.08
    assert summary["n_dividing"] >= 3
    for rec in result["per_track"]:
        if np.isfinite(rec["interdivision_frames"]):
            assert rec["interdivision_frames"] > 0


# ---------------------------------------------------------------------------
# detect_septum
# ---------------------------------------------------------------------------


def test_detect_septum_finds_central_dip() -> None:
    w = np.full(41, 3.0)
    w[20] = 2.0
    pos = ch.detect_septum(w, min_prominence=0.5)
    assert pos == pytest.approx(0.5, abs=1e-9)


def test_detect_septum_off_center_with_noise() -> None:
    rng = np.random.default_rng(7)
    w = 3.0 + rng.normal(0.0, 0.02, 61)
    w[15] = 1.5  # dip at fractional position 15/60 = 0.25
    pos = ch.detect_septum(w, min_prominence=0.3)
    assert pos == pytest.approx(0.25, abs=0.05)


def test_detect_septum_returns_none_when_flat_or_weak() -> None:
    assert ch.detect_septum(np.full(41, 3.0), min_prominence=0.1) is None
    w = np.full(41, 3.0)
    w[20] = 2.9  # prominence 0.1 below the 0.5 threshold
    assert ch.detect_septum(w, min_prominence=0.5) is None
    assert ch.detect_septum([3.0, 3.0], min_prominence=0.1) is None


# ---------------------------------------------------------------------------
# orientation_coherence
# ---------------------------------------------------------------------------


def _coherence_table(labels: np.ndarray) -> pa.Table:
    rows = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        yy, xx = np.nonzero(labels == lab)
        rows.append(
            {
                "label": int(lab),
                "centroid": {"x": float(xx.mean()), "y": float(yy.mean())},
            }
        )
    return pa.table(
        {
            "label": pa.array([r["label"] for r in rows], type=pa.uint32()),
            "centroid": pa.array([r["centroid"] for r in rows], type=schema.CENTROID_T),
        }
    )


def test_orientation_coherence_parallel_rods() -> None:
    shape = (128, 256)
    labels = np.zeros(shape, dtype=np.uint32)
    for i, cy in enumerate((24, 64, 104)):
        labels += _capsule_labels(shape, cy, 60 + 70 * i, 0.0, 40.0, 10.0, label=i + 1)
    s = ch.orientation_coherence(labels, _coherence_table(labels))
    assert s > 0.95


def test_orientation_coherence_isotropic_rods() -> None:
    shape = (256, 256)
    labels = np.zeros(shape, dtype=np.uint32)
    angles = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    centers = ((50, 50), (50, 200), (200, 200), (200, 50))
    for i, ((cy, cx), ang) in enumerate(zip(centers, angles)):
        labels += _capsule_labels(shape, cy, cx, ang, 36.0, 8.0, label=i + 1)
    s = ch.orientation_coherence(labels, _coherence_table(labels))
    assert s < 0.5


def test_orientation_coherence_single_cell_is_zero() -> None:
    labels = _capsule_labels((64, 64), 32.0, 32.0, 0.3, 30.0, 10.0, label=1)
    assert ch.orientation_coherence(labels, _coherence_table(labels)) == 0.0


# ---------------------------------------------------------------------------
# find_puncta
# ---------------------------------------------------------------------------


def _spot_image(
    spots: list[tuple[float, float]], shape=(64, 64), sigma=1.2, amplitude=5.0
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.zeros(shape, dtype=np.float64)
    for y0, x0 in spots:
        image += amplitude * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma**2))
    return image


def test_find_puncta_subpixel_recovery() -> None:
    spots = [(20.3, 30.6), (45.7, 15.2)]
    image = _spot_image(spots)
    out = ch.find_puncta(image)
    assert out.shape == (2, 2)
    for y0, x0 in spots:
        dist = np.hypot(out[:, 0] - y0, out[:, 1] - x0).min()
        assert dist < 0.5


def test_find_puncta_respects_mask() -> None:
    spots = [(20.3, 30.6), (45.7, 15.2)]
    image = _spot_image(spots)
    mask = np.zeros_like(image, dtype=bool)
    mask[35:55, 5:25] = True  # covers only the second spot
    out = ch.find_puncta(image, mask_2d=mask)
    assert out.shape == (1, 2)
    assert np.hypot(out[0, 0] - 45.7, out[0, 1] - 15.2) < 0.5


def test_find_puncta_blank_image() -> None:
    out = ch.find_puncta(np.zeros((32, 32)))
    assert out.shape == (0, 2)
    out = ch.find_puncta(np.full((32, 32), 0.5))
    assert out.shape == (0, 2)


def test_find_puncta_with_noise_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    image = _spot_image([(32.4, 31.5)]) + rng.normal(0.0, 0.05, (64, 64))
    out1 = ch.find_puncta(image)
    out2 = ch.find_puncta(image)
    assert out1.shape[0] >= 1
    np.testing.assert_array_equal(out1, out2)
    dist = np.hypot(out1[:, 0] - 32.4, out1[:, 1] - 31.5).min()
    assert dist < 0.5


# ---------------------------------------------------------------------------
# assign_pole_age
# ---------------------------------------------------------------------------


def _pole_age_fixture() -> tuple[pa.Table, pa.Table, dict[int, tuple[int, int]]]:
    """Hand-built 2-generation lineage with known pole ages.

    R (frame 0, poles (0,5)-(10,5)) --continue--> C (frame 1, grown to
    poles (0,5)-(16,5)) --division--> D1 (left half, frame 2) and D2
    (right half, frame 2). D1 --continue--> E (frame 3), whose pole
    *indices* are flipped relative to D1 to exercise the identity swap.
    Extra cells: X (frame 1, no parent) and Y (frame 2, continue edge
    from Z whose pole coordinates are null) stay unknown (-1).
    """
    R = encode_key(0, 0, 1)
    C = encode_key(0, 1, 1)
    D1 = encode_key(0, 2, 1)
    D2 = encode_key(0, 2, 2)
    E = encode_key(0, 3, 1)
    X = encode_key(0, 1, 2)
    Z = encode_key(0, 1, 3)
    Y = encode_key(0, 2, 3)

    rows = [
        # (key, frame, pole1, pole2, centroid)
        (R, 0, (0.0, 5.0), (10.0, 5.0), (5.0, 5.0)),
        (C, 1, (0.0, 5.0), (16.0, 5.0), (8.0, 5.0)),
        (D1, 2, (0.0, 5.0), (8.0, 5.0), (4.0, 5.0)),
        (D2, 2, (8.0, 5.0), (16.0, 5.0), (12.0, 5.0)),
        # E spans 0..9 like a grown D1 but pole indices are flipped:
        # pole1 sits on D1's pole2 side, pole2 on D1's pole1 side.
        (E, 3, (9.0, 5.0), (0.0, 5.0), (4.5, 5.0)),
        (X, 1, (30.0, 30.0), (40.0, 30.0), (35.0, 30.0)),
        (Z, 1, None, None, None),
        (Y, 2, (30.0, 40.0), (40.0, 40.0), (35.0, 40.0)),
    ]
    cells = pa.table(
        {
            "cell_key": pa.array([r[0] for r in rows], type=pa.uint64()),
            "frame": pa.array([r[1] for r in rows], type=pa.uint32()),
            "pole1_x": pa.array(
                [r[2][0] if r[2] else None for r in rows], type=pa.float32()
            ),
            "pole1_y": pa.array(
                [r[2][1] if r[2] else None for r in rows], type=pa.float32()
            ),
            "pole2_x": pa.array(
                [r[3][0] if r[3] else None for r in rows], type=pa.float32()
            ),
            "pole2_y": pa.array(
                [r[3][1] if r[3] else None for r in rows], type=pa.float32()
            ),
            "centroid": pa.array(
                [
                    {"x": r[4][0], "y": r[4][1]} if r[4] else None
                    for r in rows
                ],
                type=schema.CENTROID_T,
            ),
        }
    )
    edges = schema.validate_edges(
        pa.table(
            {
                "parent_key": pa.array(
                    [R, C, C, D1, Z], type=pa.uint64()
                ),
                "child_key": pa.array(
                    [C, D1, D2, E, Y], type=pa.uint64()
                ),
                "event": pa.array(
                    ["continue", "division", "division", "continue", "continue"],
                    type=pa.string(),
                ),
                "confidence": pa.array([1.0] * 5, type=pa.float32()),
            }
        )
    )
    expected = {
        R: (0, 0),  # frame 0 seeds both poles at 0
        C: (1, 1),  # continue: both poles inherited, +1
        D1: (2, 0),  # division: pole1 (far side) inherited, pole2 new
        D2: (0, 2),  # division: pole2 (far side) inherited, pole1 new
        E: (1, 3),  # continue with identity swap relative to D1 (2, 0)
        X: (-1, -1),  # no parent edge
        Z: (-1, -1),  # frame > 0, no parent edge
        Y: (-1, -1),  # mother Z lacks pole data
    }
    return cells, edges, expected


def test_assign_pole_age_two_generation_lineage() -> None:
    cells, edges, expected = _pole_age_fixture()
    out = ch.assign_pole_age(cells, edges)

    assert isinstance(out, pl.DataFrame)
    assert out.schema["pole1_age"] == pl.Int32
    assert out.schema["pole2_age"] == pl.Int32
    assert out.height == cells.num_rows

    keys = out["cell_key"].to_numpy()
    age1 = out["pole1_age"].to_numpy()
    age2 = out["pole2_age"].to_numpy()
    got = {int(k): (int(a1), int(a2)) for k, a1, a2 in zip(keys, age1, age2)}
    assert got == expected


def test_assign_pole_age_accepts_polars_and_preserves_rows() -> None:
    cells, edges, expected = _pole_age_fixture()
    df = pl.from_arrow(cells).with_row_index("order")
    out = ch.assign_pole_age(df, pl.from_arrow(edges))
    assert isinstance(out, pl.DataFrame)
    # Row order preserved and original columns untouched.
    assert out["order"].to_list() == list(range(cells.num_rows))
    got = {
        int(k): (a1, a2)
        for k, a1, a2 in zip(
            out["cell_key"].to_list(),
            out["pole1_age"].to_list(),
            out["pole2_age"].to_list(),
        )
    }
    assert got == expected


# ---------------------------------------------------------------------------
# track_demographic_mean
# ---------------------------------------------------------------------------


def test_track_demographic_mean_matches_manual_mean() -> None:
    k1, k2, k3, k4, k5 = (encode_key(0, 0, i) for i in (1, 2, 3, 4, 5))
    m1 = np.arange(8, dtype=np.float32).reshape(4, 2)
    m2 = (np.arange(8, dtype=np.float32) ** 2).reshape(4, 2)
    m3 = np.full((4, 2), 3.5, dtype=np.float32)
    projections = {k1: m1, k2: m2, k3: m3}  # k4/k5 have no projection
    cells = pa.table(
        {
            "cell_key": pa.array([k1, k2, k3, k4, k5], type=pa.uint64()),
            "track_id": pa.array([1, 1, 2, None, 3], type=pa.uint32()),
        }
    )

    out = ch.track_demographic_mean(projections, cells)

    assert isinstance(out, pl.DataFrame)
    assert out.schema == {
        "track_id": pl.UInt32,
        "n_cells": pl.UInt32,
        "mean_map": pl.List(pl.Float32),
    }
    # Track 3 has no projections and the null-track cell is skipped.
    assert out["track_id"].to_list() == [1, 2]
    assert out["n_cells"].to_list() == [2, 1]

    mean1 = np.asarray(out["mean_map"][0].to_numpy(), dtype=np.float64)
    expected1 = ((m1.astype(np.float64) + m2.astype(np.float64)) / 2.0).ravel()
    assert mean1.shape == (8,)
    np.testing.assert_allclose(mean1, expected1, rtol=1e-6, atol=1e-6)

    mean2 = np.asarray(out["mean_map"][1].to_numpy(), dtype=np.float64)
    np.testing.assert_allclose(mean2, m3.astype(np.float64).ravel(), rtol=1e-6)


def test_track_demographic_mean_empty() -> None:
    cells = pa.table(
        {
            "cell_key": pa.array([encode_key(0, 0, 1)], type=pa.uint64()),
            "track_id": pa.array([None], type=pa.uint32()),
        }
    )
    out = ch.track_demographic_mean({}, cells)
    assert out.height == 0
    assert out.schema["mean_map"] == pl.List(pl.Float32)


# ---------------------------------------------------------------------------
# efd_shape_space
# ---------------------------------------------------------------------------


def _bent_outline(
    cx: float,
    cy: float,
    arc_angle: float,
    radius: float,
    width: float,
    n: int = 48,
) -> dict:
    """Banana-shaped closed outline struct along a circular arc."""
    phis = np.linspace(-arc_angle / 2.0, arc_angle / 2.0, n // 2)
    mx = cx + radius * np.sin(phis)
    my = cy - radius * np.cos(phis)
    tx, ty = np.cos(phis), np.sin(phis)  # arc tangent
    nx_, ny_ = -ty, tx  # arc normal
    half = width / 2.0
    s1x = mx + half * nx_
    s1y = my + half * ny_
    s2x = mx - half * nx_
    s2y = my - half * ny_
    ox = np.concatenate([s1x, s2x[::-1], s1x[:1]])
    oy = np.concatenate([s1y, s2y[::-1], s1y[:1]])
    return {"x": ox.tolist(), "y": oy.tolist()}


def test_efd_shape_space_separates_straight_and_bent() -> None:
    rng = np.random.default_rng(3)
    rows: list[dict] = []
    for _ in range(6):  # straight capsules, tiny geometry jitter
        _, outline = _capsule_geometry(
            40.0,
            40.0,
            angle=float(0.02 * rng.normal()),
            length=float(30.0 * (1.0 + 0.02 * rng.normal())),
            width=float(8.0 * (1.0 + 0.02 * rng.normal())),
        )
        rows.append({"outline": outline, "group": 0})
    for _ in range(6):  # strongly bent arcs, tiny geometry jitter
        outline = _bent_outline(
            40.0,
            40.0,
            arc_angle=float(1.2 * (1.0 + 0.02 * rng.normal())),
            radius=float(30.0 * (1.0 + 0.02 * rng.normal())),
            width=float(8.0 * (1.0 + 0.02 * rng.normal())),
        )
        rows.append({"outline": outline, "group": 1})
    rows.append({"outline": None, "group": 2})  # no outline -> null PCs

    table = pa.table(
        {
            "outline": pa.array([r["outline"] for r in rows], type=schema.OUTLINE_T),
            "group": pa.array([r["group"] for r in rows], type=pa.int32()),
        }
    )
    out, info = ch.efd_shape_space(table, n_harmonics=8, n_components=3)

    assert isinstance(out, pl.DataFrame)
    for c in (1, 2, 3):
        assert f"efd_pc{c}" in out.columns
        assert out.schema[f"efd_pc{c}"] == pl.Float32
    assert info["n_cells_used"] == 12
    assert len(info["explained_variance_ratio"]) == 3
    assert 0.0 < info["explained_variance_ratio"][0] <= 1.0

    groups = out["group"].to_numpy()
    pc1 = out["efd_pc1"].to_numpy()
    straight, bent = pc1[groups == 0], pc1[groups == 1]
    assert np.isfinite(straight).all()
    assert np.isfinite(bent).all()
    # PC1 separates the morphotypes: group-mean spread exceeds jitter.
    between = float(np.var([straight.mean(), bent.mean()]))
    within = float(0.5 * (straight.var() + bent.var()))
    assert between > within
    assert abs(straight.mean() - bent.mean()) > straight.std() + bent.std()
    # The outline-less row carries null PCs.
    assert out["efd_pc1"].is_null()[12]


# ---------------------------------------------------------------------------
# contact_metrics
# ---------------------------------------------------------------------------


def _contact_cells_table(orientations: list[float]) -> pa.Table:
    return pa.table(
        {
            "label": pa.array([1, 2], type=pa.uint32()),
            "perimeter": pa.array([40.0, 40.0], type=pa.float32()),
            "weighted_orientation": pa.array(orientations, type=pa.float32()),
        }
    )


def test_contact_metrics_isolated_cells_have_no_contact() -> None:
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:20, 10:20] = 1
    labels[10:20, 40:50] = 2
    out = ch.contact_metrics(labels, _contact_cells_table([0.0, 0.0]))

    assert isinstance(out, pl.DataFrame)
    assert out.schema["contact_fraction"] == pl.Float32
    assert out.schema["mean_neighbor_alignment"] == pl.Float32
    np.testing.assert_allclose(out["contact_fraction"].to_numpy(), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        out["mean_neighbor_alignment"].to_numpy(), 0.0, atol=1e-7
    )


def test_contact_metrics_touching_cells_share_boundary() -> None:
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:20, 10:20] = 1
    labels[10:20, 20:30] = 2  # 10 shared 4-neighbour edges with cell 1
    out = ch.contact_metrics(labels, _contact_cells_table([0.0, 0.0]))

    np.testing.assert_allclose(
        out["contact_fraction"].to_numpy(), [10.0 / 40.0, 10.0 / 40.0], rtol=1e-6
    )
    # Parallel orientations -> alignment 1.
    np.testing.assert_allclose(
        out["mean_neighbor_alignment"].to_numpy(), [1.0, 1.0], atol=1e-6
    )


def test_contact_metrics_perpendicular_cells_have_zero_alignment() -> None:
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:20, 10:20] = 1
    labels[10:20, 20:30] = 2
    out = ch.contact_metrics(labels, _contact_cells_table([0.0, float(np.pi / 2)]))

    np.testing.assert_allclose(
        out["contact_fraction"].to_numpy(), [10.0 / 40.0, 10.0 / 40.0], rtol=1e-6
    )
    np.testing.assert_allclose(
        out["mean_neighbor_alignment"].to_numpy(), [0.0, 0.0], atol=1e-6
    )
