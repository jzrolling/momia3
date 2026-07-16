"""Tests for momia3.ops.measure (§4.3)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from momia3 import schema
from momia3.keys import decode_keys
from momia3.ops import measure as meas
from tests.synth import generate_timelapse


def _frame(seed: int = 4, shape: tuple[int, int] = (200, 200)) -> dict:
    return generate_timelapse(
        n_frames=2,
        shape=shape,
        n_initial=3,
        drift_px_per_frame=0.0,
        noise_sigma=0.01,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# labels_to_cells
# ---------------------------------------------------------------------------


def test_labels_to_cells_schema_and_keys() -> None:
    data = _frame()
    labels = data["labels"][0]
    image = data["images"][0]
    table = meas.labels_to_cells(
        labels, fov_id=2, frame=5, intensity={"phase": image}, pixel_microns=0.065
    )

    # Re-validation against the canonical schema must succeed.
    validated = schema.validate_cells(table, channels=["phase"])
    assert validated.num_rows == int(labels.max())

    # Keys decode back to (fov_id, frame, label).
    fov, frame, label = decode_keys(table.column("cell_key").to_numpy())
    assert set(fov.tolist()) == {2}
    assert set(frame.tolist()) == {5}
    assert sorted(label.tolist()) == list(range(1, int(labels.max()) + 1))

    # Geometry columns are null placeholders (filled by outline/midline ops).
    for name in ("outline", "midline", "width_profile", "coords"):
        assert table.column(name).null_count == table.num_rows
    # Identity flags and intensity cluster are present.
    assert set(table.column("include").to_pylist()) == {True}
    assert "phase_mean" in table.column_names
    assert "weighted_orientation" in table.column_names


def test_labels_to_cells_area_within_5pct_of_capsule() -> None:
    # Seed 5: frame-0 cells are pairwise non-overlapping, so the rasterized
    # ground truth matches the analytic capsule area to <5% (seeds with
    # overlapping cells deviate through occlusion, not measurement error).
    data = _frame(seed=5)
    labels = data["labels"][0]
    pm = float(data["meta"]["pixel_microns"])
    table = meas.labels_to_cells(labels, fov_id=0, frame=0, pixel_microns=pm)

    areas = {
        int(lb): float(ar)
        for lb, ar in zip(
            table.column("label").to_pylist(), table.column("area").to_pylist()
        )
    }
    truth = [rec for rec in data["truth"] if rec["frame"] == 0]
    assert len(truth) == len(areas)
    for rec in truth:
        length_px = rec["length_um"] / pm
        width_px = rec["width_um"] / pm
        capsule = (length_px - width_px) * width_px + np.pi * (width_px / 2.0) ** 2
        measured = areas[rec["label"]]
        assert abs(measured - capsule) / capsule < 0.05, rec


def test_labels_to_cells_touching_edge_and_empty() -> None:
    labels = np.zeros((40, 40), dtype=np.uint32)
    labels[10:20, 10:20] = 1  # interior
    labels[0:5, 30:38] = 2  # touches the border
    table = meas.labels_to_cells(labels, fov_id=0, frame=0)
    touching = dict(
        zip(table.column("label").to_pylist(), table.column("touching_edge").to_pylist())
    )
    assert touching == {1: False, 2: True}

    empty = meas.labels_to_cells(np.zeros((16, 16), dtype=np.uint32), fov_id=0, frame=0)
    assert empty.num_rows == 0
    schema.validate_cells(empty)

    with pytest.raises(ValueError, match="pixel_microns"):
        meas.labels_to_cells(labels, fov_id=0, frame=0, pixel_microns=0.0)


def test_labels_to_cells_derived_morphology_finite() -> None:
    data = _frame(seed=6)
    labels = data["labels"][0]
    table = meas.labels_to_cells(labels, fov_id=0, frame=0)
    for name in (
        "aspect_ratio",
        "circularity",
        "compactness",
        "eccentricity",
        "solidity",
        "extent",
        "weighted_orientation",
        *[f"hu_{i}" for i in range(7)],
    ):
        arr = table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
        assert np.isfinite(arr).all(), name
    aspect = np.asarray(table.column("aspect_ratio").to_pylist(), dtype=float)
    assert np.all((aspect > 0) & (aspect <= 1.0))


# ---------------------------------------------------------------------------
# intensity_stats
# ---------------------------------------------------------------------------


def test_intensity_stats_matches_manual_computation() -> None:
    rng = np.random.default_rng(12)
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[5:20, 5:15] = 1
    labels[30:45, 30:48] = 2
    image = rng.uniform(0, 1000, (50, 50)).astype(np.float32)

    stats = meas.intensity_stats(labels, image, "gfp")
    assert set(stats) == {f"gfp_{s}" for s in schema.INTENSITY_STATS}
    for label_id in (1, 2):
        pixels = image[labels == label_id].astype(np.float64)
        i = label_id - 1
        assert stats["gfp_mean"][i] == pytest.approx(pixels.mean(), rel=1e-5)
        assert stats["gfp_max"][i] == pytest.approx(pixels.max(), rel=1e-6)
        assert stats["gfp_min"][i] == pytest.approx(pixels.min(), rel=1e-6)
        assert stats["gfp_std"][i] == pytest.approx(pixels.std(), rel=1e-4)
        assert stats["gfp_p25"][i] == pytest.approx(np.percentile(pixels, 25), rel=1e-4)
        assert stats["gfp_p75"][i] == pytest.approx(np.percentile(pixels, 75), rel=1e-4)
        assert stats["gfp_integrated"][i] == pytest.approx(pixels.sum(), rel=1e-4)


def test_intensity_stats_empty_labels() -> None:
    stats = meas.intensity_stats(
        np.zeros((8, 8), dtype=np.int32), np.zeros((8, 8)), "ch"
    )
    assert all(v.shape == (0,) for v in stats.values())


# ---------------------------------------------------------------------------
# texture features
# ---------------------------------------------------------------------------


def test_texture_haralick_shapes_and_finite() -> None:
    data = _frame(seed=8)
    labels = data["labels"][0]
    image = data["images"][0]
    n = int(labels.max())

    feats = meas.texture_haralick(labels, image, levels=64)
    assert set(feats) == {f"haralick_{p}" for p in meas.HARALICK_PROPS}
    for name, arr in feats.items():
        assert arr.shape == (n,), name
        assert np.isfinite(arr).all(), name

    # A constant crop yields zero contrast.
    labels2 = np.zeros((30, 30), dtype=np.uint32)
    labels2[5:15, 5:15] = 1
    const = np.full((30, 30), 7.0, dtype=np.float32)
    feats2 = meas.texture_haralick(labels2, const)
    assert feats2["haralick_contrast"][0] == pytest.approx(0.0)


def test_texture_lbp_histogram() -> None:
    data = _frame(seed=9)
    labels = data["labels"][0]
    image = data["images"][0]
    n = int(labels.max())

    P = 8
    feats = meas.texture_lbp(labels, image, P=P, R=1.0)
    assert set(feats) == {f"lbp_{b}" for b in range(P + 2)}
    for arr in feats.values():
        assert arr.shape == (n,)
        assert np.isfinite(arr).all()
    row_sums = np.sum([feats[f"lbp_{b}"] for b in range(P + 2)], axis=0)
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# add_features
# ---------------------------------------------------------------------------


def test_add_features_appends_and_validates() -> None:
    data = _frame(seed=10)
    labels = data["labels"][0]
    image = data["images"][0]
    table = meas.labels_to_cells(labels, fov_id=0, frame=0)
    n = table.num_rows

    feats = meas.texture_haralick(labels, image)
    out = meas.add_features(table, feats)
    assert out.num_columns == table.num_columns + len(feats)
    assert out.num_rows == n
    assert out.column("haralick_energy").type == pa.float32()

    with pytest.raises(ValueError, match="already exists"):
        meas.add_features(out, {"haralick_energy": np.zeros(n)})
    with pytest.raises(ValueError, match="rows"):
        meas.add_features(table, {"bad": np.zeros(n + 1)})
