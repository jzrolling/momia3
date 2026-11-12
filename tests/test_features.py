"""Tests for the aggregated per-cell feature entry point."""

from __future__ import annotations

import numpy as np
import pytest

from momia3.ops.features import (
    FEATURE_GROUPS,
    FeaturePlan,
    cell_features,
)
from momia3.ops.measure import labels_to_cells
from momia3.ops.midline import extract_midlines, fill_cells_midline
from momia3.ops.outline import extract_outlines, fill_cells_geometry
from tests import synth


@pytest.fixture(scope="module")
def frame():
    data = synth.generate_timelapse(
        n_frames=1, shape=(96, 96), n_initial=3, seed=7
    )
    labels = data["labels"][0]
    image = data["images"][0]
    channels = {"gfp": image, "rfp": 1.0 - image}
    cells = labels_to_cells(labels, 0, 0, channels, pixel_microns=0.065)
    outlines = extract_outlines(labels, None, {"n_points": 64})
    cells = fill_cells_geometry(cells, outlines)
    midlines = extract_midlines(labels, outlines, pixel_microns=0.065)
    cells = fill_cells_midline(cells, midlines)
    return labels, channels, cells


def test_feature_plan_validation() -> None:
    plan = FeaturePlan(("shape", "shape", "intensity"))
    assert plan.groups == ("shape", "intensity")
    assert plan.has("intensity")
    assert "shape" in plan.cache_key()
    with pytest.raises(ValueError, match="unknown feature groups"):
        FeaturePlan(("nope",))


def test_group_prerequisites(frame) -> None:
    labels, channels, _ = frame
    with pytest.raises(ValueError, match="cells_table"):
        cell_features(labels, channels, groups=["geometry"])
    with pytest.raises(ValueError, match="at least two channels"):
        cell_features(labels, {"gfp": channels["gfp"]}, groups=["coloc"])
    with pytest.raises(ValueError, match="at least one channel"):
        cell_features(labels, None, groups=["texture"])


def test_shape_features(frame) -> None:
    labels, _, _ = frame
    out = cell_features(labels, groups=["shape"], pixel_microns=0.065)
    n = len(np.unique(labels)) - 1
    for name in (
        "feret_max",
        "feret_min",
        "radius_min",
        "radius_mean",
        "radius_max",
        "convex_area",
        "convex_perimeter",
        "filled_area",
        "equivalent_diameter",
        "euler_number",
        "skeleton_length",
        "angularity",
        "roundness",
        "zernike_0",
        "zernike_29",
    ):
        assert out[name].shape == (n,)
        assert np.isfinite(out[name]).all(), name
    assert out["feret_max"].min() >= out["feret_min"].min()
    assert out["zernike_0"].min() > 0


def test_all_groups(frame) -> None:
    labels, channels, cells = frame
    out = cell_features(labels, channels, cells, pixel_microns=0.065)
    assert set(out) == set(out)
    assert len(out) > 100
    assert "gfp_mad" in out
    assert "gfp_snr" in out
    assert "gfp_haralick_entropy" in out
    assert "gfp_granularity" in out
    assert "gfp_rfp_pearson" in out
    assert "coloc_gfp_rfp_costes_tau_a" in out
    assert "coloc_gfp_rfp_object_dice" in out
    assert "coloc_gfp_rfp_angle_0" in out
    assert "constriction_depth" in out
    assert "septum_position" in out
    assert "puncta_gfp_count" in out
    assert "puncta_gfp_ld_l_mean" in out
    for name in FEATURE_GROUPS:
        assert FeaturePlan((name,)).has(name)


def test_explicit_group_subset(frame) -> None:
    labels, channels, _ = frame
    out = cell_features(
        labels, channels, groups=["intensity", "texture"], pixel_microns=0.065
    )
    assert "gfp_mad" in out
    assert "gfp_haralick_contrast" in out
    assert "feret_max" not in out
    assert "puncta_gfp_count" not in out


def test_registry_entry() -> None:
    from momia3.pipeline import registry

    spec = registry.get("measure.cell_features")
    assert spec.params_model.__name__ == "CellFeaturesParams"
    assert spec.shard == "frame"
