"""Pipeline param-model integration with momia3.ops defaults."""

from __future__ import annotations

import pytest

from momia3.ops import midline as midline_ops
from momia3.ops import outline as outline_ops
from momia3.pipeline.steps import MidlineParams, OutlineParams, _merged_params


def test_outline_params_match_ops_defaults() -> None:
    params = OutlineParams().model_dump(mode="json")
    defaults = outline_ops._DEFAULT_PARAMS
    for key in (
        "sigma",
        "level",
        "simplify",
        "tolerance",
        "smooth",
        "n_points",
        "downsample_factor",
        "smooth_scale",
        "refine_outline_iters",
        "refine_outline_mode",
        "refine_edge_step",
        "refine_edge_cutoff",
        "refine_edge_max_shift",
        "refine_edge_pre_expand",
        "constriction_iters",
        "constriction_image_iters",
    ):
        assert params[key] == defaults[key], key


def test_midline_params_match_ops_defaults() -> None:
    params = MidlineParams().model_dump(mode="json")
    defaults = midline_ops._DEFAULT_PARAMS
    for key in (
        "method",
        "min_branch_length_um",
        "n_points",
        "smooth",
        "knn",
        "pad",
        "midline_method",
        "degenerate_ar",
        "degenerate_refine",
        "degenerate_refine_points",
        "degenerate_solidity_min",
        "equidistant_guard",
        "equidistant_skeleton_from_outline",
        "equidistant_sdf_max_dim",
        "equidistant_n_points_auto",
        "equidistant_inside_clamp",
        "equidistant_branch_min_px",
        "equidistant_branch_min_um",
    ):
        assert params[key] == defaults[key], key


def test_merged_params_advanced_overrides_and_excludes() -> None:
    params = MidlineParams(
        advanced_params={
            "equidistant_axis_points": 64,
            "equidistant_n_points": 96,
        }
    )
    merged = _merged_params(params, exclude=("pixel_microns",))
    assert merged["equidistant_axis_points"] == 64
    assert merged["equidistant_n_points"] == 96
    assert "pixel_microns" not in merged
    assert "advanced_params" not in merged


def test_pipeline_params_validate_ranges() -> None:
    with pytest.raises(Exception):
        OutlineParams(refine_outline_mode="bogus")
    with pytest.raises(Exception):
        MidlineParams(midline_method="bogus")
    with pytest.raises(Exception):
        OutlineParams(refine_outline_iters=-1)
