"""Correctness-only benchmark checks: rolling-ball background.

Gated behind ``MOMIA3_BENCH=1`` (skipped by default — these load the momia2
reference and run real image-size workloads). Timing is *not* asserted; only
the correctness contracts behind the numbers in
``benchmarks/benchmark_report.md``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MOMIA3_BENCH") != "1",
    reason="benchmarks are opt-in: set MOMIA3_BENCH=1",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_core import (  # noqa: E402
    UINT16_SCALE,
    _momia2_production_background,
    _pearson,
    load_momia2,
    make_rb_image,
    torch_reference_background,
)


@pytest.fixture(scope="module")
def momia2ref():
    return load_momia2()


def test_fft_fidelity_vs_momia2_torch_reference() -> None:
    """momia3's exact path matches momia2's FFT reference to float precision."""
    from momia3.ops.preprocess import RollingBallParams, rolling_ball_background_fft

    rng = np.random.default_rng(7)
    frame, _ = make_rb_image(rng, 1024, noise=0.05)
    for kind in ("parabola", "ball"):
        bg_ref = torch_reference_background(frame, 50, kind)
        bg3 = frame.astype(np.float32) - rolling_ball_background_fft(
            frame, RollingBallParams(radius=50, kind=kind, shrink_factor=None)
        )
        assert _pearson(bg_ref, bg3) >= 0.98
        assert np.abs(bg_ref - bg3).max() / UINT16_SCALE <= 1e-4


def test_interior_agreement_with_momia2_production(momia2ref) -> None:
    """Different estimators, same background: interior Pearson >= 0.98."""
    from momia3.ops.preprocess import RollingBallParams, rolling_ball_background_fft

    rng = np.random.default_rng(11)
    frame, _ = make_rb_image(rng, 1024, noise=0.05)
    bg3 = frame.astype(np.float32) - rolling_ball_background_fft(
        frame, RollingBallParams(radius=50, kind="parabola", shrink_factor=None)
    )
    bg2 = _momia2_production_background(momia2ref, frame, 50)
    crop = np.s_[50:-50, 50:-50]
    assert _pearson(bg2[crop], bg3[crop]) >= 0.98


def test_shrink_path_tracks_exact_estimate() -> None:
    """Default production path (shrink=4) tracks the exact estimator.

    Fidelity contract behind the >= 50x speedup claim: background-space
    Pearson >= 0.995 full-frame and >= 0.999 in the interior (one kernel
    radius margin; the border band is where the FFT estimator itself is
    zero-padding biased, so it is excluded from the strict bound).
    """
    from momia3.ops.preprocess import RollingBallParams, rolling_ball_background_fft

    rng = np.random.default_rng(7)
    frame, _ = make_rb_image(rng, 1024, noise=0.05)
    exact = rolling_ball_background_fft(
        frame, RollingBallParams(radius=50, kind="parabola", shrink_factor=None)
    )
    fast = rolling_ball_background_fft(
        frame, RollingBallParams(radius=50, kind="parabola", shrink_factor=4)
    )
    frame32 = frame.astype(np.float32)
    bg_exact = frame32 - exact
    bg_fast = frame32 - fast
    assert _pearson(bg_fast, bg_exact) >= 0.995
    crop = np.s_[50:-50, 50:-50]
    assert _pearson(bg_fast[crop], bg_exact[crop]) >= 0.999


def test_shrink_path_agrees_with_momia2_production(momia2ref) -> None:
    """The shrink path also matches momia2's production estimator interior.

    Both sides amortize with block-min shrink + bilinear upsampling, so
    the production paths should agree at least as well as the exact-path
    cross-estimator check (>= 0.98).
    """
    from momia3.ops.preprocess import RollingBallParams, rolling_ball_background_fft

    rng = np.random.default_rng(11)
    frame, _ = make_rb_image(rng, 1024, noise=0.05)
    bg3 = frame.astype(np.float32) - rolling_ball_background_fft(
        frame, RollingBallParams(radius=50, kind="parabola", shrink_factor=4)
    )
    bg2 = _momia2_production_background(momia2ref, frame, 50)
    crop = np.s_[50:-50, 50:-50]
    assert _pearson(bg2[crop], bg3[crop]) >= 0.98
