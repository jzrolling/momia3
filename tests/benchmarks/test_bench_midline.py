"""Correctness-only benchmark checks: midline extraction.

Gated behind ``MOMIA3_BENCH=1``. Runs a small mask set through both engines
and asserts main-branch length agreement; timing is not asserted. momia2
crashes on some smooth-skeleton masks (find_poles IndexError) — those masks
are excluded from the comparison exactly as in
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
    PIXEL_MICRONS,
    _arc_length,
    _banana_mask,
    _pearson,
    load_momia2,
)


def test_midline_lengths_agree() -> None:
    from momia3.ops.midline import extract_midlines

    ref = load_momia2()
    rng = np.random.default_rng(101)
    masks = [
        _banana_mask((128, 128), 64, 64, rng.uniform(0, np.pi),
                     rng.uniform(150, 350), rng.uniform(35, 55),
                     rng.uniform(2.8, 3.4), rng.uniform(0, 6.28),
                     rng.uniform(0, 6.28))
        for _ in range(6)
    ]
    labels = np.zeros((128, 128 * 6), np.int32)
    for i, m in enumerate(masks):
        labels[:, i * 128 : (i + 1) * 128][m] = i + 1

    pm = PIXEL_MICRONS
    len2 = []
    for m in masks:
        try:
            outline = ref.contour.find_contour_marching_squares(m)
            mid = ref.midline.extract_midline(m, outline, pixel_microns=pm,
                                              method="zhang")
            main = max(mid, key=_arc_length) if len(mid) else np.empty((0, 2))
            len2.append(_arc_length(main) * pm)
        except Exception:
            len2.append(None)
    res = extract_midlines(labels)
    len3 = []
    for i in range(len(masks)):
        branches = res[i + 1]["midline"]
        main = max(branches, key=_arc_length) if branches else np.empty((0, 2))
        len3.append(_arc_length(np.asarray(main, float)) * pm)

    pairs = [(a, b) for a, b in zip(len2, len3) if a is not None and b > 0]
    assert len(pairs) >= 4, "too few masks comparable between engines"
    a = np.asarray([p[0] for p in pairs])
    b = np.asarray([p[1] for p in pairs])
    assert _pearson(a, b) >= 0.95
    assert np.median(np.abs(a - b) / np.maximum(a, 1e-9)) <= 0.15
