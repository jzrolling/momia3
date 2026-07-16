"""Correctness-only benchmark checks: overlap matrix.

Gated behind ``MOMIA3_BENCH=1``. Asserts exact equality of the
(label_t, label_t1, pixels) overlap triples between momia2's
``_calculate_overlap_matrix`` and momia3's ``overlap_matrix``; timing is not
asserted.
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

from bench_core import load_momia2, make_overlap_pair  # noqa: E402


def test_overlap_pairs_exact_match() -> None:
    from momia3.ops.track import overlap_matrix

    ref = load_momia2()
    rng = np.random.default_rng(31)
    f1, f2 = make_overlap_pair(rng, size=512, n_cells=40)

    df = ref.tracker._calculate_overlap_matrix(f1, f2, "0", "1")
    ref_pairs = {
        (int(a.split("_")[1]), int(b.split("_")[1])): int(o)
        for a, b, o in zip(df["frame1_id"], df["frame2_id"], df["overlap_area"])
    }
    coo = overlap_matrix(f1, f2)
    got_pairs = {
        (int(r), int(c)): int(v)
        for r, c, v in zip(coo.row, coo.col, coo.data)
    }
    assert got_pairs == ref_pairs
