"""End-to-end robustness/usability matrix (ARCHITECTURE.md §7, Phase 4).

Runs the canonical chain (``io.import_synth`` -> ``preprocess.drift`` ->
``segment.threshold`` -> ``measure.cells`` -> ``outline.extract`` ->
``midline.extract`` -> ``track.link``) programmatically through
:func:`momia3.pipeline.engine.run` over the full condition matrix of
:mod:`tests.endtoend_matrix` (3 noise levels x 3 drift levels x 2 founder
counts x 2 division lengths x 3 seeds = 108 runs, ~2-3 min total).

Acceptance policy
-----------------
- **Hard assertions** on the clean baseline condition
  (``noise_sigma=0.01``, ``drift=0``, ``n_initial=3``,
  ``division_length_um=4.0``), aggregated over the three seeds:
  foreground IoU >= 0.85, detection-aware tracking edge recall >= 0.95,
  median midline length relative error <= 5%. The evaluation is fully
  deterministic (fixed seeds), so these thresholds cannot flake.
- **Soft-failure logging** for every other matrix cell: conditions whose
  seed-aggregated metrics fall below the thresholds are logged as warnings
  (and aggregated into the report data) without failing the suite — they
  are the degradation profile analysed in ``benchmarks/robustness_report.md``.

The raw per-run rows are persisted to ``benchmarks/robustness_matrix.json``
so the report generator never has to re-run the matrix.

Evaluation notes (see module docstrings of :mod:`tests.endtoend_matrix`):
predicted labels live in drift-corrected coordinates, so truth labels are
aligned by the estimated cumulative shifts before comparison; predicted
instances are matched to truth instances with track-consistent Hungarian
matching (IoU >= 0.25); ``edge_recall`` is detection-aware (its denominator
holds only truth edges whose endpoints were both detected) while
``edge_recall_raw`` scores against all truth edges.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import endtoend_matrix as em  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_JSON = _REPO_ROOT / "benchmarks" / "robustness_matrix.json"

_N_CONDITIONS = (
    len(em.NOISE_LEVELS)
    * len(em.DRIFT_LEVELS)
    * len(em.N_INITIAL_LEVELS)
    * len(em.DIVISION_LENGTHS)
)


@pytest.fixture(scope="session")
def matrix_rows(tmp_path_factory) -> list[dict]:
    """Run the full condition matrix once per test session (deterministic)."""
    root = tmp_path_factory.mktemp("endtoend_matrix")
    rows = em.run_matrix(root)
    em.write_results_json(rows, _RESULTS_JSON)
    return rows


@pytest.fixture(scope="session")
def matrix_summary(matrix_rows) -> list[dict]:
    """Seed-aggregated per-condition summary of the matrix run."""
    return em.summarize(matrix_rows)


# ---------------------------------------------------------------------------
# hard acceptance: clean baseline
# ---------------------------------------------------------------------------


def test_clean_baseline_acceptance(matrix_summary) -> None:
    """The clean baseline condition must meet the §7 usability thresholds."""
    clean = [r for r in matrix_summary if r["clean"]]
    assert len(clean) == 1, f"expected exactly one clean condition, got {len(clean)}"
    agg = clean[0]
    assert agg["n_runs"] == len(em.SEEDS)
    assert agg["fg_iou"] >= em.THRESHOLDS["fg_iou"], (
        f"clean foreground IoU {agg['fg_iou']:.3f} < {em.THRESHOLDS['fg_iou']}"
    )
    assert agg["edge_recall"] >= em.THRESHOLDS["edge_recall"], (
        f"clean detection-aware tracking recall {agg['edge_recall']:.3f} "
        f"< {em.THRESHOLDS['edge_recall']}"
    )
    assert agg["midline_len_relerr"] <= em.THRESHOLDS["midline_len_relerr"], (
        f"clean midline length error {agg['midline_len_relerr']:.4f} "
        f"> {em.THRESHOLDS['midline_len_relerr']}"
    )


def test_clean_baseline_divisions_and_precision(matrix_summary) -> None:
    """Division events and edge precision on the clean baseline stay high."""
    agg = next(r for r in matrix_summary if r["clean"])
    assert agg["division_recall"] >= 0.9
    assert agg["division_precision"] >= 0.9
    assert agg["edge_precision"] >= 0.9


# ---------------------------------------------------------------------------
# matrix integrity
# ---------------------------------------------------------------------------


def test_matrix_completeness(matrix_rows) -> None:
    """Every matrix cell ran end-to-end and produced finite metrics."""
    assert len(matrix_rows) == _N_CONDITIONS * len(em.SEEDS)
    required = (
        "fg_iou",
        "detection",
        "edge_recall",
        "edge_recall_raw",
        "edge_precision",
        "division_recall",
        "division_precision",
        "midline_len_relerr",
    )
    stage_keys = tuple(f"stage_{k}_s" for k in ("import", "drift", "segment", "cells", "outlines", "midlines", "track"))
    for row in matrix_rows:
        for key in required:
            value = row[key]
            assert value == value, f"{row['condition']} s{row['seed']}: {key} is NaN"
            assert 0.0 <= value <= 1.0, f"{row['condition']} s{row['seed']}: {key}={value}"
        for key in stage_keys:
            assert row[key] >= 0.0


def test_results_json_written(matrix_rows) -> None:
    """The raw matrix rows are persisted for the report generator."""
    assert _RESULTS_JSON.is_file()
    loaded = json.loads(_RESULTS_JSON.read_text(encoding="utf-8"))
    assert len(loaded) == len(matrix_rows)


# ---------------------------------------------------------------------------
# soft-failure logging for non-clean conditions
# ---------------------------------------------------------------------------


def test_degraded_conditions_soft_log(matrix_summary, caplog) -> None:
    """Non-clean conditions below threshold are logged, never asserted.

    The degradation profile (which axis hurts which metric) is analysed in
    ``benchmarks/robustness_report.md`` from the same data.
    """
    degraded: list[str] = []
    with caplog.at_level(logging.WARNING, logger=logger.name):
        for agg in matrix_summary:
            if agg["clean"]:
                continue
            failures = []
            if agg["fg_iou"] < em.THRESHOLDS["fg_iou"]:
                failures.append(f"fg_iou={agg['fg_iou']:.3f}")
            if agg["edge_recall"] < em.THRESHOLDS["edge_recall"]:
                failures.append(f"edge_recall={agg['edge_recall']:.3f}")
            if agg["midline_len_relerr"] > em.THRESHOLDS["midline_len_relerr"]:
                failures.append(f"midline_err={agg['midline_len_relerr']:.3f}")
            if failures:
                degraded.append(agg["condition"])
                logger.warning(
                    "soft-failure %s (noise=%.2f drift=%.1f n_init=%d L=%.1f): %s",
                    agg["condition"],
                    agg["noise_sigma"],
                    agg["drift_px_per_frame"],
                    agg["n_initial"],
                    agg["division_length_um"],
                    ", ".join(failures),
                )
    # informational invariants: the matrix indeed contains stress conditions
    assert degraded, "expected at least one degraded condition in the matrix"
    # crowding must be visible in detection coverage (sister-cell merges)
    crowded = [
        a
        for a in matrix_summary
        if a["n_initial"] == 8 and a["noise_sigma"] == 0.01 and a["drift_px_per_frame"] == 0.0
    ]
    sparse = [
        a
        for a in matrix_summary
        if a["n_initial"] == 3 and a["noise_sigma"] == 0.01 and a["drift_px_per_frame"] == 0.0
    ]
    assert crowded and sparse
    assert min(a["detection"] for a in crowded) < min(a["detection"] for a in sparse)
