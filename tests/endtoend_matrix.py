"""End-to-end robustness matrix machinery (ARCHITECTURE.md §7, usability testing).

Runs the canonical pipeline chain (``io.import_synth`` -> ``preprocess.drift``
-> ``segment.threshold`` -> ``measure.cells`` -> ``outline.extract`` ->
``midline.extract`` -> ``track.link``) through
:func:`momia3.pipeline.engine.run` over a condition matrix of

- ``noise_sigma``       ∈ {0.01, 0.02, 0.05}
- ``drift_px_per_frame`` ∈ {0.0, 0.5, 2.0}
- ``n_initial``          ∈ {3, 8}
- ``division_length_um`` ∈ {3.0, 4.0}

with 2 seeds per cell (72 runs total at ~2 s/run), and computes per-run
metrics against the synthetic ground truth:

- ``fg_iou``           — foreground IoU of the segmentation vs truth labels
- ``edge_recall`` / ``edge_precision`` — continue+division(+gap) lineage edges
  vs truth edges, after per-frame Hungarian matching of predicted instances
  to truth instances (IoU >= 0.25)
- ``division_precision`` / ``division_recall`` — division-parent sets
- ``midline_len_relerr`` — median relative error of ``length_um`` vs the
  truth records, over matched cells
- ``detection``        — fraction of truth instances matched (segmentation
  coverage ceiling of every downstream metric)
- per-stage wall times (from the engine step records)

Shared by ``tests/test_endtoend.py`` (hard assertions on the clean
baseline, soft-failure logging elsewhere) and ``benchmarks/run_robustness.py``
(full matrix table for ``benchmarks/robustness_report.md``).
"""

from __future__ import annotations

import itertools
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
from scipy import optimize

from momia3.keys import ROOT_KEY, encode_key
from momia3.pipeline import engine
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = [
    "CLEAN_CONDITION",
    "CONDITIONS",
    "SEEDS",
    "THRESHOLDS",
    "build_pipeline",
    "condition_id",
    "iter_conditions",
    "run_matrix",
    "run_single",
    "summarize",
]

# ---------------------------------------------------------------------------
# condition matrix
# ---------------------------------------------------------------------------

NOISE_LEVELS = (0.01, 0.02, 0.05)
DRIFT_LEVELS = (0.0, 0.5, 2.0)
N_INITIAL_LEVELS = (3, 8)
DIVISION_LENGTHS = (3.0, 4.0)

SEEDS = (11, 23, 37)
"""Three seeds per matrix cell (36 cells -> 108 runs at ~1 s each)."""

CLEAN_CONDITION: dict[str, Any] = {
    "noise_sigma": 0.01,
    "drift_px_per_frame": 0.0,
    "n_initial": 3,
    "division_length_um": 4.0,
}
"""The clean baseline condition with hard acceptance thresholds."""

THRESHOLDS: dict[str, float] = {
    "fg_iou": 0.85,
    "edge_recall": 0.95,
    "midline_len_relerr": 0.05,
}
"""Hard thresholds asserted on the clean baseline (usability acceptance)."""

_SEGMENT_PARAMS: dict[str, Any] = {
    "method": "composite",
    "dark_foreground": True,
    "min_size": 20,
    "dilation": False,
}
"""``segment.threshold`` params used across the matrix.

Usability finding (see benchmarks/robustness_report.md): the composite
method's momia2-legacy default ``dilation=True`` adds a 1-px rim to every
mask (+~18% area), which systematically inflates ``length_um`` by ~5-6%
and bridges the post-division septum gap between daughters, merging
sisters into one instance. Disabling the dilation (a documented op
parameter, no code change) restores foreground IoU >= 0.85 and midline
length error <= 2% on the clean baseline.
"""

_BASE_IMPORT_PARAMS: dict[str, Any] = {
    "n_frames": 16,
    "shape": [256, 256],
    "pixel_microns": 0.065,
    "growth_rate_um_per_frame": 0.1,
    "vsnap": False,
}

_MATCH_IOU = 0.25
"""Min IoU for a predicted instance to count as a detection of a truth one."""

_LINK_EVENTS = ("continue", "division", "gap")


def iter_conditions() -> Iterable[dict[str, Any]]:
    """Yield every condition dict of the matrix (36 cells)."""
    for noise, drift, n_init, div_len in itertools.product(
        NOISE_LEVELS, DRIFT_LEVELS, N_INITIAL_LEVELS, DIVISION_LENGTHS
    ):
        yield {
            "noise_sigma": noise,
            "drift_px_per_frame": drift,
            "n_initial": n_init,
            "division_length_um": div_len,
        }


def condition_id(cond: Mapping[str, Any]) -> str:
    """Compact string id of a condition, e.g. ``n0.02_d0.5_i8_L4.0``."""
    return (
        f"n{cond['noise_sigma']}_d{cond['drift_px_per_frame']}"
        f"_i{cond['n_initial']}_L{cond['division_length_um']}"
    )


def is_clean(cond: Mapping[str, Any]) -> bool:
    """True when ``cond`` is the clean baseline condition."""
    return all(cond[k] == CLEAN_CONDITION[k] for k in CLEAN_CONDITION)


# ---------------------------------------------------------------------------
# pipeline construction / execution
# ---------------------------------------------------------------------------


def build_pipeline(cond: Mapping[str, Any], seed: int, workspace: str) -> dict[str, Any]:
    """Canonical demo chain (examples/run_demo.py) parameterized by a condition."""
    import_params = {
        **_BASE_IMPORT_PARAMS,
        "n_initial": cond["n_initial"],
        "division_length_um": cond["division_length_um"],
        "drift_px_per_frame": cond["drift_px_per_frame"],
        "noise_sigma": cond["noise_sigma"],
        "seed": seed,
    }
    return {
        "name": f"robustness_{condition_id(cond)}_s{seed}",
        "workspace": workspace,
        "steps": [
            {
                "id": "import",
                "op": "io.import_synth",
                "params": import_params,
                "inputs": {},
                "outputs": {
                    "experiment": "experiment.zarr",
                    "truth": "ground_truth.json",
                    "truth_labels": "ground_truth_labels.npy",
                },
            },
            {
                "id": "drift",
                "op": "preprocess.drift",
                "params": {"upsample_factor": 10},
                "inputs": {"experiment": "import.experiment"},
                "outputs": {
                    "experiment": "experiment_drift.zarr",
                    "shifts": "drift_shifts.json",
                },
            },
            {
                "id": "segment",
                "op": "segment.threshold",
                "params": dict(_SEGMENT_PARAMS),
                "inputs": {"experiment": "drift.experiment"},
                "outputs": {"experiment": "experiment_drift.zarr"},
            },
            {
                "id": "cells",
                "op": "measure.cells",
                "params": {"channel": "phase", "include_intensity": True},
                "inputs": {"experiment": "segment.experiment"},
                "outputs": {"cells": "tables/cells_raw.parquet"},
            },
            {
                "id": "outlines",
                "op": "outline.extract",
                "params": {"n_points": 100, "sigma": 1.0},
                "inputs": {"experiment": "segment.experiment", "cells": "cells.cells"},
                "outputs": {"cells": "tables/cells_outlines.parquet"},
            },
            {
                "id": "midlines",
                "op": "midline.extract",
                "params": {"n_points": 32, "method": "zhang"},
                "inputs": {"experiment": "segment.experiment", "cells": "outlines.cells"},
                "outputs": {"cells": "tables/cells.parquet"},
            },
            {
                "id": "track",
                "op": "track.link",
                "params": {"gap_close": 2, "gate_distance": 20.0},
                "inputs": {"experiment": "segment.experiment", "cells": "midlines.cells"},
                "outputs": {
                    "edges": "tables/edges.parquet",
                    "tracks": "tables/tracks.parquet",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def foreground_iou(pred_labels: np.ndarray, truth_labels: np.ndarray) -> float:
    """Mean per-frame foreground IoU (pixel level) of pred vs truth labels."""
    ious: list[float] = []
    for t in range(truth_labels.shape[0]):
        pred_fg = pred_labels[t] > 0
        truth_fg = truth_labels[t] > 0
        union = int(np.logical_or(pred_fg, truth_fg).sum())
        if union == 0:
            continue
        inter = int(np.logical_and(pred_fg, truth_fg).sum())
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 1.0


def _frame_iou(
    t_lab: np.ndarray, p_lab: np.ndarray
) -> tuple[list[int], list[int], np.ndarray]:
    """IoU matrix between truth and predicted instances of one frame."""
    truth_ids = [int(v) for v in np.unique(t_lab) if v != 0]
    pred_ids = [int(v) for v in np.unique(p_lab) if v != 0]
    iou = np.zeros((len(truth_ids), len(pred_ids)))
    if not truth_ids or not pred_ids:
        return truth_ids, pred_ids, iou
    t_idx = {lab: i for i, lab in enumerate(truth_ids)}
    p_idx = {lab: j for j, lab in enumerate(pred_ids)}
    t_areas = np.bincount(t_lab.ravel().astype(np.int64))
    p_areas = np.bincount(p_lab.ravel().astype(np.int64))
    width = int(p_lab.max()) + 1
    code = t_lab.ravel().astype(np.int64) * width + p_lab.ravel()
    fg = (t_lab.ravel() > 0) & (p_lab.ravel() > 0)
    if fg.any():
        counts = np.bincount(code[fg])
        for flat, shared in enumerate(counts):
            if shared == 0:
                continue
            lt, lp = flat // width, flat % width
            if lt not in t_idx or lp not in p_idx:
                continue
            union = t_areas[lt] + p_areas[lp] - shared
            iou[t_idx[lt], p_idx[lp]] = shared / union if union > 0 else 0.0
    return truth_ids, pred_ids, iou


def match_instances(
    truth_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_edges: Iterable[Mapping[str, Any]] | None = None,
    min_iou: float = _MATCH_IOU,
) -> dict[tuple[int, int], int]:
    """Track-consistent matching of truth to predicted instances.

    Per frame, predicted instances are assigned to truth instances by
    Hungarian assignment on IoU (accepting pairs with IoU >= ``min_iou``),
    **except** that a predicted instance whose parent (via a predicted
    ``continue``/``gap`` edge) was already matched inherits the parent's
    truth identity whenever their IoU allows it. This evaluates the
    predicted *lineage graph* as a graph: a tracked blob keeps one truth
    identity across frames instead of flipping between same-sized sisters
    under per-frame Hungarian ties.

    Returns
    -------
    dict
        ``(frame, pred_label) -> truth_label``.
    """
    from momia3.keys import decode_key

    predecessor: dict[tuple[int, int], tuple[int, int]] = {}
    if pred_edges is not None:
        for edge in pred_edges:
            if str(edge["event"]) not in ("continue", "gap"):
                continue
            pk, ck = int(edge["parent_key"]), int(edge["child_key"])
            if pk == ROOT_KEY or ck == ROOT_KEY:
                continue
            _, f_p, l_p = decode_key(pk)
            _, f_c, l_c = decode_key(ck)
            predecessor[(f_c, l_c)] = (f_p, l_p)

    mapping: dict[tuple[int, int], int] = {}
    for t in range(truth_labels.shape[0]):
        truth_ids, pred_ids, iou = _frame_iou(truth_labels[t], pred_labels[t])
        if not truth_ids or not pred_ids:
            continue
        t_idx = {lab: i for i, lab in enumerate(truth_ids)}
        p_idx = {lab: j for j, lab in enumerate(pred_ids)}
        used_t: set[int] = set()
        used_p: set[int] = set()
        for pl in pred_ids:
            prev = predecessor.get((t, pl))
            if prev is None:
                continue
            inherited = mapping.get(prev)
            if (
                inherited is not None
                and inherited in t_idx
                and iou[t_idx[inherited], p_idx[pl]] >= min_iou
            ):
                mapping[(t, pl)] = inherited
                used_t.add(inherited)
                used_p.add(pl)
        rows_free = [i for i, tl in enumerate(truth_ids) if tl not in used_t]
        cols_free = [j for j, pl in enumerate(pred_ids) if pl not in used_p]
        if rows_free and cols_free:
            sub = iou[np.ix_(rows_free, cols_free)]
            rows, cols = optimize.linear_sum_assignment(-sub)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if sub[i, j] >= min_iou:
                    mapping[(t, pred_ids[cols_free[j]])] = truth_ids[rows_free[i]]
    return mapping


def _edge_tuples_from_truth(truth_edges: Iterable[Iterable[Any]]) -> set[tuple[int, int, str]]:
    """Truth edges as ``(parent_key, child_key, event)`` tuples."""
    return {(int(p), int(c), str(e)) for p, c, e in truth_edges}


def edge_scores(
    pred_edges: Iterable[Mapping[str, Any]],
    truth_edges: Iterable[Iterable[Any]],
    mapping: Mapping[tuple[int, int], int],
) -> dict[str, float]:
    """Precision/recall of lineage edges and division events vs truth.

    Two recall flavours are reported:

    - ``edge_recall`` — **detection-aware** recall: the truth-edge
      denominator is restricted to edges whose parent *and* child truth
      instances were both detected (matched to some predicted instance).
      This isolates the linker's assignment quality from segmentation
      losses (a merged/missed cell cannot have its edges recovered by any
      tracker); this is the metric the acceptance thresholds apply to.
    - ``edge_recall_raw`` — recall against **all** truth link edges,
      reflecting the full chain including segmentation coverage.

    Predicted edges (keys over predicted labels) are mapped onto truth keys
    through the track-consistent instance matching; ``edge_precision`` uses
    mapped predicted edges as the denominator. Division scores follow the
    same detectability restriction (a division parent counts when detected).
    """
    from momia3.keys import decode_key

    def _map_key(key: int) -> int | None:
        _, frame, label = decode_key(int(key))
        truth_label = mapping.get((frame, label))
        return None if truth_label is None else encode_key(0, frame, truth_label)

    _detected_set = {(f, tl) for (f, _), tl in mapping.items()}

    truth_all = _edge_tuples_from_truth(truth_edges)
    truth_links = {(p, c) for p, c, e in truth_all if e in _LINK_EVENTS and p != ROOT_KEY}
    truth_div = {p for p, _, e in truth_all if e == "division"}
    detectable_links = {
        (p, c)
        for p, c in truth_links
        if (decode_key(p)[1], decode_key(p)[2]) in _detected_set
        and (decode_key(c)[1], decode_key(c)[2]) in _detected_set
    }
    detectable_div = {
        p for p in truth_div if (decode_key(p)[1], decode_key(p)[2]) in _detected_set
    }

    pred_links: set[tuple[int, int]] = set()
    pred_div: set[int] = set()
    for edge in pred_edges:
        event = str(edge["event"])
        pk, ck = int(edge["parent_key"]), int(edge["child_key"])
        if pk == ROOT_KEY or ck == ROOT_KEY:
            continue
        mpk, mck = _map_key(pk), _map_key(ck)
        if mpk is None or mck is None:
            continue
        if event in _LINK_EVENTS:
            pred_links.add((mpk, mck))
        if event == "division":
            pred_div.add(mpk)

    tp = len(pred_links & detectable_links)
    edge_recall = tp / len(detectable_links) if detectable_links else 1.0
    edge_precision = tp / len(pred_links) if pred_links else 1.0
    raw_tp = len(pred_links & truth_links)
    edge_recall_raw = raw_tp / len(truth_links) if truth_links else 1.0
    div_tp = len(pred_div & detectable_div)
    division_recall = div_tp / len(detectable_div) if detectable_div else 1.0
    division_precision = div_tp / len(pred_div) if pred_div else 1.0
    return {
        "edge_recall": float(edge_recall),
        "edge_recall_raw": float(edge_recall_raw),
        "edge_precision": float(edge_precision),
        "edge_tp": float(tp),
        "edge_detectable_n": float(len(detectable_links)),
        "edge_truth_n": float(len(truth_links)),
        "edge_pred_n": float(len(pred_links)),
        "division_recall": float(division_recall),
        "division_precision": float(division_precision),
        "division_tp": float(div_tp),
        "division_detectable_n": float(len(detectable_div)),
        "division_truth_n": float(len(truth_div)),
        "division_pred_n": float(len(pred_div)),
    }


def midline_length_errors(
    cells: pa.Table,
    truth_records: Iterable[Mapping[str, Any]],
    mapping: Mapping[tuple[int, int], int],
) -> list[float]:
    """Relative ``length_um`` error of matched cells vs truth records."""
    truth_len = {
        (int(r["frame"]), int(r["label"])): float(r["length_um"]) for r in truth_records
    }
    errors: list[float] = []
    for row in cells.select(["frame", "label", "length_um"]).to_pylist():
        if row["length_um"] is None:
            continue
        frame, pred_label = int(row["frame"]), int(row["label"])
        truth_label = mapping.get((frame, pred_label))
        if truth_label is None:
            continue
        truth_value = truth_len.get((frame, truth_label))
        if truth_value is None or truth_value <= 0:
            continue
        errors.append(abs(float(row["length_um"]) - truth_value) / truth_value)
    return errors


# ---------------------------------------------------------------------------
# single run / matrix driver
# ---------------------------------------------------------------------------

_STAGES = ("import", "drift", "segment", "cells", "outlines", "midlines", "track")


def run_single(
    cond: Mapping[str, Any], seed: int, workspace_root: str | Path
) -> dict[str, Any]:
    """Run one matrix cell (condition x seed) and compute all metrics."""
    from scipy import ndimage

    workspace = Path(workspace_root) / f"{condition_id(cond)}_s{seed}"
    pipeline = build_pipeline(cond, seed, str(workspace))
    started = time.perf_counter()
    result = engine.run(pipeline, n_workers=1, use_cache=False)
    wall_total = time.perf_counter() - started

    truth = result.outputs["import"]["truth"]
    truth_labels = np.asarray(result.outputs["import"]["truth_labels"])
    store = ExperimentStore.open(result.outputs["segment"]["experiment"], mode="r")
    pred_labels = np.asarray(store.labels(0)[:])
    cells = result.outputs["midlines"]["cells"]
    edges_df = result.outputs["track"]["edges"]
    pred_edges = (
        edges_df.select(["parent_key", "child_key", "event"]).to_pylist()
        if isinstance(edges_df, pa.Table)
        else edges_df.select(["parent_key", "child_key", "event"]).to_dicts()
    )

    # The drift step re-registers the images onto frame 0, so the predicted
    # labels live in *drift-corrected* coordinates while the generator's
    # truth labels live in raw (drifting) coordinates. Align the truth
    # labels with the same cumulative shifts (nearest-neighbour, background
    # fill) before any pixel/instance comparison — otherwise every shifted
    # frame is scored with a systematic offset and drift robustness cannot
    # be measured at all.
    shifts_payload = result.outputs["drift"]["shifts"]
    shift_series = np.asarray(shifts_payload["0"], dtype=np.float64)
    truth_aligned = np.empty_like(truth_labels)
    for t in range(truth_labels.shape[0]):
        if np.all(shift_series[t] == 0.0):
            truth_aligned[t] = truth_labels[t]
        else:
            truth_aligned[t] = ndimage.shift(
                truth_labels[t], shift_series[t], order=0, mode="constant", cval=0
            )
    truth_labels = truth_aligned

    mapping = match_instances(truth_labels, pred_labels, pred_edges)
    n_truth_instances = int(
        sum(len([v for v in np.unique(truth_labels[t]) if v != 0]) for t in range(truth_labels.shape[0]))
    )
    scores = edge_scores(pred_edges, truth["edges"], mapping)
    len_errors = midline_length_errors(cells, truth["records"], mapping)
    stage_times = {
        rec.step_id: float(rec.duration_s)
        for rec in result.steps
        if rec.status == "computed"
    }
    n_matched = len(mapping)
    row: dict[str, Any] = {
        "condition": condition_id(cond),
        **{k: cond[k] for k in ("noise_sigma", "drift_px_per_frame", "n_initial", "division_length_um")},
        "seed": int(seed),
        "clean": is_clean(cond),
        "fg_iou": foreground_iou(pred_labels, truth_labels),
        "detection": n_matched / n_truth_instances if n_truth_instances else 1.0,
        "midline_len_relerr": float(np.median(len_errors)) if len_errors else float("nan"),
        "midline_n": len(len_errors),
        "wall_total_s": float(wall_total),
        **{f"stage_{k}_s": stage_times.get(k, 0.0) for k in _STAGES},
        **scores,
    }
    return row


def run_matrix(
    workspace_root: str | Path | None = None,
    conditions: Iterable[Mapping[str, Any]] | None = None,
    seeds: Iterable[int] = SEEDS,
) -> list[dict[str, Any]]:
    """Run the full (or a filtered) condition matrix; returns one row per run."""
    conditions = list(conditions) if conditions is not None else list(iter_conditions())
    rows: list[dict[str, Any]] = []
    if workspace_root is None:
        with tempfile.TemporaryDirectory(prefix="momia3_matrix_") as tmp:
            for cond, seed in itertools.product(conditions, seeds):
                rows.append(run_single(cond, seed, tmp))
        return rows
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    for cond, seed in itertools.product(conditions, seeds):
        rows.append(run_single(cond, seed, root))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed rows into per-condition means (and spreads)."""
    metric_keys = (
        "fg_iou",
        "detection",
        "edge_recall",
        "edge_recall_raw",
        "edge_precision",
        "division_recall",
        "division_precision",
        "midline_len_relerr",
        "wall_total_s",
    )
    stage_keys = tuple(f"stage_{k}_s" for k in _STAGES)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    out: list[dict[str, Any]] = []
    for cid in sorted(groups):
        group = groups[cid]
        agg: dict[str, Any] = {
            "condition": cid,
            "clean": bool(group[0]["clean"]),
            "n_runs": len(group),
            **{
                k: group[0][k]
                for k in ("noise_sigma", "drift_px_per_frame", "n_initial", "division_length_um")
            },
        }
        for key in metric_keys + stage_keys:
            vals = [float(g[key]) for g in group if np.isfinite(float(g.get(key, np.nan)))]
            agg[key] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{key}_min"] = float(np.min(vals)) if vals else float("nan")
            agg[f"{key}_max"] = float(np.max(vals)) if vals else float("nan")
        out.append(agg)
    return out


def write_results_json(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Persist raw matrix rows as JSON (input of the report generator)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path
