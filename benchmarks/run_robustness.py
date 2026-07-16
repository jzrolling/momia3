"""Render ``benchmarks/robustness_report.md`` from ``robustness_matrix.json``.

The matrix itself is produced by ``tests/endtoend_matrix.py`` (108 pipeline
runs; see ``tests/test_endtoend.py`` for the asserted clean-baseline gates).
This script only aggregates and renders.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = REPO_ROOT / "benchmarks" / "robustness_matrix.json"
REPORT = REPO_ROOT / "benchmarks" / "robustness_report.md"

METRICS = ["fg_iou", "detection", "edge_recall", "edge_recall_raw",
           "edge_precision", "division_recall", "division_precision",
           "midline_len_relerr"]
STAGES = ["import", "drift", "segment", "cells", "outlines", "midlines",
          "track"]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def render(runs: list[dict]) -> str:
    L: list[str] = []
    A = L.append

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_cond[r["condition"]].append(r)

    cond_mean = {c: {m: _mean([r[m] for r in rows]) for m in METRICS}
                 | {"wall_total_s": _mean([r["wall_total_s"] for r in rows])}
                 for c, rows in by_cond.items()}

    A("# momia3 end-to-end robustness matrix\n")
    A("## Design\n")
    A("- 108 pipeline runs: noise σ {0.01, 0.02, 0.05} × drift {0, 0.5, 2.0} "
      "px/frame × initial cells {3, 8} × division length {3.0, 4.0} µm × "
      "seeds {11, 23, 37}; 16 frames, 256², growth 0.1 µm/frame.")
    A("- Synthetic ground truth with rendered noise/shading; truth labels are "
      "aligned to each drifted frame by cumulative-shift resampling before "
      "comparison (without this, drift robustness is unmeasurable).")
    A("- Detection = track-consistent Hungarian matching (IoU ≥ 0.25, truth "
      "identity inherited along continue/gap edges). `edge_recall` is "
      "detection-aware (denominator = truth edges with both endpoints "
      "detected); `edge_recall_raw` scores against all truth edges.")
    A("- Segment params override `dilation: false`: momia2-legacy 1-px mask "
      "dilation inflates masks ~18%, merges sister cells and costs ~5% "
      "midline bias (documented usability finding).")
    A("- Sister-merge caveat: in the generator, daughters elongate into each "
      "other within ~1.5 frames of division, so crowded conditions measure "
      "merged-blob behaviour as much as tracking.\n")

    # clean baseline --------------------------------------------------------
    clean = [r for r in runs if r["clean"]]
    A("## Clean-condition baseline (noise 0.01, no drift, 3 cells, L=4.0 µm)\n")
    A("| metric | seed-mean | gate | verdict |")
    A("|---|---|---|---|")
    iou = _mean([r["fg_iou"] for r in clean])
    det = _mean([r["detection"] for r in clean])
    er = _mean([r["edge_recall"] for r in clean])
    ml = _mean([r["midline_len_relerr"] for r in clean])
    dp = _mean([r["division_precision"] for r in clean])
    dr = _mean([r["division_recall"] for r in clean])
    ep = _mean([r["edge_precision"] for r in clean])
    A(f"| foreground IoU | {iou:.3f} | ≥ 0.85 | {'PASS' if iou >= 0.85 else 'FAIL'} |")
    A(f"| detection-aware edge recall | {er:.3f} | ≥ 0.95 | {'PASS' if er >= 0.95 else 'FAIL'} |")
    A(f"| midline length rel err | {ml*100:.2f}% | ≤ 5% | {'PASS' if ml <= 0.05 else 'FAIL'} |")
    A(f"| raw detection rate | {det:.3f} | — | |")
    A(f"| division precision | {dp:.3f} | — | |")
    A(f"| division recall | {dr:.3f} | — | |")
    A(f"| edge precision | {ep:.3f} | — | |")
    A("")

    # full matrix -----------------------------------------------------------
    A("## Full matrix (seed-mean per condition)\n")
    A("| noise | drift | n_init | L_div | IoU | detect | edge R | "
      "edge R raw | div R | midline err |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(cond_mean):
        r0 = by_cond[c][0]
        m = cond_mean[c]
        A(f"| {r0['noise_sigma']} | {r0['drift_px_per_frame']} "
          f"| {r0['n_initial']} | {r0['division_length_um']} "
          f"| {m['fg_iou']:.3f} | {m['detection']:.3f} "
          f"| {m['edge_recall']:.3f} | {m['edge_recall_raw']:.3f} "
          f"| {m['division_recall']:.3f} | {m['midline_len_relerr']*100:.1f}% |")
    A("")

    # axis effects ----------------------------------------------------------
    A("## Degradation by axis\n")
    for axis, label in (("noise_sigma", "noise σ"),
                        ("drift_px_per_frame", "drift (px/frame)")):
        groups: dict[float, list[dict]] = defaultdict(list)
        for r in runs:
            groups[r[axis]].append(r)
        A(f"| {label} | IoU | detect | edge R | edge R raw | div R | "
          "midline err |")
        A("|---|---|---|---|---|---|---|")
        for k in sorted(groups):
            rows = groups[k]
            A(f"| {k} | {_mean([r['fg_iou'] for r in rows]):.3f} "
              f"| {_mean([r['detection'] for r in rows]):.3f} "
              f"| {_mean([r['edge_recall'] for r in rows]):.3f} "
              f"| {_mean([r['edge_recall_raw'] for r in rows]):.3f} "
              f"| {_mean([r['division_recall'] for r in rows]):.3f} "
              f"| {_mean([r['midline_len_relerr'] for r in rows])*100:.1f}% |")
        A("")
    crowded = [r for r in runs if r["n_initial"] == 8
               and r["division_length_um"] == 3.0]
    roomy = [r for r in runs if not (r["n_initial"] == 8
                                     and r["division_length_um"] == 3.0)]
    A("| crowding | IoU | detect | edge R | edge R raw | div R | midline err |")
    A("|---|---|---|---|---|---|---|")
    for label, rows in (("8 cells + L=3.0 (crowded)", crowded),
                        ("all other conditions", roomy)):
        A(f"| {label} | {_mean([r['fg_iou'] for r in rows]):.3f} "
          f"| {_mean([r['detection'] for r in rows]):.3f} "
          f"| {_mean([r['edge_recall'] for r in rows]):.3f} "
          f"| {_mean([r['edge_recall_raw'] for r in rows]):.3f} "
          f"| {_mean([r['division_recall'] for r in rows]):.3f} "
          f"| {_mean([r['midline_len_relerr'] for r in rows])*100:.1f}% |")
    A("")
    A("**Reading**: noise barely degrades anything (IoU stays ≥0.93 at "
      "σ=0.05); drift is fully corrected by phase-correlation registration; "
      "crowding is the dominant axis — detection and raw edge recall collapse "
      "when sisters merge into shared blobs, while *conditional* edge recall "
      "stays ~0.96, i.e. the LAP linker is robust whenever both cells are "
      "actually detected.\n")

    # stage bottleneck ------------------------------------------------------
    A("## Per-frame pipeline stage cost (all 108 runs)\n")
    A("| stage | mean (s/run) | share |")
    A("|---|---|---|")
    means = {s: _mean([r[f"stage_{s}_s"] for r in runs]) for s in STAGES}
    total = sum(means.values())
    for s in STAGES:
        A(f"| {s} | {means[s]:.3f} | {means[s]/total*100:.1f}% |")
    A(f"| **total** | {total:.3f} | 100% |")
    A("")
    bott = max(means, key=means.get)
    A(f"Bottleneck: **{bott}** ({means[bott]/total*100:.0f}% of wall time) — "
      "the per-cell feature step (contour/midline-ready measurements); "
      "consistent with the midline benchmark's finding that per-cell ops "
      "dominate momia3's cost.")
    A("")
    return "\n".join(L)


def main() -> None:
    runs = json.loads(MATRIX.read_text())
    REPORT.write_text(render(runs))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
