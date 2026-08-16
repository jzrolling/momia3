"""Baseline midline stats on SP_003 with the CURRENT (skeleton-first) algorithm.

Runs the same segmentation chain as real_data_sp003.py, then measures
total curvature of every midline (sum of |turning angle| per resampled
branch) and saves the worst cells' midlines to JSON for before/after
comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from momia3.ops.segment import ThresholdParams, threshold_segment, declump
from momia3.ops.outline import extract_outlines
from momia3.ops.midline import extract_midlines

DATA = Path(__file__).resolve().parent.parent / "data"
PIXEL_MICRONS = 1.0 / (20128301 / 1000000)


def total_curvature(poly: np.ndarray) -> float:
    """Sum of |turning angle| (radians) along a polyline."""
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    v = np.diff(pts, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


def main() -> None:
    with tifffile.TiffFile(DATA / "SP_003.tif") as tf:
        stack = tf.series[0].asarray()
    phase = stack[0].astype(np.float32)
    phase_norm = phase / phase.max()
    mask = threshold_segment(phase_norm, ThresholdParams(method="composite", min_size=50))
    labels = declump(mask)
    outlines = extract_outlines(labels)

    t0 = time.time()
    mid = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
    print(f"[midline] {len(mid)} midlines in {time.time()-t0:.1f}s")

    curvs = {}
    for lab, res in mid.items():
        main = np.asarray(res["midline"][0], dtype=np.float64)
        curvs[lab] = total_curvature(main)
    arr = np.array(list(curvs.values()))
    print(f"[curvature] n={len(arr)} mean={arr.mean():.3f} rad  "
          f"median={np.median(arr):.3f}  max={arr.max():.3f}  "
          f"p90={np.percentile(arr, 90):.3f}")

    worst = sorted(curvs, key=curvs.get, reverse=True)[:10]
    dump = {
        "stats": {"n": len(arr), "mean": float(arr.mean()),
                  "median": float(np.median(arr)), "max": float(arr.max()),
                  "p90": float(np.percentile(arr, 90))},
        "worst_labels": [int(w) for w in worst],
        "worst_midlines": {str(w): np.asarray(mid[w]["midline"][0]).tolist()
                           for w in worst},
    }
    out = DATA / "sp003_midline_baseline.json"
    out.write_text(json.dumps(dump))
    print(f"[save] {out}")
    print(f"[worst] {[(int(w), round(curvs[w], 2)) for w in worst]}")


if __name__ == "__main__":
    sys.exit(main())
