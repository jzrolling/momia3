"""After-stats: quality distribution + curvature comparison vs baseline JSON."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

from momia3.ops.segment import ThresholdParams, threshold_segment, declump
from momia3.ops.outline import extract_outlines
from momia3.ops.midline import extract_midlines

DATA = Path(__file__).resolve().parent.parent / "data"
PIXEL_MICRONS = 1.0 / (20128301 / 1000000)


def total_curvature(poly: np.ndarray) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    v = np.diff(pts, axis=0)
    keep = np.hypot(v[:, 0], v[:, 1]) > 1e-9
    if keep.sum() < 2:
        return 0.0
    ang = np.arctan2(v[keep, 1], v[keep, 0])
    d = np.diff(ang)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(d)))


with tifffile.TiffFile(DATA / "SP_003.tif") as tf:
    stack = tf.series[0].asarray()
phase = stack[0].astype(np.float32)
mask = threshold_segment(phase / phase.max(), ThresholdParams(method="composite", min_size=50))
labels = declump(mask)
outlines = extract_outlines(labels)

t0 = time.time()
mid = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
dt = time.time() - t0

qualities = Counter(res["midline_quality"] for res in mid.values())
curvs = {lab: total_curvature(np.asarray(res["midline"][0])) for lab, res in mid.items()}
arr = np.array(list(curvs.values()))
print(f"[midline] {len(mid)} midlines in {dt:.1f}s")
print(f"[quality] {dict(qualities)}")
print(f"[curvature-new] mean={arr.mean():.3f} median={np.median(arr):.3f} "
      f"max={arr.max():.3f} p90={np.percentile(arr, 90):.3f}")

base = json.loads((DATA / "sp003_midline_baseline.json").read_text())
bs = base["stats"]
print(f"[curvature-old] mean={bs['mean']:.3f} median={bs['median']:.3f} "
      f"max={bs['max']:.3f} p90={bs['p90']:.3f}")
print(f"[reduction] mean {arr.mean()/bs['mean']:.1%} of old, "
      f"p90 {np.percentile(arr, 90)/bs['p90']:.1%}, max {arr.max()/bs['max']:.1%}")

worst_new = sorted(curvs, key=curvs.get, reverse=True)[:5]
print(f"[worst-new] {[(int(w), round(curvs[w], 2)) for w in worst_new]}")
resids = np.array([r['final_residual'] for r in mid.values()])
resids = resids[np.isfinite(resids)]
print(f"[residual] mean={resids.mean():.3f}px max={resids.max():.3f}px")
niter = [r['n_iter'] for r in mid.values()]
print(f"[n_iter] mean={np.mean(niter):.1f} max={max(niter)}")
