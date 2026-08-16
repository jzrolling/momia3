"""Real-data feasibility test: SP_003.tif (S. pneumoniae, phase + 2 FL channels).

Chain: import → normalize → threshold (phase, dark fg) → declump →
labels_to_cells → outlines → midlines → parquet + QC overlay figure.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tifffile

from momia3.ops.segment import ThresholdParams, threshold_segment, declump
from momia3.ops.measure import labels_to_cells
from momia3.ops.outline import extract_outlines, fill_cells_geometry
from momia3.ops.midline import extract_midlines, fill_cells_midline
from momia3.core.cellset import CellSet

DATA = Path(__file__).resolve().parent.parent / "data"
PIXEL_MICRONS = 1.0 / (20128301 / 1000000)  # XResolution tag: 20.128 px/µm

t0 = time.time()
with tifffile.TiffFile(DATA / "SP_003.tif") as tf:
    stack = tf.series[0].asarray()  # (c=3, y, x) uint16, axes CYX
phase, ch1, ch2 = [stack[i].astype(np.float32) for i in range(3)]
print(f"[load] {stack.shape} uint16, pixel = {PIXEL_MICRONS:.4f} µm  ({time.time()-t0:.1f}s)")

# --- segmentation on phase channel (dark cells on bright background) ---
t0 = time.time()
phase_norm = phase / phase.max()
mask = threshold_segment(phase_norm, ThresholdParams(method="composite", min_size=50))
labels = declump(mask)
n_cells = int(labels.max())
print(f"[segment] mask={mask.mean()*100:.1f}% fg, {n_cells} cells after declump ({time.time()-t0:.1f}s)")

# --- measurement ---
t0 = time.time()
cells = labels_to_cells(labels, fov_id=0, frame=0,
                        intensity={"phase": phase, "ch1": ch1, "ch2": ch2},
                        pixel_microns=PIXEL_MICRONS)
print(f"[measure] {len(cells)} rows x {len(cells.columns)} cols ({time.time()-t0:.1f}s)")

# --- outlines ---
t0 = time.time()
outlines = extract_outlines(labels)
cells = fill_cells_geometry(cells, outlines)
print(f"[outline] {len(outlines)} outlines ({time.time()-t0:.1f}s)")

# --- midlines ---
t0 = time.time()
mid = extract_midlines(labels, outlines, pixel_microns=PIXEL_MICRONS)
cells = fill_cells_midline(cells, mid)
print(f"[midline] {len(mid)} midlines ({time.time()-t0:.1f}s)")

# --- save ---
import polars as pl

out = DATA / "sp003_cells.parquet"
cells = CellSet(pl.from_arrow(cells))  # ops return pyarrow; CellSet wraps polars
cells.to_parquet(out)
print(f"[save] {out}")

# --- QC summary ---
cs = cells.df
L = cs["length_um"].drop_nulls()
W = cs["width_median_um"].drop_nulls()
edge_frac = float(cs["touching_edge"].mean())
print(f"[QC] n={len(cs)}, touching_edge={edge_frac:.1%}")
print(f"[QC] length_um: n={len(L)} median={L.median():.2f} q25={L.quantile(0.25):.2f} q75={L.quantile(0.75):.2f}")
print(f"[QC] width_um : n={len(W)} median={W.median():.2f} q25={W.quantile(0.25):.2f} q75={W.quantile(0.75):.2f}")

# --- overlay figure (full frame downsampled + one 512px crop) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
lo, hi = np.percentile(phase, [0.5, 99.5])
pn = np.clip((phase - lo) / (hi - lo), 0, 1)
axes[0].imshow(pn[::2, ::2], cmap="gray")
axes[0].set_title("phase (2048², downsampled)")
axes[1].imshow(labels[::2, ::2] > 0, cmap="gray")
axes[1].set_title(f"mask+declump ({n_cells} cells)")

crop = np.s_[500:1012, 500:1012]
axes[2].imshow(pn[crop], cmap="gray")
for lb, ol in outlines.items():
    xs, ys = ol[:, 0], ol[:, 1]
    m = (xs >= 500) & (xs < 1012) & (ys >= 500) & (ys < 1012)
    if m.sum() > 5:
        axes[2].plot(xs - 500, ys - 500, "r-", lw=0.7, alpha=0.8)
    if lb in mid:
        for br in mid[lb]["midline"]:
            bx, by = br[:, 0] - 500, br[:, 1] - 500
            if ((bx >= 0) & (bx < 512) & (by >= 0) & (by < 512)).sum() > 2:
                axes[2].plot(bx, by, "c-", lw=0.9)
        p1 = mid[lb]["pole1"]; p2 = mid[lb]["pole2"]
        axes[2].plot([p1[0] - 500, p2[0] - 500], [p1[1] - 500, p2[1] - 500], "y.", ms=4)
axes[2].set_title("crop [500:1012]² outlines+midlines+poles")
axes[2].set_xlim(0, 512)
axes[2].set_ylim(512, 0)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(DATA / "sp003_qc_overlay.png", dpi=110, bbox_inches="tight")

# --- feature histograms ---
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
ax2[0].hist(L.to_numpy(), bins=40, color="steelblue")
ax2[0].set_xlabel("length (µm)"); ax2[0].set_ylabel("cells")
ax2[1].hist(W.to_numpy(), bins=40, color="darkorange")
ax2[1].set_xlabel("median width (µm)")
fig2.tight_layout()
fig2.savefig(DATA / "sp003_qc_hist.png", dpi=110, bbox_inches="tight")
print(f"[fig] sp003_qc_overlay.png, sp003_qc_hist.png  total {time.time()-t0:.0f}s")
