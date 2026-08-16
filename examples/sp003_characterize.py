"""Post-pipeline characterization + degenerate_ar sweep for SP_003.

Runs the frame-level characterize ops directly on the pipeline workspace
(store + cells parquet), then sweeps midline degenerate_ar to pick a
threshold on real data. Outputs a QC report figure + sweep table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from momia3.storage.zarr_store import ExperimentStore
from momia3.ops import characterize as cz
from momia3.ops.midline import extract_midlines

WS = Path(__file__).resolve().parent / "data" / "sp003_workspace"
store = ExperimentStore.open(WS / "experiment.zarr", mode="r")
labels = store.get_labels_frame(0, 0)
ch1 = store.get_frame(0, "ch1", 0).astype(np.float32)
ch2 = store.get_frame(0, "ch2", 0).astype(np.float32)
cells = pl.read_parquet(WS / "tables" / "cells.parquet")

# --- frame-level characterize ops (public API path) ---
contact = cz.contact_metrics(labels, cells)
coherence = cz.orientation_coherence(labels, cells)
proj1 = cz.demographic_projection(cells, ch1, labels, n_l=64, n_d=16)
proj2 = cz.demographic_projection(cells, ch2, labels, n_l=64, n_d=16)

cf = contact["contact_fraction"].drop_nulls()
print(f"[contact] fraction median={cf.median():.3f} (chained cells should be >0)")
print(f"[coherence] {coherence}")
print(f"[projection] ch1 {len(proj1)} cells, ch2 {len(proj2)} cells, map 64x16")

# population mean demograph (all cells, mirrored to canonical orientation)
def mean_map(proj):
    arrs = []
    for v in proj.values():
        m = np.asarray(v)
        # orient so the brighter half is on the right (polarity normalization)
        if m[:, : m.shape[1] // 2].mean() > m.mean():
            m = m[:, ::-1]
        arrs.append(m)
    return np.nanmean(np.stack(arrs), axis=0)

mm1, mm2 = mean_map(proj1), mean_map(proj2)

# --- degenerate_ar sweep on real data ---
sweep = {}
for thr in (1.2, 1.3, 1.4):
    res = extract_midlines(labels, params={"n_points": 32, "degenerate_ar": thr},
                           pixel_microns=0.04968)
    q = pl.Series("q", [r["midline_quality"] for r in res.values()])
    curvs = np.array([r["max_curvature"] for r in res.values() if r["midline_quality"] == "refined"])
    sweep[thr] = {
        "refined": int((q == "refined").sum()),
        "geodesic": int((q == "geodesic").sum()),
        "degenerate": int((q == "degenerate").sum()),
        "failed": int((q == "failed").sum()),
        "refined_max_curv_median": float(np.median(curvs)) if len(curvs) else None,
    }
    print(f"[sweep] ar={thr}: {sweep[thr]}")
(WS / "tables" / "degenerate_ar_sweep.json").write_text(json.dumps(sweep, indent=2))

# --- QC report figure ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
axes[0].imshow(mm1, cmap="magma", aspect="auto")
axes[0].set_title("ch1 demograph (mean, n=%d)" % len(proj1))
axes[0].set_xlabel("l (pole→pole)")
axes[1].imshow(mm2, cmap="magma", aspect="auto")
axes[1].set_title("ch2 demograph (mean, n=%d)" % len(proj2))
axes[1].set_xlabel("l (pole→pole)")
efd = pl.read_parquet(WS / "tables" / "cells_efd.parquet")
sc = axes[2].scatter(efd["efd_pc1"], efd["efd_pc2"], c=efd["length_um"], s=8, cmap="viridis")
axes[2].set_xlabel("EFD PC1"); axes[2].set_ylabel("EFD PC2")
fig.colorbar(sc, ax=axes[2], label="length (µm)")
axes[2].set_title("EFD shape space")
axes[3].hist(cf.to_numpy(), bins=40, color="seagreen")
axes[3].set_xlabel("contact fraction"); axes[3].set_ylabel("cells")
axes[3].set_title("cell-cell contact (chains)")
for ax in axes[:2]:
    ax.set_ylabel("d (lateral)")
fig.tight_layout()
fig.savefig(WS / "sp003_characterize_report.png", dpi=110, bbox_inches="tight")
print(f"[fig] {WS/'sp003_characterize_report.png'}")
