"""Launch napari with the SP_003 workspace preloaded + momia3 dock widgets.

Usage: .venv/bin/python examples/launch_napari_sp003.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

WS = Path(__file__).resolve().parent / "data" / "sp003_workspace"


def main() -> None:
    import napari

    from momia3.storage.zarr_store import ExperimentStore
    from momia3.gui.layers import cells_to_midline_shapes, cells_to_outline_shapes

    store = ExperimentStore.open(WS / "experiment.zarr", mode="r")
    cells = pl.read_parquet(WS / "tables" / "cells.parquet")

    viewer = napari.Viewer(title="momia3 — SP_003 (S. pneumoniae)")

    # image channels (t axis present; slider at bottom)
    for ch in store.channels:
        arr = store.get_frame(0, ch, 0)
        viewer.add_image(
            arr, name=ch, colormap="gray",
            contrast_limits=(float(arr.min()), float(arr.max())),
            visible=(ch == "phase"),
        )
    viewer.add_labels(store.get_labels_frame(0, 0).astype(int), name="labels", opacity=0.4)

    outlines, oprops, _ = cells_to_outline_shapes(cells)
    ok = [i for i, s in enumerate(outlines) if len(np.unique(s, axis=0)) >= 3]
    if ok:
        sh = viewer.add_shapes(
            [outlines[i] for i in ok], shape_type="polygon", name="outlines",
            edge_color="#ff3d00", edge_width=1.5, face_color="transparent",
        )
        sh.properties = {k: [v[i] for i in ok] for k, v in oprops.items()}

    midlines, mprops, _ = cells_to_midline_shapes(cells)
    okm = [i for i, s in enumerate(midlines) if len(np.unique(s, axis=0)) >= 2]
    if okm:
        sh = viewer.add_shapes(
            [midlines[i] for i in okm], shape_type="path", name="midlines",
            edge_color="#00e5ff", edge_width=2.0,
        )
        sh.properties = {k: [v[i] for i in okm] for k, v in mprops.items()}

    # momia3 dock widgets (Dataset / Pipeline / Results)
    for wname in ("momia3: Dataset", "momia3: Pipeline", "momia3: Results"):
        try:
            viewer.window.add_plugin_dock_widget("momia3", wname)
        except Exception as exc:  # keep viewer usable even if a widget fails
            print(f"dock widget {wname} failed: {exc}")

    napari.run()


if __name__ == "__main__":
    main()
