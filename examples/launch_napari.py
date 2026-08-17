"""Launch a clean napari instance with the momia3 dock widgets.

No data preloaded — use the "momia3: Dataset" panel to open any
experiment.zarr, or File > Open to drop in a TIFF directly.

Usage: .venv/bin/python examples/launch_napari.py
"""
from __future__ import annotations


def main() -> None:
    import napari

    viewer = napari.Viewer(title="momia3")
    for wname in ("momia3: Dataset", "momia3: Pipeline", "momia3: Results"):
        try:
            viewer.window.add_plugin_dock_widget("momia3", wname)
        except Exception as exc:  # keep viewer usable even if a widget fails
            print(f"dock widget {wname} failed: {exc}")
    napari.run()


if __name__ == "__main__":
    main()
