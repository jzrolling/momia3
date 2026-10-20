"""Launch a clean napari instance with the momia3 dock widgets.

No data preloaded — use the "momia3: Panels" dock (Dataset tab) to open
any experiment.zarr, or File > Open to drop in a TIFF directly.

Usage: .venv/bin/python examples/launch_napari.py
"""
from __future__ import annotations


def main() -> None:
    import napari
    from qtpy.QtCore import Qt

    from momia3.gui.widgets import MainPanel

    viewer = napari.Viewer(title="momia3")
    # Give the canvas a sane size before first paint: a degenerate (few-px)
    # viewport during layout can fail FBO creation on macOS + PyQt6.
    viewer.window.resize(1400, 900)
    try:
        dock = viewer.window.add_dock_widget(
            MainPanel(viewer), name="momia3: Panels", area="right"
        )
        # The right sidebar is ~20% wider than napari's default.
        viewer.window.resizeDocks(
            [dock], [max(int(dock.width() * 1.2), 460)], Qt.Orientation.Horizontal
        )
    except Exception as exc:  # keep viewer usable even if a widget fails
        print(f"momia3 panels failed: {exc}")
    napari.run()


if __name__ == "__main__":
    main()
