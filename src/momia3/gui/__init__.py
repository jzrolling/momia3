"""napari plugin package of momia3 (ARCHITECTURE.md §8).

Imported lazily: npe2 discovers the plugin through the
``napari.manifest`` entry point (``momia3.gui:napari.yaml``) and only
loads :mod:`momia3.gui.widgets` when a dock widget is actually
requested, so ``import momia3`` never pulls in napari/Qt. The Qt-free
layer builders live in :mod:`momia3.gui.layers`.
"""
