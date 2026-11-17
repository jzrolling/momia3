import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from momia3.gui.project import ProjectSidebar

_APP = QApplication.instance() or QApplication([])


def test_project_sidebar_lists_supported_images_and_folders(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "image.tif").write_bytes(b"")
    (tmp_path / "demo.ome.zarr").mkdir()
    (tmp_path / "demo.ome.zarr" / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    sidebar = ProjectSidebar(tmp_path)
    names = [sidebar.tree.topLevelItem(i).text(0) for i in range(sidebar.tree.topLevelItemCount())]
    assert "nested" in names
    assert "demo.ome.zarr" in names
    assert "notes.txt" not in names
    image_item = sidebar.tree.topLevelItem(names.index("demo.ome.zarr"))
    selected = []
    sidebar.image_selected.connect(selected.append)
    sidebar._on_clicked(image_item, 0)
    assert selected and selected[0].endswith("demo.ome.zarr")


def test_add_loose_images_uses_same_tree(tmp_path: Path):
    image = tmp_path / "single.tif"
    image.write_bytes(b"")
    sidebar = ProjectSidebar(tmp_path)
    sidebar.add_images([image])
    names = [sidebar.tree.topLevelItem(i).text(0) for i in range(sidebar.tree.topLevelItemCount())]
    assert "Loose images" in names
    loose = sidebar.tree.topLevelItem(names.index("Loose images"))
    assert loose.childCount() == 1
    selected = []
    sidebar.image_selected.connect(selected.append)
    sidebar._on_clicked(loose.child(0), 0)
    assert selected == [str(image)]
