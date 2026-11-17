"""Filesystem-backed Project sidebar.

A project is a logical workspace, not necessarily a folder.  It can contain
folder sources and loose image files; both appear in one tree and use the same
single-click-to-load path.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSize, QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io import is_supported
from .icons import line_icon
from .theme import Tokens, build_stylesheet

__all__ = ["ProjectSidebar"]

PATH_ROLE = Qt.UserRole
IMAGE_ROLE = Qt.UserRole + 1
PENDING_ROLE = Qt.UserRole + 2


class _ProjectTree(QTreeWidget):
    paths_dropped = Signal(list)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ProjectSidebar(QWidget):
    image_selected = Signal(str)

    def __init__(self, path: Path | None = None, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tokens = tokens or Tokens()
        self.setObjectName("ProjectSidebar")
        self.setStyleSheet(build_stylesheet(self.tokens))
        self.project_path: Path | None = (path or (Path.cwd() / "data")).resolve()
        self.loose_files: list[Path] = []
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.tokens.space(0.6))
        header = QHBoxLayout()
        title = QLabel("PROJECT")
        title.setProperty("role", "section")
        header.addWidget(title)
        header.addStretch(1)
        self.folder_button = self._tool_button("folder", "Open project folder", self._choose_project)
        header.addWidget(self.folder_button)
        self.add_files_button = self._tool_button("plus", "Add image files", self._add_images_dialog)
        header.addWidget(self.add_files_button)
        self.refresh_button = self._tool_button("refresh", "Refresh project tree", self.refresh)
        header.addWidget(self.refresh_button)
        root.addLayout(header)

        self.path_label = QLabel()
        self.path_label.setProperty("role", "dim")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.path_label)

        self.tree = _ProjectTree()
        self.tree.setObjectName("ProjectTree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(self.tokens.px(14))
        self.tree.setAnimated(True)
        self.tree.setIconSize(QSize(14, 14))
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.paths_dropped.connect(self._drop_paths)
        self.tree.setAcceptDrops(True)
        root.addWidget(self.tree, 1)
        self.open_project(self.project_path)

    def _tool_button(self, icon: str, tooltip: str, callback) -> QToolButton:
        button = QToolButton()
        button.setObjectName("ProjectToolButton")
        button.setIcon(line_icon(icon, "#b8b8b8"))
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(24, 24)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def open_project(self, path: str | Path) -> None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_dir():
            return
        self.project_path = candidate
        self.loose_files = []
        self.refresh()

    def add_images(self, paths: list[str | Path]) -> None:
        selected = [Path(path).expanduser().resolve() for path in paths]
        for path in selected:
            if path.is_dir() and not is_supported(path):
                # Dropping/choosing a normal folder starts a new project.
                self.open_project(path)
                return
            if not is_supported(path):
                continue
            if path not in self.loose_files:
                self.loose_files.append(path)
        self.refresh()

    def refresh(self) -> None:
        self._refresh_label()
        self.tree.clear()
        if self.loose_files:
            loose = QTreeWidgetItem(["Loose images"])
            loose.setFlags(loose.flags() & ~Qt.ItemIsSelectable)
            for path in self.loose_files:
                loose.addChild(self._image_item(path))
            self.tree.addTopLevelItem(loose)
            loose.setExpanded(True)
        if self.project_path is not None:
            self._populate(self.tree.invisibleRootItem(), self.project_path)
        if not self.tree.topLevelItemCount():
            placeholder = QTreeWidgetItem(["Open a folder or add images"])
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(placeholder)
        self.tree.scrollToTop()
        QTimer.singleShot(0, self.tree.scrollToTop)

    def _refresh_label(self) -> None:
        if self.project_path is None:
            text = "Unsaved project"
            tooltip = text
        else:
            text = str(self.project_path)
            tooltip = text
        if self.loose_files:
            text += f" · {len(self.loose_files)} loose"
        self.path_label.setText(text)
        self.path_label.setToolTip(tooltip)

    def _choose_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Open project folder", str(self.project_path or Path.cwd()))
        if selected:
            self.open_project(selected)

    def _add_images_dialog(self) -> None:
        start = str(self.project_path or Path.cwd())
        files, _ = QFileDialog.getOpenFileNames(self, "Add image files", start, "Images (*.tif *.tiff);;All files (*)")
        if files:
            self.add_images(files)

    def _drop_paths(self, paths: list[Path]) -> None:
        self.add_images(paths)

    def _populate(self, parent: QTreeWidgetItem, directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir() and is_supported(entry):
                item = self._image_item(entry)
                parent.addChild(item)
            elif entry.is_dir():
                item = QTreeWidgetItem([entry.name])
                item.setIcon(0, line_icon("folder", "#9e9e9e"))
                item.setData(0, PATH_ROLE, str(entry))
                item.setData(0, IMAGE_ROLE, False)
                item.setData(0, PENDING_ROLE, True)
                item.addChild(QTreeWidgetItem(["…"]))
                item.setToolTip(0, str(entry))
                parent.addChild(item)
            elif is_supported(entry):
                parent.addChild(self._image_item(entry))

    def _image_item(self, path: Path) -> QTreeWidgetItem:
        item = QTreeWidgetItem([path.name])
        item.setIcon(0, line_icon("image", "#b8b8b8"))
        item.setData(0, PATH_ROLE, str(path))
        item.setData(0, IMAGE_ROLE, True)
        item.setToolTip(0, str(path))
        return item

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        raw = item.data(0, PATH_ROLE)
        if not item.data(0, PENDING_ROLE) or raw is None:
            return
        path = Path(raw)
        if not path.is_dir():
            return
        item.setData(0, PENDING_ROLE, False)
        item.takeChildren()
        self._populate(item, path)

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        raw = item.data(0, PATH_ROLE)
        if raw is None:
            return
        path = Path(raw)
        if item.data(0, IMAGE_ROLE):
            self.image_selected.emit(str(path))
            return
        if path.is_dir():
            item.setExpanded(not item.isExpanded())
