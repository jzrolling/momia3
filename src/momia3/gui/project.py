"""Filesystem-backed Project sidebar for image loading."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io import is_supported
from .theme import Tokens, build_stylesheet

__all__ = ["ProjectSidebar"]

PATH_ROLE = Qt.UserRole
IMAGE_ROLE = Qt.UserRole + 1
PENDING_ROLE = Qt.UserRole + 2


class ProjectSidebar(QWidget):
    image_selected = Signal(str)

    def __init__(self, path: Path | None = None, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tokens = tokens or Tokens()
        self.setObjectName("ProjectSidebar")
        self.setStyleSheet(build_stylesheet(self.tokens))
        self.project_path = (path or (Path.cwd() / "data")).resolve()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.tokens.space(0.6))
        header = QHBoxLayout()
        title = QLabel("PROJECT")
        title.setProperty("role", "section")
        header.addWidget(title)
        header.addStretch(1)
        refresh = QPushButton("↻")
        refresh.setProperty("variant", "ghost")
        refresh.setFixedWidth(self.tokens.px(24))
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        open_button = QPushButton("Open")
        open_button.setProperty("variant", "ghost")
        open_button.clicked.connect(self._choose_project)
        header.addWidget(open_button)
        root.addLayout(header)

        self.path_label = QLabel()
        self.path_label.setProperty("role", "dim")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.path_label)

        self.tree = QTreeWidget()
        self.tree.setObjectName("ProjectTree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(self.tokens.px(14))
        self.tree.setAnimated(True)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemExpanded.connect(self._on_expanded)
        root.addWidget(self.tree, 1)
        self.open_project(self.project_path)

    def open_project(self, path: str | Path) -> None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_dir():
            return
        self.project_path = candidate
        self.path_label.setText(str(candidate))
        self.path_label.setToolTip(str(candidate))
        self.tree.clear()
        self._populate(self.tree.invisibleRootItem(), candidate)
        self.tree.scrollToTop()
        QTimer.singleShot(0, self.tree.scrollToTop)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.tree.scrollToTop)

    def refresh(self) -> None:
        self.open_project(self.project_path)

    def _choose_project(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Open project folder", str(self.project_path))
        if selected:
            self.open_project(selected)

    def _populate(self, parent: QTreeWidgetItem, directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.is_dir() and is_supported(entry):
                item = QTreeWidgetItem([entry.name])
                item.setData(0, PATH_ROLE, str(entry))
                item.setData(0, IMAGE_ROLE, True)
                item.setToolTip(0, str(entry))
                parent.addChild(item)
            elif entry.is_dir():
                item = QTreeWidgetItem([entry.name])
                item.setData(0, PATH_ROLE, str(entry))
                item.setData(0, IMAGE_ROLE, False)
                item.setData(0, PENDING_ROLE, True)
                item.addChild(QTreeWidgetItem(["…"]))
                item.setToolTip(0, str(entry))
                parent.addChild(item)
            elif is_supported(entry):
                item = QTreeWidgetItem([entry.name])
                item.setData(0, PATH_ROLE, str(entry))
                item.setData(0, IMAGE_ROLE, True)
                item.setToolTip(0, str(entry))
                parent.addChild(item)

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        path = Path(item.data(0, PATH_ROLE))
        if not item.data(0, PENDING_ROLE) or not path.is_dir():
            return
        item.setData(0, PENDING_ROLE, False)
        item.takeChildren()
        self._populate(item, path)

    def _on_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = Path(item.data(0, PATH_ROLE))
        if item.data(0, IMAGE_ROLE):
            self.image_selected.emit(str(path))
            return
        if path.is_dir():
            item.setExpanded(not item.isExpanded())
