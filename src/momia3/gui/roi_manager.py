"""ROI Manager draft session and pop-out window.

The manager keeps a committed ROI set and a draft ROI set.  Edits touch only
the draft; ``Update`` commits the draft and keeps the immediately previous set
for a single rollback.  Cell/track tables are intentionally untouched here:
applying ROI overrides to a characterization session is a separate action.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal
from uuid import uuid4

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import Tokens, build_stylesheet, initialize_application

__all__ = ["ROI", "ROIAction", "ROISession", "ROIManagerWindow", "demo_roi_session"]

ROIKind = Literal["box", "polygon", "polyline", "point"]
MAX_ACTIONS = 50


@dataclass
class ROI:
    roi_id: str
    name: str
    kind: ROIKind
    geometry: dict
    cell_key: int | None = None
    track_id: int | None = None
    frame: int | None = None
    state: str = "auto"
    visible: bool = True
    color: str = "#2196f3"

    def clone(self) -> "ROI":
        return replace(self, geometry=deepcopy(self.geometry))

    def summary(self) -> str:
        if self.kind == "box":
            g = self.geometry
            return f"x={g['x']} y={g['y']} w={g['w']} h={g['h']}"
        pts = self.geometry.get("points", [])
        return f"{len(pts)} vertex/vertices"


@dataclass
class ROIAction:
    kind: str
    roi_id: str
    before: ROI | None
    after: ROI | None
    label: str
    at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class ROISet:
    rois: dict[str, ROI] = field(default_factory=dict)

    def clone(self) -> "ROISet":
        return ROISet({key: roi.clone() for key, roi in self.rois.items()})

    def __len__(self) -> int:
        return len(self.rois)


class ROISession:
    """Committed baseline + editable draft + 50-action undo stack."""

    def __init__(self, rois: ROISet | None = None) -> None:
        self.base = (rois or ROISet()).clone()
        self.current = self.base.clone()
        self.previous_roi_set: ROISet | None = None
        self.actions: deque[ROIAction] = deque(maxlen=MAX_ACTIONS)
        self.record: list[ROIAction] = []
        self.record_enabled = False

    def _log(self, action: ROIAction) -> None:
        if self.record_enabled:
            self.record.append(action)

    def add_roi(self, roi: ROI) -> None:
        self.current.rois[roi.roi_id] = roi.clone()
        action = ROIAction("add", roi.roi_id, None, roi.clone(), f"add {roi.name}")
        self.actions.append(action)
        self._log(action)

    def delete_roi(self, roi_id: str) -> None:
        before = self.current.rois.pop(roi_id, None)
        if before is None:
            return
        action = ROIAction("delete", roi_id, before, None, f"delete {before.name}")
        self.actions.append(action)
        self._log(action)

    def replace_roi(self, roi: ROI) -> None:
        before = self.current.rois.get(roi.roi_id)
        if before is None:
            self.add_roi(roi)
            return
        self.current.rois[roi.roi_id] = roi.clone()
        action = ROIAction("modify", roi.roi_id, before.clone(), roi.clone(), f"modify {roi.name}")
        self.actions.append(action)
        self._log(action)

    def undo(self) -> bool:
        if not self.actions:
            return False
        action = self.actions.pop()
        if action.before is None:
            self.current.rois.pop(action.roi_id, None)
        else:
            self.current.rois[action.roi_id] = action.before.clone()
        return True

    def update(self) -> bool:
        """Commit the draft and keep one previous committed set."""
        if self.current.rois == self.base.rois:
            return False
        self.previous_roi_set = self.base.clone()
        self.base = self.current.clone()
        self.actions.clear()
        return True

    def rollback_update(self) -> bool:
        if self.previous_roi_set is None:
            return False
        self.base = self.previous_roi_set.clone()
        self.current = self.base.clone()
        self.previous_roi_set = None
        self.actions.clear()
        return True

    def set_record(self, enabled: bool) -> None:
        self.record_enabled = enabled


class ROIManagerWindow(QWidget):
    """Pop-out ROI Manager."""

    closed = Signal()

    def __init__(self, session: ROISession, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tokens = tokens or Tokens()
        self.session = session
        self.setWindowFlag(Qt.Window, True)
        self.setObjectName("ROIManagerWindow")
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setWindowTitle("momia3 · ROI Manager")
        default_width = self.tokens.px(520)
        if parent is not None and parent.width() > 0:
            default_width = max(self.tokens.px(420), parent.width() // 3)
        self.resize(default_width, self.tokens.px(660))
        self.setStyleSheet(build_stylesheet(self.tokens))
        initialize_application()

        root = QVBoxLayout(self)
        root.setContentsMargins(self.tokens.space(1.5), self.tokens.space(1.2), self.tokens.space(1.5), self.tokens.space(1.2))
        root.setSpacing(self.tokens.space(0.8))

        header = QHBoxLayout()
        title = QLabel("ROI Manager")
        title.setStyleSheet("font-weight: bold; font-size: 15px; color: #e8e8e8;")
        header.addWidget(title)
        self.summary = QLabel()
        self.summary.setProperty("role", "dim")
        header.addWidget(self.summary)
        header.addStretch(1)
        root.addLayout(header)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(self.tokens.space(0.6))
        self.record_button = QPushButton("Record")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self._toggle_record)
        toolbar.addWidget(self.record_button)
        self.add_button = QPushButton("New ROI")
        self.add_button.clicked.connect(self._add_demo_roi)
        toolbar.addWidget(self.add_button)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected)
        toolbar.addWidget(self.delete_button)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._undo)
        toolbar.addWidget(self.undo_button)
        toolbar.addStretch(1)
        self.rollback_button = QPushButton("Rollback update")
        self.rollback_button.clicked.connect(self._rollback)
        toolbar.addWidget(self.rollback_button)
        self.update_button = QPushButton("Update")
        self.update_button.setProperty("variant", "primary")
        self.update_button.clicked.connect(self._update)
        toolbar.addWidget(self.update_button)
        root.addLayout(toolbar)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter ROI by name, cell, track...")
        self.filter_edit.textChanged.connect(self._refresh)
        filter_row.addWidget(self.filter_edit, 1)
        self.kind_filter = QComboBox()
        self.kind_filter.addItems(["all types", "box", "polygon", "polyline", "point"])
        self.kind_filter.currentTextChanged.connect(self._refresh)
        filter_row.addWidget(self.kind_filter)
        root.addLayout(filter_row)

        self.table = QTableWidget(0, 6)
        self.table.setObjectName("ROITable")
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Cell/Track", "Frame", "Geometry", "State"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in range(6):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        header.setStretchLastSection(False)
        widths = (170, 70, 80, 48, 190, 55)
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, self.tokens.px(width))
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        root.addWidget(self.table, 3)

        record_title = QLabel("SESSION RECORD")
        record_title.setProperty("role", "section")
        root.addWidget(record_title)
        self.record_table = QTableWidget(0, 3)
        self.record_table.setObjectName("ROIRecordTable")
        self.record_table.setHorizontalHeaderLabels(["Time", "Action", "ROI"])
        self.record_table.verticalHeader().setVisible(False)
        self.record_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.record_table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.record_table, 1)

        footer = QHBoxLayout()
        self.status = QLabel()
        self.status.setProperty("role", "dim")
        footer.addWidget(self.status)
        footer.addStretch(1)
        close = QPushButton("Close")
        close.setProperty("variant", "ghost")
        close.setMinimumSize(self.tokens.px(64), self.tokens.px(26))
        close.setStyleSheet("color: #e8e8e8; padding: 4px 10px;")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        root.addLayout(footer)
        self._refresh()

    def _toggle_record(self, enabled: bool) -> None:
        self.session.set_record(enabled)
        self.status.setText("Recording ROI actions." if enabled else "Recording stopped.")
        self._refresh_record()

    def _add_demo_roi(self) -> None:
        index = len(self.session.current.rois) + 1
        roi = ROI(
            roi_id=f"roi-{uuid4().hex[:8]}",
            name=f"rect-{index:03d}",
            kind="box",
            geometry={"x": 100 + index * 12, "y": 80 + index * 8, "w": 64, "h": 48},
            cell_key=index,
            frame=0,
            state="manual",
            color="#f2b45e",
        )
        self.session.add_roi(roi)
        self._refresh()

    def _delete_selected(self) -> None:
        ids = []
        for row in sorted({index.row() for index in self.table.selectedIndexes()}):
            item = self.table.item(row, 0)
            if item is not None:
                ids.append(item.data(Qt.UserRole))
        for roi_id in ids:
            self.session.delete_roi(roi_id)
        self._refresh()

    def _undo(self) -> None:
        if self.session.undo():
            self.status.setText("Undid last ROI action.")
        self._refresh()

    def _update(self) -> None:
        if self.session.update():
            self.status.setText("ROI set committed. Cell/track tables unchanged; previous set available for one rollback.")
        self._refresh()

    def _rollback(self) -> None:
        if self.session.rollback_update():
            self.status.setText("Restored the previous committed ROI set.")
        self._refresh()

    def _refresh(self) -> None:
        term = self.filter_edit.text().strip().lower()
        kind = self.kind_filter.currentText()
        rois = list(self.session.current.rois.values())
        rows = []
        for roi in rois:
            text = f"{roi.name} {roi.cell_key or ''} {roi.track_id or ''}".lower()
            if term and term not in text:
                continue
            if kind != "all types" and roi.kind != kind:
                continue
            rows.append(roi)
        self.table.setRowCount(len(rows))
        for row, roi in enumerate(rows):
            link = "/".join(str(value) if value is not None else "-" for value in (roi.cell_key, roi.track_id))
            values = [roi.name, roi.kind, link, str(roi.frame if roi.frame is not None else "-"), roi.summary(), roi.state]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, roi.roi_id)
                self.table.setItem(row, column, item)
        self.summary.setText(f"{len(self.session.current.rois)} ROIs · draft {len(self.session.actions)}/{MAX_ACTIONS} actions")
        self.undo_button.setEnabled(bool(self.session.actions))
        self.rollback_button.setEnabled(self.session.previous_roi_set is not None)
        self.update_button.setEnabled(self.session.current.rois != self.session.base.rois)
        self._refresh_record()

    def _refresh_record(self) -> None:
        self.record_table.setRowCount(len(self.session.record))
        for row, action in enumerate(reversed(self.session.record)):
            self.record_table.setItem(row, 0, QTableWidgetItem(action.at))
            self.record_table.setItem(row, 1, QTableWidgetItem(action.label))
            self.record_table.setItem(row, 2, QTableWidgetItem(action.roi_id))


def demo_roi_session() -> ROISession:
    rois = ROISet()
    samples = [
        ROI("cell-0001-outline", "cell-0001 outline", "polygon", {"points": [[120, 90], [210, 88], [250, 160], [205, 220], [110, 205]]}, cell_key=1, track_id=12, frame=0, color="#5ad7c4"),
        ROI("cell-0001-bbox", "cell-0001 bbox", "box", {"x": 108, "y": 87, "w": 144, "h": 136}, cell_key=1, track_id=12, frame=0, state="auto", color="#6ba8ff"),
        ROI("cell-0021-midline", "cell-0021 midline", "polyline", {"points": [[420, 120], [450, 210], [430, 300]]}, cell_key=21, track_id=12, frame=0, color="#f2b45e"),
        ROI("cell-0021-pole", "cell-0021 pole", "point", {"points": [[448, 212]]}, cell_key=21, track_id=12, frame=0, color="#b59cff"),
    ]
    for roi in samples:
        rois.rois[roi.roi_id] = roi
    return ROISession(rois)
