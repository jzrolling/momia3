"""ROI Manager draft session and pop-out window.

The manager keeps a committed ROI set and a draft ROI set.  Edits touch only
the draft; ``Update`` commits the draft and keeps the immediately previous set
for a single rollback.  Cell/track tables are intentionally untouched here:
applying ROI overrides to a characterization session is a separate action.
"""

from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QCursor, QIcon, QPainter, QPainterPath, QPalette, QPixmap
from qtpy.QtCore import QRect, QRectF, QSize
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..roi_geometry import ROIBuffer, ROIRecord
from .icons import color_square_icon, line_icon
from .theme import Tokens, build_stylesheet, initialize_application

__all__ = ["ROI", "ROIAction", "ROISession", "ROIManagerWindow", "demo_roi_session"]

ROIKind = Literal["box", "polygon", "polyline", "point", "ellipse"]
MAX_ACTIONS = 50
COLOR_ROLE = Qt.UserRole + 2
SWATCH_SIZE = 11
SWATCH_LEFT = 6
TOOL_SIZE = 26
TOOL_ICON_SIZE = 16


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
    label: str = "default"

    def clone(self) -> "ROI":
        return replace(self, geometry=deepcopy(self.geometry))

    def summary(self) -> str:
        if self.kind == "box":
            g = self.geometry
            return f"x={g['x']} y={g['y']} w={g['w']} h={g['h']}"
        if self.kind == "ellipse":
            g = self.geometry
            return f"center=({g['cx']}, {g['cy']}) r=({g['rx']}, {g['ry']})"
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
    def add_roi(self, roi: ROI) -> None:
        self.current.rois[roi.roi_id] = roi.clone()
        action = ROIAction("add", roi.roi_id, None, roi.clone(), f"add {roi.name}")
        self.actions.append(action)

    def delete_roi(self, roi_id: str) -> None:
        before = self.current.rois.pop(roi_id, None)
        if before is None:
            return
        action = ROIAction("delete", roi_id, before, None, f"delete {before.name}")
        self.actions.append(action)

    def replace_roi(self, roi: ROI) -> None:
        before = self.current.rois.get(roi.roi_id)
        if before is None:
            self.add_roi(roi)
            return
        self.current.rois[roi.roi_id] = roi.clone()
        action = ROIAction("modify", roi.roi_id, before.clone(), roi.clone(), f"modify {roi.name}")
        self.actions.append(action)

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


def _roi_points(roi: ROI) -> np.ndarray:
    if roi.kind == "box":
        g = roi.geometry
        x, y, w, h = float(g["x"]), float(g["y"]), float(g["w"]), float(g["h"])
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    if roi.kind == "ellipse":
        g = roi.geometry
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        return np.column_stack([g["cx"] + g["rx"] * np.cos(angles), g["cy"] + g["ry"] * np.sin(angles)]).astype(np.float32)
    return np.asarray(roi.geometry.get("points", []), dtype=np.float32)


def _roi_record(roi: ROI) -> ROIRecord:
    kind = {"box": "rectangle", "point": "dot"}.get(roi.kind, roi.kind)
    return ROIRecord(roi.roi_id, kind, _roi_points(roi), frame=roi.frame or 0, color=roi.color)


class _ColorDelegate(QStyledItemDelegate):
    """Paint a color swatch + text without creating a widget per row."""

    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        color = index.data(COLOR_ROLE) or "#67d9c4"
        text = opt.text
        opt.text = ""
        super().paint(painter, opt, index)
        rect = option.rect
        swatch = QRect(rect.left() + SWATCH_LEFT, rect.center().y() - SWATCH_SIZE // 2, SWATCH_SIZE, SWATCH_SIZE)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(swatch, QColor(color))
        painter.setPen(QColor("#111111"))
        painter.drawRect(swatch.adjusted(0, 0, -1, -1))
        selected = bool(option.state & QStyle.State_Selected)
        painter.setPen(opt.palette.color(QPalette.HighlightedText if selected else QPalette.Text))
        text_rect = rect.adjusted(SWATCH_LEFT + SWATCH_SIZE + 14, 0, -4, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, painter.fontMetrics().elidedText(text, Qt.ElideRight, text_rect.width()))
        painter.restore()


class _AddButton(QToolButton):
    single_clicked = Signal()
    double_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ROIToolButton")
        self.setIcon(line_icon("plus"))
        self.setIconSize(QSize(TOOL_ICON_SIZE, TOOL_ICON_SIZE))
        self.setCheckable(True)
        self.setToolTip("Click: add one ROI\nDouble-click: continuous ROI drawing")
        self.setFixedSize(TOOL_SIZE, TOOL_SIZE)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.single_clicked.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()
        event.accept()


class ROIOverlayController:
    """Links the ROI session and table to the custom Vispy overlay."""

    def __init__(self, viewer: Any, session: ROISession, manager: "ROIManagerWindow") -> None:
        from .roi_overlay import ROIVispyOverlay

        self.viewer = viewer
        self.session = session
        self.manager = manager
        self.overlay = ROIVispyOverlay(viewer.window._qt_viewer.canvas.view)
        self.buffer = ROIBuffer()
        self.id_to_index: dict[str, int] = {}
        self.selected: set[str] = set()
        self.hover_id: str | None = None
        self._drag: dict[str, Any] | None = None
        self._add_mode: dict[str, Any] | None = None
        self._add_points: list[tuple[float, float]] = []
        self._add_last_click = 0.0
        self._add_last_xy: tuple[float, float] | None = None
        self._add_continuous_active = False
        self._on_move_cb = self._on_mouse_move
        self._on_drag_cb = self._on_mouse_drag
        viewer.mouse_move_callbacks.append(self._on_move_cb)
        viewer.mouse_drag_callbacks.append(self._on_drag_cb)
        self.refresh()

    def detach(self) -> None:
        try:
            self.viewer.mouse_move_callbacks.remove(self._on_move_cb)
        except ValueError:
            pass
        try:
            self.viewer.mouse_drag_callbacks.remove(self._on_drag_cb)
        except ValueError:
            pass
        for visual in (self.overlay.fill, self.overlay.line, self.overlay.dots, self.overlay.selection, self.overlay.hover, self.overlay.handles, self.overlay.draft, self.overlay.draft_points):
            visual.visible = False

    def refresh(self) -> None:
        records = [_roi_record(roi) for roi in self.session.current.rois.values() if roi.visible]
        self.buffer.build(records)
        self.id_to_index = {roi_id: index for index, roi_id in enumerate(self.buffer.ids)}
        self.selected &= set(self.id_to_index)
        self.overlay.update(self.buffer)
        self._apply_selection()
        self._clear_hover()
        self._show_handles()

    def start_add_mode(self, kind: str, continuous: bool = False) -> None:
        self._add_mode = {"kind": kind, "continuous": bool(continuous)}
        self._add_continuous_active = bool(continuous)
        self._add_points = []
        self._add_last_click = 0.0
        self._add_last_xy = None
        self._drag = None
        self.overlay.clear_draft()
        self.select_ids(set())

    def cancel_add_mode(self) -> None:
        self._add_mode = None
        self._add_points = []
        self._add_continuous_active = False
        self.overlay.clear_draft()

    def select_ids(self, ids: set[str], sync_manager: bool = True) -> None:
        self.selected = set(ids) & set(self.id_to_index)
        self._apply_selection()
        self._show_handles()
        if sync_manager:
            self.manager._select_rows_from_overlay(self.selected)

    def _indices(self, ids: set[str]) -> list[int]:
        return [self.id_to_index[roi_id] for roi_id in ids if roi_id in self.id_to_index]

    def _apply_selection(self) -> None:
        self.overlay.set_selection(self.buffer.selected_segments(self._indices(self.selected)))

    def _clear_hover(self) -> None:
        self.hover_id = None
        self.overlay.set_hover(np.empty((0, 2, 2), dtype=np.float32))

    def _xy(self, event: Any) -> tuple[float, float] | None:
        position = getattr(event, "position", None)
        if position is None or len(position) < 2:
            return None
        return float(position[-1]), float(position[-2])

    def _on_mouse_move(self, _viewer: Any, event: Any) -> None:
        if self._drag is not None or self._add_mode is not None:
            return
        xy = self._xy(event)
        if xy is None:
            return
        hit = self._hit(*xy)
        if hit == self.hover_id:
            return
        self.hover_id = hit
        if hit is None:
            self.overlay.set_hover(np.empty((0, 2, 2), dtype=np.float32))
        else:
            self.overlay.set_hover(self.buffer.selected_segments([self.id_to_index[hit]]))

    def _on_mouse_drag(self, _viewer: Any, event: Any):
        """Napari drag callbacks are generators: press → move* → release."""
        if getattr(event, "type", "") != "mouse_press":
            return
        xy = self._xy(event)
        if xy is None:
            return
        if self._add_mode is not None:
            kind = self._add_mode["kind"]
            if kind in ("box", "ellipse"):
                yield from self._add_drag_shape(event, kind, xy)
            else:
                self._add_click(kind, xy)
                event.handled = True
                yield
            return
        modifiers = {str(item).lower() for item in getattr(event, "modifiers", ())}
        shift = any("shift" in item for item in modifiers)
        handle = self._hit_handle(*xy)
        if handle is not None and len(self.selected) == 1:
            roi_id = next(iter(self.selected))
            roi = self.session.current.rois[roi_id]
            self._drag = {"mode": "handle", "roi": roi.clone(), "handle": handle, "moved": False}
            event.handled = True
            yield
            while getattr(event, "type", "") == "mouse_move":
                point = self._xy(event)
                if point is not None:
                    self._drag_to(*point)
                    event.handled = True
                yield
            self._finish_drag()
            return
        hit = self._hit(*xy)
        if shift and hit is not None:
            self.select_ids({hit})
            roi = self.session.current.rois[hit]
            self._drag = {"mode": "move", "roi": roi.clone(), "start": (xy[0], xy[1]), "moved": False}
            event.handled = True
            yield
            while getattr(event, "type", "") == "mouse_move":
                point = self._xy(event)
                if point is not None:
                    self._move_to(*point)
                    event.handled = True
                yield
            self._finish_drag()
            return
        multi = any("shift" in item or "control" in item or item == "ctrl" for item in modifiers)
        if hit is None:
            if not multi:
                self.select_ids(set())
        elif multi:
            selected = set(self.selected)
            selected.symmetric_difference_update({hit})
            self.select_ids(selected)
        else:
            self.select_ids({hit})
        yield

    def _add_drag_shape(self, event: Any, kind: str, start: tuple[float, float]):
        self._add_points = [start]
        event.handled = True
        yield
        while getattr(event, "type", "") == "mouse_move":
            point = self._xy(event)
            if point is not None:
                self._add_points = [start, point]
                self._show_shape_draft(kind, start, point)
                event.handled = True
            yield
        if len(self._add_points) == 2:
            x0, y0 = self._add_points[0]
            x1, y1 = self._add_points[1]
            if abs(x1 - x0) >= 2 and abs(y1 - y0) >= 2:
                if kind == "box":
                    geometry = {"x": min(x0, x1), "y": min(y0, y1), "w": abs(x1 - x0), "h": abs(y1 - y0)}
                else:
                    geometry = {"cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "rx": abs(x1 - x0) / 2, "ry": abs(y1 - y0) / 2}
                self._add_new_roi(kind, geometry)
        self._add_points = []
        self.overlay.clear_draft()

    def _add_click(self, kind: str, xy: tuple[float, float]) -> None:
        if kind == "point":
            self._add_new_roi("point", {"points": [[xy[0], xy[1]]]})
            return
        now = time.perf_counter()
        near = self._add_last_xy is not None and np.hypot(xy[0] - self._add_last_xy[0], xy[1] - self._add_last_xy[1]) <= max(5.0, self._tolerance() * 3)
        is_double = bool(self._add_points) and near and (now - self._add_last_click) * 1000 <= QApplication.doubleClickInterval()
        self._add_last_click = now
        self._add_last_xy = xy
        minimum = 3 if kind == "polygon" else 2
        if is_double and len(self._add_points) >= minimum:
            points = np.asarray(self._add_points, dtype=np.float32)
            self._add_new_roi(kind, {"points": points.tolist()})
            return
        self._add_points.append(xy)
        self._show_poly_draft(kind)

    def _show_shape_draft(self, kind: str, start: tuple[float, float], end: tuple[float, float]) -> None:
        x0, y0 = start
        x1, y1 = end
        if kind == "box":
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            segments = self._segments(points, closed=True)
            self.overlay.set_draft(segments, points)
        else:
            angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            points = np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)]).astype(np.float32)
            self.overlay.set_draft(self._segments(points, closed=True), points)

    def _show_poly_draft(self, kind: str) -> None:
        points = np.asarray(self._add_points, dtype=np.float32)
        if len(points) < 2:
            self.overlay.set_draft(np.empty((0, 2, 2), dtype=np.float32), points)
            return
        segments = self._segments(points, closed=False)
        self.overlay.set_draft(segments, points)

    @staticmethod
    def _segments(points: np.ndarray, closed: bool) -> np.ndarray:
        if len(points) < 2:
            return np.empty((0, 2, 2), dtype=np.float32)
        if closed:
            points = np.vstack([points, points[:1]])
        return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)

    def _add_new_roi(self, kind: str, geometry: dict) -> None:
        if self._add_mode is None:
            return
        index = len(self.session.current.rois) + 1
        colors = {"box": self.manager.default_color, "ellipse": self.manager.default_color, "point": self.manager.default_color, "polygon": self.manager.default_color, "polyline": self.manager.default_color}
        roi = ROI(
            roi_id=f"roi-{uuid4().hex[:8]}",
            name=f"{kind}-{index:03d}",
            kind=kind,
            geometry=geometry,
            frame=0,
            state="manual",
            color=colors.get(kind, "#67d9c4"),
            label=self.manager.default_label,
        )
        self.session.add_roi(roi)
        self.manager._refresh()
        self.select_ids({roi.roi_id})
        continuous = bool(self._add_mode.get("continuous"))
        self._add_points = []
        self._add_last_click = 0.0
        self._add_last_xy = None
        self.overlay.clear_draft()
        if not continuous:
            self._add_mode = None
        self.manager._on_add_finished(continuous)

    def _tolerance(self) -> float:
        try:
            zoom = float(self.viewer.camera.zoom)
        except Exception:
            zoom = 1.0
        return max(2.0, 8.0 / max(abs(zoom), 1e-6))

    def _handle_points(self, roi: ROI) -> np.ndarray:
        if roi.kind == "box":
            points = _roi_points(roi)
            return points
        if roi.kind == "ellipse":
            g = roi.geometry
            return np.array([[g["cx"] - g["rx"], g["cy"]], [g["cx"] + g["rx"], g["cy"]], [g["cx"], g["cy"] - g["ry"]], [g["cx"], g["cy"] + g["ry"]]], dtype=np.float32)
        return _roi_points(roi)

    def _show_handles(self) -> None:
        if len(self.selected) != 1:
            self.overlay.set_handles(np.empty((0, 2), dtype=np.float32))
            return
        roi = self.session.current.rois.get(next(iter(self.selected)))
        if roi is None:
            self.overlay.set_handles(np.empty((0, 2), dtype=np.float32))
            return
        self.overlay.set_handles(self._handle_points(roi))

    def _hit_handle(self, x: float, y: float) -> int | None:
        if len(self.selected) != 1:
            return None
        roi = self.session.current.rois.get(next(iter(self.selected)))
        if roi is None:
            return None
        points = self._handle_points(roi)
        if not len(points):
            return None
        distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
        index = int(np.argmin(distances))
        return index if distances[index] <= self._tolerance() else None

    def _drag_to(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        original: ROI = self._drag["roi"]
        roi = original.clone()
        index = int(self._drag["handle"])
        if roi.kind == "box":
            g = original.geometry
            x0, y0, x1, y1 = g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]
            if index in (0, 3):
                x0 = x
            else:
                x1 = x
            if index in (0, 1):
                y0 = y
            else:
                y1 = y
            gx0, gx1 = sorted((x0, x1))
            gy0, gy1 = sorted((y0, y1))
            roi.geometry = {"x": gx0, "y": gy0, "w": max(2.0, gx1 - gx0), "h": max(2.0, gy1 - gy0)}
        elif roi.kind == "ellipse":
            g = dict(original.geometry)
            if index == 0:
                g["rx"] = max(1.0, g["cx"] - x)
            elif index == 1:
                g["rx"] = max(1.0, x - g["cx"])
            elif index == 2:
                g["ry"] = max(1.0, g["cy"] - y)
            else:
                g["ry"] = max(1.0, y - g["cy"])
            roi.geometry = g
        elif roi.kind == "point":
            roi.geometry = {"points": [[x, y]]}
        else:
            points = np.asarray(original.geometry["points"], dtype=np.float32).copy()
            if 0 <= index < len(points):
                points[index] = (x, y)
            roi.geometry = {"points": points.tolist()}
        self.session.current.rois[roi.roi_id] = roi
        self._drag["moved"] = True
        self.buffer.build([_roi_record(item) for item in self.session.current.rois.values() if item.visible])
        self.id_to_index = {roi_id: i for i, roi_id in enumerate(self.buffer.ids)}
        self.overlay.update(self.buffer)
        self._apply_selection()
        self._show_handles()

    def _move_to(self, x: float, y: float) -> None:
        if self._drag is None or self._drag.get("mode") != "move":
            return
        original: ROI = self._drag["roi"]
        sx, sy = self._drag["start"]
        dx, dy = x - sx, y - sy
        roi = original.clone()
        if roi.kind == "box":
            roi.geometry = {**original.geometry, "x": original.geometry["x"] + dx, "y": original.geometry["y"] + dy}
        elif roi.kind == "ellipse":
            roi.geometry = {**original.geometry, "cx": original.geometry["cx"] + dx, "cy": original.geometry["cy"] + dy}
        elif roi.kind == "point":
            point = np.asarray(original.geometry["points"], dtype=np.float32) + (dx, dy)
            roi.geometry = {"points": point.tolist()}
        else:
            points = np.asarray(original.geometry["points"], dtype=np.float32) + (dx, dy)
            roi.geometry = {"points": points.tolist()}
        self.session.current.rois[roi.roi_id] = roi
        self._drag["moved"] = True
        self.buffer.build([_roi_record(item) for item in self.session.current.rois.values() if item.visible])
        self.id_to_index = {roi_id: i for i, roi_id in enumerate(self.buffer.ids)}
        self.overlay.update(self.buffer)
        self._apply_selection()
        self._show_handles()

    def _finish_drag(self) -> None:
        if self._drag is None:
            return
        original: ROI = self._drag["roi"]
        moved = bool(self._drag["moved"])
        current = self.session.current.rois.get(original.roi_id)
        self._drag = None
        if not moved or current is None or current.geometry == original.geometry:
            return
        # Restore baseline, then commit exactly one undoable modification.
        self.session.current.rois[original.roi_id] = original
        self.session.replace_roi(current)
        self.manager._refresh()

    def _hit(self, x: float, y: float) -> str | None:
        tol = 1.5
        candidates = self.buffer.index.query((x - tol, y - tol, x + tol, y + tol))
        for roi_id in sorted(candidates, key=lambda item: self.id_to_index[item], reverse=True):
            roi = self.session.current.rois.get(roi_id)
            if roi is not None and self._contains(roi, x, y, tol):
                return roi_id
        return None

    def _contains(self, roi: ROI, x: float, y: float, tol: float) -> bool:
        if roi.kind == "box":
            g = roi.geometry
            return g["x"] - tol <= x <= g["x"] + g["w"] + tol and g["y"] - tol <= y <= g["y"] + g["h"] + tol
        if roi.kind == "ellipse":
            g = roi.geometry
            dx = (x - g["cx"]) / max(g["rx"] + tol, 1e-6)
            dy = (y - g["cy"]) / max(g["ry"] + tol, 1e-6)
            return dx * dx + dy * dy <= 1.0
        points = _roi_points(roi)
        if roi.kind == "point":
            return bool(np.hypot(points[0, 0] - x, points[0, 1] - y) <= max(4.0, tol))
        if roi.kind == "polyline":
            return _distance_to_polyline(points, x, y) <= tol
        if roi.kind == "polygon":
            return _point_in_polygon(points, x, y) or _distance_to_polyline(np.vstack([points, points[:1]]), x, y) <= tol
        return False


def _point_in_polygon(points: np.ndarray, x: float, y: float) -> bool:
    inside = False
    x0, y0 = points[-1]
    for x1, y1 in points:
        if (y1 > y) != (y0 > y):
            if x < (x0 - x1) * (y - y1) / ((y0 - y1) or 1e-12) + x1:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def _distance_to_polyline(points: np.ndarray, x: float, y: float) -> float:
    if len(points) < 2:
        return float(np.hypot(points[0, 0] - x, points[0, 1] - y))
    p0 = points[:-1]
    p1 = points[1:]
    delta = p1 - p0
    denom = np.sum(delta * delta, axis=1)
    t = np.divide(((x - p0[:, 0]) * delta[:, 0] + (y - p0[:, 1]) * delta[:, 1]), denom, out=np.zeros_like(denom), where=denom != 0)
    t = np.clip(t, 0.0, 1.0)
    nearest = p0 + t[:, None] * delta
    return float(np.min(np.hypot(nearest[:, 0] - x, nearest[:, 1] - y)))


class ROIManagerWindow(QWidget):
    """Pop-out ROI Manager."""

    closed = Signal()

    def __init__(self, session: ROISession, tokens: Tokens | None = None, parent: QWidget | None = None, viewer: Any = None) -> None:
        super().__init__(parent)
        self.tokens = tokens or Tokens()
        self.session = session
        self.viewer = viewer
        self.default_color = "#67d9c4"
        self.default_label = "default"
        self.controller: ROIOverlayController | None = None
        self._row_ids: list[str] = []
        self._refreshing = False
        self._syncing_label = False
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
        toolbar.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.setInsertPolicy(QComboBox.NoInsert)
        self.label_combo.setMinimumWidth(self.tokens.px(110))
        self.label_combo.setToolTip("Annotation label for new/selected ROIs")
        self.label_combo.lineEdit().editingFinished.connect(self._apply_current_label)
        self.label_combo.activated.connect(lambda _index: self._apply_current_label())
        toolbar.addWidget(self.label_combo)
        self.add_type = QComboBox()
        for label, kind in (("Rectangle", "box"), ("Ellipse", "ellipse"), ("Dot", "point"), ("Polygon", "polygon"), ("Polyline", "polyline")):
            self.add_type.addItem(label, kind)
        toolbar.addWidget(self.add_type)
        self.add_type.currentTextChanged.connect(self._update_add_tooltip)
        self.add_button = _AddButton()
        self._update_add_tooltip()
        self.add_button.single_clicked.connect(lambda: self._start_add(False))
        self.add_button.double_clicked.connect(lambda: self._start_add(True))
        toolbar.addWidget(self.add_button)
        self.delete_button = QToolButton()
        self.delete_button.setObjectName("ROIToolButton")
        self.delete_button.setIcon(line_icon("minus"))
        self.delete_button.setIconSize(QSize(TOOL_ICON_SIZE, TOOL_ICON_SIZE))
        self.delete_button.setFixedSize(TOOL_SIZE, TOOL_SIZE)
        self.delete_button.setToolTip("Delete selected ROI")
        self.delete_button.clicked.connect(self._delete_selected)
        toolbar.addWidget(self.delete_button)
        self.undo_button = QToolButton()
        self.undo_button.setObjectName("ROIToolButton")
        self.undo_button.setIcon(line_icon("undo"))
        self.undo_button.setIconSize(QSize(TOOL_ICON_SIZE, TOOL_ICON_SIZE))
        self.undo_button.setFixedSize(TOOL_SIZE, TOOL_SIZE)
        self.undo_button.setToolTip("Undo last ROI action")
        self.undo_button.clicked.connect(self._undo)
        toolbar.addWidget(self.undo_button)
        self.color_button = QToolButton()
        self.color_button.setObjectName("ROIToolButton")
        self.color_button.setIconSize(QSize(TOOL_ICON_SIZE, TOOL_ICON_SIZE))
        self.color_button.setFixedSize(TOOL_SIZE, TOOL_SIZE)
        self.color_button.setToolTip("Default color for new ROIs · click to choose")
        self.color_button.clicked.connect(self._choose_default_color)
        self._update_default_color_icon()
        toolbar.addWidget(self.color_button)
        toolbar.addStretch(1)
        self.rollback_button = QPushButton("Rollback update")
        self.rollback_button.setObjectName("ROISecondaryButton")
        self.rollback_button.clicked.connect(self._rollback)
        toolbar.addWidget(self.rollback_button)
        self.update_button = QPushButton("Update")
        self.update_button.setObjectName("ROIPrimaryButton")
        self.update_button.clicked.connect(self._update)
        toolbar.addWidget(self.update_button)
        root.addLayout(toolbar)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter ROI by name, cell, track...")
        self.filter_edit.textChanged.connect(self._refresh)
        filter_row.addWidget(self.filter_edit, 1)
        self.kind_filter = QComboBox()
        self.kind_filter.addItems(["all types", "box", "ellipse", "polygon", "polyline", "point"])
        self.kind_filter.currentTextChanged.connect(self._refresh)
        filter_row.addWidget(self.kind_filter)
        popup_qss = (
            f"QAbstractItemView {{ background-color: {self.tokens.palette.surface}; color: {self.tokens.palette.text}; "
            f"border: 1px solid {self.tokens.palette.border}; selection-background-color: {self.tokens.palette.accent}; "
            "selection-color: #ffffff; outline: 0; }"
        )
        self.kind_filter.view().setStyleSheet(popup_qss)
        self.add_type.view().setStyleSheet(popup_qss)
        self.label_combo.view().setStyleSheet(popup_qss)
        root.addLayout(filter_row)

        self.table = QTableWidget(0, 8)
        self.table.setObjectName("ROITable")
        self.table.setHorizontalHeaderLabels(["", "Name", "Type", "Cell/Track", "Frame", "Geometry", "Label", "State"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in range(8):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        header.setStretchLastSection(False)
        widths = (28, 160, 65, 70, 42, 150, 72, 50)
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, self.tokens.px(width))
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.setItemDelegateForColumn(0, _ColorDelegate(self.table))
        self.table.cellClicked.connect(self._on_cell_clicked)
        root.addWidget(self.table, 3)

        footer = QHBoxLayout()
        self.status = QLabel()
        self.status.setProperty("role", "dim")
        footer.addWidget(self.status)
        footer.addStretch(1)
        close = QPushButton("Close")
        close.setObjectName("ROICloseButton")
        close.setMinimumSize(self.tokens.px(64), self.tokens.px(26))
        close.clicked.connect(self.close)
        footer.addWidget(close)
        root.addLayout(footer)
        self._refresh()
        if viewer is not None:
            self.controller = ROIOverlayController(viewer, session, self)
            self.status.setText("Click color square to recolor · drag handles to edit · Shift+drag to move.")
        else:
            self.status.setText("Open from HyperView to edit ROI endpoints on the canvas.")

    def closeEvent(self, event) -> None:
        if self.controller is not None:
            self.controller.detach()
            self.controller = None
        self.closed.emit()
        super().closeEvent(event)

    def attach_viewer(self, viewer: Any) -> None:
        if self.controller is None:
            self.controller = ROIOverlayController(viewer, self.session, self)

    def _on_table_selection(self) -> None:
        if self._refreshing:
            return
        if self.controller is not None:
            self.controller.select_ids(set(self._selected_ids()), sync_manager=False)
        self._sync_label_combo()

    def _sync_label_combo(self) -> None:
        ids = self._selected_ids()
        labels = {roi.label for roi in self.session.current.rois.values()} | {"default", self.default_label}
        target = self.session.current.rois[ids[0]].label if ids else self.default_label
        self._syncing_label = True
        self.label_combo.blockSignals(True)
        try:
            self.label_combo.clear()
            self.label_combo.addItems(sorted(labels, key=str.lower))
            self.label_combo.setCurrentText(target or "default")
        finally:
            self.label_combo.blockSignals(False)
            self._syncing_label = False

    def _apply_current_label(self) -> None:
        if self._syncing_label:
            return
        label = self.label_combo.currentText().strip() or "default"
        self.default_label = label
        ids = self._selected_ids()
        if not ids:
            self.status.setText(f"Next ROI label: {label}")
            return
        changed = False
        for roi_id in ids:
            roi = self.session.current.rois[roi_id].clone()
            if roi.label == label:
                continue
            roi.label = label
            self.session.replace_roi(roi)
            changed = True
        if changed:
            self._refresh()

    def _selected_ids(self) -> list[str]:
        ids = []
        for row in sorted({index.row() for index in self.table.selectedIndexes()}):
            if 0 <= row < len(self._row_ids):
                ids.append(self._row_ids[row])
        return ids

    def _select_rows_from_overlay(self, roi_ids: set[str]) -> None:
        self._refreshing = True
        try:
            self.table.clearSelection()
            for row, roi_id in enumerate(self._row_ids):
                if roi_id in roi_ids:
                    self.table.selectRow(row)
                    self.table.scrollToItem(self.table.item(row, 0))
        finally:
            self._refreshing = False
        self._sync_label_combo()

    def _start_add(self, continuous: bool) -> None:
        if self.controller is None:
            self.status.setText("Open ROI Manager from HyperView to draw on the canvas.")
            self.add_button.setChecked(False)
            return
        if continuous and self.controller._add_continuous_active:
            self.controller.cancel_add_mode()
            self.add_button.setChecked(False)
            self.status.setText("Continuous add mode off.")
            return
        if not continuous and self.controller._add_continuous_active:
            return
        kind = self.add_type.currentData()
        self.default_label = self.label_combo.currentText().strip() or "default"
        self.add_button.setChecked(continuous)
        self.controller.start_add_mode(kind, continuous)
        mode = "continuous" if continuous else "single"
        self.status.setText(f"{mode} add · {self.add_type.currentText()} · click canvas to start")

    def _on_add_finished(self, continuous: bool) -> None:
        self.add_button.setChecked(continuous)
        self.status.setText("Continuous add mode." if continuous else "ROI added.")

    def _delete_selected(self) -> None:
        ids = []
        for row in sorted({index.row() for index in self.table.selectedIndexes()}):
            item = self.table.item(row, 0)
            if item is not None:
                ids.append(item.data(Qt.UserRole))
        for roi_id in ids:
            self.session.delete_roi(roi_id)
        self._refresh()

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if column != 0 or row < 0 or row >= len(self._row_ids):
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        rect = self.table.visualItemRect(item)
        position = self.table.viewport().mapFromGlobal(QCursor.pos())
        swatch = QRect(rect.left() + SWATCH_LEFT, rect.center().y() - SWATCH_SIZE // 2, SWATCH_SIZE, SWATCH_SIZE)
        if swatch.adjusted(-3, -3, 3, 3).contains(position):
            self._choose_color([self._row_ids[row]])

    def _update_add_tooltip(self, _text: str | None = None) -> None:
        kind = self.add_type.currentData()
        label = self.add_type.currentText()
        if kind in ("box", "ellipse"):
            action = "click +, then drag on the canvas"
        elif kind == "point":
            action = "click +, then click on the canvas"
        else:
            action = "click +, then click vertices; double-click to finish"
        self.add_button.setToolTip(f"Single click: {action}.\nDouble-click: toggle continuous add mode.\nType: {label}")

    def _update_default_color_icon(self) -> None:
        self.color_button.setIcon(color_square_icon(self.default_color))
        self.color_button.setIconSize(QSize(self.tokens.px(18), self.tokens.px(18)))

    def _choose_default_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.default_color), self, "Default ROI color")
        if color.isValid():
            self.default_color = color.name()
            self._update_default_color_icon()
            self.status.setText(f"Default ROI color: {self.default_color}")

    def _choose_color(self, ids: list[str]) -> None:
        if not ids:
            return
        current = QColor(self.session.current.rois[ids[0]].color)
        color = QColorDialog.getColor(current, self, "ROI color")
        if not color.isValid():
            return
        for roi_id in ids:
            roi = self.session.current.rois[roi_id].clone()
            roi.color = color.name()
            self.session.replace_roi(roi)
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
            text = f"{roi.name} {roi.cell_key or ''} {roi.track_id or ''} {roi.label}".lower()
            if term and term not in text:
                continue
            if kind != "all types" and roi.kind != kind:
                continue
            rows.append(roi)
        selected_before = set(self._selected_ids())
        self._refreshing = True
        try:
            self.table.setRowCount(len(rows))
            self._row_ids = [roi.roi_id for roi in rows]
            for row, roi in enumerate(rows):
                link = "/".join(str(value) if value is not None else "-" for value in (roi.cell_key, roi.track_id))
                values = ["", roi.name, roi.kind, link, str(roi.frame if roi.frame is not None else "-"), roi.summary(), roi.label, roi.state]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.UserRole, roi.roi_id)
                    if column == 0:
                        item.setData(COLOR_ROLE, roi.color)
                    self.table.setItem(row, column, item)
            for row, roi_id in enumerate(self._row_ids):
                if roi_id in selected_before:
                    self.table.selectRow(row)
        finally:
            self._refreshing = False
        self.summary.setText(f"{len(self.session.current.rois)} ROIs · draft {len(self.session.actions)}/{MAX_ACTIONS} actions")
        self.undo_button.setEnabled(bool(self.session.actions))
        self.rollback_button.setEnabled(self.session.previous_roi_set is not None)
        self.update_button.setEnabled(self.session.current.rois != self.session.base.rois)
        if self.controller is not None:
            self.controller.refresh()
        self._sync_label_combo()


def demo_roi_session() -> ROISession:
    rois = ROISet()
    samples = [
        ROI("cell-0001-outline", "cell-0001 outline", "polygon", {"points": [[120, 90], [210, 88], [250, 160], [205, 220], [110, 205]]}, cell_key=1, track_id=12, frame=0, color="#5ad7c4"),
        ROI("cell-0001-bbox", "cell-0001 bbox", "box", {"x": 108, "y": 87, "w": 144, "h": 136}, cell_key=1, track_id=12, frame=0, state="auto", color="#6ba8ff"),
        ROI("cell-0021-midline", "cell-0021 midline", "polyline", {"points": [[420, 120], [450, 210], [430, 300]]}, cell_key=21, track_id=12, frame=0, color="#f2b45e"),
        ROI("cell-0021-pole", "cell-0021 pole", "point", {"points": [[448, 212]]}, cell_key=21, track_id=12, frame=0, color="#b59cff"),
        ROI("cell-0033-ellipse", "cell-0033 ellipse", "ellipse", {"cx": 260, "cy": 170, "rx": 28, "ry": 18}, cell_key=33, track_id=15, frame=0, color="#8be28b"),
    ]
    for roi in samples:
        rois.rois[roi.roi_id] = roi
    return ROISession(rois)


