"""HyperView: native multi-dimensional image interaction for momia3.

napari supplies the canvas.  HyperView owns the display semantics around it:
a compact channel list in the dataset panel, a pop-out full channel sheet, and
a T/Z bar below the canvas.  The napari layer-list and dims bars are hidden;
they are not a second source of UI truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..io import ChannelData, LoadedImage
from .theme import Tokens, build_stylesheet, initialize_application

__all__ = ["HyperViewChannelPanel", "HyperViewDimensionBar", "HyperViewWidget", "demo_volume"]

CHANNELS = (
    ("DAPI", "blue"),
    ("CD3", "green"),
    ("PanCK", "magenta"),
    ("Ki67", "yellow"),
)

LUT_COLORS = {
    "gray": "#e8e8e8",
    "blue": "#4c8dff",
    "green": "#65d68b",
    "magenta": "#f078d0",
    "yellow": "#f2c45e",
    "cyan": "#63e0e0",
    "red": "#ff6d6d",
}


def demo_volume() -> np.ndarray:
    t, c, z = 8, len(CHANNELS), 6
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((t, c, z, h, w), dtype=np.uint16)
    for ti in range(t):
        cx = w / 2 + 55 * np.sin(ti * 0.45)
        cy = h / 2 + 28 * np.cos(ti * 0.35)
        for zi in range(z):
            zf = np.exp(-((zi - (z - 1) / 2) ** 2) / 2.2)
            for ci in range(c):
                r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                ring = np.exp(-((r - (35 + ci * 9)) ** 2) / (2 * (3.5 + ci) ** 2))
                core = np.exp(-(r**2) / (2 * (18 + ci * 4) ** 2))
                pattern = ring * (0.75 + 0.25 * np.sin(xx / (10 + ci))) + core * (0.25 + ci * 0.1)
                out[ti, ci, zi] = np.clip(pattern * zf * 45000 + np.random.default_rng(ci + zi + ti).normal(150, 90, (h, w)), 0, 65535).astype(np.uint16)
    return out


def _lut_pixmap(name: str, size: int = 18) -> QPixmap:
    color = QColor(LUT_COLORS.get(name, "#e8e8e8"))
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0.0, QColor("#0b0b0b"))
    grad.setColorAt(1.0, color)
    painter.setBrush(grad)
    painter.setPen(QPen(QColor("#5f5f5f"), 1))
    painter.drawRoundedRect(0, 0, size - 1, size - 1, 2, 2)
    painter.end()
    return pix


class ClickJumpSlider(QSlider):
    """Horizontal slider whose groove click jumps to the nearest value."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self)
            handle = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
            pos = event.position().toPoint()
            if not handle.contains(pos) and groove.contains(pos):
                span = max(1, groove.width() - handle.width())
                offset = pos.x() - groove.x() - handle.width() // 2
                self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), offset, span))
                event.accept()
                return
        super().mousePressEvent(event)


class AutoContrastButton(QToolButton):
    single_clicked = Signal()
    double_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setText("AUTO")
        self.setToolTip("Click: auto contrast this frame\nDouble-click: keep auto on T/Z changes")
        self.setProperty("role", "auto")
        self._suppress_release = False

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not self._suppress_release:
            self.single_clicked.emit()
        self._suppress_release = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self._suppress_release = True
        self.double_clicked.emit()
        event.accept()


def _eye_icon(color: str = "#9e9e9e", visible: bool = True) -> QIcon:
    pix = QPixmap(18, 18)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(QColor(color), 1.4)
    painter.setPen(pen)
    if visible:
        painter.drawEllipse(2, 5, 14, 8)
        painter.setBrush(QColor(color))
        painter.drawEllipse(7, 7, 4, 4)
    else:
        painter.drawLine(3, 14, 15, 4)
        painter.drawEllipse(2, 5, 14, 8)
    painter.end()
    return QIcon(pix)


class _ChannelRow(QWidget):
    def __init__(self, layer: Any, tokens: Tokens, full: bool, solo_cb=None, auto_cb=None) -> None:
        super().__init__()
        self.layer = layer
        self.tokens = tokens
        self.full = full
        self.solo_cb = solo_cb
        self.auto_cb = auto_cb
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(tokens.space(0.7))
        self.eye = QToolButton()
        self.eye.setCheckable(True)
        self.eye.setAutoRaise(True)
        self.eye.setIconSize(QSize(18, 18))
        self.eye.setChecked(layer.visible)
        self.eye.setIcon(_eye_icon("#e8e8e8", layer.visible))
        self.eye.toggled.connect(self._set_visible)
        row.addWidget(self.eye)
        self.name = QLabel(layer.name)
        self.name.setMinimumWidth(tokens.px(62))
        if not layer.visible:
            self.name.setStyleSheet("color: #6b6b6b;")
        row.addWidget(self.name, 1 if not full else 0)
        self.lut = QToolButton()
        self.lut.setAutoRaise(True)
        self.lut.setIconSize(QSize(18, 18))
        self.lut.setIcon(QIcon(_lut_pixmap(self._lut_name())))
        self.lut.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.lut)
        for name in LUT_COLORS:
            action = menu.addAction(QIcon(_lut_pixmap(name)), name)
            action.triggered.connect(lambda _checked=False, name=name: self._set_lut(name))
        self.lut.setMenu(menu)
        row.addWidget(self.lut)
        self.auto_button = AutoContrastButton()
        self.auto_button.single_clicked.connect(lambda: self.auto_cb(layer, False) if self.auto_cb else None)
        self.auto_button.double_clicked.connect(lambda: self.auto_cb(layer, True) if self.auto_cb else None)
        row.addWidget(self.auto_button)
        if full:
            data = np.asarray(layer.data)
            lo, hi = np.percentile(data, (0.5, 99.5)) if data.size else (0, 1)
            self.low = self._spin(float(lo), 0, 65535, 0, lambda: self._apply_limits())
            self.high = self._spin(float(hi), 0, 65535, 0, lambda: self._apply_limits())
            row.addWidget(self._tag("min"))
            row.addWidget(self.low)
            row.addWidget(self._tag("max"))
            row.addWidget(self.high)
            self.gamma = self._spin(float(layer.gamma), 0.1, 4.0, 2, lambda v: setattr(layer, "gamma", float(v)))
            self.alpha = self._spin(float(layer.opacity), 0.0, 1.0, 2, lambda v: setattr(layer, "opacity", float(v)))
            row.addWidget(self._tag("γ"))
            row.addWidget(self.gamma)
            row.addWidget(self._tag("α"))
            row.addWidget(self.alpha)
            solo = QPushButton("S")
            solo.setCheckable(True)
            solo.setFixedWidth(tokens.px(24))
            if self.solo_cb is not None:
                solo.clicked.connect(lambda _checked=False: self.solo_cb(layer))
            row.addWidget(solo)
            row.addStretch(1)

    def _tag(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #6b6b6b; font-size: 10px;")
        return label

    def _spin(self, value, low, high, decimals, callback):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(1 if decimals == 0 else 0.05)
        spin.setValue(value)
        spin.setFixedWidth(self.tokens.px(76))
        spin.valueChanged.connect(callback)
        return spin

    def _lut_name(self) -> str:
        try:
            return str(self.layer.colormap.name)
        except Exception:
            return "gray"

    def _set_lut(self, name: str) -> None:
        self.layer.colormap = name
        self.lut.setIcon(QIcon(_lut_pixmap(name)))

    def _apply_limits(self) -> None:
        low, high = sorted((float(self.low.value()), float(self.high.value())))
        if high <= low:
            high = low + 1
        self.layer.contrast_limits = (low, high)

    def _set_visible(self, checked: bool) -> None:
        self.layer.visible = checked
        self.eye.setIcon(_eye_icon("#e8e8e8", checked))
        self.name.setStyleSheet("" if checked else "color: #6b6b6b;")

    def set_auto(self, enabled: bool) -> None:
        self.auto_button.setProperty("autoOn", enabled)
        self.auto_button.style().unpolish(self.auto_button)
        self.auto_button.style().polish(self.auto_button)

    def sync_visibility(self) -> None:
        self.eye.blockSignals(True)
        self.eye.setChecked(self.layer.visible)
        self.eye.setIcon(_eye_icon("#e8e8e8", self.layer.visible))
        self.eye.blockSignals(False)
        self.name.setStyleSheet("" if self.layer.visible else "color: #6b6b6b;")


class HyperViewChannelPanel(QWidget):
    popout_requested = Signal()

    def __init__(self, viewer: Any, tokens: Tokens | None = None, compact: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.tokens = tokens or Tokens()
        self.compact = compact
        self.layers: list[Any] = []
        self.rows: list[_ChannelRow] = []
        self._solo = None
        self.setObjectName("HyperViewChannelPanel")
        self.setStyleSheet(build_stylesheet(self.tokens))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(self.tokens.space(0.5))
        if not compact:
            title = QLabel("HYPERVIEW CHANNELS")
            title.setStyleSheet("font-weight: bold; font-size: 13px; color: #e8e8e8;")
            root.addWidget(title)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(2, 2, 2, 2)
        self.body_layout.setSpacing(self.tokens.space(0.4))
        self.body_layout.addStretch(1)
        self.scroll.setWidget(self.body)
        self.viewer.dims.events.current_step.connect(self._on_dim_step)
        if compact:
            self.scroll.setMaximumHeight(self.tokens.px(136))
        root.addWidget(self.scroll, 1)
        if compact:
            popout = QPushButton("Pop out")
            popout.setProperty("variant", "ghost")
            popout.clicked.connect(self.popout_requested.emit)
            root.addWidget(popout)

    def bind_layers(self, layers: list[Any]) -> None:
        self.layers = layers
        for row in self.rows:
            row.deleteLater()
        self.rows = []
        for index in range(len(layers)):
            row = _ChannelRow(layers[index], self.tokens, full=not self.compact, solo_cb=self._solo_channel, auto_cb=self._auto_channel)
            self.body_layout.insertWidget(index, row)
            self.rows.append(row)
            row.set_auto(bool(layers[index].metadata.get("hyperview_auto", False)))

    def _current_slice(self, layer: Any):
        data = layer.data
        axes = str(layer.metadata.get("hyperview_axes", ""))
        step = self.viewer.dims.current_step
        selectors = []
        for index, axis in enumerate(axes):
            if axis in ("T", "Z") and index < len(step):
                selectors.append(int(step[index]))
            else:
                selectors.append(slice(None))
        if selectors:
            return data[tuple(selectors)]
        return data

    def _apply_auto(self, layer: Any) -> None:
        frame = self._current_slice(layer)
        if not frame.size:
            return
        lo, hi = np.percentile(frame, (0.5, 99.5))
        if hi <= lo:
            hi = lo + 1
        layer.contrast_limits = (float(lo), float(hi))

    def _auto_channel(self, layer: Any, persistent: bool) -> None:
        if persistent:
            enabled = not bool(layer.metadata.get("hyperview_auto", False))
            layer.metadata["hyperview_auto"] = enabled
            if enabled:
                self._apply_auto(layer)
        else:
            self._apply_auto(layer)
            enabled = bool(layer.metadata.get("hyperview_auto", False))
        for row in self.rows:
            row.set_auto(bool(row.layer.metadata.get("hyperview_auto", False)))
        if enabled:
            self._apply_auto(layer)

    def _on_dim_step(self, _event=None) -> None:
        for layer in self.layers:
            if layer.metadata.get("hyperview_auto", False):
                self._apply_auto(layer)

    def _solo_channel(self, layer: Any) -> None:
        if self._solo is layer:
            for item in self.layers:
                item.visible = True
            self._solo = None
        else:
            for item in self.layers:
                item.visible = item is layer
            self._solo = layer
        for row in self.rows:
            row.sync_visibility()


class HyperViewDimensionBar(QWidget):
    def __init__(self, viewer: Any, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.tokens = tokens or Tokens()
        self.layers: list[Any] = []
        self._syncing = False
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self.setObjectName("HyperViewDimensionBar")
        self.setStyleSheet(build_stylesheet(self.tokens))
        root = QVBoxLayout(self)
        root.setContentsMargins(self.tokens.space(1.0), self.tokens.space(0.5), self.tokens.space(1.0), self.tokens.space(0.5))
        root.setSpacing(self.tokens.space(0.4))
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("T"))
        self.time_slider = ClickJumpSlider(Qt.Horizontal)
        self.time_label = QLabel("0 / 0")
        self.time_label.setProperty("role", "dim")
        time_row.addWidget(self.time_slider, 1)
        time_row.addWidget(self.time_label)
        root.addLayout(time_row)
        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z"))
        self.z_slider = ClickJumpSlider(Qt.Horizontal)
        self.z_label = QLabel("0 / 0")
        self.z_label.setProperty("role", "dim")
        z_row.addWidget(self.z_slider, 1)
        z_row.addWidget(self.z_label)
        root.addLayout(z_row)
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        fit = QPushButton("Fit channels")
        fit.clicked.connect(self._fit)
        controls.addWidget(fit)
        controls.addStretch(1)
        root.addLayout(controls)
        self.time_slider.valueChanged.connect(self._set_time)
        self.z_slider.valueChanged.connect(self._set_z)
        self.viewer.dims.events.current_step.connect(self._sync)

    def bind_layers(self, layers: list[Any]) -> None:
        self.layers = layers
        axes = str(layers[0].metadata.get("hyperview_axes", "")) if layers else ""
        shape = layers[0].data.shape if layers else (0, 0)
        self._t_index = axes.find("T")
        self._z_index = axes.find("Z")
        self.time_slider.setRange(0, max(0, shape[self._t_index] - 1) if self._t_index >= 0 else 0)
        self.z_slider.setRange(0, max(0, shape[self._z_index] - 1) if self._z_index >= 0 else 0)
        self.time_slider.setEnabled(self._t_index >= 0)
        self.z_slider.setEnabled(self._z_index >= 0)
        self._sync()

    def _set_time(self, value: int) -> None:
        if self._syncing or not self.layers or self._t_index < 0:
            return
        self._syncing = True
        try:
            self.viewer.dims.set_point(self._t_index, int(value))
        finally:
            self._syncing = False
        self.time_label.setText(f"{value} / {self.time_slider.maximum()}")

    def _set_z(self, value: int) -> None:
        if self._syncing or not self.layers or self._z_index < 0:
            return
        self._syncing = True
        try:
            self.viewer.dims.set_point(self._z_index, int(value))
        finally:
            self._syncing = False
        self.z_label.setText(f"{value} / {self.z_slider.maximum()}")

    def _sync(self, _event=None) -> None:
        if self._syncing or not self.layers:
            return
        step = self.viewer.dims.current_step
        self._syncing = True
        try:
            if self._t_index >= 0 and self._t_index < len(step):
                self.time_slider.setValue(int(step[self._t_index]))
                self.time_label.setText(f"{int(step[self._t_index])} / {self.time_slider.maximum()}")
            else:
                self.time_label.setText("–")
            if self._z_index >= 0 and self._z_index < len(step):
                self.z_slider.setValue(int(step[self._z_index]))
                self.z_label.setText(f"{int(step[self._z_index])} / {self.z_slider.maximum()}")
            else:
                self.z_label.setText("–")
        finally:
            self._syncing = False

    def _toggle_play(self, enabled: bool) -> None:
        self._playing = enabled
        self.play_button.setText("Pause" if enabled else "Play")
        if enabled:
            self._timer.start(150)
        else:
            self._timer.stop()

    def _next_frame(self) -> None:
        if not self.layers:
            return
        self.time_slider.setValue((self.time_slider.value() + 1) % max(1, self.time_slider.maximum() + 1))

    def _fit(self) -> None:
        for layer in self.layers:
            data = np.asarray(layer.data)
            if data.size:
                lo, hi = np.percentile(data, (0.5, 99.5))
                layer.contrast_limits = (float(lo), float(hi))
                layer.visible = True


class HyperViewWidget(QWidget):
    def __init__(self, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import napari

        self.tokens = tokens or Tokens()
        initialize_application()
        self.setObjectName("HyperViewWidget")
        self.viewer = napari.Viewer(show=False)
        self.layers = []
        data = demo_volume()
        channels = [
            ChannelData(
                data[:, ci],
                name,
                lut,
                tuple(np.percentile(data[:, ci], (0.5, 99.5))),
            )
            for ci, (name, lut) in enumerate(CHANNELS)
        ]
        self.load_image(LoadedImage(Path("demo"), "TZYX", channels, "demo"))
        qt_window = self.viewer.window._qt_window
        qt_window.setWindowFlags(Qt.Widget)
        qt_window.setAttribute(Qt.WA_DeleteOnClose, False)
        if qt_window.menuBar() is not None:
            qt_window.menuBar().hide()
        if qt_window.statusBar() is not None:
            qt_window.statusBar().hide()
        for dock in qt_window.findChildren(QWidget):
            if type(dock).__name__ == "QtViewerDockWidget":
                dock.hide()
        dims = getattr(self.viewer.window._qt_viewer, "dims", None)
        if dims is not None:
            dims.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(qt_window)
        self.viewer.window._qt_window.show()

    def load_image(self, loaded: LoadedImage) -> None:
        self.viewer.layers.clear()
        self.layers = []
        axes = "".join(loaded.axes)
        for channel in loaded.channels:
            layer = self.viewer.add_image(
                channel.data,
                name=channel.name,
                colormap=channel.lut,
                blending="additive",
                opacity=0.8,
                contrast_limits=channel.contrast_limits,
            )
            layer.metadata["hyperview_axes"] = axes
            layer.metadata["hyperview_auto"] = False
            self.layers.append(layer)
        self.viewer.dims.axis_labels = tuple(axis.lower() for axis in axes)
        self.viewer.dims.ndisplay = 2
        self.viewer.reset_view()
        self.loaded = loaded
