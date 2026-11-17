"""Design tokens and QSS for the momia3 GUI shell.

The palette, Inter typography and 4 px spacing / scale presets are aligned with
``dummy_microscope_control``.  This module deliberately contains no layout
logic: the shell composes widgets from :mod:`momia3.gui.shell`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from string import Template

from qtpy.QtGui import QFont, QFontDatabase
from qtpy.QtWidgets import QApplication

__all__ = [
    "DARK",
    "SCALE_PRESETS",
    "Palette",
    "Tokens",
    "build_stylesheet",
    "initialize_application",
]

SCALE_PRESETS = {"small": 0.85, "normal": 1.0, "large": 1.15, "xlarge": 1.3}


@dataclass(frozen=True)
class Palette:
    bg: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    text_dim: str
    text_disabled: str
    accent: str
    success: str
    danger: str
    warning: str


DARK = Palette(
    bg="#2b2b2b",
    surface="#333333",
    surface_alt="#3c3c3c",
    border="#4a4a4a",
    border_strong="#5f5f5f",
    text="#e8e8e8",
    text_dim="#9e9e9e",
    text_disabled="#6b6b6b",
    accent="#2196f3",
    success="#4caf50",
    danger="#f44336",
    warning="#ffb300",
)


@dataclass(frozen=True)
class Tokens:
    scale: float = 1.0
    family: str = "Inter"
    base_size: float = 12.0
    palette: Palette = DARK
    space_unit: float = 4.0
    radius: float = 4.0

    def px(self, value: float) -> int:
        return max(1, round(value * self.scale))

    def space(self, steps: float = 1.0) -> int:
        return max(1, round(self.space_unit * steps * self.scale))

    def size(self, key: str = "md") -> float:
        offsets = {"xs": -2.0, "sm": -1.0, "md": 0.0, "lg": 2.0, "xl": 4.0, "xxl": 7.0}
        return max(6.0, (self.base_size + offsets.get(key, 0.0)) * self.scale)

    def scaled(self, factor: float) -> "Tokens":
        return replace(self, scale=factor)


def initialize_application(app: QApplication | None = None) -> None:
    app = app or QApplication.instance()
    if app is None or app.property("momia3Typography"):
        return
    folder = Path(__file__).with_name("fonts")
    for name in ("Inter-Regular.otf", "Inter-SemiBold.otf", "Inter-Bold.otf"):
        path = folder / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    families = set(QFontDatabase.families())
    family = next((f for f in ("Inter", "Segoe UI", "Helvetica Neue", "Noto Sans") if f in families), app.font().family())
    font = QFont(family)
    font.setPixelSize(12)
    app.setFont(font)
    app.setStyle("Fusion")
    app.setProperty("momia3Typography", True)


_QSS = Template(
    """
* { font-size: ${base_size}px; font-family: "${family}"; }
QWidget#ShellRoot, QWidget#ROIManagerWindow, QWidget#PluginManagerDialog, QWidget#HyperViewPanel { background-color: $bg; color: $text; }
QDialog { background-color: $bg; color: $text; }
QFrame#TopBar { background-color: $surface; border: 1px solid $border; border-radius: ${radius}px; }
QLabel#BrandMark { background-color: $accent; color: white; border-radius: ${radius}px; font-weight: bold; }
QLabel#BrandName { font-weight: bold; }
QLabel[role="chip"] { color: $dim; background-color: #2b2b2b; border: 1px solid $border; border-radius: ${radius}px; padding: ${badge_v}px ${badge_h}px; font-size: ${badge_size}px; }
QLabel[role="chip-ok"] { color: $success; background-color: #233323; border: 1px solid #355635; border-radius: ${radius}px; padding: ${badge_v}px ${badge_h}px; font-size: ${badge_size}px; }
QFrame[role="panel"] { background-color: $surface; border: 1px solid $border; border-radius: ${radius}px; }
QFrame[role="strip"] { background-color: $surface; border: 1px solid $border; border-radius: ${radius}px; }
QFrame[role="viewer"] { background-color: #232323; border: 1px solid $border; border-radius: ${radius}px; }
QFrame[role="rule"] { background-color: $border; border: none; }
QSplitter::handle { background-color: $bg; width: ${gap}px; height: ${gap}px; }
QTabWidget::pane { background-color: $surface; border: 1px solid $border; border-radius: ${radius}px; }
QTabBar::tab { background: transparent; color: $dim; padding: ${tab_v}px ${tab_h}px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: $text; border-bottom-color: $accent; }
QTabBar::tab:hover { color: $text; }
QLabel { background: transparent; color: $text; }
QLabel[role="section"] { color: $dim; font-size: ${section_size}px; font-weight: bold; }
QLabel[role="dim"] { color: $dim; }
QLabel[role="title"] { font-weight: bold; }
QLabel[role="value"] { color: $text; font-weight: bold; }
QLabel[role="stage"] { color: $dim; padding: ${stage_pad_v}px ${stage_pad_h}px; }
QLabel[role="stage"][state="done"] { color: $success; }
QLabel[role="stage"][state="current"] { color: $accent; background-color: ${accent_soft}; border: 1px solid ${accent_border}; border-radius: ${radius}px; }
QLabel[role="stage"][state="future"] { color: $disabled; }
QLabel[role="badge"] { color: $accent; background-color: ${accent_soft}; border: 1px solid ${accent_border}; border-radius: ${radius}px; padding: ${badge_v}px ${badge_h}px; font-size: ${badge_size}px; }
QLabel[role="ok"] { color: $success; }
QPushButton { background-color: $alt; color: $text; border: none; border-radius: 6px; padding: ${btn_v}px ${btn_h}px; }
QPushButton:hover { background-color: $alt_hover; }
QPushButton:pressed { background-color: ${accent_soft}; }
QPushButton:focus { border: 1px solid $accent; }
QPushButton[variant="primary"] { background-color: $accent; color: white; font-weight: bold; }
QPushButton[variant="primary"]:hover { background-color: #2aa7ff; }
QPushButton[variant="ghost"] { background-color: transparent; }
QPushButton[variant="ghost"]:hover { background-color: $alt; }
QPushButton[variant="primary"]:disabled, QPushButton:disabled { color: $disabled; background-color: $surface; border: none; }
QPushButton[variant="ghost"]:disabled { background-color: transparent; color: $disabled; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2b2b2b; color: $text; border: 1px solid $border; border-radius: ${radius}px; padding: ${input_v}px ${input_h}px; selection-background-color: $accent; }
QComboBox { padding-right: 22px; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border: none; background: transparent; }
QComboBox::down-arrow { image: url("${chevron}"); width: 10px; height: 6px; }
QComboBox QAbstractItemView { background-color: $surface; color: $text; border: 1px solid $border; outline: 0; selection-background-color: $accent; selection-color: #ffffff; }
QComboBox QAbstractItemView::item { min-height: 22px; padding: 3px 8px; }
QComboBox:disabled { color: $disabled; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: $accent; }
QToolButton { background: transparent; border: none; padding: 1px; }
QToolButton:hover { background-color: $alt; border-radius: ${radius}px; }
QToolButton::menu-indicator { image: none; }
QToolButton#ROIToolButton, QToolButton#ProjectToolButton { background: $alt; border: none; border-radius: 6px; padding: 0; }
QToolButton#ProjectToolButton:hover { background: $alt_hover; }
QToolButton#ProjectToolButton:pressed { background: ${accent_soft}; }
QToolButton#ROIToolButton { background: $alt; border: none; border-radius: 6px; padding: 0; }
QToolButton#ROIToolButton:hover { background: $alt_hover; }
QToolButton#ROIToolButton:pressed { background: ${accent_soft}; }
QToolButton#ROIToolButton:checked { background: ${accent_soft}; }
QToolButton#ROIToolButton:disabled { background: $surface; color: $disabled; }
QPushButton#ROISecondaryButton { background: transparent; border: none; color: $dim; padding: ${btn_v}px ${btn_h}px; }
QPushButton#ROISecondaryButton:hover { background: $alt; color: $text; }
QPushButton#ROISecondaryButton:disabled { background: transparent; color: $disabled; }
QPushButton#ROIPrimaryButton { background: $accent; border: none; color: white; font-weight: bold; padding: ${btn_v}px ${btn_h}px; }
QPushButton#ROIPrimaryButton:hover { background: #2aa7ff; }
QPushButton#ROIPrimaryButton:disabled { background: $surface; color: $disabled; }
QPushButton#ROICloseButton { background: transparent; border: none; color: $text; padding: 4px 10px; }
QPushButton#ROICloseButton:hover { background: $alt; }
QToolButton[role="auto"] { color: $dim; border: none; border-radius: 6px; padding: 2px 5px; font-size: 9px; background: $alt; }
QToolButton[role="auto"]:hover { color: $text; background: $alt_hover; }
QToolButton[role="auto"][autoOn="true"] { color: $accent; background: ${accent_soft}; }
QSlider { min-height: 18px; }
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4c7896; border-radius: 2px; }
QSlider::add-page:horizontal { background: $border; border-radius: 2px; }
QSlider::handle:horizontal { width: 10px; margin: -4px 0; border-radius: 5px; background: #cfcfcf; border: 1px solid #6b6b6b; }
QSlider::handle:horizontal:hover { background: $text; border-color: $accent; }
QToolTip { background-color: $surface; color: $text; border: 1px solid $border; padding: 4px 6px; }
QMenu { background-color: $surface; color: $text; border: 1px solid $border; padding: 4px; }
QMenu::item { padding: 4px 18px; border-radius: ${radius}px; }
QMenu::item:selected { background-color: $accent; color: #ffffff; }
QScrollArea { background: transparent; border: none; }
QTreeWidget, QListWidget { background-color: $surface; color: $text; border: none; outline: none; }
QTreeWidget::item, QListWidget::item { color: $text; padding: ${item_v}px ${item_h}px; border-radius: 2px; }
QTreeWidget::item:selected, QListWidget::item:selected { background-color: ${accent_soft}; color: $text; }
QTreeWidget::item:hover, QListWidget::item:hover { background-color: $alt_hover; }
QHeaderView::section { background-color: $alt; color: $dim; border: 0; border-right: 1px solid $border; padding: ${input_v}px ${input_h}px; }
QTableView { background-color: $surface; alternate-background-color: $alt; color: $text; border: none; gridline-color: $border; }
QScrollBar:vertical, QScrollBar:horizontal { background: $bg; border: none; margin: 0; }
QScrollBar:vertical { width: ${scroll}px; }
QScrollBar:horizontal { height: ${scroll}px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: $strong; border-radius: ${scroll_radius}px; }
QScrollBar::handle:vertical { min-height: ${scroll_handle}px; }
QScrollBar::handle:horizontal { min-width: ${scroll_handle}px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: $accent; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: transparent; border: none; width: 0; height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QAbstractScrollArea::corner { background: $bg; border: none; }
QStatusBar { background: $surface; color: $dim; border-top: 1px solid $border; }
QSplitter::handle:hover { background-color: $border; }
"""
)


def build_stylesheet(tokens: Tokens = Tokens()) -> str:
    p = tokens.palette
    return _QSS.safe_substitute(
        bg=p.bg,
        surface=p.surface,
        alt=p.surface_alt,
        alt_hover="#474747",
        border=p.border,
        strong=p.border_strong,
        text=p.text,
        dim=p.text_dim,
        disabled=p.text_disabled,
        accent=p.accent,
        accent_soft="#1d3448",
        accent_border="#2b5b83",
        success=p.success,
        danger=p.danger,
        warning=p.warning,
        family=tokens.family,
        radius=tokens.px(tokens.radius),
        base_size=round(tokens.size("md")),
        section_size=round(tokens.size("xs")),
        gap=tokens.px(4),
        tab_v=tokens.px(8),
        tab_h=tokens.px(12),
        stage_pad_v=tokens.px(4),
        stage_pad_h=tokens.px(6),
        badge_v=tokens.px(2),
        badge_h=tokens.px(5),
        badge_size=round(tokens.size("xs")),
        btn_v=tokens.px(5),
        btn_h=tokens.px(9),
        input_v=tokens.px(4),
        input_h=tokens.px(6),
        item_v=tokens.px(3),
        item_h=tokens.px(5),
        scroll=tokens.px(10),
        scroll_radius=tokens.px(3),
        scroll_handle=tokens.px(24),
        chevron=(Path(__file__).resolve().parent / "assets" / "chevron-down.svg").as_posix(),
    )
