"""In-process GUI plugin contract for the momia3 shell.

GUI plugins are deliberately not napari plugins: they receive a small
``GuiContext`` and contribute shell-level items (inspector tabs, ROI tools,
menu actions).  A broken plugin is recorded in ``PluginRegistry.errors`` and
never prevents the shell from starting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable
from uuid import uuid4

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import Tokens, build_stylesheet

__all__ = [
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "GuiContext",
    "GuiPluginSpec",
    "InspectorTab",
    "LoadedPlugin",
    "MenuAction",
    "PluginManagerDialog",
    "PluginRegistry",
    "RoiTool",
    "ToolbarAction",
    "builtin_plugins",
]

API_VERSION = "1"
ENTRY_POINT_GROUP = "momia3.gui.plugins"


@dataclass
class GuiContext:
    """Controlled access to the shell for plugins."""

    tokens: Tokens
    roi_session_getter: Callable[[], Any]
    open_window: Callable[[str, QWidget], None]
    notify: Callable[[str], None]

    def roi_session(self):
        return self.roi_session_getter()


@dataclass(frozen=True)
class InspectorTab:
    title: str
    factory: Callable[[GuiContext], QWidget]


@dataclass(frozen=True)
class ToolbarAction:
    label: str
    callback: Callable[[GuiContext], None]


@dataclass(frozen=True)
class RoiTool:
    name: str
    callback: Callable[[GuiContext], None]


@dataclass(frozen=True)
class MenuAction:
    label: str
    callback: Callable[[GuiContext], None]


Contribution = InspectorTab | ToolbarAction | RoiTool | MenuAction


@dataclass(frozen=True)
class GuiPluginSpec:
    id: str
    name: str
    version: str
    api_version: str
    contributions: tuple[Contribution, ...] = ()

    def validate(self) -> None:
        if self.api_version != API_VERSION:
            raise ValueError(f"{self.id}: plugin API {self.api_version!r} != {API_VERSION!r}")
        if not self.id or not self.name:
            raise ValueError("plugin id and name are required")
        for item in self.contributions:
            if not isinstance(item, (InspectorTab, ToolbarAction, RoiTool, MenuAction)):
                raise TypeError(f"{self.id}: unsupported contribution {item!r}")


@dataclass
class LoadedPlugin:
    spec: GuiPluginSpec | None
    error: str | None = None
    source: str = ""


@dataclass
class PluginRegistry:
    plugins: list[LoadedPlugin] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def load(self, include_entry_points: bool = True) -> None:
        self.plugins.clear()
        self.errors.clear()
        for spec in builtin_plugins():
            self._add(spec, source="builtin")
        if not include_entry_points:
            return
        try:
            entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as exc:
            self.errors.append(f"entry point discovery failed: {exc}")
            return
        for entry in entry_points:
            try:
                loaded = entry.load()
                spec = loaded() if callable(loaded) and not isinstance(loaded, GuiPluginSpec) else loaded
                if not isinstance(spec, GuiPluginSpec):
                    raise TypeError(f"expected GuiPluginSpec, got {type(spec).__name__}")
                self._add(spec, source=f"entry point {entry.name}")
            except Exception as exc:
                self.errors.append(f"{entry.name}: {exc}")

    def _add(self, spec: GuiPluginSpec, source: str) -> None:
        try:
            spec.validate()
            if any(item.spec and item.spec.id == spec.id for item in self.plugins):
                raise ValueError(f"duplicate plugin id {spec.id!r}")
            self.plugins.append(LoadedPlugin(spec=spec, source=source))
        except Exception as exc:
            self.errors.append(f"{source}: {exc}")
            self.plugins.append(LoadedPlugin(spec=None, error=str(exc), source=source))

    def specs(self) -> list[GuiPluginSpec]:
        return [item.spec for item in self.plugins if item.spec is not None]


class PluginManagerDialog(QDialog):
    def __init__(self, registry: PluginRegistry, context: GuiContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.context = context
        self.setWindowFlag(Qt.Window, True)
        self.setObjectName("PluginManagerDialog")
        self.setWindowTitle("momia3 · Plugins")
        self.resize(context.tokens.px(720), context.tokens.px(520))
        self.setStyleSheet(build_stylesheet(context.tokens))

        root = QVBoxLayout(self)
        root.setContentsMargins(context.tokens.space(1.5), context.tokens.space(1.2), context.tokens.space(1.5), context.tokens.space(1.2))
        header = QHBoxLayout()
        title = QLabel("Plugins")
        title.setStyleSheet("font-weight: bold; font-size: 15px; color: #e8e8e8;")
        header.addWidget(title)
        count = QLabel(f"{len(registry.specs())} loaded · {len(registry.errors)} error(s)")
        count.setProperty("role", "dim")
        header.addWidget(count)
        header.addStretch(1)
        root.addLayout(header)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Plugin", "Version", "Contributions", "Source"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for plugin in registry.plugins:
            row = self.table.rowCount()
            self.table.insertRow(row)
            spec = plugin.spec
            values = [spec.name if spec else plugin.source, spec.version if spec else "-", str(len(spec.contributions)) if spec else "0", plugin.source]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.itemSelectionChanged.connect(self._refresh_contributions)
        root.addWidget(self.table, 2)

        self.contributions = QListWidget()
        root.addWidget(self.contributions, 2)
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run selected contribution")
        self.run_button.clicked.connect(self._run_selected)
        run_row.addWidget(self.run_button)
        self.status = QLabel()
        self.status.setProperty("role", "dim")
        run_row.addWidget(self.status, 1)
        close = QPushButton("Close")
        close.setProperty("variant", "ghost")
        close.clicked.connect(self.close)
        run_row.addWidget(close)
        root.addLayout(run_row)
        if registry.errors:
            details = QLabel("\n".join(registry.errors))
            details.setProperty("role", "dim")
            root.addWidget(details)
        if self.table.rowCount():
            self.table.selectRow(0)
        self._refresh_contributions()

    def _selected_spec(self) -> GuiPluginSpec | None:
        row = self.table.currentRow()
        specs = [item.spec for item in self.registry.plugins]
        return specs[row] if 0 <= row < len(specs) else None

    def _refresh_contributions(self) -> None:
        self.contributions.clear()
        spec = self._selected_spec()
        if spec is None:
            self.run_button.setEnabled(False)
            return
        for contribution in spec.contributions:
            self.contributions.addItem(f"{type(contribution).__name__}: {getattr(contribution, 'title', getattr(contribution, 'name', getattr(contribution, 'label', '')))}")
        self.run_button.setEnabled(any(isinstance(item, (RoiTool, ToolbarAction, MenuAction)) for item in spec.contributions))

    def _run_selected(self) -> None:
        spec = self._selected_spec()
        row = self.contributions.currentRow()
        if spec is None or row < 0:
            return
        contribution = spec.contributions[row]
        if not isinstance(contribution, (RoiTool, ToolbarAction, MenuAction)):
            self.status.setText("Inspector tabs are installed automatically.")
            return
        try:
            contribution.callback(self.context)
            self.status.setText("Contribution ran.")
        except Exception as exc:
            self.status.setText(f"Contribution failed: {exc}")


def builtin_plugins() -> list[GuiPluginSpec]:
    """Small built-in plugin used to exercise the GUI contract."""

    def demo_tab(context: GuiContext) -> QWidget:
        from qtpy.QtWidgets import QLabel, QVBoxLayout

        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("Built-in GUI plugin demo")
        label.setStyleSheet("color: #e8e8e8; font-weight: bold;")
        layout.addWidget(label)
        hint = QLabel("Contributes an inspector tab and a ROI tool through GuiContext.")
        hint.setStyleSheet("color: #9e9e9e;")
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def add_demo_roi(context: GuiContext) -> None:
        from .roi_manager import ROI

        session = context.roi_session()
        session.add_roi(
            ROI(
                roi_id=f"plugin-{uuid4().hex[:8]}",
                name="plugin demo box",
                kind="box",
                geometry={"x": 300, "y": 180, "w": 96, "h": 64},
                cell_key=None,
                track_id=None,
                frame=0,
                state="manual",
                color="#e8e8e8",
            )
        )
        context.notify("Plugin added a demo ROI.")

    return [
        GuiPluginSpec(
            id="momia3.builtin.demo",
            name="Built-in GUI demo",
            version="0.1.0",
            api_version=API_VERSION,
            contributions=(
                InspectorTab("Plugin demo", demo_tab),
                RoiTool("Add demo ROI", add_demo_roi),
            ),
        )
    ]
