"""Layout-only GUI shell for momia3.

The shell is deliberately inert: it provides the persistent workflow strip,
the three-column workspace, the inspector tabs and the status row, but no
widget is wired to a store, pipeline or napari viewer yet.  This file is the
frame that the real Dataset / Pipeline / Results panels will be placed into.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .theme import Tokens, build_stylesheet, initialize_application

__all__ = ["MainShell"]


class _Panel(QFrame):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tokens = tokens
        self.setProperty("role", "panel")
        self.setFrameShape(QFrame.NoFrame)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(tokens.space(1.5), tokens.space(1.5), tokens.space(1.5), tokens.space(1.5))
        self._layout.setSpacing(tokens.space(1.0))


class _TopBar(QFrame):
    roi_requested = Signal()
    plugins_requested = Signal()

    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TopBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(tokens.space(1.5), tokens.space(0.8), tokens.space(1.5), tokens.space(0.8))
        layout.setSpacing(tokens.space(0.8))
        mark = QLabel("m")
        mark.setObjectName("BrandMark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(tokens.px(24), tokens.px(24))
        layout.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        name = QLabel("momia3")
        name.setObjectName("BrandName")
        sub = _dim("analysis workstation")
        brand.addWidget(name)
        brand.addWidget(sub)
        layout.addLayout(brand)
        layout.addWidget(_rule(tokens))
        layout.addWidget(QLabel("exemplar-001"))
        layout.addWidget(_dim("/"))
        layout.addWidget(QLabel("cycle-06"))
        layout.addWidget(_dim("/"))
        layout.addWidget(_dim("mosaic"))
        layout.addSpacing(tokens.space(0.5))
        layout.addWidget(_chip("READY", ok=True))
        layout.addWidget(_chip("6 tiles"))
        layout.addWidget(_chip("4 channels"))
        layout.addWidget(_chip("0.65 µm/px"))
        layout.addStretch(1)
        plugins_button = QPushButton("Plugins")
        plugins_button.setProperty("variant", "ghost")
        plugins_button.clicked.connect(self.plugins_requested.emit)
        layout.addWidget(plugins_button)
        roi_button = QPushButton("ROI Manager")
        roi_button.setProperty("variant", "ghost")
        roi_button.clicked.connect(self.roi_requested.emit)
        layout.addWidget(roi_button)
        for label in ("Settings", "Share"):
            button = QPushButton(label)
            button.setProperty("variant", "ghost")
            button.setEnabled(False)
            layout.addWidget(button)
        run = QPushButton("Run pipeline")
        run.setProperty("variant", "primary")
        run.setEnabled(False)
        layout.addWidget(run)


class _WorkflowStrip(QFrame):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowStrip")
        self.setProperty("role", "strip")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(tokens.space(2.0), tokens.space(1.0), tokens.space(2.0), tokens.space(1.0))
        layout.setSpacing(tokens.space(0.7))
        layout.addWidget(QLabel("WORKFLOW"))
        stages = [("01", "Import", "done"), ("02", "Preprocess", "done"), ("03", "Stitch", "current"), ("04", "Segment", "future"), ("05", "Track", "future"), ("06", "Results", "future")]
        for index, (number, name, state) in enumerate(stages):
            label = QLabel(f"{number}  {name}")
            label.setProperty("role", "stage")
            label.setProperty("state", state)
            layout.addWidget(label)
            if index < len(stages) - 1:
                rule = QFrame()
                rule.setProperty("role", "rule")
                rule.setFixedSize(tokens.px(12), 1)
                layout.addWidget(rule)
        layout.addStretch(1)
        badge = QLabel("stage 03 / 06")
        badge.setProperty("role", "badge")
        layout.addWidget(badge)
        layout.addWidget(QLabel("7 edges solved"))
        layout.addWidget(QLabel("next: segment"))


class _DatasetPanel(_Panel):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(tokens, parent)
        self.setObjectName("DatasetPanel")
        self.setMinimumWidth(tokens.px(220))
        self._layout.addWidget(_section("Dataset"))
        self._layout.addWidget(_dim("2.8 GB · 6 tiles · 4 channels"))
        search = QLineEdit()
        search.setPlaceholderText("Filter datasets...")
        search.setReadOnly(True)
        self._layout.addWidget(search)
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        root = QTreeWidgetItem(["exemplar-001"])
        cycle = QTreeWidgetItem(["cycle-06"])
        raw = QTreeWidgetItem(["raw"])
        for name in ("#761", "#762", "#802", "#803", "#843", "#844"):
            raw.addChild(QTreeWidgetItem([name]))
        cycle.addChild(raw)
        cycle.addChild(QTreeWidgetItem(["illumination"]))
        cycle.addChild(QTreeWidgetItem(["stitched.zarr"]))
        root.addChild(cycle)
        root.addChild(QTreeWidgetItem(["cycle-07"]))
        root.addChild(QTreeWidgetItem(["cycle-08"]))
        tree.addTopLevelItem(root)
        tree.expandAll()
        self._layout.addWidget(tree, 1)
        self._layout.addWidget(_section("Channels"))
        for name, color, enabled in (("DAPI", "#4c8dff", True), ("CD3", "#65d68b", True), ("PanCK", "#f0788d", True), ("Ki67", "#b59cff", False)):
            row = QHBoxLayout()
            swatch = QFrame()
            swatch.setFixedSize(tokens.px(8), tokens.px(8))
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
            row.addWidget(swatch)
            row.addWidget(QLabel(name))
            row.addStretch(1)
            state = QLabel("on" if enabled else "off")
            state.setProperty("role", "ok" if enabled else "dim")
            row.addWidget(state)
            self._layout.addLayout(row)
        self._layout.addWidget(_section("Tiles"))
        thumbs = QGridLayout()
        thumbs.setSpacing(tokens.space(0.6))
        for index in range(4):
            cell = QLabel(f"tile {index + 1}")
            cell.setAlignment(Qt.AlignCenter)
            cell.setFixedHeight(tokens.px(38))
            cell.setStyleSheet("background-color: #2b2b2b; color: #6b6b6b; border: 1px solid #4a4a4a;")
            thumbs.addWidget(cell, index // 2, index % 2)
        self._layout.addLayout(thumbs)


class _ViewerPanel(_Panel):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(tokens, parent)
        self.setObjectName("ViewerPanel")
        self._layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QFrame()
        toolbar.setProperty("role", "strip")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(tokens.space(1.0), tokens.space(0.6), tokens.space(1.0), tokens.space(0.6))
        tb.addWidget(QLabel("cycle-06 · stitched mosaic"))
        tb.addWidget(_dim("3138 × 2511 px"))
        tb.addStretch(1)
        for name in ("DAPI", "CD3", "PanCK", "+1"):
            button = QPushButton(name)
            button.setProperty("variant", "ghost")
            button.setEnabled(False)
            tb.addWidget(button)
        self._layout.addWidget(toolbar)
        viewer = QFrame()
        viewer.setProperty("role", "viewer")
        vl = QVBoxLayout(viewer)
        vl.setAlignment(Qt.AlignCenter)
        title = QLabel("napari canvas")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; color: #9e9e9e;")
        vl.addWidget(title)
        subtitle = QLabel("OME-Zarr multiscale mosaic · image layer goes here")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setProperty("role", "dim")
        vl.addWidget(subtitle)
        self._layout.addWidget(viewer, 1)
        timeline = QFrame()
        timeline.setProperty("role", "strip")
        tl = QVBoxLayout(timeline)
        tl.setContentsMargins(tokens.space(1.0), tokens.space(0.7), tokens.space(1.0), tokens.space(0.7))
        head = QHBoxLayout()
        head.addWidget(QLabel("Timeline"))
        head.addWidget(_dim("frame 12 / 24 · 0.20 s / frame"))
        head.addStretch(1)
        head.addWidget(_dim("filmstrip placeholder"))
        tl.addLayout(head)
        strip = QHBoxLayout()
        strip.setSpacing(tokens.space(0.6))
        for index in range(6):
            frame = QLabel(str(index + 1))
            frame.setAlignment(Qt.AlignCenter)
            frame.setFixedHeight(tokens.px(34))
            frame.setStyleSheet("background-color: #2b2b2b; color: #6b6b6b; border: 1px solid #4a4a4a;")
            if index == 2:
                frame.setStyleSheet("background-color: #24384d; color: #e8e8e8; border: 1px solid #2196f3;")
            strip.addWidget(frame, 1)
        tl.addLayout(strip)
        self._layout.addWidget(timeline)


class _InspectorPanel(_Panel):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(tokens, parent)
        self.setObjectName("InspectorPanel")
        self.setMinimumWidth(tokens.px(300))
        self._layout.addWidget(_section("Context"))
        self.tabs = QTabWidget()
        self.tabs.setObjectName("InspectorTabs")
        self.tabs.addTab(self._pipeline_page(tokens), "Pipeline")
        self.tabs.addTab(self._results_page(tokens), "Results")
        self.tabs.addTab(self._inspector_page(tokens), "Inspector")
        self._layout.addWidget(self.tabs, 1)

    def add_tab(self, title: str, widget: QWidget) -> None:
        self.tabs.addTab(widget, title)

    def _pipeline_page(self, tokens: Tokens) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(tokens.space(1.0), tokens.space(1.0), tokens.space(1.0), tokens.space(1.0))
        layout.setSpacing(tokens.space(0.8))
        layout.addWidget(_section("Run configuration"))
        for index, name in enumerate(("io.import_tiff", "preprocess.illumination_basic", "stitch.torch", "segment.omnipose", "track.lineage")):
            row = QPushButton(f"{index + 1:02d}  {name}")
            row.setProperty("variant", "ghost")
            row.setEnabled(False)
            row.setStyleSheet("text-align: left; padding: 5px 6px;" + (" border-color: #2b5b83; background-color: #24384d;" if name == "stitch.torch" else ""))
            layout.addWidget(row)
        layout.addWidget(_section("stitch.torch"))
        for label, value in (("Reference channel", "DAPI · channel 0"), ("Max shift", "100 µm"), ("Window limit", "2"), ("Output", "stitched.zarr / cycle-06")):
            field = QVBoxLayout()
            field.setSpacing(tokens.space(0.3))
            field.addWidget(_dim(label))
            edit = QLineEdit(value)
            edit.setReadOnly(True)
            field.addWidget(edit)
            layout.addLayout(field)
        row = QHBoxLayout()
        preview = QPushButton("Preview")
        preview.setProperty("variant", "ghost")
        preview.setEnabled(False)
        run = QPushButton("Run stitch")
        run.setProperty("variant", "primary")
        run.setEnabled(False)
        row.addWidget(preview)
        row.addWidget(run)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _results_page(self, tokens: Tokens) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(tokens.space(1.0), tokens.space(1.0), tokens.space(1.0), tokens.space(1.0))
        layout.addWidget(_section("Results"))
        layout.addWidget(_dim("cells · edges · tracks · QC"))
        table = QLabel("table placeholder")
        table.setAlignment(Qt.AlignCenter)
        table.setMinimumHeight(tokens.px(120))
        table.setStyleSheet("background-color: #2b2b2b; color: #6b6b6b; border: 1px solid #4a4a4a;")
        layout.addWidget(table)
        layout.addStretch(1)
        return page

    def _inspector_page(self, tokens: Tokens) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(tokens.space(1.0), tokens.space(1.0), tokens.space(1.0), tokens.space(1.0))
        layout.addWidget(_section("Properties"))
        layout.addWidget(_dim("Selected layer / ROI properties go here."))
        layout.addStretch(1)
        return page


class MainShell(QWidget):
    """Layout frame for the next momia3 GUI."""

    def __init__(self, tokens: Tokens | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tokens = tokens or Tokens()
        initialize_application()
        self.setObjectName("ShellRoot")
        self.setWindowTitle("momia3")
        self.setMinimumSize(self.tokens.px(1100), self.tokens.px(700))
        self.setStyleSheet(build_stylesheet(self.tokens))
        self._roi_window = None
        self._roi_session = None
        self._plugin_dialog = None
        self._plugins = None
        self._plugin_context = None
        root = QVBoxLayout(self)
        root.setContentsMargins(self.tokens.space(1.0), self.tokens.space(1.0), self.tokens.space(1.0), self.tokens.space(1.0))
        root.setSpacing(self.tokens.space(0.8))
        top_bar = _TopBar(self.tokens)
        top_bar.roi_requested.connect(self.open_roi_manager)
        top_bar.plugins_requested.connect(self.open_plugins)
        root.addWidget(top_bar)
        root.addWidget(_WorkflowStrip(self.tokens))
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setChildrenCollapsible(False)
        dataset = _DatasetPanel(self.tokens)
        viewer = _ViewerPanel(self.tokens)
        inspector = _InspectorPanel(self.tokens)
        self._inspector = inspector
        splitter.addWidget(dataset)
        splitter.addWidget(viewer)
        splitter.addWidget(inspector)
        splitter.setSizes([self.tokens.px(260), self.tokens.px(760), self.tokens.px(360)])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        root.addWidget(splitter, 1)
        status = QFrame()
        status.setProperty("role", "strip")
        sl = QHBoxLayout(status)
        sl.setContentsMargins(self.tokens.space(1.2), self.tokens.space(0.4), self.tokens.space(1.2), self.tokens.space(0.4))
        sl.addWidget(_dim("● connected"))
        sl.addWidget(_dim("workspace: exemplar-001"))
        sl.addWidget(_dim("tile 6 / 6"))
        sl.addStretch(1)
        self._status_message = _dim("")
        sl.addWidget(self._status_message)
        sl.addWidget(_dim("GPU · MPS"))
        sl.addWidget(_dim("zoom 62%"))
        root.addWidget(status)
        self._load_plugins()

    def _ensure_roi_session(self):
        from .roi_manager import demo_roi_session

        if self._roi_session is None:
            self._roi_session = demo_roi_session()
        return self._roi_session

    def _notify(self, message: str) -> None:
        if self._status_message is not None:
            self._status_message.setText(message)

    def _open_plugin_window(self, title: str, widget: QWidget) -> None:
        widget.setWindowTitle(title)
        widget.setParent(self, widget.windowFlags())
        widget.show()
        widget.raise_()
        widget.activateWindow()

    def _load_plugins(self) -> None:
        from .plugins import GuiContext, InspectorTab, PluginRegistry

        self._plugins = PluginRegistry()
        self._plugins.load()
        self._plugin_context = GuiContext(
            tokens=self.tokens,
            roi_session_getter=self._ensure_roi_session,
            open_window=self._open_plugin_window,
            notify=self._notify,
        )
        for spec in self._plugins.specs():
            for contribution in spec.contributions:
                if isinstance(contribution, InspectorTab):
                    try:
                        self._inspector.add_tab(contribution.title, contribution.factory(self._plugin_context))
                    except Exception as exc:
                        self._plugins.errors.append(f"{spec.id}: inspector tab failed: {exc}")
        if self._plugins.errors:
            self._notify(f"plugin error: {self._plugins.errors[-1]}")

    def open_plugins(self) -> None:
        from .plugins import PluginManagerDialog

        if self._plugins is None or self._plugin_context is None:
            self._load_plugins()
        if self._plugin_dialog is None:
            self._plugin_dialog = PluginManagerDialog(self._plugins, self._plugin_context, parent=self)
        self._plugin_dialog.show()
        self._plugin_dialog.raise_()
        self._plugin_dialog.activateWindow()

    def open_roi_manager(self) -> None:
        from .roi_manager import ROIManagerWindow

        if self._roi_window is None:
            self._roi_window = ROIManagerWindow(self._ensure_roi_session(), self.tokens, parent=self)
        self._roi_window.show()
        self._roi_window.raise_()
        self._roi_window.activateWindow()


def _chip(text: str, ok: bool = False) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "chip-ok" if ok else "chip")
    return label


def _rule(tokens: Tokens) -> QFrame:
    rule = QFrame()
    rule.setProperty("role", "rule")
    rule.setFixedSize(1, tokens.px(24))
    return rule


def _section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("role", "section")
    return label


def _dim(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "dim")
    return label
