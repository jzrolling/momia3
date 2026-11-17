"""Layout-only GUI shell for momia3.

The shell is deliberately inert: it provides the persistent workflow strip,
the three-column workspace, the inspector tabs and the status row, but no
widget is wired to a store, pipeline or napari viewer yet.  This file is the
frame that the real Dataset / Pipeline / Results panels will be placed into.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QMessageBox,
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
        from pathlib import Path
        from .project import ProjectSidebar

        self.setObjectName("DatasetPanel")
        self.project = ProjectSidebar(Path.cwd() / "data", tokens, self)
        self._layout.addWidget(self.project, 1)
        self._layout.addWidget(_section("HyperView"))
        self.hyperview_host = QWidget()
        self._hyperview_layout = QVBoxLayout(self.hyperview_host)
        self._hyperview_layout.setContentsMargins(0, 0, 0, 0)
        self._hyperview_layout.setSpacing(tokens.space(0.4))
        self._layout.addWidget(self.hyperview_host)

    def set_hyperview_controls(self, widget: QWidget) -> None:
        while self._hyperview_layout.count():
            item = self._hyperview_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._hyperview_layout.addWidget(widget)


class _ViewerPanel(_Panel):
    def __init__(self, tokens: Tokens, parent: QWidget | None = None) -> None:
        super().__init__(tokens, parent)
        self.setObjectName("ViewerPanel")
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._hyperview = None
        toolbar = QFrame()
        toolbar.setProperty("role", "strip")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(tokens.space(1.0), tokens.space(0.6), tokens.space(1.0), tokens.space(0.6))
        self._source_label = QLabel("HyperView · native viewer")
        tb.addWidget(self._source_label)
        tb.addWidget(_dim("T · C · Z · Y · X"))
        tb.addStretch(1)
        tb.addWidget(_chip("demo 5D"))
        self._layout.addWidget(toolbar)
        self._placeholder = QLabel("HyperView loads on first show")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #9e9e9e;")
        self._layout.addWidget(self._placeholder, 1)
        self._dimension_host = QWidget()
        self._dimension_layout = QVBoxLayout(self._dimension_host)
        self._dimension_layout.setContentsMargins(0, 0, 0, 0)
        self._dimension_layout.setSpacing(0)
        self._layout.addWidget(self._dimension_host)

    def ensure_hyperview(self):
        if self._hyperview is not None:
            return self._hyperview
        from .hyperview import HyperViewWidget

        self._placeholder.hide()
        self._hyperview = HyperViewWidget(self.tokens, self)
        self._layout.insertWidget(1, self._hyperview, 1)
        return self._hyperview

    def set_dimension_controls(self, widget: QWidget) -> None:
        while self._dimension_layout.count():
            item = self._dimension_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._dimension_layout.addWidget(widget)

    def set_source_label(self, text: str) -> None:
        self._source_label.setText(text)


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
        self._hyperview_controls = None
        self._hyperview_dimensions = None
        self._hyperview_popout = None
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
        dataset.project.image_selected.connect(self._load_project_image)
        self._dataset_panel = dataset
        viewer = _ViewerPanel(self.tokens)
        self._viewer_panel = viewer
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

    def _ensure_hyperview(self) -> None:
        widget = self._viewer_panel.ensure_hyperview()
        if widget is None or self._hyperview_controls is not None:
            return
        from .hyperview import HyperViewChannelPanel, HyperViewDimensionBar

        panel = HyperViewChannelPanel(widget.viewer, self.tokens, compact=True)
        panel.bind_layers(widget.layers)
        panel.popout_requested.connect(self._popout_hyperview_controls)
        self._dataset_panel.set_hyperview_controls(panel)
        dimensions = HyperViewDimensionBar(widget.viewer, self.tokens)
        dimensions.bind_layers(widget.layers)
        self._viewer_panel.set_dimension_controls(dimensions)
        self._hyperview_controls = panel
        self._hyperview_dimensions = dimensions

    def _load_project_image(self, path_text: str) -> None:
        from ..io import open_image

        path = Path(path_text)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._ensure_hyperview()
            loaded = open_image(path)
            widget = self._viewer_panel._hyperview
            widget.load_image(loaded)
            if self._hyperview_controls is not None:
                self._hyperview_controls.bind_layers(widget.layers)
            if self._hyperview_dimensions is not None:
                self._hyperview_dimensions.bind_layers(widget.layers)
            if self._hyperview_popout is not None:
                self._hyperview_popout.close()
                self._hyperview_popout = None
            axes = "".join(loaded.axes)
            self._viewer_panel.set_source_label(f"{path.name} · {loaded.source}")
            self._notify(f"loaded {path.name} · {len(loaded.channels)} channels · axes {axes}")
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"{path}\n\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _popout_hyperview_controls(self) -> None:
        widget = self._viewer_panel._hyperview
        if widget is None:
            return
        from .hyperview import HyperViewChannelPanel

        if self._hyperview_popout is None:
            dialog = QDialog(self)
            dialog.setObjectName("HyperViewPopout")
            dialog.setWindowTitle("momia3 · HyperView channels")
            dialog.resize(self.tokens.px(760), self.tokens.px(300))
            dialog.setStyleSheet(build_stylesheet(self.tokens))
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(self.tokens.space(1.0), self.tokens.space(1.0), self.tokens.space(1.0), self.tokens.space(1.0))
            panel = HyperViewChannelPanel(widget.viewer, self.tokens, compact=False)
            panel.bind_layers(widget.layers)
            layout.addWidget(panel)
            self._hyperview_popout = dialog
        self._hyperview_popout.show()
        self._hyperview_popout.raise_()
        self._hyperview_popout.activateWindow()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._ensure_hyperview)

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

        self._ensure_hyperview()
        viewer = self._viewer_panel._hyperview.viewer
        if self._roi_window is None:
            self._roi_window = ROIManagerWindow(self._ensure_roi_session(), self.tokens, parent=self, viewer=viewer)
        else:
            self._roi_window.attach_viewer(viewer)
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
