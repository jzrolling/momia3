"""Qt dock widgets of the momia3 napari plugin (ARCHITECTURE.md §8).

Loaded lazily by npe2 when a dock widget is requested (or explicitly in
tests); ``import momia3`` never imports this module.

Widgets
-------
:class:`DatasetWidget`
    Open an ``experiment.zarr``, browse FOVs/channels, load Image/Labels
    layers (lazy via dask when available; the time axis maps to napari's
    slider), and re-run labels_to_cells + outlines + midlines for the
    current frame after a manual labels edit.
:class:`PipelineWidget`
    Registry-driven pipeline editor: op dropdown, parameter form
    auto-generated from the op's pydantic model via magicgui's type map,
    editable step list, JSON export, background execution through
    :func:`momia3.pipeline.engine.run` with a progress bar.
:class:`ResultsWidget`
    Open cells/edges/tracks parquets of a workspace and render outlines
    and midlines as Shapes layers and lineages as a Tracks layer, with a
    clickable track table that filters the Tracks layer.

One GUI-only registry extension is added on import: ``gui.open_store``
(public :func:`momia3.pipeline.registry.register` API — no pipeline
internals are modified) lets a GUI pipeline start from an already-opened
experiment store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union, get_args, get_origin

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import PydanticUndefined

from magicgui.widgets import Container, create_widget
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from momia3.gui import layers as _layers
from momia3.ops import measure as _measure_ops
from momia3.ops import midline as _midline_ops
from momia3.ops import outline as _outline_ops
from momia3.pipeline import engine
from momia3.pipeline.registry import REGISTRY, StepContext, get, list_ops, register
from momia3.storage.tables import read_table
from momia3.storage.zarr_store import ExperimentStore

logger = logging.getLogger(__name__)

__all__ = ["DatasetWidget", "PipelineWidget", "ResultsWidget"]

_LAYER_OUTLINES = "momia3 outlines"
_LAYER_MIDLINES = "momia3 midlines"
_LAYER_TRACKS = "momia3 tracks"


# ---------------------------------------------------------------------------
# GUI-only helper op: open an existing store as a pipeline entry point
# ---------------------------------------------------------------------------


class _OpenStoreParams(BaseModel):
    """Parameters of the ``gui.open_store`` helper op."""

    path: str = Field(description="Path of an existing experiment.zarr directory")


def _run_open_store(
    inputs: dict[str, Any], params: _OpenStoreParams, ctx: StepContext
) -> dict[str, Any]:
    """Runner of ``gui.open_store`` (passes the store path through)."""
    del inputs, ctx
    path = Path(params.path)
    if not path.is_dir():
        raise FileNotFoundError(f"experiment store not found: {path}")
    return {"experiment": path}


if "gui.open_store" not in REGISTRY:
    register(
        "gui.open_store",
        params_model=_OpenStoreParams,
        inputs={},
        outputs={"experiment": "store"},
        description=(
            "Open an existing experiment.zarr store by path (GUI helper "
            "step; registered by the napari plugin, not part of the "
            "canonical pipeline)."
        ),
        runner=_run_open_store,
    )


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _sibling_widget(viewer: Any, cls: type) -> Any:
    """Find another momia3 dock widget attached to the same viewer."""
    if viewer is None:
        return None
    try:
        window = viewer.window
        public = getattr(window, "dock_widgets", None)
        if public is not None:  # napari >= 0.8 public mapping of inner widgets
            candidates = list(public.values())
        else:
            candidates = [
                dock.widget() for dock in window._dock_widgets.values()  # noqa: SLF001
            ]
        for widget in candidates:
            inner = getattr(widget, "widget", None)
            if callable(inner):  # QDockWidget wrapper -> inner widget
                try:
                    widget = inner() or widget
                except Exception:  # pragma: no cover - defensive
                    pass
            if isinstance(widget, cls):
                return widget
            native = getattr(widget, "native", None)  # magicgui wrapper
            if isinstance(native, cls):
                return native
    except Exception:  # pragma: no cover - defensive against napari internals
        logger.debug("dock widget scan failed", exc_info=True)
    return None


def _lazy_array(zarr_array: Any) -> Any:
    """Dask wrapper of a zarr array when dask is available, else eager."""
    try:
        import dask.array as da

        return da.from_zarr(zarr_array)
    except Exception:
        logger.debug("dask unavailable; loading array eagerly", exc_info=True)
        return np.asarray(zarr_array)


def _status(label: QLabel, text: str, error: bool = False) -> None:
    label.setText(text)
    label.setStyleSheet(f"color: {'#c0392b' if error else '#557733'};")


def _ask_directory(parent: QWidget, caption: str) -> str | None:
    path = QFileDialog.getExistingDirectory(parent, caption)
    return path or None


class _ImportTiffDialog(QDialog):
    """Collect ``io.import_tiff`` parameters for a one-click TIFF import."""

    def __init__(
        self, parent: QWidget, tiff_path: Path, pixel_microns: float
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import TIFF")
        form = QFormLayout(self)
        path_label = QLabel(str(tiff_path))
        path_label.setWordWrap(True)
        form.addRow("TIFF:", path_label)
        self.pixel_spin = QDoubleSpinBox()
        self.pixel_spin.setDecimals(5)
        self.pixel_spin.setRange(0.0001, 100.0)
        self.pixel_spin.setSingleStep(0.001)
        self.pixel_spin.setValue(pixel_microns)
        form.addRow("Pixel size (µm):", self.pixel_spin)
        self.channels_edit = QLineEdit()
        self.channels_edit.setPlaceholderText(
            "optional, comma-separated (default: ch0, ch1, …)"
        )
        form.addRow("Channels:", self.channels_edit)
        self.workspace_edit = QLineEdit(
            str(tiff_path.parent / f"{tiff_path.stem}_workspace")
        )
        form.addRow("Workspace:", self.workspace_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def channels(self) -> list[str] | None:
        text = self.channels_edit.text().strip()
        if not text:
            return None
        return [c.strip() for c in text.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# generic pydantic-model parameter form (magicgui type map; no per-op code)
# ---------------------------------------------------------------------------

_NoneType = type(None)
_UNION_ORIGINS = {Union}
try:  # PEP 604 unions (``X | None``) have their own origin type
    import types as _types

    _UNION_ORIGINS.add(_types.UnionType)
except (ImportError, AttributeError):  # pragma: no cover - Python < 3.10
    pass


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[X]``/``X | None`` -> ``(X, True)``; else unchanged."""
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [a for a in get_args(annotation) if a is not _NoneType]
        if len(args) < len(get_args(annotation)):
            if len(args) == 1:
                return args[0], True
            return Union[tuple(args)], True  # type: ignore[index]
    return annotation, False


def _constraint_options(field: Any) -> dict[str, Any]:
    """``ge``/``le``/``gt``/``lt`` metadata -> magicgui min/max options."""
    options: dict[str, Any] = {}
    for meta in field.metadata:
        ge = getattr(meta, "ge", None)
        gt = getattr(meta, "gt", None)
        le = getattr(meta, "le", None)
        lt = getattr(meta, "lt", None)
        if ge is not None:
            options["min"] = ge
        elif gt is not None:
            options["min"] = gt
        if le is not None:
            options["max"] = le
        elif lt is not None:
            options["max"] = lt
    return options


def _json_line_parser(edit: Any, nullable: bool) -> Callable[[], Any]:
    """Parser of a JSON LineEdit: empty -> None (if nullable) else error."""

    def parse() -> Any:
        text = str(edit.value).strip()
        if not text or text.lower() in ("none", "null"):
            if nullable:
                return None
            raise ValueError("a JSON value is required")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text  # bare word -> plain string

    return parse


class ParamForm:
    """magicgui form auto-generated from a pydantic params model.

    Field mapping (generic, model-driven — never per-op):

    - ``Literal[...]`` -> ComboBox, ``bool`` -> CheckBox, ``int`` ->
      SpinBox, ``float`` -> FloatSpinBox, ``str`` -> LineEdit (via
      :func:`magicgui.type_map.get_widget_class` /
      :func:`magicgui.widgets.create_widget`);
    - ``ge``/``le``/``gt``/``lt`` constraints -> widget min/max;
    - ``Optional[...]`` with a ``None`` default and tuple/list
      annotations -> JSON LineEdit (empty string = None), so nullable
      parameters round-trip exactly;
    - ``Field(description=...)`` -> widget tooltip (§3.1).
    """

    def __init__(self, params_model: type[BaseModel]) -> None:
        self.params_model = params_model
        self._parsers: dict[str, Callable[[], Any]] = {}
        widgets: list[Any] = []
        for name, field in params_model.model_fields.items():
            widget, parser = self._build_field(name, field)
            widgets.append(widget)
            self._parsers[name] = parser
        self.container = Container(widgets=widgets, layout="vertical", labels=True)

    @property
    def native(self) -> QWidget:
        return self.container.native

    def _build_field(self, name: str, field: Any) -> tuple[Any, Callable[[], Any]]:
        annotation, optional_ann = _strip_optional(field.annotation)
        has_default = field.default is not PydanticUndefined
        default = field.default if has_default else None
        nullable = optional_ann or (has_default and default is None)
        origin = get_origin(annotation)
        is_sequence = origin in (list, tuple) or annotation in (list, tuple)
        description = field.description or ""

        if is_sequence or (nullable and default is None):
            # JSON LineEdit: exact round-trip for sequences and nullables.
            text = "" if default is None else json.dumps(default)
            widget = create_widget(value=text, annotation=str, name=name)
            parser = _json_line_parser(widget, nullable or not has_default)
        else:
            # magicgui's type map: Literal->ComboBox, bool->CheckBox,
            # int->SpinBox, float->FloatSpinBox, str->LineEdit; field
            # constraints become widget min/max.
            widget = create_widget(
                value=default,
                annotation=annotation,
                name=name,
                options=_constraint_options(field),
            )
            if annotation is float and hasattr(widget.native, "setDecimals"):
                # QDoubleSpinBox defaults to 2 decimals and would round
                # e.g. 0.065 -> 0.07; keep sub-pixel precision and
                # re-assign the full-precision default.
                widget.native.setDecimals(6)
                if default is not None:
                    widget.value = default
            parser = lambda w=widget: w.value  # noqa: E731
        if description:
            widget.tooltip = description
        return widget, parser

    def values(self) -> dict[str, Any]:
        """Current form values as a plain dict (JSON-parse errors raised)."""
        values: dict[str, Any] = {}
        problems: list[str] = []
        for name, parse in self._parsers.items():
            try:
                values[name] = parse()
            except Exception as exc:
                problems.append(f"{name}: {exc}")
        if problems:
            raise ValueError("; ".join(problems))
        return values

    def validated_params(self) -> dict[str, Any]:
        """Form values validated against the pydantic model (JSON-safe)."""
        model = self.params_model.model_validate(self.values())
        return model.model_dump(mode="json")


# ---------------------------------------------------------------------------
# background engine execution
# ---------------------------------------------------------------------------


@thread_worker
def _engine_worker(
    pipeline: Mapping[str, Any], workspace: str, n_workers: int
) -> engine.RunResult:
    """Run the pipeline engine off the GUI thread."""
    return engine.run(pipeline, workspace=workspace, n_workers=n_workers)


# ---------------------------------------------------------------------------
# DatasetWidget
# ---------------------------------------------------------------------------


class DatasetWidget(QWidget):
    """Dataset browser dock widget ("momia3: Dataset")."""

    def __init__(self, napari_viewer: Any = None) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._store: ExperimentStore | None = None
        self._labels_layer: Any = None
        self._last_pixel_microns = 0.04968

        layout = QVBoxLayout(self)
        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open experiment.zarr…")
        self.open_button.clicked.connect(self._on_open_clicked)
        open_row.addWidget(self.open_button)
        self.import_button = QPushButton("Import TIFF…")
        self.import_button.setToolTip(
            "Import a raw TIFF (io.import_tiff) into a new experiment.zarr "
            "workspace, then open it here."
        )
        self.import_button.clicked.connect(self._on_import_clicked)
        open_row.addWidget(self.import_button)
        layout.addLayout(open_row)
        self.path_label = QLabel("(no store open)")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        fov_row = QHBoxLayout()
        fov_row.addWidget(QLabel("FOV:"))
        self.fov_combo = QComboBox()
        fov_row.addWidget(self.fov_combo, stretch=1)
        layout.addLayout(fov_row)

        layout.addWidget(QLabel("Channels:"))
        self.channel_list = QListWidget()
        self.channel_list.setMaximumHeight(90)
        layout.addWidget(self.channel_list)

        self.load_image_button = QPushButton("Load channel as Image layer")
        self.load_image_button.clicked.connect(lambda: self.load_channel())
        layout.addWidget(self.load_image_button)
        self.load_labels_button = QPushButton("Load labels as Labels layer")
        self.load_labels_button.clicked.connect(lambda: self.load_labels())
        layout.addWidget(self.load_labels_button)
        self.recompute_button = QPushButton("Recompute cells (current frame)")
        self.recompute_button.setToolTip(
            "After editing the Labels layer, re-run labels_to_cells + "
            "outlines + midlines for the current frame and refresh the "
            "Results widget."
        )
        self.recompute_button.clicked.connect(self.recompute_current_frame)
        layout.addWidget(self.recompute_button)

        layout.addWidget(QLabel("Store attributes:"))
        self.attrs_panel = QPlainTextEdit()
        self.attrs_panel.setReadOnly(True)
        self.attrs_panel.setMaximumHeight(140)
        layout.addWidget(self.attrs_panel)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # store access
    # ------------------------------------------------------------------
    def _on_open_clicked(self) -> None:
        path = _ask_directory(self, "Open experiment.zarr")
        if path:
            self.open_store(path)

    def _on_import_clicked(self) -> None:
        """One-click TIFF import: io.import_tiff into a fresh workspace."""
        from momia3.pipeline.io_ops import ImportTiffParams, run_import_tiff

        tiff, _ = QFileDialog.getOpenFileName(
            self,
            "Import TIFF",
            "",
            "TIFF images (*.tif *.tiff *.ome.tif *.ome.tiff);;All files (*)",
        )
        if not tiff:
            return
        tiff_path = Path(tiff)
        dialog = _ImportTiffDialog(self, tiff_path, self._last_pixel_microns)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        workspace = Path(dialog.workspace_edit.text()).expanduser()
        store_path = workspace / "experiment.zarr"
        if store_path.exists():
            _status(
                self.status_label,
                f"refusing to overwrite existing store: {store_path}",
                error=True,
            )
            return
        self._last_pixel_microns = float(dialog.pixel_spin.value())
        params = ImportTiffParams(
            path=str(tiff_path),
            pixel_microns=self._last_pixel_microns,
            channels=dialog.channels(),
        )
        workspace.mkdir(parents=True, exist_ok=True)
        ctx = StepContext(
            step_id="import",
            workspace=workspace,
            outputs={"experiment": store_path},
        )
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            run_import_tiff({}, params, ctx)
        except Exception as exc:
            _status(self.status_label, f"import failed: {exc}", error=True)
            return
        finally:
            self.unsetCursor()
        self.open_store(store_path)
        try:  # show the first channel right away
            self.load_channel()
        except Exception as exc:
            logger.debug("auto-load after import failed: %s", exc)

    def open_store(self, path: str | Path) -> ExperimentStore:
        """Open an experiment store and populate the FOV/channel panels."""
        self._store = ExperimentStore.open(path, mode="a")
        store = self._store
        self.path_label.setText(str(store.path))
        self.fov_combo.clear()
        for fov_id in store.fov_ids:
            self.fov_combo.addItem(str(fov_id))
        self.channel_list.clear()
        for channel in store.channels:
            self.channel_list.addItem(channel)
        if self.channel_list.count():
            self.channel_list.setCurrentRow(0)
        attrs = dict(store.attrs.asdict()) if hasattr(store.attrs, "asdict") else dict(store.attrs)
        info: dict[str, Any] = {"attrs": attrs, "fovs": {}}
        for fov_id in store.fov_ids:
            entry: dict[str, Any] = {}
            for channel in store.channels:
                try:
                    entry[f"images/{channel}"] = list(store.shape(fov_id, channel))
                except KeyError:
                    pass
            try:
                entry["labels"] = list(store.labels(fov_id).shape)
            except KeyError:
                entry["labels"] = None
            info["fovs"][str(fov_id)] = entry
        self.attrs_panel.setPlainText(json.dumps(info, indent=2, default=str))
        _status(self.status_label, f"opened {store.path}")
        return store

    def _current_fov(self) -> int:
        text = self.fov_combo.currentText()
        if not text:
            raise RuntimeError("no FOV selected (open a store first)")
        return int(text)

    def _current_channel(self) -> str | None:
        item = self.channel_list.currentItem()
        if item is not None:
            return item.text()
        if self._store is not None and self._store.channels:
            return self._store.channels[0]
        return None

    # ------------------------------------------------------------------
    # layer loading
    # ------------------------------------------------------------------
    def load_channel(self, fov_id: int | None = None, channel: str | None = None) -> Any:
        """Load one channel stack ``(t, y, x)`` as an Image layer (lazy)."""
        if self.viewer is None:
            raise RuntimeError("no napari viewer attached")
        store = self._require_store()
        fov_id = self._current_fov() if fov_id is None else int(fov_id)
        channel = channel or self._current_channel()
        if channel is None:
            raise RuntimeError("no channel available in this store")
        data = _lazy_array(store.images(fov_id, channel))
        layer = self.viewer.add_image(data, name=f"fov{fov_id}:{channel}")
        _status(self.status_label, f"loaded fov{fov_id}:{channel} {tuple(data.shape)}")
        return layer

    def load_labels(self, fov_id: int | None = None) -> Any:
        """Load the labels stack ``(t, y, x)`` as an editable Labels layer."""
        if self.viewer is None:
            raise RuntimeError("no napari viewer attached")
        store = self._require_store()
        fov_id = self._current_fov() if fov_id is None else int(fov_id)
        data = _lazy_array(store.labels(fov_id))
        self._labels_layer = self.viewer.add_labels(data, name=f"fov{fov_id}:labels")
        _status(self.status_label, f"loaded labels fov{fov_id} {tuple(data.shape)}")
        return self._labels_layer

    def _require_store(self) -> ExperimentStore:
        if self._store is None:
            raise RuntimeError("open an experiment.zarr first")
        return self._store

    # ------------------------------------------------------------------
    # manual-edit hook (§8 interactive correction)
    # ------------------------------------------------------------------
    def recompute_current_frame(self) -> Any:
        """Re-run labels_to_cells + outlines + midlines for the current frame.

        Takes the (possibly hand-edited) Labels layer at the viewer's
        current time point, persists it into the store, recomputes the
        frame's cells table, and asks the sibling Results widget to
        refresh its layers.
        """
        store = self._require_store()
        if self._labels_layer is None:
            _status(self.status_label, "load the labels layer first", error=True)
            return None
        fov_id = self._current_fov()
        steps = tuple(self.viewer.dims.current_step) if self.viewer is not None else ()
        t = int(steps[0]) if steps else 0
        try:
            edited = np.asarray(self._labels_layer.data[t]).astype(np.uint32)
            store.set_labels_frame(fov_id, t, edited)
            channel = self._current_channel()
            intensity = None
            if channel is not None and channel in store.channels:
                intensity = {channel: store.get_frame(fov_id, channel, t)}
            pixel_microns = store.pixel_microns
            cells = _measure_ops.labels_to_cells(
                edited, fov_id, t, intensity=intensity, pixel_microns=pixel_microns
            )
            outlines = _outline_ops.extract_outlines(edited, None, {})
            cells = _outline_ops.fill_cells_geometry(cells, outlines)
            results = _midline_ops.extract_midlines(
                edited, outlines, {}, pixel_microns=pixel_microns
            )
            cells = _midline_ops.fill_cells_midline(cells, results)
        except Exception as exc:
            _status(self.status_label, f"recompute failed: {exc}", error=True)
            logger.exception("recompute failed")
            return None
        results_widget = _sibling_widget(self.viewer, ResultsWidget)
        if results_widget is not None:
            results_widget.update_frame_cells(fov_id, t, cells)
        _status(
            self.status_label,
            f"recomputed {cells.num_rows} cell(s) at fov{fov_id} t={t}",
        )
        return cells


# ---------------------------------------------------------------------------
# PipelineWidget
# ---------------------------------------------------------------------------


def _kinds_compatible(src: str, dst: str) -> bool:
    """Local copy of the engine's lenient artifact-type compatibility."""
    dst = dst.rstrip("?")
    if dst == "any" or src == dst:
        return True
    if dst == "table" and src.startswith("table"):
        return True
    return bool(dst.startswith("table") and src == "table")


def _params_summary(params: Mapping[str, Any], max_items: int = 3) -> str:
    items = [f"{k}={v!r}" for k, v in list(params.items())[:max_items]]
    suffix = ", …" if len(params) > max_items else ""
    return ", ".join(items) + suffix


class PipelineWidget(QWidget):
    """Pipeline editor dock widget ("momia3: Pipeline")."""

    def __init__(self, napari_viewer: Any = None) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._form: ParamForm | None = None
        self._worker: Any = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self._specs = {spec.name: spec for spec in list_ops()}
        self.op_combo.addItems(sorted(self._specs))
        self.op_combo.currentTextChanged.connect(self._on_op_changed)
        layout.addWidget(self.op_combo)
        self.op_info = QLabel("")
        self.op_info.setWordWrap(True)
        layout.addWidget(self.op_info)

        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setMinimumHeight(140)
        layout.addWidget(self.form_scroll)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Step id:"))
        self.step_id_edit = QLineEdit()
        id_row.addWidget(self.step_id_edit, stretch=1)
        layout.addLayout(id_row)

        add_row = QHBoxLayout()
        self.add_button = QPushButton("Add step")
        self.add_button.clicked.connect(lambda: self.add_step())
        add_row.addWidget(self.add_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected_step)
        add_row.addWidget(self.remove_button)
        self.up_button = QPushButton("↑")
        self.up_button.clicked.connect(lambda: self.move_step(-1))
        add_row.addWidget(self.up_button)
        self.down_button = QPushButton("↓")
        self.down_button.clicked.connect(lambda: self.move_step(1))
        add_row.addWidget(self.down_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.step_list_clear)
        add_row.addWidget(self.clear_button)
        layout.addLayout(add_row)

        self.step_list = QListWidget()
        self.step_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.step_list)

        ws_row = QHBoxLayout()
        ws_row.addWidget(QLabel("Workspace:"))
        self.workspace_edit = QLineEdit("momia3_gui_workspace")
        ws_row.addWidget(self.workspace_edit, stretch=1)
        self.ws_browse = QPushButton("…")
        self.ws_browse.clicked.connect(self._on_browse_workspace)
        ws_row.addWidget(self.ws_browse)
        layout.addLayout(ws_row)

        store_row = QHBoxLayout()
        store_row.addWidget(QLabel("Input store:"))
        self.store_edit = QLineEdit()
        self.store_edit.setPlaceholderText(
            "optional experiment.zarr for the first store input"
        )
        store_row.addWidget(self.store_edit, stretch=1)
        self.store_browse = QPushButton("…")
        self.store_browse.clicked.connect(self._on_browse_store)
        store_row.addWidget(self.store_browse)
        layout.addLayout(store_row)

        run_row = QHBoxLayout()
        self.export_button = QPushButton("Export pipeline JSON")
        self.export_button.clicked.connect(lambda: self.export_json())
        run_row.addWidget(self.export_button)
        run_row.addWidget(QLabel("workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(1)
        run_row.addWidget(self.workers_spin)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_pipeline)
        run_row.addWidget(self.run_button)
        layout.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(400)
        self._poll_timer.timeout.connect(self._poll_progress)

        self._on_op_changed(self.op_combo.currentText())

    # ------------------------------------------------------------------
    # op form
    # ------------------------------------------------------------------
    def _on_op_changed(self, name: str) -> None:
        spec = self._specs.get(name)
        if spec is None:
            return
        self.op_info.setText(
            f"{spec.description}\n"
            f"inputs: {spec.inputs}  outputs: {spec.outputs}"
        )
        self.op_info.setToolTip(spec.description)
        self._form = ParamForm(spec.params_model)
        self.form_scroll.setWidget(self._form.native)
        self.step_id_edit.setText(self._next_step_id(name))

    def _next_step_id(self, op_name: str) -> str:
        base = op_name.replace(".", "_")
        existing = {self.step_list.item(i).data(Qt.UserRole)["id"] for i in range(self.step_list.count())}
        if base not in existing:
            return base
        n = 2
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def select_op(self, name: str) -> None:
        """Select an op programmatically (builds its parameter form)."""
        self.op_combo.setCurrentText(name)

    # ------------------------------------------------------------------
    # step list
    # ------------------------------------------------------------------
    def add_step(self, step_id: str | None = None) -> dict[str, Any] | None:
        """Validate the form and append ``{id, op, params}`` to the list."""
        if self._form is None:
            return None
        op_name = self.op_combo.currentText()
        sid = (step_id or self.step_id_edit.text()).strip()
        if not sid:
            _status(self.status_label, "step id required", error=True)
            return None
        if "." in sid:
            _status(self.status_label, "step ids must not contain '.'", error=True)
            return None
        existing = {self.step_list.item(i).data(Qt.UserRole)["id"] for i in range(self.step_list.count())}
        if sid in existing:
            _status(self.status_label, f"duplicate step id {sid!r}", error=True)
            return None
        try:
            params = self._form.validated_params()
        except (ValidationError, ValueError) as exc:
            _status(self.status_label, f"invalid params: {exc}", error=True)
            return None
        step = {"id": sid, "op": op_name, "params": params}
        item = QListWidgetItem(f"{sid}: {op_name}({_params_summary(params)})")
        item.setData(Qt.UserRole, step)
        self.step_list.addItem(item)
        self.step_id_edit.setText(self._next_step_id(op_name))
        _status(self.status_label, f"added step {sid!r}")
        return step

    def remove_selected_step(self) -> None:
        row = self.step_list.currentRow()
        if row >= 0:
            self.step_list.takeItem(row)

    def move_step(self, delta: int) -> None:
        row = self.step_list.currentRow()
        new_row = row + delta
        if row < 0 or not 0 <= new_row < self.step_list.count():
            return
        item = self.step_list.takeItem(row)
        self.step_list.insertItem(new_row, item)
        self.step_list.setCurrentRow(new_row)

    def step_list_clear(self) -> None:
        self.step_list.clear()

    def steps(self) -> list[dict[str, Any]]:
        """Current step dicts in list order."""
        return [
            dict(self.step_list.item(i).data(Qt.UserRole))
            for i in range(self.step_list.count())
        ]

    # ------------------------------------------------------------------
    # pipeline assembly (auto-wiring of step inputs)
    # ------------------------------------------------------------------
    def _on_browse_workspace(self) -> None:
        path = _ask_directory(self, "Choose workspace directory")
        if path:
            self.workspace_edit.setText(path)

    def _on_browse_store(self) -> None:
        path = _ask_directory(self, "Choose input experiment.zarr")
        if path:
            self.store_edit.setText(path)

    def _find_producer(
        self, arg: str, kind: str, available: dict[tuple[str, str], str]
    ) -> str | None:
        """Most recent compatible upstream output reference for one input."""
        want = kind.rstrip("?")
        items = list(available.items())
        for (sid, oname), okind in reversed(items):  # same-name preferred
            if oname == arg and _kinds_compatible(okind, want):
                return f"{sid}.{oname}"
        for (sid, oname), okind in reversed(items):
            if _kinds_compatible(okind, want):
                return f"{sid}.{oname}"
        if want == "store" and self.store_edit.text().strip():
            return "open_store.experiment"
        return None

    def build_pipeline(self) -> dict[str, Any]:
        """Assemble the pipeline dict (auto-wired inputs + helper step)."""
        steps = self.steps()
        if not steps:
            raise ValueError("no steps in the pipeline")
        wired: list[dict[str, Any]] = []
        available: dict[tuple[str, str], str] = {}
        uses_open_store = False
        for step in steps:
            spec = get(step["op"])
            inputs = dict(step.get("inputs") or {})
            for arg, kind in spec.inputs.items():
                if arg in inputs or kind.endswith("?"):
                    continue
                ref = self._find_producer(arg, kind, available)
                if ref is not None:
                    inputs[arg] = ref
                    uses_open_store = uses_open_store or ref.startswith("open_store.")
            wired.append({"id": step["id"], "op": step["op"], "params": step["params"], "inputs": inputs})
            for out_name, out_kind in spec.outputs.items():
                available[(step["id"], out_name)] = out_kind
        if uses_open_store:
            wired.insert(
                0,
                {
                    "id": "open_store",
                    "op": "gui.open_store",
                    "params": {"path": self.store_edit.text().strip()},
                    "inputs": {},
                },
            )
        return {
            "name": "momia3_gui_pipeline",
            "workspace": self.workspace_edit.text().strip() or "momia3_gui_workspace",
            "steps": wired,
        }

    def export_json(self, path: str | Path | None = None) -> Path | None:
        """Export the assembled pipeline to a JSON file."""
        try:
            pipeline = self.build_pipeline()
        except ValueError as exc:
            _status(self.status_label, str(exc), error=True)
            return None
        if path is None:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Export pipeline JSON", "pipeline.json", "JSON (*.json)"
            )
            if not chosen:
                return None
            path = chosen
        path = Path(path)
        path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
        _status(self.status_label, f"exported {path}")
        return path

    # ------------------------------------------------------------------
    # execution (background thread + progress polling)
    # ------------------------------------------------------------------
    def run_pipeline(self) -> bool:
        """Validate and execute the pipeline in a napari thread worker."""
        if self._worker is not None:
            _status(self.status_label, "a run is already in progress", error=True)
            return False
        try:
            pipeline = self.build_pipeline()
        except ValueError as exc:
            _status(self.status_label, str(exc), error=True)
            return False
        errors = engine.validate(pipeline)
        if errors:
            _status(self.status_label, f"invalid pipeline: {errors[0]}", error=True)
            return False
        workspace = str(pipeline["workspace"])
        self.progress.setRange(0, len(pipeline["steps"]))
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        _status(self.status_label, "running…")
        worker = _engine_worker(pipeline, workspace, int(self.workers_spin.value()))
        worker.returned.connect(self._on_run_returned)
        worker.errored.connect(self._on_run_errored)
        self._worker = worker
        self._run_workspace = workspace
        worker.start()
        self._poll_timer.start()
        return True

    def _poll_progress(self) -> None:
        """Progress = steps with a provenance record (written per step)."""
        try:
            prov = Path(self._run_workspace) / "provenance"
            done = len(list(prov.glob("*.json"))) if prov.is_dir() else 0
            self.progress.setValue(min(done, self.progress.maximum()))
        except Exception:  # pragma: no cover - cosmetic
            logger.debug("progress poll failed", exc_info=True)

    def _finish_run(self) -> None:
        self._poll_timer.stop()
        self._worker = None
        self.run_button.setEnabled(True)

    def _on_run_errored(self, exc: BaseException) -> None:
        self._finish_run()
        _status(self.status_label, f"run failed: {exc}", error=True)
        logger.exception("pipeline run failed", exc_info=exc)

    def _on_run_returned(self, result: engine.RunResult) -> None:
        self._finish_run()
        self.progress.setValue(self.progress.maximum())
        _status(
            self.status_label,
            f"done — {result.stats['computed']} computed, "
            f"{result.stats['cached']} cached → {result.workspace}",
        )
        self._auto_load_results(result)

    def _auto_load_results(self, result: engine.RunResult) -> None:
        """Load resulting tables into Results and labels into the viewer."""
        if self.viewer is None:
            return
        tables: dict[str, Any] = {}
        store_paths: list[Path] = []
        for step in result.steps:
            for out_name, value in result.outputs.get(step.step_id, {}).items():
                if isinstance(value, Path):
                    store_paths.append(value)
                elif out_name in ("cells", "edges", "tracks"):
                    tables[out_name] = value
        results_widget = _sibling_widget(self.viewer, ResultsWidget)
        if results_widget is not None and tables:
            results_widget.open_dataframes(
                cells=tables.get("cells"),
                edges=tables.get("edges"),
                tracks=tables.get("tracks"),
            )
        if store_paths:
            dataset_widget = _sibling_widget(self.viewer, DatasetWidget)
            try:
                if dataset_widget is not None:
                    dataset_widget.open_store(store_paths[-1])
                    dataset_widget.load_labels()
                else:
                    store = ExperimentStore.open(store_paths[-1], mode="r")
                    fov_id = store.fov_ids[0]
                    self.viewer.add_labels(
                        _lazy_array(store.labels(fov_id)), name=f"fov{fov_id}:labels"
                    )
            except Exception:
                logger.debug("auto-load of result store failed", exc_info=True)


# ---------------------------------------------------------------------------
# ResultsWidget
# ---------------------------------------------------------------------------


class ResultsWidget(QWidget):
    """Results viewer dock widget ("momia3: Results")."""

    def __init__(self, napari_viewer: Any = None) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._cells: pl.DataFrame | None = None
        self._edges: pl.DataFrame | None = None
        self._tracks: pl.DataFrame | None = None
        self._tracks_full: tuple[np.ndarray, dict[str, np.ndarray], dict[int, int]] | None = None

        layout = QVBoxLayout(self)
        self.open_button = QPushButton("Open workspace…")
        self.open_button.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_button)
        self.path_label = QLabel("(no workspace open)")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color tracks by:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("track_id")
        self.color_combo.currentTextChanged.connect(self._on_color_by_changed)
        color_row.addWidget(self.color_combo, stretch=1)
        layout.addLayout(color_row)

        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Max displayed cells:"))
        self.max_cells_spin = QSpinBox()
        self.max_cells_spin.setRange(100, 1_000_000)
        self.max_cells_spin.setValue(_layers.MAX_SHAPES_CELLS)
        sub_row.addWidget(self.max_cells_spin, stretch=1)
        self.rebuild_button = QPushButton("Rebuild layers")
        self.rebuild_button.clicked.connect(self.rebuild_layers)
        sub_row.addWidget(self.rebuild_button)
        layout.addLayout(sub_row)

        layout.addWidget(QLabel("Tracks (click to isolate):"))
        self.tracks_table = QTableWidget(0, 5)
        self.tracks_table.setHorizontalHeaderLabels(
            ["track_id", "birth", "last", "n_frames", "fate"]
        )
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tracks_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tracks_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tracks_table.itemSelectionChanged.connect(self._on_track_selected)
        layout.addWidget(self.tracks_table)

        self.show_all_button = QPushButton("Show all tracks")
        self.show_all_button.clicked.connect(self.show_all_tracks)
        layout.addWidget(self.show_all_button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # data access
    # ------------------------------------------------------------------
    def _on_open_clicked(self) -> None:
        path = _ask_directory(self, "Open results workspace")
        if path:
            self.open_workspace(path)

    @staticmethod
    def _find_table(workspace: Path, name: str) -> Path | None:
        for candidate in (workspace / "tables" / f"{name}.parquet", workspace / f"{name}.parquet"):
            if candidate.is_file():
                return candidate
        return None

    def open_workspace(self, path: str | Path) -> None:
        """Open cells/edges/tracks parquets of a workspace directory."""
        workspace = Path(path)
        cells_path = self._find_table(workspace, "cells")
        if cells_path is None:
            _status(self.status_label, f"no cells.parquet under {workspace}", error=True)
            return
        cells = read_table(cells_path)
        edges_path = self._find_table(workspace, "edges")
        tracks_path = self._find_table(workspace, "tracks")
        edges = read_table(edges_path) if edges_path else None
        tracks = read_table(tracks_path) if tracks_path else None
        self.path_label.setText(str(workspace))
        self.open_dataframes(cells=cells, edges=edges, tracks=tracks)

    def open_dataframes(
        self,
        cells: Any = None,
        edges: Any = None,
        tracks: Any = None,
    ) -> None:
        """Set the in-memory tables (polars/pyarrow) and rebuild layers."""
        if cells is not None:
            self._cells = cells if isinstance(cells, pl.DataFrame) else pl.from_arrow(cells)
        if edges is not None:
            self._edges = edges if isinstance(edges, pl.DataFrame) else pl.from_arrow(edges)
        if tracks is not None:
            self._tracks = tracks if isinstance(tracks, pl.DataFrame) else pl.from_arrow(tracks)
        self.rebuild_layers()

    def update_frame_cells(self, fov_id: int, frame: int, cells: Any) -> None:
        """Replace one (fov, frame)'s rows (manual-edit hook refresh)."""
        new = cells if isinstance(cells, pl.DataFrame) else pl.from_arrow(cells)
        if self._cells is None:
            self._cells = new
        else:
            kept = self._cells.filter(
                ~((pl.col("fov_id") == int(fov_id)) & (pl.col("frame") == int(frame)))
            )
            self._cells = pl.concat([kept, new], how="diagonal_relaxed")
        self.rebuild_layers()

    # ------------------------------------------------------------------
    # layer building
    # ------------------------------------------------------------------
    def _remove_layer(self, name: str) -> None:
        if self.viewer is not None and name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[name])

    def _color_choices(self) -> list[str]:
        choices = ["track_id"]
        if self._cells is not None:
            for col in ("length_um", "width_median_um", "area"):
                if col in self._cells.columns:
                    choices.append(col)
            choices += sorted(
                c for c in self._cells.columns if "age" in c and c not in choices
            )
        return choices

    def _refresh_color_combo(self) -> None:
        current = self.color_combo.currentText() or "track_id"
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        self.color_combo.addItems(self._color_choices())
        idx = self.color_combo.findText(current)
        self.color_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.color_combo.blockSignals(False)

    def rebuild_layers(self) -> None:
        """(Re)build outlines/midlines Shapes and the Tracks layer."""
        if self.viewer is None or self._cells is None:
            return
        max_cells = int(self.max_cells_spin.value())
        notes: list[str] = [f"{self._cells.height} cells"]

        shapes, props, sub = _layers.cells_to_outline_shapes(self._cells, max_cells)
        self._remove_layer(_LAYER_OUTLINES)
        if shapes:
            self.viewer.add_shapes(
                shapes,
                shape_type="polygon",
                edge_color="lime",
                face_color="transparent",
                edge_width=1.0,
                properties=props,
                name=_LAYER_OUTLINES,
            )
        if sub:
            notes.append(f"outlines subsampled to {len(shapes)}")

        mid_shapes, mid_props, mid_sub = _layers.cells_to_midline_shapes(
            self._cells, max_cells
        )
        self._remove_layer(_LAYER_MIDLINES)
        if mid_shapes:
            self.viewer.add_shapes(
                mid_shapes,
                shape_type="path",
                edge_color="magenta",
                edge_width=1.5,
                properties=mid_props,
                name=_LAYER_MIDLINES,
            )

        data, tprops, graph = _layers.cells_edges_to_tracks(
            self._cells, self._edges, self._tracks
        )
        self._tracks_full = (data, tprops, graph)
        self._show_tracks(data, tprops, graph)
        notes.append(f"{data.shape[0]} track points, {len(graph)} lineage links")
        self._fill_tracks_table()
        self._refresh_color_combo()
        _status(self.status_label, "; ".join(notes))

    def _show_tracks(
        self,
        data: np.ndarray,
        props: Mapping[str, np.ndarray],
        graph: Mapping[int, int],
    ) -> None:
        """(Re)create the Tracks layer with the given (sub)set of data."""
        self._remove_layer(_LAYER_TRACKS)
        if not data.shape[0]:
            return
        color_by = self.color_combo.currentText() or "track_id"
        if color_by not in props:
            color_by = "track_id"
        self.viewer.add_tracks(
            data,
            properties={k: np.asarray(v) for k, v in props.items()},
            graph=dict(graph),
            color_by=color_by,
            tail_length=20,
            name=_LAYER_TRACKS,
        )

    def _fill_tracks_table(self) -> None:
        self.tracks_table.blockSignals(True)
        self.tracks_table.setRowCount(0)
        rows: list[dict[str, Any]] = []
        if self._tracks is not None and self._tracks.height:
            rows = self._tracks.to_dicts()
        elif self._tracks_full is not None and self._tracks_full[0].shape[0]:
            data = self._tracks_full[0]
            for tid in np.unique(data[:, 0]):
                frames = data[data[:, 0] == tid, 1]
                rows.append(
                    {
                        "track_id": int(tid),
                        "birth_frame": int(frames.min()),
                        "last_frame": int(frames.max()),
                        "n_frames": int(frames.size),
                        "fate": "-",
                    }
                )
        rows.sort(key=lambda r: (int(r.get("fov_id", 0)), int(r.get("track_id", 0))))
        self.tracks_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [
                row.get("track_id"),
                row.get("birth_frame"),
                row.get("last_frame"),
                row.get("n_frames"),
                row.get("fate", "-"),
            ]
            for j, value in enumerate(values):
                self.tracks_table.setItem(i, j, QTableWidgetItem(str(value)))
        self.tracks_table.resizeColumnsToContents()
        self.tracks_table.blockSignals(False)

    # ------------------------------------------------------------------
    # track selection / coloring
    # ------------------------------------------------------------------
    def _on_track_selected(self) -> None:
        if self._tracks_full is None or self.viewer is None:
            return
        if _LAYER_TRACKS not in self.viewer.layers:
            return
        row = self.tracks_table.currentRow()
        if row < 0:
            return
        item = self.tracks_table.item(row, 0)
        if item is None:
            return
        tid = float(item.text())
        data, props, graph = self._tracks_full
        mask = data[:, 0] == tid
        # napari requires every graph node to exist in the visible data.
        visible = {tid}
        sub_graph = {
            child: parent
            for child, parent in graph.items()
            if child in visible and parent in visible
        }
        self._show_tracks(
            data[mask], {k: np.asarray(v)[mask] for k, v in props.items()}, sub_graph
        )
        _status(self.status_label, f"track {int(tid)} isolated ({int(mask.sum())} points)")

    def show_all_tracks(self) -> None:
        """Reset the Tracks layer to the full, unfiltered data."""
        if self._tracks_full is None or self.viewer is None:
            return
        data, props, graph = self._tracks_full
        self._show_tracks(data, props, graph)
        # clearSelection keeps the current row; block signals so the
        # selection-changed handler does not immediately re-isolate it.
        self.tracks_table.blockSignals(True)
        self.tracks_table.clearSelection()
        self.tracks_table.blockSignals(False)
        _status(self.status_label, "showing all tracks")

    def _on_color_by_changed(self, name: str) -> None:
        if self.viewer is None or _LAYER_TRACKS not in self.viewer.layers:
            return
        if not name:
            return
        layer = self.viewer.layers[_LAYER_TRACKS]
        if name in layer.properties or name == "track_id":
            layer.color_by = name
