"""Headless smoke tests of the momia3 napari plugin (ARCHITECTURE.md §8).

The whole module is skipped when the ``gui`` extra (napari) is not
installed, so the base suite stays green without it. Run with::

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui.py -x -q

Headless policy: ``QT_QPA_PLATFORM=offscreen`` is honored when set
explicitly (CI/Linux) and is the default on non-macOS platforms. On
macOS the offscreen platform has no usable GL context (vispy segfaults
in ``glGetParameter``), so the native platform is used instead — the
viewer still never shows a window (``Viewer(show=False)``).

The Qt-free layer builders of :mod:`momia3.gui.layers` are tested here
without a viewer (plain function calls); the widget tests create an
in-process napari Viewer.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if "QT_QPA_PLATFORM" not in os.environ and sys.platform != "darwin":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import polars as pl
import pytest

napari = pytest.importorskip("napari", reason="momia3 gui extra (napari) not installed")
pytest.importorskip("qtpy", reason="momia3 gui extra (qtpy) not installed")

from qtpy.QtWidgets import QApplication

from momia3.gui import layers as gui_layers
from momia3.gui.widgets import (
    DatasetWidget,
    MainPanel,
    PipelineWidget,
    ResultsWidget,
)
from momia3.keys import encode_key
from momia3.pipeline import engine
from momia3.storage.zarr_store import ExperimentStore
from tests.synth import generate_timelapse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_WORKSPACE = REPO_ROOT / "examples" / "demo_workspace"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    """Session QApplication (offscreen)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture()
def viewer(qapp):
    """In-process napari viewer with proper teardown."""
    v = napari.Viewer(show=False)
    yield v
    v.close()
    qapp.processEvents()


@pytest.fixture()
def synth_store(tmp_path):
    """Small synth-derived experiment.zarr in a tmp dir (4 frames, 96 px)."""
    data = generate_timelapse(
        n_frames=4,
        shape=(96, 96),
        n_initial=2,
        pixel_microns=0.065,
        growth_rate_um_per_frame=0.1,
        division_length_um=3.0,
        noise_sigma=0.02,
        seed=3,
    )
    path = tmp_path / "experiment.zarr"
    store = ExperimentStore.create(
        path, pixel_microns=0.065, channels=["phase"], time_interval_s=60.0
    )
    store.write_images(0, "phase", (data["images"] * 65535.0).astype(np.uint16))
    store.write_labels(0, data["labels"])
    return path


def _set_form_values(widget: PipelineWidget, **values) -> None:
    """Set values on the current magicgui param form by field name."""
    assert widget._form is not None
    for w in widget._form.container:
        if w.name in values:
            w.value = values[w.name]


def _pump_until(timeout: float, predicate) -> bool:
    """Process Qt events until ``predicate()`` or timeout (seconds)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Qt-free layer builders
# ---------------------------------------------------------------------------


def _toy_cells() -> pl.DataFrame:
    """Three cells over two frames with outline/midline geometry."""
    outlines = [
        {"x": [0.0, 4.0, 4.0, 0.0], "y": [0.0, 0.0, 8.0, 8.0]},
        {"x": [1.0, 5.0, 5.0, 1.0], "y": [0.0, 0.0, 8.0, 8.0]},
        {"x": [10.0, 14.0, 14.0, 10.0], "y": [10.0, 10.0, 18.0, 18.0]},
    ]
    midlines = [
        [{"x": [2.0, 2.0], "y": [1.0, 7.0]}],
        [{"x": [3.0, 3.0], "y": [1.0, 7.0]}],
        [{"x": [12.0, 12.0], "y": [11.0, 17.0]}],
    ]
    return pl.DataFrame(
        {
            "cell_key": [encode_key(0, 0, 1), encode_key(0, 1, 1), encode_key(0, 1, 2)],
            "fov_id": [0, 0, 0],
            "frame": [0, 1, 1],
            "label": [1, 1, 2],
            "track_id": [None, None, None],
            "centroid": [
                {"x": 2.0, "y": 4.0},
                {"x": 3.0, "y": 4.0},
                {"x": 12.0, "y": 14.0},
            ],
            "outline": outlines,
            "midline": midlines,
            "length_um": [1.0, 1.2, 0.9],
        }
    )


def test_cells_to_outline_shapes_qt_free() -> None:
    cells = _toy_cells()
    shapes, props, subsampled = gui_layers.cells_to_outline_shapes(cells)
    assert len(shapes) == 3
    assert not subsampled
    first = shapes[0]
    assert first.shape == (4, 3)  # (vertices, (t, y, x))
    assert np.all(first[:, 0] == 0.0)  # frame column
    # (x, y) storage -> (t, y, x) vertices
    assert np.allclose(first[0], [0.0, 0.0, 0.0])
    assert np.allclose(first[1], [0.0, 0.0, 4.0])
    assert props["cell_key"] == [encode_key(0, 0, 1), encode_key(0, 1, 1), encode_key(0, 1, 2)]


def test_cells_to_outline_shapes_subsample() -> None:
    cells = _toy_cells()
    shapes, _props, subsampled = gui_layers.cells_to_outline_shapes(
        cells, max_cells=2, seed=1
    )
    assert subsampled
    assert len(shapes) == 2


def test_cells_to_midline_shapes_qt_free() -> None:
    shapes, props, _ = gui_layers.cells_to_midline_shapes(_toy_cells())
    assert len(shapes) == 3  # one branch per cell here
    assert shapes[0].shape == (2, 3)
    assert np.allclose(shapes[0][0], [0.0, 1.0, 2.0])  # (t, y, x)
    assert len(props["cell_key"]) == 3


def test_cells_edges_to_tracks_qt_free() -> None:
    cells = _toy_cells()
    k1, k2, k3 = (encode_key(0, 0, 1), encode_key(0, 1, 1), encode_key(0, 1, 2))
    edges = pl.DataFrame(
        {
            "parent_key": [k1, k2],
            "child_key": [k2, k3],
            "event": ["continue", "division"],
            "confidence": [1.0, 1.0],
        }
    )
    tracks = pl.DataFrame(
        {
            "track_id": [1, 2],
            "fov_id": [0, 0],
            "birth_frame": [0, 1],
            "last_frame": [1, 1],
            "division_frame": [None, None],
            "parent_track": [None, 1],
            "n_frames": [2, 1],
            "fate": ["divide", "disappear"],
        }
    )
    data, props, graph = gui_layers.cells_edges_to_tracks(cells, edges, tracks)
    assert data.shape == (3, 4)  # (N, (track_id, t, y, x))
    # canonical ids matched: track 1 spans frames 0-1, track 2 frame 1
    assert set(data[:, 0].tolist()) == {1.0, 2.0}
    track1 = data[data[:, 0] == 1]
    assert track1[:, 1].tolist() == [0.0, 1.0]
    assert graph == {2: 1}
    assert "track_id" in props and "length_um" in props
    assert props["length_um"].shape == (3,)


def test_cells_edges_to_tracks_uses_track_id_column() -> None:
    cells = _toy_cells().with_columns(pl.Series("track_id", [5, 5, 6]))
    data, _props, graph = gui_layers.cells_edges_to_tracks(cells)
    assert set(data[:, 0].tolist()) == {1.0, 2.0}  # remapped to contiguous ids
    assert data[data[:, 0] == 1].shape[0] == 2
    assert graph == {}


# ---------------------------------------------------------------------------
# DatasetWidget
# ---------------------------------------------------------------------------


def test_dataset_widget_loads_layers(viewer, synth_store) -> None:
    widget = DatasetWidget(viewer)
    widget.open_store(synth_store)
    assert "pixel_microns" in widget.attrs_panel.toPlainText()
    assert widget.fov_combo.count() == 1
    assert widget.channel_list.count() == 1

    image_layer = widget.load_channel(0, "phase")
    assert isinstance(image_layer, napari.layers.Image)
    assert tuple(image_layer.data.shape) == (4, 96, 96)

    labels_layer = widget.load_labels(0)
    assert isinstance(labels_layer, napari.layers.Labels)
    assert tuple(labels_layer.data.shape) == (4, 96, 96)
    assert viewer.dims.ndim == 3  # time axis mapped to the napari slider


def test_dock_widgets_scrollable_vertically(qapp) -> None:
    """Right-dock panels wrap their content in a QScrollArea (small screens)."""
    from qtpy.QtWidgets import QScrollArea

    for widget in (DatasetWidget(), PipelineWidget(), ResultsWidget()):
        scrolls = [
            w for w in widget.findChildren(QScrollArea) if w.parent() is widget
        ]
        assert scrolls, f"{type(widget).__name__} lacks a scroll wrapper"
        assert scrolls[0].widgetResizable()


def test_main_panel_hosts_all_three_tabs(viewer) -> None:
    """MainPanel: Dataset / Pipeline / Results as tabs of one dock."""
    panel = MainPanel(viewer)
    viewer.window.add_dock_widget(panel, name="momia3: Panels", area="right")

    assert panel.count() == 3
    labels = [panel.tabText(i) for i in range(panel.count())]
    assert labels == ["Dataset", "Pipeline", "Results"]
    assert isinstance(panel.widget(0), DatasetWidget)
    assert isinstance(panel.widget(1), PipelineWidget)
    assert isinstance(panel.widget(2), ResultsWidget)


def test_sibling_widget_finds_panels_inside_tab_dock(viewer) -> None:
    """Cross-panel lookups descend into the tabbed MainPanel."""
    from momia3.gui.widgets import _sibling_widget

    panel = MainPanel(viewer)
    viewer.window.add_dock_widget(panel, name="momia3: Panels", area="right")

    found_dataset = _sibling_widget(viewer, DatasetWidget)
    assert found_dataset is panel.dataset_widget
    found_results = _sibling_widget(viewer, ResultsWidget)
    assert found_results is panel.results_widget


def test_import_tiff_runs_off_gui_thread(qapp, tmp_path) -> None:
    """start_import_tiff returns immediately; the store is written in background."""
    import tifffile

    tif = tmp_path / "input.tif"
    arr = np.zeros((2, 1, 96, 96), dtype=np.uint16)
    arr[:, 0, 20:80, 20:80] = 800
    tifffile.imwrite(str(tif), arr, metadata={"axes": "TCYX"})

    widget = DatasetWidget()  # no viewer: only open_store/status assertions
    ws = tmp_path / "ws"
    assert widget.start_import_tiff(tif, 0.065, None, ws) is True
    assert widget._importers_busy  # returned immediately, not blocking
    assert _pump_until(60.0, lambda: not widget._importers_busy)
    store_path = ws / "experiment.zarr"
    assert store_path.is_dir()
    store = ExperimentStore.open(store_path, mode="r")
    assert list(store.channels) == ["ch0"]
    assert list(store.shape(0, "ch0")) == [2, 96, 96]
    assert "imported" in widget.status_label.text()


def test_import_labels_runs_off_gui_thread(qapp, synth_store, tmp_path) -> None:
    """start_import_labels writes the mask into the open store in background."""
    import tifffile

    labels = np.zeros((96, 96), dtype=np.uint32)
    labels[10:40, 10:40] = 1
    mask_tif = tmp_path / "mask.tif"
    tifffile.imwrite(str(mask_tif), labels)

    widget = DatasetWidget()
    widget.open_store(synth_store)
    assert widget.start_import_labels(mask_tif, 0) is True
    assert widget._importers_busy
    assert _pump_until(60.0, lambda: not widget._importers_busy)
    store = ExperimentStore.open(synth_store, mode="r")
    frame = np.asarray(store.get_labels_frame(0, 0))
    assert frame.max() == 1
    assert (frame == 1).sum() == labels.sum()


# ---------------------------------------------------------------------------
# PipelineWidget
# ---------------------------------------------------------------------------


def test_pipeline_widget_build_and_export(viewer, tmp_path) -> None:
    widget = PipelineWidget(viewer)
    op_names = [widget.op_combo.itemText(i) for i in range(widget.op_combo.count())]
    assert "segment.threshold" in op_names and "io.import_synth" in op_names

    widget.select_op("io.import_synth")
    assert widget._form is not None  # form auto-generated from the pydantic model
    step = widget.add_step(step_id="import")
    assert step is not None and step["op"] == "io.import_synth"
    widget.select_op("segment.threshold")
    assert widget.add_step(step_id="segment") is not None
    assert widget.step_list.count() == 2
    item_text = widget.step_list.item(1).text()
    assert item_text.startswith("segment: segment.threshold(")

    out = tmp_path / "pipeline.json"
    assert widget.export_json(out) == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [s["op"] for s in data["steps"]] == ["io.import_synth", "segment.threshold"]
    # auto-wired store input
    assert data["steps"][1]["inputs"] == {"experiment": "import.experiment"}
    assert engine.validate(data) == []


def test_pipeline_widget_param_validation_error_shown(viewer) -> None:
    widget = PipelineWidget(viewer)
    widget.select_op("io.import_synth")
    _set_form_values(widget, shape="not_a_shape")
    assert widget.add_step(step_id="bad") is None
    assert "invalid params" in widget.status_label.text()


def test_pipeline_widget_run_background(viewer, tmp_path) -> None:
    """Full engine run through the thread worker (small synth pipeline)."""
    widget = PipelineWidget(viewer)
    widget.workspace_edit.setText(str(tmp_path / "ws"))
    widget.select_op("io.import_synth")
    _set_form_values(
        widget, n_frames=3, shape="[64, 64]", n_initial=2, noise_sigma=0.02
    )
    assert widget.add_step(step_id="import") is not None
    widget.select_op("segment.threshold")
    _set_form_values(widget, min_size=5)
    assert widget.add_step(step_id="segment") is not None

    assert widget.run_pipeline() is True
    assert _pump_until(120.0, lambda: widget._worker is None)
    assert "done" in widget.status_label.text()
    # default store output path of the import step: <step_id>/<output>
    assert (tmp_path / "ws" / "import" / "experiment").is_dir()
    # resulting labels were auto-loaded into the viewer
    labels = [lyr for lyr in viewer.layers if isinstance(lyr, napari.layers.Labels)]
    assert labels, "expected an auto-loaded Labels layer"


# ---------------------------------------------------------------------------
# ResultsWidget
# ---------------------------------------------------------------------------


def test_results_widget_demo_workspace(viewer) -> None:
    cells = pl.read_parquet(DEMO_WORKSPACE / "tables" / "cells.parquet")
    n_with_outline = cells.filter(pl.col("outline").is_not_null()).height
    n_with_midline = cells.filter(pl.col("midline").is_not_null()).height

    widget = ResultsWidget(viewer)
    widget.open_workspace(DEMO_WORKSPACE)

    assert "momia3 outlines" in viewer.layers
    assert "momia3 midlines" in viewer.layers
    assert "momia3 tracks" in viewer.layers
    outlines = viewer.layers["momia3 outlines"]
    assert len(outlines.data) == n_with_outline
    midlines = viewer.layers["momia3 midlines"]
    assert len(midlines.data) >= n_with_midline  # branched cells add paths

    tracks_layer = viewer.layers["momia3 tracks"]
    n_points = tracks_layer.data.shape[0]
    assert tracks_layer.data.ndim == 2 and tracks_layer.data.shape[1] == 4
    assert n_points == cells.height  # one vertex per cell
    assert widget.tracks_table.rowCount() >= 2

    # clicking a track row isolates it in the Tracks layer
    widget.tracks_table.selectRow(0)
    QApplication.processEvents()
    tid = float(widget.tracks_table.item(0, 0).text())
    # filtering re-creates the layer, so re-fetch it from the viewer
    tracks_layer = viewer.layers["momia3 tracks"]
    assert set(np.unique(tracks_layer.data[:, 0]).tolist()) == {tid}
    widget.show_all_tracks()
    QApplication.processEvents()
    tracks_layer = viewer.layers["momia3 tracks"]
    assert tracks_layer.data.shape[0] == n_points


def test_results_widget_missing_cells(viewer, tmp_path) -> None:
    widget = ResultsWidget(viewer)
    widget.open_workspace(tmp_path)  # no parquets here
    assert "no cells.parquet" in widget.status_label.text()
    assert "momia3 tracks" not in viewer.layers


def test_results_refine_lineage_requires_store_and_tables(viewer) -> None:
    """Refine button is guarded: needs tables + an open Dataset store."""
    widget = ResultsWidget(viewer)
    assert widget.refine_lineage_action() is False
    assert "cells + edges" in widget.status_label.text()

    widget.open_workspace(DEMO_WORKSPACE)  # has tables, but no store panel
    assert widget.refine_lineage_action() is False
    assert "Dataset panel" in widget.status_label.text()


def test_results_refine_lineage_splits_converged_cell(viewer, tmp_path) -> None:
    """Refine button: splits the converged cell, writes labels back, refreshes."""
    from momia3.keys import ROOT_KEY, decode_key, encode_key
    from momia3.ops import measure as m_ops
    from momia3.storage.zarr_store import ExperimentStore

    labels = np.zeros((3, 60, 70), dtype=np.uint32)
    labels[0, 28:33, 10:50] = 1
    labels[1, 28:33, 12:29] = 1
    labels[1, 28:33, 32:49] = 2
    labels[2, 28:33, 12:49] = 3
    ka = encode_key(0, 0, 1)
    kb = encode_key(0, 1, 1)
    kc = encode_key(0, 1, 2)
    kd = encode_key(0, 2, 3)
    edges = pl.DataFrame(
        {
            "parent_key": [ROOT_KEY, ka, ka, kb, kc],
            "child_key": [ka, kb, kc, kd, kd],
            "event": ["appear", "division", "division", "continue", "continue"],
            "confidence": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    cells = pl.concat(
        [
            pl.from_arrow(
                m_ops.labels_to_cells(labels[t], 0, t, pixel_microns=0.065)
            )
            for t in range(3)
        ],
        how="diagonal_relaxed",
    )

    store_path = tmp_path / "experiment.zarr"
    store = ExperimentStore.create(
        store_path, pixel_microns=0.065, channels=["phase"], time_interval_s=60.0
    )
    store.write_labels(0, labels)

    dataset = DatasetWidget(viewer)
    results = ResultsWidget(viewer)
    viewer.window.add_dock_widget(dataset, name="momia3: Dataset", area="right")
    viewer.window.add_dock_widget(results, name="momia3: Results", area="right")
    dataset.open_store(store_path)
    results.open_dataframes(cells=cells, edges=edges)

    assert results.refine_lineage_action() is True
    assert _pump_until(60.0, lambda: not results._refine_busy)
    assert "refined" in results.status_label.text()

    # Edges rewired: each mother keeps one fresh-fragment child, not kd.
    pairs = {
        (int(r["parent_key"]), int(r["child_key"])): r["event"]
        for r in results._edges.to_dicts()
    }
    assert (kb, kd) not in pairs and (kc, kd) not in pairs
    b_children = [c for (p, c) in pairs if p == kb]
    c_children = [c for (p, c) in pairs if p == kc]
    assert len(b_children) == 1 and len(c_children) == 1
    assert b_children[0] != c_children[0]
    frag_labels = {decode_key(k)[2] for k in b_children + c_children}

    # Split fragments were written back into the store's labels (frame 2).
    reopened = ExperimentStore.open(store_path, mode="r")
    frame2 = np.asarray(reopened.get_labels_frame(0, 2))
    assert (frame2 == 3).sum() == 0
    assert set(np.unique(frame2[frame2 > 3]).tolist()) == frag_labels

    # The in-memory cells table gained rows for the fresh fragments.
    assert results._cells is not None
    frag_rows = results._cells.filter(
        (pl.col("fov_id") == 0) & (pl.col("frame") == 2) & (pl.col("label") > 3)
    )
    assert frag_rows.height == 2


# ---------------------------------------------------------------------------
# manual-edit hook (DatasetWidget -> ResultsWidget)
# ---------------------------------------------------------------------------


def test_recompute_current_frame_after_labels_edit(viewer, synth_store) -> None:
    dataset = DatasetWidget(viewer)
    results = ResultsWidget(viewer)
    viewer.window.add_dock_widget(dataset, name="momia3: Dataset", area="right")
    viewer.window.add_dock_widget(results, name="momia3: Results", area="right")

    dataset.open_store(synth_store)
    dataset.load_channel(0, "phase")
    dataset.load_labels(0)

    # simulate a manual edit on frame 1: erase the largest label
    frame1 = np.asarray(dataset._labels_layer.data[1])
    victim = int(frame1.max())
    assert victim > 0
    edited = np.asarray(dataset._labels_layer.data).copy()
    edited[1][edited[1] == victim] = 0
    dataset._labels_layer.data = edited
    expected_cells = int(np.unique(edited[1]).size) - 1  # minus background

    viewer.dims.set_current_step(0, 1)
    cells = dataset.recompute_current_frame()
    assert cells is not None

    # store was updated with the edited frame
    assert int(store_frame_max(synth_store)) <= victim
    # results widget refreshed with the recomputed frame
    assert results._cells is not None
    frame1_rows = results._cells.filter(
        (pl.col("fov_id") == 0) & (pl.col("frame") == 1)
    )
    assert frame1_rows.height == expected_cells
    assert victim not in set(frame1_rows["label"].to_list())
    # geometry was recomputed too (outlines/midlines present as layers)
    assert "momia3 outlines" in viewer.layers
    assert "momia3 midlines" in viewer.layers
    # recomputed rows carry geometry columns
    assert frame1_rows["outline"].is_not_null().all()


def store_frame_max(store_path: Path) -> int:
    store = ExperimentStore.open(store_path, mode="r")
    return int(np.asarray(store.get_labels_frame(0, 1)).max())


def test_pipeline_preview_step_on_current_frame(viewer, synth_store) -> None:
    """Preview button: sandbox run on current frame, source store untouched."""
    dataset = DatasetWidget(viewer)
    pipeline = PipelineWidget(viewer)
    viewer.window.add_dock_widget(dataset, name="momia3: Dataset", area="right")
    viewer.window.add_dock_widget(pipeline, name="momia3: Pipeline", area="right")

    dataset.open_store(synth_store)
    dataset.load_channel(0, "phase")
    labels_before = np.asarray(
        ExperimentStore.open(synth_store, mode="r").labels(0)
    ).copy()

    # profiling preview: outline.extract (implicit measure.cells on labels)
    pipeline.select_op("outline.extract")
    assert pipeline.add_step("ol") is not None
    pipeline.step_list.setCurrentRow(0)
    viewer.dims.set_current_step(0, 0)
    assert pipeline.preview_selected_step()
    assert "preview: outlines" in viewer.layers

    # segmentation preview: segment.threshold into the sandbox only
    pipeline.select_op("segment.threshold")
    pipeline._form  # form built by select_op
    assert pipeline.add_step("seg") is not None
    pipeline.step_list.setCurrentRow(1)
    assert pipeline.preview_selected_step()
    assert "preview: labels" in viewer.layers
    labels_after = np.asarray(
        ExperimentStore.open(synth_store, mode="r").labels(0)
    )
    np.testing.assert_array_equal(labels_before, labels_after)

    # re-preview replaces rather than accumulates preview layers
    assert pipeline.preview_selected_step()
    names = [layer.name for layer in viewer.layers]
    assert "preview: labels" in names  # current step's output stays
    assert "preview: outlines" not in names  # previous step's layers cleared
    assert "preview" in pipeline.status_label.text()
