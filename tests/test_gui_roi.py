import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from momia3.gui.roi_manager import ROI, ROIManagerWindow, ROISession, ROISet, demo_roi_session


def _roi(name: str, x: int) -> ROI:
    return ROI(
        roi_id=name,
        name=name,
        kind="box",
        geometry={"x": x, "y": 4, "w": 8, "h": 6},
        cell_key=1,
        track_id=1,
        frame=0,
    )


def test_action_undo_is_limited_to_fifty_and_has_no_redo():
    session = ROISession()
    for index in range(60):
        session.add_roi(_roi(f"r{index}", index))
    assert len(session.actions) == 50
    assert len(session.current) == 60
    for _ in range(50):
        assert session.undo()
    assert not session.undo()
    assert len(session.current) == 10


def test_update_keeps_one_previous_set_and_rollback():
    session = ROISession(ROISet({"a": _roi("a", 1)}))
    session.add_roi(_roi("b", 2))
    assert session.update()
    assert session.previous_roi_set is not None
    assert len(session.base) == 2
    assert session.undo() is False
    assert session.rollback_update()
    assert len(session.current) == 1
    assert session.previous_roi_set is None
    assert session.rollback_update() is False


def test_roi_manager_window_renders_demo_set():
    app = QApplication.instance() or QApplication([])
    window = ROIManagerWindow(demo_roi_session())
    assert window.table.rowCount() == 5
    assert not hasattr(window, "record_table")
    window.delete_button.click()
    assert window.undo_button.isEnabled() is False
    window.close()


def test_label_edits_selected_roi_and_defaults():
    app = QApplication.instance() or QApplication([])
    session = demo_roi_session()
    window = ROIManagerWindow(session)
    assert window.label_combo.currentText() == "default"
    window.table.selectRow(0)
    window.label_combo.setEditText("T cell")
    window._apply_current_label()
    first = next(iter(session.current.rois.values()))
    assert first.label == "T cell"
    assert "T cell" in [window.label_combo.itemText(i) for i in range(window.label_combo.count())]
    window.close()
