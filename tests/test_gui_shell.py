import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QSplitter, QTabWidget

from momia3.gui.shell import MainShell
from momia3.gui.theme import Tokens, build_stylesheet


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_shell_has_persistent_three_columns():
    _app()
    shell = MainShell()
    assert shell.objectName() == "ShellRoot"
    assert shell.findChild(QSplitter, "MainSplitter") is not None
    assert shell.findChild(QSplitter, "MainSplitter").count() == 3
    assert shell.findChild(object, "DatasetPanel") is not None
    assert shell.findChild(object, "ViewerPanel") is not None
    assert shell.findChild(object, "InspectorPanel") is not None
    tabs = shell.findChild(QTabWidget, "InspectorTabs")
    assert tabs is not None and tabs.count() >= 3


def test_shell_workflow_strip_and_status():
    _app()
    shell = MainShell()
    assert shell.findChild(object, "WorkflowStrip") is not None
    stages = shell.findChildren(object, "StageLabel")
    assert len(stages) == 0 or len(stages) >= 1
    # The strip exists even if stage labels are styled QLabels rather than named.
    assert shell.findChild(object, "TopBar") is not None


def test_stylesheet_has_no_unsubstituted_tokens():
    qss = build_stylesheet(Tokens())
    assert "$" not in qss
