import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QTabWidget

from momia3.gui.plugins import API_VERSION, PluginRegistry, RoiTool
from momia3.gui.shell import MainShell


def test_builtin_plugin_loads():
    registry = PluginRegistry()
    registry.load(include_entry_points=False)
    assert registry.errors == []
    specs = registry.specs()
    assert len(specs) == 1
    assert specs[0].api_version == API_VERSION
    assert any(isinstance(item, RoiTool) for item in specs[0].contributions)


def test_shell_installs_plugin_tab():
    app = QApplication.instance() or QApplication([])
    shell = MainShell()
    tabs = shell.findChild(QTabWidget, "InspectorTabs")
    assert tabs is not None
    assert tabs.count() >= 4
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Plugin demo" in titles
    shell.close()


def test_plugin_roi_tool_uses_roi_session():
    app = QApplication.instance() or QApplication([])
    shell = MainShell()
    spec = shell._plugins.specs()[0]
    tool = next(item for item in spec.contributions if isinstance(item, RoiTool))
    before = len(shell._ensure_roi_session().current)
    tool.callback(shell._plugin_context)
    assert len(shell._ensure_roi_session().current) == before + 1
    shell.close()
