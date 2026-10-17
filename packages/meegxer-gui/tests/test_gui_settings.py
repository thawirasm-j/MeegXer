"""Settings: colour maps, the dialog, and settings + remembered state in the
main window."""

import json
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from meegxer.core import settings as core_settings
from meegxer.gui.colormaps import DIVERGING, SEQUENTIAL, Maps, hex_rgb
from meegxer.gui.settings_dialog import SettingsDialog
from meegxer.gui.theme import load_theme

DARK, LIGHT = load_theme("dark"), load_theme("light")


def dispose(qapp: QApplication, *widgets: QWidget) -> None:
    for w in widgets:
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


# -- colour maps --------------------------------------------------------------------


def test_every_map_gives_colours_from_low_to_high() -> None:
    s = np.linspace(0, 1, 5)
    for name in SEQUENTIAL:
        rgb = Maps(sequential=name).sequential_rgb(s, DARK)
        assert rgb.shape == (5, 3) and rgb.dtype == np.uint8
        assert not np.array_equal(rgb[0], rgb[-1])
    viridis = Maps().sequential_rgb([0.0, 1.0], DARK)
    assert viridis.tolist() == [[68, 1, 84], [253, 231, 37]]  # viridis' own ends


def test_maps_stay_fixed_unless_flipped_on_a_light_theme() -> None:
    s = np.array([0.0, 1.0])
    fixed = Maps()
    assert np.array_equal(fixed.sequential_rgb(s, DARK), fixed.sequential_rgb(s, LIGHT))
    flipped = Maps(flip_on_light=True)
    assert np.array_equal(
        flipped.sequential_rgb(s, DARK), fixed.sequential_rgb(s, DARK)
    )
    assert np.array_equal(flipped.sequential_rgb(s, LIGHT),
                          fixed.sequential_rgb(s[::-1], LIGHT))  # fmt: skip
    assert np.array_equal(flipped.peak_rgb(s, LIGHT), fixed.peak_rgb(s[::-1], LIGHT))


def test_diverging_maps_run_blue_to_red_through_their_centre() -> None:
    centres = {"blue-white-red": "white", "blue-black-red": "black", "meegxer": "bg"}
    assert set(centres) == set(DIVERGING)
    for name, centre in centres.items():
        neg, mid, pos = Maps(diverging=name).diverging_rgb([-1.0, 0.0, 1.0], DARK)
        assert neg[2] > neg[0] and pos[0] > pos[2]  # blue, then red
        if centre == "white":
            assert mid.min() > 200
        elif centre == "black":
            assert mid.max() < 60
        else:
            assert np.abs(mid.astype(int) - hex_rgb(DARK.plot_background)).max() <= 2
    # never flipped: that would swap what the colours mean
    flip = Maps(flip_on_light=True)
    assert np.array_equal(flip.diverging_rgb([-1.0], LIGHT),
                          Maps().diverging_rgb([-1.0], LIGHT))  # fmt: skip


def test_the_head_plot_paints_with_the_chosen_map() -> None:
    from meegxer.gui.headplot import HeadInput, HeadMap, render

    labels = ("Fz", "Cz", "Pz", "O1", "O2")
    f = np.arange(0, 60.5, 0.5)
    psd = 1 + np.array([[0], [5], [20], [50], [50]]) * np.exp(
        -0.5 * ((f - 10) / 0.8) ** 2
    )
    data = HeadInput(labels=labels, f=f, psd=psd, band=(8.0, 13.0))
    for name in ("viridis", "magma"):
        m = render(data, DARK, maps=Maps(sequential=name))
        assert isinstance(m, HeadMap)
        n = len(m.scale[2])
        expected = Maps(sequential=name).sequential_rgb((np.arange(n) + 0.5) / n, DARK)
        assert m.scale[2] == [tuple(int(c) for c in rgb) for rgb in expected]


# -- the dialog ---------------------------------------------------------------------


class FakePlugins:
    def __init__(self) -> None:
        self.entries = [
            {"name": "script.notch", "version": "0.1.0", "where": "python -m x",
             "unsandboxed": True, "description": "Notch"},
            {"name": "meegxer.eeglab.asr", "version": "1.0.0", "where": "img:1",
             "unsandboxed": False, "description": "ASR"},
        ]  # fmt: skip
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> tuple[int, str, str]:
        self.calls.append(args)
        if args[0] == "remove":
            self.entries = [e for e in self.entries if e["name"] != args[1]]
            return 0, f"removed {args[1]}", ""
        return 0, json.dumps(self.entries), ""


def dialog(qapp: QApplication, **changes: Any) -> tuple[SettingsDialog, FakePlugins]:
    current = core_settings.defaults()
    for key, value in changes.items():
        core_settings.put(current, key.replace("__", "."), value)
    plugins = FakePlugins()
    return SettingsDialog(current, DARK, plugins=plugins), plugins


def test_changes_preview_at_once_and_cancel_goes_back(qapp: QApplication) -> None:
    d, _ = dialog(qapp)
    seen: list[dict[str, Any]] = []
    d.changed.connect(seen.append)
    assert d.pages.count() == 5
    from PySide6.QtWidgets import QComboBox

    seq = next(c for c in d.findChildren(QComboBox)
               if c.findData("cividis") >= 0 and "heat" in c.toolTip())  # fmt: skip
    seq.setCurrentIndex(seq.findData("cividis"))
    assert seen[-1]["visualization"]["sequential"] == "cividis"
    assert d.settings()["visualization"]["sequential"] == "cividis"
    d.reject()
    assert seen[-1]["visualization"]["sequential"] == "viridis"  # back as it was
    dispose(qapp, d)


def test_reset_restores_the_defaults_and_settings_stay_valid(
    qapp: QApplication,
) -> None:
    d, _ = dialog(qapp, appearance__scale=150, general__advanced=True,
                  visualization__spectrum__band=[4.0, 8.0])  # fmt: skip
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox

    advanced = next(c for c in d.findChildren(QCheckBox) if "advanced" in c.text())
    assert advanced.isChecked()
    spins = [s for s in d.findChildren(QDoubleSpinBox) if s.suffix() == " Hz"]
    assert [s.value() for s in spins] == [4.0, 8.0]
    d.reset_defaults()
    assert d.settings() == core_settings.defaults()
    assert not advanced.isChecked() and [s.value() for s in spins] == [8.0, 13.0]
    spins[1].setValue(20.0)
    spins[1].editingFinished.emit()
    assert d.settings()["visualization"]["spectrum"]["band"] == [8.0, 20.0]
    jsonschema.validate(d.settings(), core_settings.schema())  # savable
    dispose(qapp, d)


def test_the_plugins_page_lists_and_removes(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    d, plugins = dialog(qapp)
    rows = [d.plugin_list.item(i).text() for i in range(d.plugin_list.count())]
    assert rows == ["script.notch  0.1.0  (local, unsandboxed)",
                    "meegxer.eeglab.asr  1.0.0  (container)"]  # fmt: skip
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)  # fmt: skip
    d.plugin_list.setCurrentRow(0)
    d._remove_plugin()
    assert ["remove", "script.notch"] in plugins.calls
    assert d.plugin_list.count() == 1
    asked: list[bool] = []
    d.add_script_requested.connect(lambda: asked.append(True))
    from PySide6.QtWidgets import QPushButton

    add = next(b for b in d.findChildren(QPushButton) if b.text().startswith("Add a"))
    add.click()
    assert asked == [True]
    dispose(qapp, d)


def test_the_plugin_list_comes_from_the_plugin_command(tmp_path: Path) -> None:
    from meegxer.gui.settings_dialog import plugin_cli

    code, out, err = plugin_cli(["list", "--json"])
    assert code == 0, err
    assert isinstance(json.loads(out), list)


# -- the main window ----------------------------------------------------------------


def _window(control: str, **kw: Any) -> Any:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    return MainWindow(DaemonLink(control), **kw)


def test_the_window_starts_from_settings_then_remembered_state(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    control, _ = daemon
    s = core_settings.defaults()
    core_settings.put(s, "general.advanced", True)
    core_settings.put(s, "general.start_mode", "offline")
    core_settings.put(s, "visualization.span", 20)
    core_settings.put(s, "visualization.sequential", "magma")
    core_settings.put(s, "visualization.spectrum.scheme", "peak")
    win = _window(control, settings=s, state={})
    try:
        assert win.advanced and win.doc.mode == "offline"
        assert win.view.seconds() == 20.0
        assert win.view.scheme.currentData() == "peak"
        assert win.view.maps.sequential == "magma"
    finally:
        dispose(qapp, win)
    remembered = {"view": {"span": 5, "spectrum": {"band": [4.0, 8.0]}},
                  "advanced": False}  # fmt: skip
    core_settings.put(s, "appearance.theme", "light")
    win = _window(control, settings=s, state=remembered)
    try:
        assert win.view.seconds() == 5.0 and win.view.band_range() == (4.0, 8.0)
        assert not win.advanced and not win.theme.dark
        assert win.view.scheme.currentData() == "peak"  # not remembered: default
    finally:
        dispose(qapp, win)


def test_closing_remembers_the_view_and_settings_replace_it(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    own_config: Path,
) -> None:
    control, _ = daemon
    win = _window(control)
    try:
        win.view.span.setCurrentIndex(win.view.span.findData(30.0))
        win.toggle_theme()  # the quick switch is a setting
        saved = json.loads((own_config / "settings.json").read_text())
        assert saved["appearance"]["theme"] == "light"
        win.close()
        state = json.loads((own_config / "state.json").read_text())
        assert state["view"]["span"] == 30 and "theme" not in state
        assert isinstance(state["geometry"], str)
    finally:
        dispose(qapp, win)
    win = _window(control)
    try:
        assert win.view.seconds() == 30.0 and not win.theme.dark
        new = core_settings.defaults()
        core_settings.put(new, "visualization.span", 2)
        core_settings.put(new, "appearance.theme", "dark")
        win.accept_settings(new)  # a changed default wins over what was remembered
        assert win.view.seconds() == 2.0 and win.theme.dark
        assert "span" not in win.state["view"]
        saved = json.loads((own_config / "settings.json").read_text())
        assert saved["visualization"]["span"] == 2
        win.view.span.setCurrentIndex(win.view.span.findData(20.0))
        win.forget_view()
        assert win.state == {} and not (own_config / "state.json").exists()
        assert win.view.seconds() == 2.0  # the setting's default again
    finally:
        dispose(qapp, win)


def test_the_last_pipeline_is_reopened_and_dialogs_start_there(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PySide6.QtWidgets import QFileDialog

    control, _ = daemon
    pipeline = {"spec_version": "0.2.9", "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic", "params": {}},
        {"id": "notch", "node": "meegxer.nodes.notch", "params": {}},
    ]}  # fmt: skip
    path = tmp_path / "mine" / "p.json"
    path.parent.mkdir()
    path.write_text(json.dumps(pipeline))
    win = _window(control, state={"last_pipeline": str(path),
                                  "last_dir": str(path.parent)})  # fmt: skip
    try:
        assert [s.node for s in win.doc.strips][-1] == "meegxer.nodes.notch"
        assert "Reopened" in win.logbox.toPlainText()
        starts: list[str] = []

        def pick(_w: object, _t: str, start: str, *_: object) -> tuple[str, str]:
            starts.append(start)
            return "", ""

        monkeypatch.setattr(QFileDialog, "getOpenFileName", pick)
        win.open_pipeline()
        assert starts == [str(path.parent)]
    finally:
        dispose(qapp, win)


def test_closing_while_running_asks_first(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _ = daemon
    win = _window(control)
    try:
        win.settings["general"]["confirm_quit_running"] = True
        win.running = True
        monkeypatch.setattr(win, "stop", lambda: setattr(win, "running", False))
        monkeypatch.setattr(win, "_confirm_close", lambda: "cancel")
        win.close()
        assert win.isVisible() or win.running  # cancelled: still open, still running
        monkeypatch.setattr(win, "_confirm_close", lambda: "stop")
        win.close()
        assert not win.running
    finally:
        dispose(qapp, win)
