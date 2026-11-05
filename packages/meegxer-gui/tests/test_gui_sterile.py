"""The Sterile switch on the top bar (docs/STERILE.md): it is the
general.sterile setting; switched on with no image (or one built from other
code), the GUI offers to build it, follows the build in the log, and does not
start the run until it is built."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from meegxer.core import settings


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_the_switch_offers_to_build_then_runs(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    missing = (
        "the Sterile image meegxer-sterile:0.5.0 is not built yet: run "
        "`meegxer sterile build`"
    )
    status: dict[str, Any] = {"enabled": True, "problem": "", "build": "idle",
                              "lines": [], "error": "", "image": missing}  # fmt: skip
    calls: list[str] = []
    real = win.call

    def call(method: str, **kw: Any) -> Any:
        calls.append(method)
        if method == "sterile_status":
            return dict(status)
        if method == "sterile_build":
            status.update(build="running", lines=["Building…", "step 1/9"])
            return dict(status)
        return real(method, **kw)

    asked: list[str] = []

    def question(parent: Any, title: str, text: str, *a: Any) -> Any:
        asked.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    try:
        win.call = call  # type: ignore[method-assign]
        assert not win.sterile_btn.isChecked()  # off by default
        win.sterile_btn.setChecked(True)
        assert settings.load_settings()["general"]["sterile"] is True  # saved
        # switched on: the image is checked at once, and its build offered
        assert asked and "is not built yet" in asked[0]
        assert "needs the network once" in asked[0]
        assert calls[-1] == "sterile_build"
        pump(qapp, 1.3)
        log = win.logbox.toPlainText()
        assert "Building the Sterile image…" in log and "step 1/9" in log
        # a run asked for meanwhile waits for the build
        assert win.sterile_ready() is False
        status.update(build="done", image="ready",
                      lines=["Building…", "step 1/9", "built"])  # fmt: skip
        pump(qapp, 1.3)
        log = win.logbox.toPlainText()
        assert "built" in log and "The Sterile image is built" in log
        assert win.sterile_ready() is True
        win.sterile_btn.setChecked(False)
        assert settings.load_settings()["general"]["sterile"] is False
        assert "Sterile mode off" in win.logbox.toPlainText()
    finally:
        win.close()


def test_the_daemon_started_local_says_so(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    """The switch on, a daemon started with --no-sterile: the run is local,
    and the log says why."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.call = lambda method, **kw: {"enabled": False, "problem": "",  # type: ignore[method-assign]
                                         "image": "ready", "build": "idle"}  # fmt: skip
        win.sterile_btn.blockSignals(True)
        win.sterile_btn.setChecked(True)
        win.sterile_btn.blockSignals(False)
        assert win.sterile_ready() is True
        assert "started with --no-sterile" in win.logbox.toPlainText()
    finally:
        win.close()
