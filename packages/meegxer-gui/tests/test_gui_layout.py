"""The window fits a laptop screen (1024 px wide and up): the session
commands sit on Save's arrow, the run's details shorten with "…" instead of
widening the window, and the signal view keeps its height first."""

from __future__ import annotations

import subprocess
import time

from PySide6.QtWidgets import QApplication


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def test_the_window_fits_a_laptop_screen(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        assert win.minimumSizeHint().width() <= 1024
        menu = win.save_btn.menu()
        assert menu is not None
        assert [a.text().split("\t")[0] for a in menu.actions()] == [
            "Save pipeline…", "Save session…", "Share session…"]  # fmt: skip
        win.start()
        pump(qapp, 3.0)
        win.resize(1024, 700)
        pump(qapp, 0.5)
        assert win.width() == 1024  # it does not grow past the screen
        assert win.state_label.text() == "● running"  # always whole
        full = win.meter_details.full_text()
        assert full.startswith("processing load")
        assert win.meter_details.toolTip() == full
        shown = win.meter_details.text()
        assert shown in ("", full) or shown.endswith("…")  # "": no room at all
        win.resize(1600, 900)
        pump(qapp, 0.5)
        assert win.meter_details.text() == win.meter_details.full_text()
        assert win.views.height() >= 200  # the view keeps its height
        assert win.inspector.width() > win.logbox.width()  # settings first
        win.stop()
        pump(qapp, 1.0)
    finally:
        win.close()
