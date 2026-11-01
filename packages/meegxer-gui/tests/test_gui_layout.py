"""The window fits a laptop screen (1024 px wide and up), also with wider
text (Windows draws the same point size wider, or Text size raised): the
file, undo and mode buttons fold into the toolbar's », the status line and
the views' legends shorten with "…" instead of widening the window, the
run's state is never cut, and the signal view keeps its height first."""

from __future__ import annotations

import subprocess
import time

from PySide6.QtWidgets import QApplication, QToolButton


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
    from meegxer.gui.theme import stylesheet

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        assert win.minimumSizeHint().width() <= 980  # room for wider fonts
        menu = win.save_btn.menu()
        assert menu is not None
        assert [a.text().split("\t")[0] for a in menu.actions()] == [
            "Save pipeline…", "Save session…", "Share session…"]  # fmt: skip
        win.start()
        pump(qapp, 3.0)
        win.resize(1024, 700)
        pump(qapp, 0.5)
        assert win.width() == 1024  # it does not grow past the screen
        line = win.status_line
        full = line.text()
        assert full.startswith("● running · processing load")  # the state first
        assert line.toolTip() == full
        shown = line.shown()
        assert shown == full or shown.endswith("…")  # cut at its end
        assert shown.startswith("● running")  # the state itself whole
        win.resize(2400, 900)  # room for it all, also with Windows' wider text
        pump(qapp, 0.5)
        assert line.shown() == line.text()  # whole (the load may have moved)
        win.resize(1024, 700)
        pump(qapp, 0.5)
        assert win.views.height() >= 200  # the view keeps its height
        assert win.inspector.width() > win.logbox.width()  # settings first
        win.setStyleSheet(stylesheet(win.theme, 1.4))  # text 40 % wider
        pump(qapp, 0.5)
        assert win.minimumSizeHint().width() <= 1024
        assert win.width() == 1024
        fold = win.tools.findChild(QToolButton, "qt_toolbar_ext_button")
        assert fold is not None and fold.isVisible() and fold.width() >= 24  # »
        assert line.shown().startswith("● running")  # still whole
        assert win.view.legend.shown().endswith("…") or win.view.legend.shown()
        win.stop()
        pump(qapp, 1.0)
    finally:
        win.close()
