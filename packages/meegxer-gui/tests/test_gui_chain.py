"""Clicking a strip selects it, which rebuilds the chain while Qt is still
delivering the click to the old strip widget: the old widgets must be deleted
later, not on the spot."""

import subprocess
import time

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def test_clicking_a_strip_selects_it(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.chain import StripWidget
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            qapp.processEvents()
        strip = next(w for w in win.findChildren(StripWidget)
                     if w.strip_id == "iirfilter1")  # fmt: skip
        # before the fix: "Internal C++ object (StripWidget) already deleted"
        QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(20, 60))
        qapp.processEvents()
        assert win.selected == "iirfilter1"
    finally:
        win.close()
