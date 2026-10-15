"""Clicking a strip selects it, which rebuilds the chain while Qt is still
delivering the click to the old strip widget: the old widgets must be deleted
later, not on the spot."""

import subprocess
import time

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QWidget


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


def _wheel(widget: QWidget, steps: int = 1) -> None:
    pos = QPointF(widget.rect().center())
    event = QWheelEvent(
        pos, widget.mapToGlobal(pos), QPoint(), QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )  # fmt: skip
    QApplication.sendEvent(widget, event)


def test_wheel_changes_a_value_only_after_it_is_clicked(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            qapp.processEvents()
        win.select("iirfilter1")
        qapp.processEvents()
        spin = next(w for w in win.inspector.findChildren(QDoubleSpinBox)
                    if w.isVisible() and w.isEnabled())  # fmt: skip
        before = spin.value()
        _wheel(spin)  # pointer passes over the field while scrolling the panel
        assert spin.value() == before
        spin.setFocus()
        qapp.processEvents()
        if not spin.hasFocus():  # offscreen windows may not take focus
            win.activateWindow()
            spin.setFocus()
            qapp.processEvents()
        assert spin.hasFocus()
        _wheel(spin)
        assert spin.value() != before
    finally:
        win.close()


def test_no_widget_has_a_pixel_sized_font(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    """A font-size in px leaves pointSize() at -1, and Qt styles that copy the
    point size warn "QFont::setPointSize: Point size <= 0 (-1)" (seen on Windows)."""
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        qapp.processEvents()
        bad = [w for w in win.findChildren(QWidget) if w.font().pointSize() <= 0]
        assert not bad, [(type(w).__name__, w.styleSheet()) for w in bad]
    finally:
        win.close()
