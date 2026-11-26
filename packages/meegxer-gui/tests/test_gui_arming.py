"""Arming the keyboard from the GUI (red team L3, docs/SECURITY.md): the log
says what it presses; a map that can type text asks first, and only a yes
arms it (the daemon refuses it otherwise)."""

import subprocess
import time
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


def test_a_map_that_types_text_asks_before_arming(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    asked: list[str] = []
    answer = [QMessageBox.StandardButton.No]

    def question(parent: Any, title: str, text: str, *a: Any) -> Any:
        asked.append(text)
        return answer[0]

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.show()
        if win.doc.mode != "live":
            win.toggle_mode()
        win.change_source("meegxer.core.synthetic")
        strip = win.doc.insert(len(win.doc.strips), "meegxer.nodes.keyboard")
        win.change_param(strip.id, "keys", "a=r, b=m, c=x, go=enter")
        win.start()
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            qapp.processEvents()
        win.arm(strip.id, True)
        log = win.logbox.toPlainText()
        assert asked and "can type text" in asked[0] and "presses a → r" in asked[0]
        assert "presses a → r, b → m, c → x, go → enter" in log
        assert log.rstrip().endswith("not armed.")
        answer[0] = QMessageBox.StandardButton.Yes
        win.arm(strip.id, True)  # confirmed: the daemon no longer refuses it
        assert "confirm to arm it" not in win.logbox.toPlainText()
        win.stop()
    finally:
        win.close()
