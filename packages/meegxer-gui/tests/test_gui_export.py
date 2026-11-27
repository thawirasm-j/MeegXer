"""*Save → Export as Python script…*: the chain as a script that runs it as
MeegXer does (meegxer.api.export_script)."""

import subprocess
from pathlib import Path

from PySide6.QtWidgets import QApplication


def test_the_chain_exports_as_a_script(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        target = tmp_path / "mine"
        win.export_script(str(target))
        text = (tmp_path / "mine.py").read_text()
        assert "from meegxer.api import run" in text
        for strip in win.doc.strips:
            assert strip.node in text
        assert "Exported" in win.logbox.toPlainText()
    finally:
        win.close()
