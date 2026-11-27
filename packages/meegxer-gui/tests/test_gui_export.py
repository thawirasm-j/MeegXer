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


def test_the_chain_exports_as_standalone_mne_code(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        source = win.doc.source
        assert source is not None
        for strip in list(win.doc.strips):
            if strip.id != source.id:
                win.doc.remove(strip.id)
        win.doc.insert(0, "meegxer.nodes.iir_filter")
        win.export_script(str(tmp_path / "mine"), mne=True)
        text = (tmp_path / "mine.py").read_text()
        assert "signal.butter(" in text and "from meegxer" not in text
        win.doc.insert(1, "meegxer.nodes.eog_regression")
        win.export_script(str(tmp_path / "eogreg"), mne=True)
        assert not (tmp_path / "eogreg.py").exists()
        log = win.logbox.toPlainText()
        assert "no standalone MNE code yet:" in log and "eog_regression)" in log
    finally:
        win.close()
