"""Adding a custom script from the GUI: notice first, then a strip."""

import socket
import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

SCRIPT = '''
    """Scale the signal (a custom script)."""


    def process(raw, gain: float = 2.0):
        """Multiply.

        Parameters
        ----------
        gain : float
            Factor.
        """
        return raw.get_data() * gain
'''


@pytest.fixture
def isolated_daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A daemon and plugin index of its own (the GUI's helper process and the
    daemon must see the same index)."""
    from meegxer.gui.connection import spawn_daemon

    def free_endpoint() -> str:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return f"tcp://127.0.0.1:{s.getsockname()[1]}"

    monkeypatch.setenv("MEEGXER_PLUGIN_INDEX", str(tmp_path / "plugins.json"))
    monkeypatch.delenv("MEEGXER_ALLOW_UNSANDBOXED", raising=False)
    control = free_endpoint()
    proc = spawn_daemon(control, free_endpoint())
    yield control
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_a_custom_script_is_added_only_after_the_notice(
    qapp: QApplication, isolated_daemon: str, tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    script = tmp_path / "scale.py"
    script.write_text(textwrap.dedent(SCRIPT))
    win = MainWindow(DaemonLink(isolated_daemon))
    try:
        count = len(win.doc.strips)
        win._accept_local_code = lambda what: False  # type: ignore[method-assign]
        win.add_script(1, str(script))
        assert "script.scale" not in win.doc.manifests
        assert len(win.doc.strips) == count
        assert not (tmp_path / "plugins.json").exists()

        win._accept_local_code = lambda what: True  # type: ignore[method-assign]
        win.add_script(1, str(script))
        assert "script.scale" in win.doc.manifests  # the daemon found it
        assert win.doc.strips[1].node == "script.scale"
        assert "unsandboxed" in win.logbox.toPlainText()
    finally:
        win.close()
        for w in (win.stacked, win.recording_window, win):
            w.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
