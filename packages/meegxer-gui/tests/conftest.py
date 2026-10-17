import os
import socket
import subprocess
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def free_endpoint() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


@pytest.fixture(autouse=True)
def own_config(tmp_path, monkeypatch):
    """Each test (and the daemon it starts) gets empty settings and state, so a
    window never reopens what another test left behind."""
    folder = tmp_path / "config"
    monkeypatch.setenv("MEEGXER_CONFIG_DIR", str(folder))
    return folder


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def daemon() -> Iterator[tuple[str, subprocess.Popen[bytes]]]:
    from meegxer.gui.connection import spawn_daemon

    control = free_endpoint()
    proc = spawn_daemon(control, free_endpoint())
    yield control, proc
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
