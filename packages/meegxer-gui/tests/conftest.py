import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

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
    folder.mkdir()
    # no "close while running?" question: a modal box would wait forever offscreen
    (folder / "settings.json").write_text(
        '{"general": {"confirm_quit_running": false}}', encoding="utf-8"
    )
    monkeypatch.setenv("MEEGXER_CONFIG_DIR", str(folder))
    return folder


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def spawn_slow_daemon(control: str, display: str) -> subprocess.Popen[bytes]:
    """The daemon of ``slow_daemon.py``: each finished offline run's first
    window read answers after the GUI has stopped waiting."""
    from meegxer.gui.connection import daemon_running

    script = Path(__file__).with_name("slow_daemon.py")
    proc = subprocess.Popen([sys.executable, str(script), "--control", control,
                             "--display", display],
                            stdout=subprocess.DEVNULL)  # fmt: skip
    deadline = time.monotonic() + 15
    while not daemon_running(control):
        assert proc.poll() is None, f"the slow daemon exited ({proc.returncode})"
        assert time.monotonic() < deadline, "the slow daemon did not answer"
    return proc


@pytest.fixture
def daemon(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[str, subprocess.Popen[bytes]]]:
    """A daemon of its own; ``"slow"`` (indirect parametrization) for one that
    is slow to answer once an offline run is done (``slow_daemon.py``)."""
    from meegxer.gui.connection import spawn_daemon

    control = free_endpoint()
    if getattr(request, "param", "normal") == "slow":
        proc = spawn_slow_daemon(control, free_endpoint())
    else:
        proc = spawn_daemon(control, free_endpoint())
    yield control, proc
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
