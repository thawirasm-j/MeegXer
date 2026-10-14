"""The GUI's two channels to the daemon (rule 3): control calls and display frames.

The GUI is only a client: closing or crashing it never interrupts processing
(rule 2). Display frames arrive on a background thread and are handed to the GUI
thread as Qt signals; if the GUI falls behind, the daemon drops frames for it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from meegxer.core.control import ControlClient, RpcError
from meegxer.core.display import DisplaySubscriber


class DisplayThread(QThread):
    frame = Signal(object)  # DisplayFrame

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.endpoint = endpoint
        self._running = True

    def run(self) -> None:
        subscriber = DisplaySubscriber(self.endpoint)
        try:
            while self._running:
                frame = subscriber.recv(timeout=0.2)
                if frame is not None:
                    self.frame.emit(frame)
        finally:
            subscriber.close()

    def stop(self) -> None:
        self._running = False
        self.wait(2000)


class DaemonLink(QObject):
    """Control calls plus a status poll; display frames via ``frame``."""

    status = Signal(dict)
    frame = Signal(object)
    lost = Signal(str)

    def __init__(self, endpoint: str, poll_ms: int = 300) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.client = ControlClient(endpoint, timeout=3.0)
        info = self.client.call("ping")
        self.version: str = info["version"]
        self.display_endpoint: str = info["display"]
        self._display = DisplayThread(self.display_endpoint)
        self._display.frame.connect(self.frame)
        self._display.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.poll)
        self._timer.start(poll_ms)
        self._lost = False

    def call(self, method: str, **params: Any) -> Any:
        return self.client.call(method, **params)

    def poll(self) -> None:
        try:
            status = self.client.call("status")
        except TimeoutError as exc:
            if not self._lost:
                self._lost = True
                self.lost.emit(str(exc))
            return
        self._lost = False
        self.status.emit(status)

    def close(self) -> None:
        self._timer.stop()
        self._display.stop()
        self.client.close()


def daemon_running(endpoint: str, timeout: float = 0.5) -> bool:
    client = ControlClient(endpoint, timeout=timeout)
    try:
        client.call("ping")
        return True
    except (TimeoutError, RpcError):
        return False
    finally:
        client.close()


def spawn_daemon(
    control: str, display: str, wait: float = 15.0
) -> subprocess.Popen[bytes]:
    """Start a daemon in its own session, so it outlives the GUI (rule 2)."""
    detach: dict[str, Any] = (
        {
            "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "meegxer.core.daemon",
            "--control",
            control,
            "--display",
            display,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **detach,
    )
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the daemon exited with status {proc.returncode}")
        if daemon_running(control):
            return proc
    raise TimeoutError("the daemon did not answer in time")
