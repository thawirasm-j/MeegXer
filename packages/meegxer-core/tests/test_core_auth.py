"""The daemon's token (meegxer.core.auth): only clients with it get through, on
the control and the display socket."""

import os
import sys
import threading
import time
from collections.abc import Iterator

import pytest

from meegxer.core.auth import read_token, token_path
from meegxer.core.control import ControlClient, Daemon
from meegxer.core.display import DisplaySubscriber

PIPELINE = {
    "spec_version": "0.2.0",
    "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 0, "n_channels": 4, "sfreq": 200}},
        {"id": "main", "node": "meegxer.core.probe"},
    ],
}  # fmt: skip


@pytest.fixture
def daemon() -> Iterator[Daemon]:
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    thread = threading.Thread(target=d.serve_forever, daemon=True)
    thread.start()
    yield d
    d.request_shutdown()
    thread.join(5)


def test_token_file_is_private_and_removed_on_exit() -> None:
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    path = token_path(d.control_endpoint)
    assert d.token_file == path and path.is_file()
    assert read_token(d.control_endpoint) == path.read_text()
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o077 == 0
    d.close()
    assert not path.exists()


def test_control_needs_the_token(daemon: Daemon) -> None:
    good = ControlClient(daemon.control_endpoint, timeout=5)
    assert good.call("ping")["display"] == daemon.display_endpoint
    good.close()

    wrong = ControlClient(daemon.control_endpoint, timeout=0.5, token="not-it")
    with pytest.raises(TimeoutError):
        wrong.call("ping")
    wrong.close()

    token_path(daemon.control_endpoint).rename(
        token_path(daemon.control_endpoint).with_suffix(".moved")
    )
    try:
        missing = ControlClient(daemon.control_endpoint, timeout=0.5)
        with pytest.raises(TimeoutError, match=r"no token for .*your account"):
            missing.call("ping")
        missing.close()
    finally:
        token_path(daemon.control_endpoint).with_suffix(".moved").rename(
            token_path(daemon.control_endpoint)
        )


def test_token_from_the_environment(
    daemon: Daemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = token_path(daemon.control_endpoint).read_text()
    monkeypatch.setenv("MEEGXER_TOKEN", token)
    os.replace(token_path(daemon.control_endpoint),
               token_path(daemon.control_endpoint).with_suffix(".moved"))  # fmt: skip
    client = ControlClient(daemon.control_endpoint, timeout=5)
    assert client.call("ping")
    client.close()


def test_display_needs_the_token(daemon: Daemon) -> None:
    client = ControlClient(daemon.control_endpoint, timeout=10)
    client.call("load_pipeline", pipeline=PIPELINE)
    outsider = DisplaySubscriber(daemon.display_endpoint, token="not-it")
    member = DisplaySubscriber(daemon.display_endpoint, token=client.token)
    client.call("start")
    try:
        deadline = time.monotonic() + 10
        frame = None
        while frame is None and time.monotonic() < deadline:
            frame = member.recv(timeout=0.2)
        assert frame is not None
        assert outsider.recv(timeout=1.0) is None
    finally:
        client.call("stop")
        outsider.close()
        member.close()
        client.close()
