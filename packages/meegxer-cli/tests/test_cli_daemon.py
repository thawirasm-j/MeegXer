import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core.control import Daemon

runner = CliRunner()


@pytest.fixture
def control() -> Iterator[str]:
    daemon = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    yield daemon.control_endpoint
    daemon.request_shutdown()
    thread.join(5)


def cli(*args: str) -> str:
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_drive_the_daemon_from_the_cli(control: str, tmp_path: Path) -> None:
    pipeline = tmp_path / "live.json"
    pipeline.write_text(json.dumps({"spec_version": "0.2.0", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 0, "n_channels": 4, "sfreq": 200}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "main", "node": "meegxer.core.probe"},
    ]}))  # fmt: skip
    c = ["--control", control]
    assert "loaded" in cli("load", str(pipeline), *c)
    assert "started" in cli("start", *c)
    assert "'low': 0.5" in cli("set", "hp", "low=0.5", "order=2", *c)
    out = cli("status", *c)
    assert "running: True" in out and "hp" in out
    status = json.loads(cli("status", "--json", *c))
    assert {n["id"]: n["params"] for n in status["nodes"]}["hp"]["order"] == 2
    assert "stopped" in cli("stop", *c)


def test_cli_reports_daemon_errors(control: str) -> None:
    result = runner.invoke(app, ["calibrate", "hp", "--control", control])
    assert result.exit_code == 1 and "error" in result.output
    bad = runner.invoke(app, ["set", "hp", "oops", "--control", control])
    assert bad.exit_code == 1 and "name=value" in bad.output


def test_cli_times_out_without_daemon() -> None:
    from meegxer.core import control as control_module

    original = control_module.ControlClient.__init__

    def short_timeout(
        self: control_module.ControlClient,
        endpoint: str = "",
        timeout: float = 5.0,
        **kw: object,
    ) -> None:
        original(self, endpoint, 0.2)

    control_module.ControlClient.__init__ = short_timeout  # type: ignore[method-assign]
    try:
        result = runner.invoke(app, ["status", "--control", "tcp://127.0.0.1:1"])
    finally:
        control_module.ControlClient.__init__ = original  # type: ignore[method-assign]
    assert result.exit_code == 1 and "no reply" in result.output
