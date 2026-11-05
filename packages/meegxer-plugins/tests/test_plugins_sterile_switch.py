"""The Sterile switch end to end (docs/STERILE.md): an offline run, a live run,
the daemon and ``meegxer run`` with Sterile mode on run their built-in strips
in the Sterile worker and give the local run's output; the local processors are
back when the run ends. The worker runs as a local process here
(``MEEGXER_STERILE_LOCAL=1``); CI's container job runs the same tests in the
image."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block
from meegxer.core.control import ControlClient, Daemon
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage
from meegxer.core.sterile import ENV
from meegxer.plugins.sterile import LOCAL_ENV, SterileProcessor

PIPELINE: dict[str, Any] = {"spec_version": SPEC_VERSION, "nodes": [
    {"id": "input", "node": "meegxer.core.synthetic",
     "params": {"n_channels": 4, "sfreq": 250.0, "duration": 3.0}},
    {"id": "bp", "node": "meegxer.nodes.iir_filter",
     "params": {"kind": "bandpass", "low": 1.0, "high": 40.0}},
    {"id": "notch", "node": "meegxer.nodes.notch"},
    {"id": "tap", "node": "meegxer.core.probe", "input": "notch"},
]}  # fmt: skip


@pytest.fixture
def local_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    monkeypatch.setenv(LOCAL_ENV, "1")


def graph() -> Graph:
    return Graph.from_dict(PIPELINE, Registry.from_entry_points())


def tapped(g: Graph) -> np.ndarray:
    probe = g["tap"].impl
    assert isinstance(probe, MemoryProbe)
    return np.hstack([b.data for b in probe.blocks if isinstance(b, Block)])


def processors(g: Graph) -> list[type]:
    return [type(n.impl.processor) for n in g.nodes if isinstance(n.impl, Stage)]


def test_an_offline_run_switched_on(local_worker: None) -> None:
    local, isolated = graph(), graph()
    BatchScheduler(block_size=100, sterile=False).run(local)
    BatchScheduler(block_size=100, sterile=True).run(isolated)
    np.testing.assert_array_equal(tapped(isolated), tapped(local), strict=True)
    assert SterileProcessor not in processors(isolated)  # local again after


def test_the_setting_switches_an_offline_run(
    local_worker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No flag: the variable (or the setting) decides."""
    monkeypatch.setenv(ENV, "1")
    g = graph()
    scheduler = BatchScheduler(block_size=100)
    calls: list[str] = []
    from meegxer.plugins import sterile

    real = sterile.SterileWorker.call

    def counted(self: Any, processor: Any, method: str, *rest: Any) -> Any:
        calls.append(method)
        return real(self, processor, method, *rest)

    monkeypatch.setattr(sterile.SterileWorker, "call", counted)
    scheduler.run(g)
    assert "transform" in calls  # the strips ran in the worker


def test_a_live_run_switched_on(local_worker: None) -> None:
    """Live: the stream processors' output through the worker is the local
    one (block by block, state carried by the host), and the status says
    Sterile while it runs."""

    def live(sterile: bool) -> tuple[np.ndarray, bool, Graph]:
        g = graph()
        scheduler = RealtimeScheduler(g, max_block=25, sterile=sterile)
        scheduler.start()
        deadline = time.monotonic() + 20
        said = False
        while not scheduler.wait(0.2):
            said = said or bool(scheduler.status().get("sterile"))
            assert time.monotonic() < deadline, "the live run did not end"
        assert not scheduler.metrics.errors, scheduler.metrics.errors
        return tapped(g), said, g

    want, said_local, _ = live(False)
    got, said, g = live(True)
    assert said and not said_local
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    assert SterileProcessor not in processors(g)  # local again after the run


def test_the_daemon_says_and_builds(
    local_worker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sterile_status: on or off, the image, the build; sterile_build runs
    the image build in the background (a stand-in build here)."""
    from meegxer.core import sterile as core_sterile

    def fake_build(log: Any = print, **_: Any) -> str:
        for line in ("step 1/2", "step 2/2"):
            log(line)
            time.sleep(0.05)
        return "meegxer-sterile:test"

    monkeypatch.setattr(core_sterile, "build_image", fake_build)
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    thread = threading.Thread(target=d.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(d.control_endpoint, timeout=10)
    try:
        status = client.call("sterile_status")
        assert status["enabled"] is False and status["build"] == "idle"
        assert status["image"] == "ready"  # the local worker needs no image
        monkeypatch.setenv(ENV, "1")
        assert client.call("sterile_status")["enabled"] is True
        client.call("sterile_build")
        deadline = time.monotonic() + 10
        while (status := client.call("sterile_status"))["build"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert status["build"] == "done"
        assert status["lines"] == ["step 1/2", "step 2/2"]
        client.call("load_pipeline", pipeline=PIPELINE)
        client.call("start")
        assert client.call("status")["sterile"] is True
        client.call("stop")
    finally:
        client.call("shutdown")
        client.close()
        thread.join(5)


def test_meegxer_run_sterile(
    local_worker: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--sterile runs it in the worker and says so; Sterile with no image is
    refused at once, with what to do."""
    import json

    from meegxer.cli import app

    monkeypatch.delenv(ENV, raising=False)  # --sterile sets it: undone after

    pipeline = tmp_path / "p.json"
    pipeline.write_text(json.dumps(PIPELINE))
    runner = CliRunner()
    done = runner.invoke(app, ["run", str(pipeline), "--sterile"])
    assert done.exit_code == 0, done.output
    assert "Sterile mode: the processing runs in its container" in done.output
    monkeypatch.delenv(LOCAL_ENV)
    monkeypatch.setenv("MEEGXER_STERILE_IMAGE", "meegxer-sterile:none-such")
    monkeypatch.setenv("PATH", str(tmp_path))  # no Docker either
    refused = runner.invoke(app, ["run", str(pipeline), "--sterile"])
    assert refused.exit_code == 1
    assert "Sterile mode needs Docker or Podman" in refused.output
