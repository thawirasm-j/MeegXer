"""Offline runs through the daemon: background batch run, scrollable probe windows."""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from meegxer.core.control import Daemon
from meegxer.core.manifest import SPEC_VERSION


@pytest.fixture
def daemon() -> Iterator[Daemon]:
    d = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    yield d
    d.close()


def wait_done(daemon: Daemon) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status: dict[str, Any] = daemon.status()["offline"]
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("offline run did not finish")


PIPELINE = {
    "spec_version": SPEC_VERSION,
    "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic",
         "params": {"duration": 6.0, "n_channels": 4, "sfreq": 500.0}},
        {"id": "tap_input", "node": "meegxer.core.probe", "input": "input",
         "params": {"keep_seconds": 0.0}},
        {"id": "bp", "node": "meegxer.sterile.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "tap_bp", "node": "meegxer.core.probe", "input": "bp",
         "params": {"keep_seconds": 0.0}},
    ],
}  # fmt: skip


def test_offline_run_serves_aligned_probe_windows(daemon: Daemon) -> None:
    started = daemon.run_offline(PIPELINE)
    assert started["state"] == "running"
    status = wait_done(daemon)
    assert status["state"] == "done", status["error"]
    probes = status["probes"]
    assert set(probes) == {"tap_input", "tap_bp"}
    assert probes["tap_bp"]["bin_size"] == 2 and probes["tap_bp"]["n_bins"] == 1500
    a = daemon.offline_window("tap_input", 100, 50)
    b = daemon.offline_window("tap_bp", 100, 50)
    assert a["start_bin"] == b["start_bin"] == 100
    assert np.array(a["mins"]).shape == (4, 50)
    assert np.all(np.array(a["mins"]) <= np.array(a["maxs"]))
    clipped = daemon.offline_window("tap_bp", 1490, 100)
    assert len(clipped["mins"][0]) == 10


def test_offline_refuses_live_only_nodes_and_reports_errors(daemon: Daemon) -> None:
    live = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.lsl_in"}]}  # fmt: skip
    with pytest.raises(ValueError, match="online-only"):
        daemon.run_offline(live)
    endless = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic",
         "params": {"duration": 0.0}}]}  # fmt: skip
    daemon.run_offline(endless)
    status = wait_done(daemon)
    assert status["state"] == "error" and "finite" in status["error"]
