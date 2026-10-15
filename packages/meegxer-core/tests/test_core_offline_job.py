"""Offline runs through the daemon: background batch run, scrollable probe windows."""

import time
from collections.abc import Iterator
from pathlib import Path
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


def test_offline_run_serves_aligned_probe_windows(
    daemon: Daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEEGXER_OFFLINE_DIR", str(tmp_path))
    started = daemon.run_offline(PIPELINE)
    assert started["state"] == "running"
    status = wait_done(daemon)
    assert status["state"] == "done", status["error"]
    probes = status["probes"]
    assert set(probes) == {"tap_input", "tap_bp"}
    assert probes["tap_bp"]["n_filled"] == probes["tap_bp"]["n_samples"] == 3000
    a = daemon.offline_window("tap_input", 200, 100, 50)
    b = daemon.offline_window("tap_bp", 200, 100, 50)
    assert a["start_bin"] == b["start_bin"] == 100 and a["bin_size"] == 2
    assert np.array(a["mins"]).shape == (4, 50)
    assert np.all(np.array(a["mins"]) <= np.array(a["maxs"]))
    clipped = daemon.offline_window("tap_bp", 2980, 100, 50)
    assert len(clipped["mins"][0]) == 10
    # zoomed out: a coarser level; zoomed in: single samples
    whole = daemon.offline_window("tap_bp", 0, 3000, 100)
    assert whole["bin_size"] >= 30 and len(whole["mins"][0]) <= 100
    fine = daemon.offline_window("tap_bp", 1000, 20, 100)
    assert fine["bin_size"] == 1 and fine["mins"] == fine["maxs"]
    # the stored envelope matches the samples
    coarse = np.array(daemon.offline_window("tap_bp", 1000, 400, 200)["maxs"])
    samples = np.array(daemon.offline_window("tap_bp", 1000, 400, 400)["maxs"])
    np.testing.assert_allclose(coarse, samples.reshape(4, 200, 2).max(axis=2),
                               rtol=1e-6)  # fmt: skip
    assert list((tmp_path).glob("run-*/tap_bp/samples.npy"))


def test_probe_store_is_readable_while_it_fills(tmp_path: Path) -> None:
    from meegxer.core.block import Block, SignalInfo
    from meegxer.core.offline import ProbeStore

    store = ProbeStore(tmp_path / "p", SignalInfo(("Cz",), 500.0), 5000)
    data = np.sin(np.arange(5000) / 10.0)[None, :]
    store.write(Block(data[:, :1000], SignalInfo(("Cz",), 500.0), 0, 0.0))
    early = store.window(0, 5000, 100)
    assert store.filled == 1000
    assert len(early["mins"][0]) <= 1000 // early["bin_size"] + 1
    store.write(Block(data[:, 1000:], SignalInfo(("Cz",), 500.0), 1000, 2.0))
    late = store.window(0, 5000, 100)
    assert len(late["mins"][0]) > len(early["mins"][0])


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
