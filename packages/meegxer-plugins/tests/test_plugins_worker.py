from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.plugins.adapter import plugin_class, probe_manifest
from meegxer.plugins.worker import (
    ContainerLauncher,
    LocalLauncher,
    PluginCrashed,
    PluginError,
    PluginTimeout,
    PluginWorker,
)

INFO = SignalInfo(("Fz", "Cz"), 100.0)
DATA = np.random.default_rng(0).standard_normal((2, 400)) * 1e-5
BLOCK = Block(DATA, INFO, 0, 0.0)
FP = signal_fingerprint(INFO)


def test_local_plugins_need_explicit_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEEGXER_ALLOW_UNSANDBOXED", raising=False)
    with pytest.raises(PermissionError, match="unsandboxed"):
        LocalLauncher(("python", "x.py")).command("n")
    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    assert LocalLauncher(("python", "x.py")).command("n") == ["python", "x.py"]


def test_container_command_is_locked_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    cmd = ContainerLauncher("lab/plugin:1", engine="podman", cpus=2).command("w1")
    assert cmd[:2] == ["podman", "run"] and cmd[-1] == "lab/plugin:1"
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--rm", "-i",
                 "--security-opt no-new-privileges", "--pids-limit", "--memory",
                 "--cpus 2"):  # fmt: skip
        assert flag in joined


def test_handshake_and_errors(plugin: Callable[..., LocalLauncher]) -> None:
    worker = PluginWorker(plugin("ema"), timeout=10)
    manifest = worker.start()
    assert manifest["name"] == "test.ema" and worker.sdk.startswith("meegxer-python/")
    bad = worker.raw_exchange({"type": "nonsense"})
    assert bad.type == "error" and bad.header["error"] == "bad_request"
    with pytest.raises(PluginError, match="plugin_error"):
        worker.request(
            {"type": "transform", "params": {}, "info": {}}
        )  # missing fields
    worker.close()
    assert worker.last_exit_code == 0 and not worker.alive


def test_crash_is_survived_with_host_held_state(
    plugin: Callable[..., LocalLauncher], tmp_path: Path
) -> None:
    marker = tmp_path / "crash"
    launcher = plugin("ema", CRASH_MARKER=str(marker))
    cls = plugin_class(
        probe_manifest(launcher), launcher, worker_options={"backoff": 0}
    )
    one_pass = Stage(cls()).transform(BLOCK, FP).block
    stage = Stage(cls())
    first = stage.transform(BLOCK.slice(0, 200), FP).block
    marker.touch()  # the next transform crashes the process
    second = stage.transform(BLOCK.slice(200, 400), FP).block
    assert isinstance(first, Block) and isinstance(second, Block)
    assert isinstance(one_pass, Block)
    joined = np.hstack([first.data, second.data])
    np.testing.assert_allclose(joined, one_pass.data, rtol=1e-6, atol=1e-12)
    processor = stage.processor
    assert processor.worker.restarts == 1  # type: ignore[attr-defined]
    processor.close()


def test_hang_times_out_and_restarts(
    plugin: Callable[..., LocalLauncher], tmp_path: Path
) -> None:
    marker = tmp_path / "hang"
    launcher = plugin("ema", HANG_MARKER=str(marker))
    cls = plugin_class(probe_manifest(launcher), launcher,
                       worker_options={"timeout": 1.0, "backoff": 0})  # fmt: skip
    stage = Stage(cls())
    stage.transform(BLOCK.slice(0, 100), FP)
    marker.touch()
    out = stage.transform(BLOCK.slice(100, 200), FP).block  # times out, retried
    assert isinstance(out, Block) and out.n_samples == 100
    stage.processor.close()


def test_retries_are_bounded(
    plugin: Callable[..., LocalLauncher], tmp_path: Path
) -> None:
    launcher = plugin("ema", HANG_MARKER=str(tmp_path / "hang"))
    worker = PluginWorker(launcher, timeout=0.5, max_restarts=1, backoff=0)
    worker.start()
    (tmp_path / "hang").touch()
    header = {
        "type": "transform", "params": {"alpha": 0.1}, "info": {}, "sample_index": 0,
        "timestamp": 0.0, "model_id": None, "state": None,
    }  # fmt: skip
    data = {"data": DATA[:, :10].astype(np.float32)}
    # first attempt hangs (marker consumed), the retry succeeds
    assert worker.request(header, data).type == "result"
    worker.close()


def test_stdout_pollution_is_a_protocol_error(
    plugin: Callable[..., LocalLauncher],
) -> None:
    launcher = plugin("printer")
    cls = plugin_class(probe_manifest(launcher), launcher,
                       worker_options={"max_restarts": 0})  # fmt: skip
    with pytest.raises(PluginError, match=r"protocol error|magic"):
        Stage(cls()).transform(BLOCK, FP)


def test_dead_worker_reports_crash() -> None:
    worker = PluginWorker(LocalLauncher(("false",), allow=True), max_restarts=0)
    with pytest.raises(PluginCrashed):
        worker.start()
    with pytest.raises((PluginCrashed, PluginTimeout)):
        worker.request({"type": "transform"})
