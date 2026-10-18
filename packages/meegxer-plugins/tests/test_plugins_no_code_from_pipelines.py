"""Pipelines and sessions never run code (docs/ROADMAP.md M9.6, rule 9).

A pipeline or session only names nodes. A local plugin or custom script runs only
from an index entry the user added after accepting the local-code notice; naming
it, or recording its command in a session, starts nothing. The trap below writes a
marker file as soon as anything launches it, so its absence proves nothing ran.
"""

import json
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core.control import Daemon
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.session import SessionError, load_session
from meegxer.plugins import index
from meegxer.plugins.adapter import plugin_class
from meegxer.plugins.worker import UNSANDBOXED_ENV

NAME = "lab.trap"
MANIFEST = {
    "spec_version": "0.2.1", "name": NAME, "version": "0.1.0", "type": "stream",
    "inputs": [{"name": "in", "type": "signal"}],
    "outputs": [{"name": "out", "type": "signal"}], "params": [],
}  # fmt: skip


@pytest.fixture
def trap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A local plugin entry added without consent, whose code leaves a marker."""
    marker = tmp_path / "ran"
    script = tmp_path / "trap.py"
    script.write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    monkeypatch.setenv(index.INDEX_ENV, str(tmp_path / "plugins.json"))
    monkeypatch.delenv(UNSANDBOXED_ENV, raising=False)
    entry = {"command": [sys.executable, str(script)], "unsandboxed": True,
             "manifest": MANIFEST}  # fmt: skip
    index.save_index([entry])
    return {"marker": marker, "entry": entry, "tmp": tmp_path}


def pipeline(rec: Path | None = None) -> dict[str, Any]:
    source: dict[str, Any] = (
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(rec)}}
        if rec
        else {"id": "input", "node": "meegxer.core.synthetic"}
    )
    return {"spec_version": SPEC_VERSION,
            "nodes": [source, {"id": "trap", "node": NAME}]}  # fmt: skip


def recording(path: Path) -> Path:
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    data = np.random.default_rng(0).standard_normal((3, 1000)) * 1e-5
    mne.io.RawArray(data, info, verbose="error").save(path, verbose="error")
    return path


def test_naming_a_local_plugin_does_not_load_it(trap: dict[str, Any]) -> None:
    registry = Registry.from_entry_points()
    assert NAME not in registry.names()
    with pytest.raises(LookupError):
        Graph.from_dict(pipeline(), registry)
    assert not trap["marker"].exists()


def test_a_session_recording_the_command_does_not_run_it(
    trap: dict[str, Any],
) -> None:
    session = {
        "meegxer_session": "1.0", "spec_version": SPEC_VERSION,
        "pipeline": pipeline(),
        "nodes": {"trap": {"node": NAME, "version": "0.1.0", "provenance": {
            "command": trap["entry"]["command"], "unsandboxed": True}}},
    }  # fmt: skip
    with pytest.raises(SessionError):
        load_session(session, Registry.from_entry_points())
    assert not trap["marker"].exists()


def test_the_cli_and_the_daemon_refuse_without_running(trap: dict[str, Any]) -> None:
    tmp: Path = trap["tmp"]
    rec = recording(tmp / "rec_raw.fif")
    file = tmp / "pipeline.json"
    file.write_text(json.dumps(pipeline(rec)), encoding="utf-8")
    result = CliRunner().invoke(app, ["run", str(file), "-o", str(tmp / "o.set")])
    assert result.exit_code == 1, result.output
    daemon = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    try:
        for method, doc in (("load_pipeline", pipeline()),
                            ("run_offline", pipeline(rec))):  # fmt: skip
            request = {"jsonrpc": "2.0", "id": 1, "method": method,
                       "params": {"pipeline": doc}}  # fmt: skip
            assert "error" in daemon.handle(json.dumps(request).encode()), method
    finally:
        daemon.close()
    assert not trap["marker"].exists()


def test_a_class_built_anyway_cannot_start_its_code(trap: dict[str, Any]) -> None:
    """Even a plugin class made from the entry directly (no registry) refuses to
    launch its command without the recorded consent."""
    cls = plugin_class(MANIFEST, index.launcher_for_entry(trap["entry"]))
    with pytest.raises(PermissionError):
        cls().worker.start()
    assert not trap["marker"].exists()


def test_with_consent_the_trap_does_run(trap: dict[str, Any]) -> None:
    """The control: the marker does appear once the user has consented, so its
    absence above means nothing ran."""
    index.save_index([{**trap["entry"], "consent": {"accepted": "now"}}])
    registry = Registry.from_entry_points()
    assert NAME in registry.names()  # loading still runs nothing
    assert not trap["marker"].exists()
    graph = Graph.from_dict(pipeline(), registry)
    with pytest.raises(Exception):  # noqa: B017 - the trap speaks no protocol
        graph["trap"].impl.processor.worker.start()  # type: ignore[union-attr]
    assert trap["marker"].exists()
