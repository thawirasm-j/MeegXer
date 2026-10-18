"""Share-safe sessions (spec/session.md, Sharing): no file or folder names leave
with a shared session; hashes, models and the user's own text stay."""

import json
from pathlib import Path
from typing import Any

from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.session import (
    drop_shared_sinks,
    free_text,
    is_placeholder,
    load_session,
    read_session,
    share_safe,
    shared_input_hint,
    write_session,
)

HOME = "/home/jane/Patients/JaneDoe_1980"
H1, H2, H3 = ("sha256:" + d * 64 for d in "123")


def session() -> dict[str, Any]:
    return {
        "meegxer_session": "1.0", "spec_version": SPEC_VERSION,
        "name": "Jane Doe follow-up", "description": "second visit",
        "pipeline": {"spec_version": SPEC_VERSION, "nodes": [
            {"id": "input", "node": "meegxer.core.file",
             "params": {"path": f"{HOME}/JaneDoe_2024-05-06.set"},
             "recording": {"montage": f"{HOME}/JaneDoe_digitized.elc",
                           "events": {"file": f"{HOME}/JaneDoe_events.tsv"}}},
            {"id": "hp", "node": "meegxer.nodes.iir_filter",
             "params": {"kind": "highpass", "low": 1.0}},
            {"id": "out", "node": "meegxer.core.edf_writer",
             "params": {"path": f"{HOME}/out/JaneDoe_clean.edf"}},
            {"id": "bids", "node": "meegxer.core.bids_writer",
             "params": {"root": f"{HOME}/derivatives"}},
        ]},
        "nodes": {"hp": {"node": "lab.script", "version": "1", "provenance": {
            "command": ["/home/jane/venv/bin/python", f"{HOME}/clean.py"],
            "unsandboxed": True, "sha256": "sha256:" + "a" * 64,
            "script": "clean.py"}}},
        "models": {"ica": {
            "fitted_on": {"channels": ["Fz"], "sfreq": 250.0, "bads": [],
                          "stream_id": "JaneDoe_2024-05-06.set"},
            "fingerprint": "f", "data": {}}},
        "inputs": {"input": [
            {"path": f"{HOME}/JaneDoe_2024-05-06.set", "sha256": H1},
            {"path": f"{HOME}/JaneDoe_2024-05-06.fdt", "sha256": H2}]},
        "outputs": [{"path": f"{HOME}/out/JaneDoe_clean.edf", "sha256": H3}],
    }  # fmt: skip


def test_no_file_or_folder_name_leaves() -> None:
    shared = share_safe(session())
    text = json.dumps(shared)
    assert "JaneDoe" not in text and "/home/jane" not in text
    nodes = shared["pipeline"]["nodes"]
    assert nodes[0]["params"]["path"] == "<input>.set"
    assert nodes[0]["recording"] == {"montage": "<montage>.elc",
                                     "events": {"file": "<events>.tsv"}}  # fmt: skip
    assert nodes[2]["params"]["path"] == "<output-1>.edf"
    assert nodes[3]["params"]["root"] == ""  # the default derivatives folder
    paths = [e["path"] for e in shared["inputs"]["input"]]
    assert paths == ["<input>.set", "<input>.fdt"]
    assert shared["outputs"] == [{"path": "<output-1>.edf", "sha256": H3}]
    assert [e["sha256"] for e in shared["inputs"]["input"]] == [H1,
                                                                H2]  # fmt: skip
    assert shared["nodes"]["hp"]["provenance"]["command"] == ["python", "clean.py"]
    assert shared["models"]["ica"]["fitted_on"]["stream_id"] == "<input>.set"
    assert free_text(shared) == {"name": "Jane Doe follow-up",
                                 "description": "second visit"}  # fmt: skip
    assert share_safe(shared) == shared  # sharing twice changes nothing
    assert not is_placeholder("sub-01_raw.fif") and is_placeholder("<input-root>")


def test_placeholders_survive_writing_and_reading(tmp_path: Path) -> None:
    file = write_session(tmp_path / "s.meegxer.json", share_safe(session()))
    back = read_session(file)
    assert back["pipeline"]["nodes"][0]["params"]["path"] == "<input>.set"
    assert back["outputs"][0]["path"] == "<output-1>.edf"


def test_a_shared_session_loads_and_asks_for_its_recording() -> None:
    shared = share_safe(session())
    del shared["nodes"], shared["models"]  # not installed / not a real model
    loaded = load_session(shared, Registry.from_entry_points(), strict=True)
    assert any("montage file was left out" in w for w in loaded.warnings)
    assert any("events file was left out" in w for w in loaded.warnings)
    assert not any("inputs" in w for w in loaded.warnings)  # no file to compare
    graph: Graph = loaded.graph
    assert "-i FILE" in (shared_input_hint(graph) or "")
    assert drop_shared_sinks(graph) == ["out"]
    assert "out" not in graph and "bids" in graph


def test_an_ordinary_session_is_unaffected() -> None:
    plain = session()
    plain["pipeline"]["nodes"][0].pop("recording")
    del plain["nodes"], plain["models"]
    loaded = load_session(plain, Registry.from_entry_points())
    assert shared_input_hint(loaded.graph) is None
    assert drop_shared_sinks(loaded.graph) == []
