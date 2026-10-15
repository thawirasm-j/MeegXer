"""Recording overrides on sources (spec 0.2.3): names, types, bads, positions,
events; and the channel description surviving the disk cache."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import mne
import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.cache import CachedOutput, DirectoryCache
from meegxer.core.control import Daemon
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.node import Source
from meegxer.core.registry import Registry


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    # the EOG channel is stored as EEG, as many files do; the override fixes it
    info = mne.create_info(["Fp1", "Cz", "VEOG"], 250.0, "eeg")
    raw = mne.io.RawArray(np.zeros((3, 2500)), info, verbose="error")
    raw.set_annotations(mne.Annotations([1.0, 3.0, 5.0], [0.0, 0.5, 0.0],
                                        ["stim", "rest", "stim"]))  # fmt: skip
    path = tmp_path / "rec_raw.fif"
    raw.save(path, verbose="error")
    return path


def source(path: Path, **recording: object) -> dict[str, object]:
    entry: dict[str, object] = {"id": "input", "node": "meegxer.core.file",
                                "params": {"path": str(path)}}  # fmt: skip
    if recording:
        entry["recording"] = recording
    return entry


@pytest.fixture
def daemon() -> Iterator[Daemon]:
    d = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    yield d
    d.close()


def test_describe_reads_types_and_events(daemon: Daemon, recording: Path) -> None:
    info = daemon.describe_source(source(recording))
    assert [c["label"] for c in info["channels"]] == ["Fp1", "Cz", "VEOG"]
    assert [c["type"] for c in info["channels"]] == ["eeg", "eeg", "eeg"]
    assert info["sfreq"] == 250.0 and info["duration_s"] == 10.0
    assert [e["label"] for e in info["events"]] == ["stim", "rest", "stim"]
    assert info["events"][1]["onset"] == pytest.approx(3.0)
    assert info["event_labels"] == ["rest", "stim"]


def test_overrides_rename_retype_mark_bad_and_position(
    daemon: Daemon, recording: Path
) -> None:
    rec = {
        "channels": {"VEOG": {"rename": "EOG1", "type": "eog", "bad": True},
                     "Fp1": {"position": [0.0, 0.1, 0.0]}},
        "montage": "standard_1005",
        "events": {"rename": {"stim": "target"}, "drop": ["rest"]},
    }  # fmt: skip
    info = daemon.describe_source(source(recording, **rec))
    chans = {c["label"]: c for c in info["channels"]}
    assert set(chans) == {"Fp1", "Cz", "EOG1"} and chans["EOG1"]["original"] == "VEOG"
    assert chans["EOG1"]["bad"] and chans["EOG1"]["position"] is None
    assert chans["Fp1"]["position"] == [0.0, 0.1, 0.0]  # per channel beats montage
    assert chans["Cz"]["position"] is not None  # from the montage
    assert [e["label"] for e in info["events"]] == ["target", "target"]

    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        source(recording, **rec)]}, Registry.from_entry_points())  # fmt: skip
    assert (
        graph.to_dict()["nodes"][0]["recording"]["channels"]["VEOG"]["rename"] == "EOG1"
    )
    src = graph["input"].impl
    assert isinstance(src, Source)
    block = src.read(100)
    assert block is not None
    assert block.info.ch_labels == ("Fp1", "Cz", "EOG1") and block.info.bads == (
        "EOG1",
    )
    assert block.info.ch_types == ("eeg", "eeg", "eog")


def test_events_file_replaces_the_recordings_events(
    daemon: Daemon, recording: Path, tmp_path: Path
) -> None:
    events = tmp_path / "sub-01_events.tsv"
    events.write_text("onset\tduration\ttrial_type\n2.5\t0.1\tgo\n4\tn/a\tstop\n")
    info = daemon.describe_source(source(recording, events={"file": str(events)}))
    assert info["events_origin"] == "file"
    assert [(e["onset"], e["label"]) for e in info["events"]] == [
        (2.5, "go"),
        (4.0, "stop"),
    ]


@pytest.mark.parametrize("rec", [
    {"channels": {"Nope": {"bad": True}}},
    {"montage": "no_such_montage"},
    {"channels": {"Cz": {"type": "brain"}}},
])  # fmt: skip
def test_bad_overrides_are_refused(recording: Path, rec: dict[str, object]) -> None:
    with pytest.raises((GraphError, ValueError)):
        graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
            source(recording, **rec)]}, Registry.from_entry_points())  # fmt: skip
        src = graph["input"].impl
        assert isinstance(src, Source)
        src.read(10)


def test_recording_is_for_sources_only() -> None:
    with pytest.raises(GraphError, match="sources"):
        Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
            {"id": "input", "node": "meegxer.core.synthetic"},
            {"node": "meegxer.core.probe", "recording": {"montage": "standard_1020"}},
        ]}, Registry.from_entry_points())  # fmt: skip


def test_disk_cache_keeps_the_channel_description() -> None:
    info = SignalInfo(("Cz", "Fz"), 100.0, "x", ("Fz",), ("eeg", "eog"),
                      ((0.0, 0.0, 0.09), None))  # fmt: skip
    cache = DirectoryCache(tempfile.mkdtemp())
    cache.put("k", CachedOutput(Block(np.zeros((2, 5)), info, 0, 0.0), ()))
    hit = cache.get("k")
    assert hit is not None and hit.block.info == info
