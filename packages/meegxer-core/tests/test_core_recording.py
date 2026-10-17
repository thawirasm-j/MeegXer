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


def test_recording_events_are_events_at_sample_indices(recording: Path) -> None:
    """What a labelled fit receives (spec 0.2.16): renamed and dropped as the
    overrides say, onset x rate from the first sample."""
    from meegxer.core.block import Event
    from meegxer.core.recording import recording_events

    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [source(
        recording, events={"rename": {"stim": "go"}, "drop": ["rest"]})]},
        Registry.from_entry_points())  # fmt: skip
    src = graph["input"].impl
    assert isinstance(src, Source)
    assert recording_events(src) == (Event(250, "go"), Event(1250, "go"))
    plain = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        source(recording)]}, Registry.from_entry_points())["input"].impl  # fmt: skip
    assert isinstance(plain, Source)
    assert recording_events(plain)[1] == Event(750, "rest", 125)


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


@pytest.mark.parametrize(
    ("label", "found"),
    [("Fp1", "Fp1"), ("FP1", "Fp1"), ("EEG Fp1-REF", "Fp1"), ("EEG O1-LE", "O1"),
     ("EEG_C3_A2", "C3"), ("T3", "T7"), ("EEG T5-REF", "P7"), ("Fp1-F7", None),
     ("HEOG", None)],
)  # fmt: skip
def test_label_lookup_like_eeglab(label: str, found: str | None) -> None:
    from meegxer.core.recording import lookup_positions, montage_positions

    table = montage_positions("standard_1005")
    positions, unmatched = lookup_positions([label], "standard_1005")
    if found is None:
        assert unmatched == [label] and not positions
    else:
        assert positions[label] == table[found]


def test_daemon_lookup_positions(daemon: Daemon) -> None:
    reply = daemon.lookup_positions(["Cz", "EEG Pz-REF", "Status"], "standard_1020")
    assert set(reply["positions"]) == {"Cz", "EEG Pz-REF"}
    assert reply["unmatched"] == ["Status"]
    assert len(reply["positions"]["Cz"]) == 3


def test_eeglab_ced_file(tmp_path: Path) -> None:
    from meegxer.core.recording import lookup_positions

    ced = tmp_path / "chanlocs.ced"
    ced.write_text(
        "Number\tlabels\ttheta\tradius\tX\tY\tZ\tsph_theta\tsph_phi\tsph_radius\ttype\n"
        "1\tFpz\t0\t0.5\t85\t0\t-1.8\t0\t-1.2\t85\t\n"
        "2\tT7\t-90\t0.5\t0\t85\t-1.8\t90\t-1.2\t85\t\n"
        "3\tCz\t0\t0\t0\t0\t85\t0\t90\t85\t\n"
        "4\tEOG\t\t\t\t\t\t\t\t\t\n"
    )
    positions, unmatched = lookup_positions(
        ["EEG Fpz-REF", "T7", "Cz", "EOG"], str(ced)
    )
    assert unmatched == ["EOG"]  # listed without coordinates
    fpz, t7 = positions["EEG Fpz-REF"], positions["T7"]
    assert fpz[1] == pytest.approx(0.085) and abs(fpz[0]) < 1e-9  # front
    assert t7[0] == pytest.approx(-0.085)  # left ear: negative x


@pytest.mark.parametrize(
    "change",
    [{"Fp1": {"rename": "AF7"}}, {"VEOG": {"type": "eog"}}, {"Cz": {"bad": True}},
     {"Fp1": {"position": [0.0, 0.1, 0.0]}}],
)  # fmt: skip
def test_recording_changes_are_not_served_from_the_cache(
    recording: Path, change: dict[str, object]
) -> None:
    """The overrides change what the source delivers, so outputs computed before
    an edit in the Recording window must not be reused after it."""
    from meegxer.core.batch import BatchScheduler
    from meegxer.core.cache import MemoryCache

    def pipeline(**rec: object) -> dict[str, object]:
        hp = {"id": "hp", "node": "meegxer.nodes.iir_filter",
              "params": {"kind": "highpass", "low": 1.0}}  # fmt: skip
        return {"spec_version": SPEC_VERSION, "nodes": [source(recording, **rec), hp]}

    registry, cache = Registry.from_entry_points(), MemoryCache()
    BatchScheduler(cache=cache).run(Graph.from_dict(pipeline(), registry))
    edited = Graph.from_dict(pipeline(channels=change), registry)
    result = BatchScheduler(cache=cache).run(edited)
    assert result.cached == []
    src = edited["input"].impl
    assert isinstance(src, Source)
    out = result.outputs["hp"]
    assert isinstance(out, Block) and out.info == src.info
