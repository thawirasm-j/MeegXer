"""Review at import (docs/REVIEW.md): saved per recording, applied by every
pipeline that reads it, the raw file untouched."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.control import Daemon
from meegxer.core.graph import Graph
from meegxer.core.import_review import import_review_path
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.recording import describe, recording_events
from meegxer.core.registry import Registry
from meegxer.core.review import load_review


def recording(tmp_path: Path) -> Path:
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((3, 1000)), info, verbose="error")
    raw.set_annotations(mne.Annotations([1.0, 5.0], [0.0, 0.5], ["stim", "blink"]))
    raw.info["bads"] = ["Pz"]
    path = tmp_path / "sub-01_raw.fif"
    raw.save(path, verbose="error")
    return path


def source_of(path: Path, **params: Any) -> Any:
    entry = {"id": "src", "node": "meegxer.core.file",
             "params": {"path": str(path), **params}}  # fmt: skip
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [entry]},
                            Registry.from_entry_points())  # fmt: skip
    return graph.source.impl


@pytest.fixture
def daemon() -> Any:
    d = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    yield d
    d.close()


def test_a_review_at_import_is_saved_and_every_pipeline_uses_it(
    tmp_path: Path, daemon: Daemon
) -> None:
    path = recording(tmp_path)
    node = {"node": "meegxer.core.file", "params": {"path": str(path)}}
    first = daemon.import_review_open(node)  # nothing saved yet: the raw file's
    assert [m["label"] for m in first["marks"]] == ["stim", "blink"]
    assert first["bads"] == ["Pz"] and first["crop"] is None
    assert first["duration_s"] == 10.0
    assert first["file"].endswith("sub-01_raw_desc-import_events.tsv")

    before = source_of(path).content_id()
    marks = [
        {"onset": 1.0, "duration": 0.0, "label": "stim"},  # kept
        {"onset": 2.0, "duration": 1.0, "label": "BAD_move"},  # added
        {"onset": 0.5, "duration": 0.0, "label": "early"},
    ]  # before the crop
    daemon.save_import_review(node, marks, bads=["Pz", "Fz"], crop=[0.8, 6.0],
                              viewed=[[0.0, 10.0]], reviewer="R. Rater")  # fmt: skip
    saved = load_review(import_review_path(source_of(path)))  # type: ignore[arg-type]
    assert saved is not None and saved.crop == (0.8, 6.0)
    assert saved.reviewer == "R. Rater" and saved.bads == ("Pz", "Fz")

    # any pipeline reading the file: cropped, the reviewed events, the bads
    source = source_of(path)
    assert source.n_samples == 520  # 0.8 s .. 6.0 s
    assert source.info.bads == ("Pz", "Fz")
    events = recording_events(source)
    assert [(e.sample_index, e.label, e.duration) for e in events] == [
        (20, "stim", 0), (120, "BAD_move", 100)]  # shifted by 0.8 s  # fmt: skip
    assert describe(source)["events_origin"] == "import review"
    assert source.content_id() != before  # a change asks for refits downstream
    # the input can leave it out; the raw file itself is untouched
    raw = source_of(path, import_review=False)
    assert raw.n_samples == 1000 and raw.info.bads == ("Pz",)
    assert describe(raw)["events_origin"] == "recording"
    # opening the review again starts from what was saved
    again = daemon.import_review_open(node)
    assert again["crop"] == [0.8, 6.0] and len(again["marks"]) == 3
    assert again["bads"] == ["Pz", "Fz"] and again["duration_s"] == 10.0

    # and a batch run writes the cropped recording with the reviewed events
    out = tmp_path / "out.set"
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "w", "node": "meegxer.core.set_writer", "params": {"path": str(out)}},
    ]}, Registry.from_entry_points())  # fmt: skip
    BatchScheduler().run(graph)
    written = mne.io.read_raw_eeglab(out, verbose="error")
    assert written.n_times == 520
    assert list(written.annotations.description) == ["stim", "BAD_move"]


def test_wrong_reviews_at_import_are_refused(tmp_path: Path, daemon: Daemon) -> None:
    node = {"node": "meegxer.core.file",
            "params": {"path": str(recording(tmp_path))}}  # fmt: skip
    with pytest.raises(ValueError, match="no channels"):
        daemon.save_import_review(node, [], bads=["Oz"])
    with pytest.raises(ValueError, match="crop"):
        daemon.save_import_review(node, [], crop=[3.0, 2.0])
    synthetic = {"node": "meegxer.core.synthetic", "params": {"duration": 2.0}}
    with pytest.raises(ValueError, match="no file"):
        daemon.import_review_open(synthetic)
