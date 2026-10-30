"""Epoch review (docs/REVIEW.md step 5): accept, reject or relabel each epoch of
an Epochs strip; later strips read the reviewed epochs rail."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import FeatureBlock
from meegxer.core.control import Daemon
from meegxer.core.epoch_review import EpochVerdict, load_epoch_review
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import FeatureProbe
from meegxer.core.registry import Registry
from meegxer.core.review import ReviewNeeded, ReviewRequest
from meegxer.core.session import build_session, session_schema, share_safe

SFREQ = 100.0


def pipeline(tmp_path: Path, tmax: float = 0.5) -> dict[str, Any]:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n"
                      "1.0\t0\tgo\n2.0\t0\tstop\n3.0\t0\tgo\n")  # fmt: skip
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 5.0, "n_channels": 2, "sfreq": SFREQ},
         "recording": {"events": {"file": str(events)}}},
        {"id": "ep", "node": "meegxer.nodes.epochs",
         "params": {"tmin": -0.2, "tmax": tmax}},
        {"id": "rev", "node": "meegxer.core.epoch_review",
         "params": {"labels": "go_slow", "file": str(tmp_path / "epochs.tsv")}},
        {"id": "tap", "node": "meegxer.core.feature_probe", "input": "rev:epochs",
         "params": {"keep_windows": 0}},
    ]}  # fmt: skip


def graph_of(tmp_path: Path, tmax: float = 0.5) -> Graph:
    return Graph.from_dict(pipeline(tmp_path, tmax), Registry.from_entry_points())


def tapped(graph: Graph) -> list[FeatureBlock]:
    tap = graph["tap"].impl
    assert isinstance(tap, FeatureProbe)
    return [w for w in tap.windows if isinstance(w, FeatureBlock)]


def test_reviewed_epochs_go_on_the_rail_and_the_verdicts_are_reused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReviewNeeded) as needed:
        BatchScheduler().run(graph_of(tmp_path))
    request = needed.value.request
    assert request.kind == "epochs" and request.reason == "not reviewed"
    assert [(v.onset, v.label, v.keep) for v in request.epochs] == [
        (0.8, "go", True), (1.8, "stop", True), (2.8, "go", True)]  # fmt: skip
    assert request.labels == ("go_slow", "go", "stop")
    assert len(request.windows) == 3
    wire = request.to_wire()
    assert wire["kind"] == "epochs" and len(wire["epochs"]) == 3

    asked: list[ReviewRequest] = []

    def review(req: ReviewRequest) -> list[EpochVerdict]:
        asked.append(req)
        first, second, _ = req.epochs
        return [
            EpochVerdict(first.onset, first.duration, "go_slow"),
            EpochVerdict(second.onset, second.duration, "stop", keep=False),
        ]

    graph = graph_of(tmp_path)
    result = BatchScheduler(review=review, reviewer="E. Rater").run(graph)
    got = tapped(graph)  # the second rejected, the first relabelled
    assert [(w.window, w.coords["epoch"]) for w in got] == [
        ((80, 151), ("go_slow",)), ((280, 351), ("go",))]  # fmt: skip
    saved = load_epoch_review(tmp_path / "epochs.tsv")
    assert saved is not None and saved.reviewer == "E. Rater" and saved.upstream
    assert [(v.label, v.keep, v.event) for v in saved.epochs] == [
        ("go_slow", True, "go"),
        ("stop", False, "stop"),
        ("go", True, "go"),
    ]
    # the rejected epoch is on the Events rail too, for the file writers
    (rejected,) = result.events["rev"]
    assert (rejected.sample_index, rejected.label, rejected.duration) == (
        180, "rejected_epoch", 71)  # fmt: skip
    assert dict(rejected.payload) == {"label": "stop", "event": "stop"}
    header = (tmp_path / "epochs.tsv").read_text().splitlines()[0]
    assert header == "onset\tduration\ttrial_type\tevent\tepoch_status"
    session = build_session(graph)
    jsonschema.validate(session, session_schema())
    assert session["nodes"]["rev"]["review"]["reviewer"] == "E. Rater"
    shared = share_safe(session)
    assert "reviewer" not in shared["nodes"]["rev"]["review"]
    rev = next(n for n in shared["pipeline"]["nodes"] if n["id"] == "rev")
    assert rev["params"]["file"] == ""

    # the same input: the verdicts are reused, nobody is asked
    again = graph_of(tmp_path)
    BatchScheduler(review=review).run(again)
    assert len(asked) == 1 and len(tapped(again)) == 2

    # longer epochs: asked again, the verdicts made before offered
    longer = graph_of(tmp_path, tmax=0.6)
    BatchScheduler(review=review).run(longer)
    assert len(asked) == 2 and asked[1].reason == "input changed"
    assert [(v.label, v.keep) for v in asked[1].epochs] == [
        ("go_slow", True), ("stop", False), ("go", True)]  # fmt: skip


def test_epoch_review_needs_an_epochs_strip_before_it(tmp_path: Path) -> None:
    nodes = pipeline(tmp_path)["nodes"]
    del nodes[1]  # no Epochs strip
    nodes[-1]["input"] = "src"
    with pytest.raises(GraphError, match="epochs"):
        Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes},
                        Registry.from_entry_points())  # fmt: skip
    # a second review reads the first one's rail: the nearest
    data = pipeline(tmp_path)
    data["nodes"].insert(3, {"id": "rev2", "node": "meegxer.core.epoch_review"})
    graph = Graph.from_dict(data, Registry.from_entry_points())
    assert graph.rail_sources("rev2") == {"epochs": ("rev", "epochs")}


def test_an_offline_run_pauses_for_an_epoch_review(tmp_path: Path) -> None:
    daemon = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    try:
        daemon.run_offline(pipeline(tmp_path), reviewer="E. Rater")
        status = daemon.status()["offline"]
        for _ in range(200):
            if status["state"] != "running":
                break
            time.sleep(0.05)
            status = daemon.status()["offline"]
        assert status["state"] == "review"
        assert status["review"]["kind"] == "epochs"
        assert len(status["review"]["epochs"]) == 3
        page = daemon.offline_review_epochs(indices=[0, 2], max_samples=36)
        assert page["ch_labels"] and page["unit"] == "V"
        assert len(page["times"]) == 36 and page["times"][0] == pytest.approx(-0.2)
        assert [e["index"] for e in page["epochs"]] == [0, 2]
        assert np.shape(page["epochs"][0]["data"]) == (2, 36)
        erp = daemon.offline_review_epochs(indices=[0, 2], reduce="mean",
                                           max_samples=1000)  # fmt: skip
        assert erp["n"] == 2 and np.shape(erp["mean"]) == (2, 71)
        with pytest.raises(ValueError, match="no epoch 7"):
            daemon.offline_review_epochs(indices=[7])
        verdicts = [dict(v, keep=False) if i == 1 else v
                    for i, v in enumerate(status["review"]["epochs"])]  # fmt: skip
        daemon.submit_review([], epochs=verdicts)
        for _ in range(200):
            if daemon.status()["offline"]["state"] not in ("running", "review"):
                break
            time.sleep(0.05)
        assert daemon.status()["offline"]["state"] == "done"
        saved = load_epoch_review(tmp_path / "epochs.tsv")
        assert saved is not None
        assert [v.keep for v in saved.epochs] == [True, False, True]
    finally:
        daemon.close()


def test_rejected_epochs_reach_the_written_file(tmp_path: Path) -> None:
    import mne

    out = tmp_path / "out.set"
    data = pipeline(tmp_path)
    data["nodes"][-1] = {"id": "w", "node": "meegxer.core.set_writer",
                         "params": {"path": str(out)}}  # fmt: skip
    graph = Graph.from_dict(data, Registry.from_entry_points())

    def review(req: ReviewRequest) -> list[EpochVerdict]:
        second = req.epochs[1]
        return [EpochVerdict(second.onset, second.duration, second.label, keep=False)]

    BatchScheduler(review=review).run(graph)
    raw = mne.io.read_raw_eeglab(out, verbose="error")
    rejected = [(round(a["onset"], 2), round(a["duration"], 2))
                for a in raw.annotations
                if a["description"] == "rejected_epoch"]  # fmt: skip
    assert rejected == [(1.8, 0.71)]
