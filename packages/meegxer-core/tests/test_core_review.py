"""Manual review (docs/REVIEW.md): marks in a sidecar per recording, a batch run
that asks for a review, an offline run that pauses for it."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.control import Daemon
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.review import (
    Mark,
    ReviewNeeded,
    ReviewRequest,
    load_review,
    review_path,
)


def nodes(review: dict[str, Any], low: float = 1.0) -> list[dict[str, Any]]:
    return [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 4.0, "n_channels": 2, "sfreq": 100.0}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": low}},
        {"id": "rev", "node": "meegxer.core.review", "params": review},
        {"id": "tap", "node": "meegxer.core.probe", "input": "rev",
         "params": {"keep_seconds": 0.0}},
    ]  # fmt: skip


def graph_of(file: Path, low: float = 1.0) -> Graph:
    return Graph.from_dict(
        {"spec_version": SPEC_VERSION, "nodes": nodes({"file": str(file)}, low)},
        Registry.from_entry_points(),
    )


def test_a_batch_run_asks_once_then_reuses_the_review(tmp_path: Path) -> None:
    file = tmp_path / "marks.tsv"
    with pytest.raises(ReviewNeeded) as needed:
        BatchScheduler().run(graph_of(file))
    request = needed.value.request
    assert (request.reason, request.labels, request.duration_s) == (
        "not reviewed", ("BAD",), 4.0)  # fmt: skip
    assert not file.exists()  # nothing saved without a review

    asked: list[ReviewRequest] = []

    def review(req: ReviewRequest) -> list[Mark]:
        asked.append(req)
        return [Mark(1.0, 0.5, "BAD_muscle"), Mark(3.0, 0.0, "blink")]

    result = BatchScheduler(review=review).run(graph_of(file))
    assert [(e.sample_index, e.label, e.duration) for e in result.events["rev"]] == [
        (100, "BAD_muscle", 50), (300, "blink", 0)]  # fmt: skip
    saved = load_review(file)
    assert saved is not None and len(saved.marks) == 2 and saved.upstream
    assert file.with_suffix(".json").is_file()

    again = BatchScheduler().run(graph_of(file))  # made on this input: reused
    assert again.events["rev"] == result.events["rev"] and len(asked) == 1

    # something upstream changed: the marks are offered again, not reused
    with pytest.raises(ReviewNeeded) as changed:
        BatchScheduler().run(graph_of(file, low=2.0))
    assert changed.value.request.reason == "input changed"
    assert changed.value.request.previous == saved.marks


def test_the_sidecar_sits_next_to_the_recording_or_in_its_derivatives(
    tmp_path: Path,
) -> None:
    raw = mne.io.RawArray(np.zeros((2, 500)), mne.create_info(["Fz", "Cz"], 100.0,
                                                              "eeg"))  # fmt: skip
    plain = tmp_path / "rec01.fif"
    raw.save(plain, verbose="error")
    registry = Registry.from_entry_points()

    def path_for(source: dict[str, Any], node_id: str = "rev_1") -> Path:
        review = {"id": node_id, "node": "meegxer.core.review"}
        graph = Graph.from_dict(
            {"spec_version": SPEC_VERSION, "nodes": [source, review]}, registry
        )
        stage = graph[node_id].impl
        return review_path(graph, node_id, stage.params)

    file_source = {"id": "src", "node": "meegxer.core.file",
                   "params": {"path": str(plain)}}  # fmt: skip
    assert path_for(file_source) == tmp_path / "rec01_desc-rev1_events.tsv"
    synthetic = {"id": "src", "node": "meegxer.core.synthetic"}
    with pytest.raises(ValueError, match="set the review strip's 'file'"):
        path_for(synthetic)
    with pytest.raises(GraphError, match="'file'"):
        BatchScheduler().run(Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
            {**synthetic, "params": {"duration": 1.0}},
            {"id": "rev", "node": "meegxer.core.review"}]}, registry))  # fmt: skip


@pytest.fixture
def daemon() -> Iterator[Daemon]:
    d = Daemon("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    yield d
    d.close()


def wait_for(daemon: Daemon, states: set[str]) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status: dict[str, Any] = daemon.status()["offline"]
        if status["state"] in states:
            return status
        time.sleep(0.05)
    raise AssertionError(f"offline run never reached {states}")


def test_an_offline_run_pauses_for_the_review(
    daemon: Daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEEGXER_OFFLINE_DIR", str(tmp_path / "offline"))
    file = tmp_path / "marks.tsv"
    pipeline = {"spec_version": SPEC_VERSION, "nodes": nodes({"file": str(file)})}
    daemon.run_offline(pipeline)
    status = wait_for(daemon, {"review", "done", "error"})
    assert status["state"] == "review", status["error"]
    assert status["review"]["node"] == "rev" and status["review"]["marks"] == []
    with pytest.raises(RuntimeError, match="in progress"):
        daemon.run_offline(pipeline)  # one run at a time, paused or not
    daemon.submit_review([{"onset": 0.5, "duration": 1.0, "label": "BAD_eyes"}])
    status = wait_for(daemon, {"done", "error"})
    assert status["state"] == "done", status["error"]
    assert status["probes"]["tap"]["n_filled"] == 400  # it went on past the strip
    saved = load_review(file)
    assert saved is not None and saved.marks == (Mark(0.5, 1.0, "BAD_eyes"),)

    daemon.run_offline(pipeline)  # reviewed on this input: no pause
    assert wait_for(daemon, {"review", "done", "error"})["state"] == "done"

    file.unlink()
    daemon.run_offline(pipeline)
    wait_for(daemon, {"review"})
    status = daemon.cancel_review()
    assert status["state"] == "error" and "cancelled" in status["error"]
    assert not file.exists()
