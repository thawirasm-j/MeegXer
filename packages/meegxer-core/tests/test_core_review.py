"""Manual review (docs/REVIEW.md): marks in a sidecar per recording, a batch run
that asks for a review, an offline run that pauses for it."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jsonschema
import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event
from meegxer.core.control import Daemon
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.review import (
    Mark,
    ReviewNeeded,
    ReviewRequest,
    Suggestion,
    load_review,
    review_path,
)
from meegxer.core.session import build_session, session_schema, share_safe


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

    graph = graph_of(file)
    result = BatchScheduler(review=review, reviewer="A. Rater").run(graph, keep=["rev"])
    assert [(e.sample_index, e.label, e.duration) for e in result.events["rev"]] == [
        (100, "BAD_muscle", 50), (300, "blink", 0)]  # fmt: skip
    assert not isinstance(result.outputs["rev"], list)  # events only: nothing cut
    saved = load_review(file)
    assert saved is not None and len(saved.marks) == 2 and saved.upstream
    assert saved.reviewer == "A. Rater"
    assert file.with_suffix(".json").is_file()
    entry = build_session(graph)["nodes"]["rev"]["review"]  # who, when, on what
    assert entry == {"reviewer": "A. Rater", "reviewed": saved.reviewed,
                     "upstream": saved.upstream}  # fmt: skip
    jsonschema.validate(build_session(graph), session_schema())
    shared = share_safe(build_session(graph))  # names no one, and no file
    assert "reviewer" not in shared["nodes"]["rev"]["review"]
    assert shared["pipeline"]["nodes"][2]["params"]["file"] == ""

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
    daemon.run_offline(pipeline, reviewer="B. Rater")
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
    assert saved.reviewer == "B. Rater"
    session = daemon.get_session(offline=True)
    assert session["nodes"]["rev"]["review"]["reviewer"] == "B. Rater"

    daemon.run_offline(pipeline)  # reviewed on this input: no pause
    assert wait_for(daemon, {"review", "done", "error"})["state"] == "done"

    file.unlink()
    daemon.run_offline(pipeline)
    wait_for(daemon, {"review"})
    status = daemon.cancel_review()
    assert status["state"] == "error" and "cancelled" in status["error"]
    assert not file.exists()


# -- a Reject segments strip cuts rejected spans out (docs/REVIEW.md) ----------------


def cut_graph(tmp_path: Path, after: list[dict[str, Any]]) -> Graph:
    return Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 6.0, "n_channels": 2, "sfreq": 100.0}},
        {"id": "rev", "node": "meegxer.core.review",
         "params": {"file": str(tmp_path / "marks.tsv")}},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
        *after,
    ]}, Registry.from_entry_points())  # fmt: skip


def rejecting(req: ReviewRequest) -> list[Mark]:
    return [Mark(2.0, 1.0, "BAD_muscle"), Mark(2.5, 1.0, "BAD_eyes"),
            Mark(5.0, 0.2, "blink")]  # fmt: skip


def test_rejected_spans_are_cut_with_a_boundary_and_do_not_leak(
    tmp_path: Path,
) -> None:
    from meegxer.core.stage import Stage, signal_fingerprint
    from meegxer.nodes.filters import IIRFilter

    hp = {"id": "hp", "node": "meegxer.nodes.iir_filter",
          "params": {"kind": "highpass", "low": 1.0}}  # fmt: skip
    graph = cut_graph(tmp_path, [hp])
    result = BatchScheduler(block_size=37, review=rejecting).run(
        graph, keep=["src", "hp"]
    )
    pieces = result.outputs["hp"]
    assert isinstance(pieces, list) and all(isinstance(b, Block) for b in pieces)
    pieces = [b for b in pieces if isinstance(b, Block)]
    # overlapping BAD spans merged: [200, 350) cut; samples keep their index
    assert [(b.sample_index, b.stop_index) for b in pieces] == [(0, 200), (350, 600)]
    assert result.events["rej"] == (Event(350, "boundary", 150, {"origin": "rej"}),)
    assert "blink" in [e.label for e in result.events["rev"]]  # not a reject word
    # the piece after the cut is filtered as if it were a recording of its own
    source = result.outputs["src"]
    assert not isinstance(source, list)
    alone = Stage(IIRFilter(), {"kind": "highpass", "low": 1.0}).for_mode("offline")
    piece = source.slice(350, 600)
    expected = alone.transform(piece, signal_fingerprint(piece.info)).block
    assert isinstance(expected, Block)
    np.testing.assert_array_equal(pieces[1].data, expected.data)


def test_nothing_is_cut_without_reject_words(tmp_path: Path) -> None:
    graph = cut_graph(tmp_path, [])
    graph["rej"].impl.params["reject"] = ""
    result = BatchScheduler(review=rejecting).run(graph, keep=["rej"])
    assert not isinstance(result.outputs["rej"], list)


def test_the_recordings_own_bad_spans_are_cut_too(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0.5\tBAD_acq\n")
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 3.0, "n_channels": 2, "sfreq": 100.0},
         "recording": {"events": {"file": str(events)}}},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
    ]}, Registry.from_entry_points())  # fmt: skip
    result = BatchScheduler().run(graph, keep=["rej"])  # no review needed
    pieces = result.outputs["rej"]
    assert isinstance(pieces, list)
    assert [(b.sample_index, b.stop_index) for b in pieces
            if isinstance(b, Block)] == [(0, 100), (150, 300)]  # fmt: skip


def test_a_writer_saves_the_pieces_with_boundary_events(tmp_path: Path) -> None:
    out = tmp_path / "clean.set"
    writer = {"id": "w", "node": "meegxer.core.set_writer",
              "params": {"path": str(out)}}  # fmt: skip
    BatchScheduler(review=rejecting).run(cut_graph(tmp_path, [writer]))
    raw = mne.io.read_raw_eeglab(out, verbose="error")
    assert raw.n_times == 450  # 600 samples less the 150 cut
    # the boundary, and the review's blink (5.0 s) 1.5 s earlier; the BAD
    # marks were in the cut
    assert list(raw.annotations.description) == ["boundary", "blink"]
    assert raw.annotations.onset[0] == pytest.approx(2.0, abs=0.011)
    assert raw.annotations.onset[1] == pytest.approx(3.5, abs=0.011)


def test_a_fit_after_a_cut_sees_only_the_kept_samples(tmp_path: Path) -> None:
    from typing import ClassVar

    from meegxer.core.manifest import Manifest
    from meegxer.core.node import Params, Processor, TransformResult
    from meegxer.core.serialization import State
    from meegxer.core.stage import Stage

    class Fitted(Processor):
        manifest: ClassVar[Manifest] = Manifest.from_dict({
            "spec_version": SPEC_VERSION, "name": "test.fitted", "version": "0.1.0",
            "type": "stream", "requires_fit": True, "params": [],
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}]})  # fmt: skip

        def fit(self, calibration: Block, params: Params) -> State:
            return {"n": calibration.n_samples}

        def transform(
            self, block: Block, model: State, state: State | None, params: Params
        ) -> TransformResult:
            return TransformResult(block, {})

    graph = cut_graph(tmp_path, [])
    graph.add(Stage(Fitted()), id="fitted")
    BatchScheduler(review=rejecting).run(graph)
    stage = graph["fitted"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    assert stage.model.data["n"] == 450  # the 150 samples cut are not fitted on


def test_bids_events_follow_the_processed_recordings_time() -> None:
    from meegxer.core.bids_out import _events_rows
    from meegxer.core.block import Block, SignalInfo

    info = SignalInfo(("Cz",), 100.0)
    pieces = [Block(np.zeros((1, 200)), info, 0, 0.0),
              Block(np.zeros((1, 250)), info, 350, 0.0)]  # fmt: skip
    events = [
        {"onset": 1.0, "duration": 0.0, "label": "a"},  # before the cut
        {"onset": 2.5, "duration": 0.0, "label": "b"},  # inside it: dropped
        {"onset": 4.0, "duration": 0.5, "label": "c"},
    ]  # after: 1.5 s earlier
    assert _events_rows(events, pieces) == [
        ["1.0", "0.0", "a"],
        ["2.0", "0.0", "boundary"],
        ["2.5", "0.5", "c"],
    ]


# -- display presets and bad channels (Design v2, step 2) ----------------------------


def test_bad_channels_marked_in_review_become_the_streams_bads(
    tmp_path: Path,
) -> None:
    from meegxer.core.review import ReviewAnswer

    file = tmp_path / "marks.tsv"
    graph = graph_of(file)
    graph["rev"].impl.params.update(
        montage="longitudinal bipolar", lowpass="40", notch="off"
    )  # the window opens with these
    asked: list[ReviewRequest] = []

    def review(req: ReviewRequest) -> ReviewAnswer:
        asked.append(req)
        return ReviewAnswer((), ((0.0, 4.0),), ("Fp2",))

    result = BatchScheduler(review=review).run(graph, keep=["hp", "rev"])
    # the researcher's presets reach the window; the rest is the reviewer's
    assert asked[0].display == {
        "montage": "longitudinal bipolar",
        "lowpass": 40.0,
        "notch": 0.0,
    }
    assert asked[0].to_wire()["display"] == asked[0].display
    after, before = result.outputs["rev"], result.outputs["hp"]
    assert isinstance(after, Block) and isinstance(before, Block)
    assert after.info.bads == ("Fp2",) and before.info.bads == ()
    saved = load_review(file)
    assert saved is not None and saved.bads == ("Fp2",)
    # reused on the next run, and offered again when the input changes
    again = BatchScheduler().run(graph_of(file), keep=["rev"])
    reused = again.outputs["rev"]
    assert isinstance(reused, Block) and reused.info.bads == ("Fp2",)
    with pytest.raises(ReviewNeeded) as changed:
        BatchScheduler().run(graph_of(file, low=2.0))
    assert changed.value.request.bads == ("Fp2",)
    # the bads are part of what later strips depend on
    assert (
        again.fingerprints["rev"]
        != BatchScheduler(review=lambda req: ReviewAnswer((), ((0.0, 4.0),)))
        .run(graph_of(tmp_path / "other.tsv"))
        .fingerprints["rev"]
    )


def test_an_offline_page_through_a_montage_over_rpc(
    daemon: Daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEEGXER_OFFLINE_DIR", str(tmp_path / "offline"))
    file = tmp_path / "marks.tsv"
    chain = nodes({"file": str(file)})
    chain.insert(
        2,
        {
            "id": "tap_hp",
            "node": "meegxer.core.probe",
            "input": "hp",
            "params": {"keep_seconds": 0.0},
        },
    )  # before the strip
    pipeline = {"spec_version": SPEC_VERSION, "nodes": chain}
    daemon.run_offline(pipeline)
    wait_for(daemon, {"review"})
    page = daemon.offline_window(
        "tap_hp", 0, 100, 2000, {"montage": "average", "highpass": 1.0}
    )
    assert page["derived"] and len(page["ch_labels"]) == 2
    plain = daemon.offline_window("tap_hp", 0, 100, 2000)
    assert "derived" not in plain
    with pytest.raises(ValueError, match="unknown display option"):
        daemon.offline_window("tap_hp", 0, 100, 2000, {"zoom": 2})
    daemon.submit_review([], [[0.0, 4.0]], ["Fp1"])
    assert wait_for(daemon, {"done", "error"})["state"] == "done"
    saved = load_review(file)
    assert saved is not None and saved.bads == ("Fp1",)


# -- a review's marks reach the output files and the views --------------------------


def test_writers_save_the_reviews_marks_and_the_inputs_events(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n0.5\t0\tstim\n2.6\t0\tlost\n")
    out, edf = tmp_path / "clean.set", tmp_path / "clean.edf"
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 6.0, "n_channels": 2, "sfreq": 100.0},
         "recording": {"events": {"file": str(events)}}},
        {"id": "rev", "node": "meegxer.core.review",
         "params": {"file": str(tmp_path / "marks.tsv")}},
        {"id": "set", "node": "meegxer.core.set_writer", "params": {"path": str(out)}},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
        {"id": "edf", "node": "meegxer.core.edf_writer", "params": {"path": str(edf)}},
    ]}, Registry.from_entry_points())  # fmt: skip
    BatchScheduler(review=rejecting).run(graph)
    raw = mne.io.read_raw_eeglab(out, verbose="error")  # before the cut
    got = sorted(zip(raw.annotations.onset.round(2), raw.annotations.description,
                     strict=True))  # fmt: skip
    assert got == [(0.5, "stim"), (2.0, "BAD_muscle"), (2.5, "BAD_eyes"),
                   (2.6, "lost"), (5.0, "blink")]  # fmt: skip
    cut = mne.io.read_raw_edf(edf, verbose="error")  # after it: [2.0, 3.5) gone
    got = sorted(
        (o, d)
        for o, d in zip(
            cut.annotations.onset.round(2), cut.annotations.description, strict=True
        )
        if d != "BAD_ACQ_SKIP"
    )  # MNE's EDF export pads the last record
    # events in the cut are left out; those after it move 1.5 s earlier
    assert got == [(0.5, "stim"), (2.0, "boundary"), (3.5, "blink")]


def test_the_views_get_the_runs_events(
    daemon: Daemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEEGXER_OFFLINE_DIR", str(tmp_path / "offline"))
    pipeline = {"spec_version": SPEC_VERSION,
                "nodes": nodes({"file": str(tmp_path / "marks.tsv")})}  # fmt: skip
    daemon.run_offline(pipeline)
    wait_for(daemon, {"review"})
    assert daemon.offline_events()["events"] == []  # nothing marked yet
    daemon.submit_review([{"onset": 1.0, "duration": 0.5, "label": "spike"}])
    wait_for(daemon, {"done"})
    assert daemon.offline_events()["events"] == [
        {"onset": 1.0, "duration": 0.5, "label": "spike", "origin": "review"}
    ]
    assert daemon.offline_events("hp")["events"] == []  # before the strip


def test_review_points_are_offered_and_answers_keep_their_point(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0.5\tbeta\n"
                      "2.0\t0\tstim\n3.0\t0.4\tbeta\n")  # fmt: skip
    file = tmp_path / "marks.tsv"
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 5.0, "n_channels": 2, "sfreq": 100.0},
         "recording": {"events": {"file": str(events)}}},
        {"id": "rev", "node": "meegxer.core.review",
         "params": {"file": str(file), "points": "beta"}},
    ]}, Registry.from_entry_points())  # fmt: skip
    asked: list[ReviewRequest] = []

    def review(req: ReviewRequest) -> list[Mark]:
        asked.append(req)
        first, second = req.points
        # the physician: the first is a spindle (retimed), the second is not
        return [
            Mark(0.9, 0.7, "spindle", first),
            Mark(3.0, 0.4, "not_spindle", second),
        ]

    BatchScheduler(review=review).run(graph)
    assert asked[0].points == (Suggestion(1.0, 0.5, "beta", "input"),
                               Suggestion(3.0, 0.4, "beta", "input"))  # fmt: skip
    assert asked[0].to_wire()["points"][0] == {
        "onset": 1.0, "duration": 0.5, "label": "beta", "origin": "input"}  # fmt: skip
    saved = load_review(file)
    assert saved is not None
    assert [(m.point, m.verdict) for m in saved.marks] == [(1.0, "edited"),
                                                           (3.0, "edited")]  # fmt: skip
    assert "suggestion_onset" in file.read_text().splitlines()[0]
    # a mark from the wire keeps its suggestion; one without has none
    wired = Mark.from_wire({"onset": 1, "label": "x", "suggestion": {
        "onset": 2.5, "duration": 0, "label": "beta", "origin": "det"}})  # fmt: skip
    assert wired.suggestion == Suggestion(2.5, 0.0, "beta", "det")
    assert Mark.from_wire({"onset": 1, "label": "x", "point": 2.5}).point == 2.5
    assert Mark.from_wire({"onset": 1, "label": "x"}).to_wire() == {
        "onset": 1.0,
        "duration": 0.0,
        "label": "x",
    }
