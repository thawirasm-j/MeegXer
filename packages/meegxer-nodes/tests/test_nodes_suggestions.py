"""Suggestions (docs/REVIEW.md step 4b): a detector's events reviewed in a
Review strip with priors. Each gets a verdict kept with its origin; later
strips see the verdicts only."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.review import (
    BAD_CHANNEL,
    Mark,
    ReviewAnswer,
    ReviewNeeded,
    ReviewRequest,
    load_review,
)

SFREQ = 250.0


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """10 s, two spikes on Fz (2 s and 5 s) and Oz flat throughout."""
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 5e-6, (4, 2500))
    data[0, 500] += 400e-6
    data[0, 1250] += 400e-6
    data[3] = 0.0
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], SFREQ, "eeg")
    path = tmp_path / "sub-01_raw.fif"
    mne.io.RawArray(data, info, verbose="error").save(path, verbose="error")
    return path


def graph_of(recording: Path, decision: str = "verify") -> Graph:
    return Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.file",
         "params": {"path": str(recording)}},
        {"id": "amp", "node": "meegxer.nodes.detect_amplitude"},
        {"id": "rev", "node": "meegxer.core.review",
         "params": {"points": f"BAD_peak,{BAD_CHANNEL}", "decision": decision,
                    "file": str(recording.with_name("marks.tsv"))}},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
    ]}, Registry.from_entry_points())  # fmt: skip


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_verify_review_keeps_verdicts_and_later_strips_see_only_them(
    recording: Path,
) -> None:
    asked: list[ReviewRequest] = []

    def review(req: ReviewRequest) -> ReviewAnswer:
        asked.append(req)
        first, second = (s for s in req.points if s.label == "BAD_peak")
        return ReviewAnswer(
            (Mark(first.onset, first.duration, first.label, first),),  # yes
            bads=("Oz",),  # the proposed bad channel: yes
            rejected=(second,),  # no
        )

    graph = graph_of(recording)
    scheduler = BatchScheduler(review=review)
    result = scheduler.run(graph, keep=["rev", "rej"])
    (request,) = asked
    assert request.decision == "verify"
    assert {(s.label, s.origin, s.channel) for s in request.points} == {
        ("BAD_peak", "amp", ""), (BAD_CHANNEL, "amp", "Oz")}  # fmt: skip
    peaks = [s for s in request.points if s.label == "BAD_peak"]
    assert [round(s.onset) for s in peaks] == [2, 5]

    # only the accepted spike is cut; the rejected one is not seen after rev
    (boundary,) = result.events["rej"]
    assert boundary.label == "boundary" and boundary.sample_index < 1250
    after = [e.label for e in scheduler.rail_events("rej")]
    assert after.count("BAD_peak") == 1 and BAD_CHANNEL not in after
    assert [e.label for e in scheduler.rail_events("rev")].count("BAD_peak") == 2
    rev = result.outputs["rev"]
    block = rev[0] if isinstance(rev, list) else rev
    assert isinstance(block, Block) and "Oz" in block.info.bads

    # every verdict is kept with its origin, the rejected one too
    marks_file = recording.with_name("marks.tsv")
    table = rows(marks_file)
    assert sorted((r["trial_type"], r["verdict"], r["suggestion_origin"],
                   r["channel"]) for r in table) == [
        ("BAD_peak", "accepted", "amp", "n/a"),
        ("BAD_peak", "rejected", "amp", "n/a"),
        (BAD_CHANNEL, "accepted", "amp", "Oz"),
    ]  # fmt: skip
    saved = load_review(marks_file)
    assert saved is not None and saved.bads == ("Oz",)

    # the same input again: the verdicts are reused, nothing asked
    BatchScheduler().run(graph_of(recording))


def test_verify_needs_a_verdict_for_every_suggestion(recording: Path) -> None:
    def review(req: ReviewRequest) -> ReviewAnswer:
        return ReviewAnswer((), bads=("Oz",))  # the spikes left unanswered

    with pytest.raises(GraphError, match="2 suggestion"):
        BatchScheduler(review=review).run(graph_of(recording))
    with pytest.raises(ReviewNeeded):  # headless: stops at the strip
        BatchScheduler().run(graph_of(recording))


def test_free_review_rejects_what_is_left_without_an_answer(
    recording: Path,
) -> None:
    def review(req: ReviewRequest) -> ReviewAnswer:
        first = next(s for s in req.points if s.label == "BAD_peak")
        own = Mark(8.0, 0.5, "BAD_eyes")  # a mark of the reviewer's own
        moved = Mark(first.onset - 0.1, 0.3, "BAD_muscle", first)  # edited
        return ReviewAnswer((own, moved))

    graph = graph_of(recording, "free")
    result = BatchScheduler(review=review).run(graph)
    saved = load_review(recording.with_name("marks.tsv"))
    assert saved is not None
    verdicts: list[Any] = sorted((m.label, m.verdict) for m in saved.marks)
    assert verdicts == [
        ("BAD_eyes", ""), ("BAD_muscle", "edited"),
        ("BAD_peak", "rejected"), (BAD_CHANNEL, "rejected"),
    ]  # fmt: skip
    assert saved.bads == ()
    cut = [e for e in result.events["rej"] if e.label == "boundary"]
    assert len(cut) == 2  # the edited span and the reviewer's own
