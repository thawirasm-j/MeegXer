"""Two reviewers, blind, then resolved (docs/REVIEW.md step 6b)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.review import (
    Mark,
    Review,
    ReviewAnswer,
    ReviewRequest,
    load_review,
)
from meegxer.core.review_report import review_row, review_section
from meegxer.core.two_reviewers import agreement, disagreements, rater_path

A = Review((Mark(2.0, 3.0, "BAD"), Mark(10.0, 1.0, "BAD"), Mark(20.0, 0.5, "blink")),
           reviewer="A", bads=("T7", "Pz"))  # fmt: skip
B = Review((Mark(2.0, 3.0, "BAD"), Mark(14.0, 2.0, "BAD")), reviewer="B",
           bads=("T7",))  # fmt: skip


def test_disagreements_and_agreement() -> None:
    agreed, points, both = disagreements(A, B)
    assert [(m.onset, m.label) for m in agreed] == [(2.0, "BAD")]
    assert [(p.onset, p.label, p.origin, p.channel) for p in points] == [
        (0.0, "bad_channel", "reviewer 1 (A)", "Pz"),
        (10.0, "BAD", "reviewer 1 (A)", ""),
        (14.0, "BAD", "reviewer 2 (B)", ""),
        (20.0, "blink", "reviewer 1 (A)", ""),
    ]
    assert both == ("T7",)
    kappa = agreement(A, B, 30.0)
    # BAD per 1 s window: A marks 2-5 and 10-11, B marks 2-5 and 14-16
    a = np.zeros(30, int)
    b = np.zeros(30, int)
    a[[2, 3, 4, 10]] = 1
    b[[2, 3, 4, 14, 15]] = 1
    assert kappa["BAD"] == pytest.approx(round(cohen_kappa_score(a, b), 3))
    assert kappa["blink"] == pytest.approx(0.0)  # only A used it: chance level
    assert rater_path(Path("x/r_desc-review1_events.tsv"), 2).name == (
        "r_desc-review1rater2_events.tsv")  # fmt: skip


def pipeline(tmp_path: Path) -> dict[str, Any]:
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic",
         "params": {"duration": 30.0, "n_channels": 4, "sfreq": 100.0}},
        {"id": "review1", "node": "meegxer.core.review",
         "params": {"reviewers": 2, "file": str(tmp_path / "marks.tsv")}},
    ]}  # fmt: skip


def run(
    tmp_path: Path, reviewer: str, answer: Callable[[ReviewRequest], ReviewAnswer]
) -> list[ReviewRequest]:
    asked: list[ReviewRequest] = []

    def review(request: ReviewRequest) -> ReviewAnswer:
        asked.append(request)
        return answer(request)

    graph = Graph.from_dict(pipeline(tmp_path), Registry.from_entry_points())
    scheduler = BatchScheduler(review=review)
    scheduler.reviewer = reviewer
    scheduler.run(graph)
    return asked


def test_two_blind_reviews_then_the_resolve(tmp_path: Path) -> None:
    def first(request: ReviewRequest) -> ReviewAnswer:
        assert request.reason == "reviewer 1 of 2" and request.previous == ()
        return ReviewAnswer(A.marks)

    with pytest.raises(GraphError, match="needs a second reviewer: A reviewed"):
        run(tmp_path, "A", first)  # A cannot be the second reviewer too
    assert load_review(tmp_path / "marks_rater1.tsv") is not None

    def second_then_resolve(request: ReviewRequest) -> ReviewAnswer:
        if request.reason == "reviewer 2 of 2":
            assert request.previous == ()  # blind: none of A's marks
            return ReviewAnswer(B.marks)
        assert request.reason == "resolve" and request.decision == "verify"
        assert [m.onset for m in request.previous] == [2.0]  # agreed
        assert "Cohen's kappa" in request.note
        by = {(p.onset, p.label): p for p in request.points}
        keep = by[(10.0, "BAD")]  # A's; B's 14 s and A's blink are rejected
        return ReviewAnswer(
            (*request.previous, Mark(keep.onset, keep.duration, keep.label, keep)),
            rejected=(by[(14.0, "BAD")], by[(20.0, "blink")]),
        )

    asked = run(tmp_path, "B", second_then_resolve)
    assert [r.reason for r in asked] == ["reviewer 2 of 2", "resolve"]
    resolved = load_review(tmp_path / "marks.tsv")
    assert resolved is not None and resolved.raters == ("A", "B")
    kept = sorted((m.onset, m.label) for m in resolved.marks if m.verdict != "rejected")
    assert kept == [(2.0, "BAD"), (10.0, "BAD")]
    assert resolved.agreement is not None and "BAD" in resolved.agreement
    row = review_row([review_section("review1", "Review", resolved, 30.0)])
    assert row["review1_raters"] == "A, B" and row["review1_kappa"].startswith("BAD: ")
    # resolved on this input: a later run asks no one
    assert run(tmp_path, "C", lambda r: pytest.fail("asked again")) == []
