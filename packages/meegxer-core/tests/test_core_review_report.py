"""The review part of the QC report (docs/REVIEW.md step 6): who reviewed,
what they viewed and marked, how they answered suggestions, epochs kept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np

from meegxer.core.batch import BatchScheduler
from meegxer.core.bids_out import target_for
from meegxer.core.epoch_review import EpochReviewRecord, EpochVerdict
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.qc import recording_qc
from meegxer.core.registry import Registry
from meegxer.core.review import (
    Mark,
    Review,
    ReviewAnswer,
    ReviewRequest,
    Suggestion,
)
from meegxer.core.review_report import (
    epoch_review_section,
    review_row,
    review_section,
    section_html,
)


def test_a_review_in_a_few_fields() -> None:
    blink = Suggestion(4.0, 0.5, "blink", "blinks1")
    other = Suggestion(9.0, 0.5, "blink", "blinks1")
    big = Suggestion(12.0, 1.0, "BAD_amp", "amp1")
    review = Review(
        (
            Mark(1.0, 2.0, "BAD"),
            Mark(5.0, 1.5, "BAD"),
            Mark(4.0, 0.5, "blink", blink, "accepted"),
            Mark(8.9, 0.7, "blink", other, "edited"),
            Mark(12.0, 1.0, "BAD_amp", big, "rejected"),
        ),
        reviewed="2026-10-31T12:00:00Z",
        viewed=((0.0, 30.0), (20.0, 45.0)),
        reviewer="TJ",
        bads=("T7",),
    )
    section = review_section("review1", "Review (review1)", review, 60.0)
    row = review_row([section])
    assert row["review1_reviewer"] == "TJ"
    assert row["review1_viewed_percent"] == 75.0  # 0-45 s of 60, overlaps merged
    # a rejected suggestion is no mark
    assert row["review1_marks"] == "BAD: 2 (3.5 s); blink: 2 (1.2 s)"
    assert row["review1_bad_channels"] == "T7"
    assert row["review1_verdicts"] == "amp1: rejected 1; blinks1: accepted 1, edited 1"
    page = section_html(section)
    assert "Suggestions answered" in page and "<td>BAD</td><td>2</td>" in page

    record = EpochReviewRecord(
        (
            EpochVerdict(1.0, 1.0, "go"),
            EpochVerdict(3.0, 1.0, "go", keep=False),
            EpochVerdict(5.0, 1.0, "stop"),
        ),
        reviewer="TJ",
    )
    epochs = review_row([epoch_review_section("er1", "Review epochs", record)])
    assert epochs["er1_epochs"] == "go: 1 kept, 1 rejected; stop: 1 kept, 0 rejected"


def test_the_qc_report_of_a_reviewed_recording(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    data = np.random.default_rng(0).standard_normal((3, 250 * 20)) * 1e-5
    bids = mne_bids.BIDSPath(root=root, subject="01", task="rest", datatype="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    mne_bids.write_raw_bids(raw, bids, format="EDF", allow_preload=True,
                            verbose="error")  # fmt: skip
    pipeline: dict[str, Any] = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.bids",
         "params": {"root": str(root), "subject": "01", "task": "rest"}},
        {"id": "review1", "node": "meegxer.core.review"},
        {"id": "out", "node": "meegxer.core.bids_writer"},
    ]}  # fmt: skip
    graph = Graph.from_dict(pipeline, Registry.from_entry_points())

    def reviewer(request: ReviewRequest) -> ReviewAnswer:
        return ReviewAnswer((Mark(2.0, 3.0, "BAD"),), viewed=((0.0, 10.0),),
                            bads=("Pz",))  # fmt: skip

    scheduler = BatchScheduler(review=reviewer)
    scheduler.reviewer = "TJ"
    scheduler.run(graph)
    target = target_for(graph, graph["out"].impl.params)
    row = recording_qc(graph, target, "sub-01_task-rest")
    assert row["review1_reviewer"] == "TJ"
    assert row["review1_marks"] == "BAD: 1 (3.0 s)"
    assert row["review1_viewed_percent"] == 50.0
    assert row["review1_bad_channels"] == "Pz"
    page = target.path("report.html").read_text(encoding="utf-8")
    assert "Review (review1)" in page and "Reviewer" in page
