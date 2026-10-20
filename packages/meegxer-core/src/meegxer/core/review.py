"""Manual review (docs/REVIEW.md): a strip whose marks a person makes.

The review strip passes its input on unchanged and puts the marks made for the
recording on the Events rail (labelled spans: ``BAD_muscle``, ``eyes_closed``,
…), where later strips see them like the recording's own events. The marks live
in a sidecar per recording, a BIDS events file (onset and duration in seconds
from the recording's start, ``trial_type``) with a JSON file that records what
they were made on: the fingerprint of the strip's input. So the pipeline stays a
template for every recording, and other tools read the marks.

A batch run reaching the strip uses the sidecar when it was made on the same
input. Otherwise (no sidecar yet, or something upstream changed since) it asks
for a review through its ``review`` callback: the GUI's offline run pauses and
shows the recording; with no callback (the CLI) the recording stops there with
:class:`ReviewNeeded`, so nothing is processed past an unreviewed strip. Marks
made on another input are offered again to be confirmed, never reused silently
(rule 7).
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from meegxer.core.block import CHANNEL_TYPES, Block, Event
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, Source, TransformResult
from meegxer.core.references import credits
from meegxer.core.serialization import State

REVIEW_NODE = "meegxer.core.review"
DEFAULT_LABELS = "BAD"
BOUNDARY = "boundary"  # the event at a cut (EEGLAB's name for it)


@dataclass(frozen=True, slots=True)
class Mark:
    """A labelled span, in seconds from the recording's start."""

    onset: float
    duration: float
    label: str

    def to_wire(self) -> dict[str, Any]:
        return {"onset": self.onset, "duration": self.duration, "label": self.label}

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Mark:
        onset, duration = float(data["onset"]), float(data.get("duration", 0.0))
        label = str(data["label"]).strip()
        if not label:
            raise ValueError("a mark needs a label")
        if onset < 0 or duration < 0 or not math.isfinite(onset + duration):
            raise ValueError(f"mark {label!r}: onset and duration must be >= 0")
        return cls(onset, duration, label)


@dataclass(frozen=True, slots=True)
class Review:
    """The marks of one recording and what they were made on."""

    marks: tuple[Mark, ...]
    upstream: str = ""  # fingerprint of the strip's input
    reviewed: str = ""  # when (ISO 8601, UTC)


@dataclass(frozen=True, slots=True)
class ReviewRequest:
    """A strip waiting for a review."""

    node_id: str
    path: Path
    upstream: str
    reason: str  # "not reviewed" or "input changed"
    previous: tuple[Mark, ...] = ()  # marks made on another input, to confirm
    labels: tuple[str, ...] = ()  # offered: the strip's, then the rail's
    duration_s: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {
            "node": self.node_id, "file": str(self.path), "reason": self.reason,
            "marks": [m.to_wire() for m in self.previous],
            "labels": list(self.labels), "duration_s": self.duration_s,
        }  # fmt: skip


class ReviewNeeded(RuntimeError):
    """A recording reached a review strip nobody has reviewed it at."""

    def __init__(self, request: ReviewRequest) -> None:
        self.request = request
        super().__init__(
            f"{request.node_id}: needs a manual review ({request.reason}); open "
            "the pipeline in the GUI and run it offline to review "
            f"(the marks are saved to {request.path})"
        )


class ReviewCancelled(RuntimeError):
    """The person reviewing stopped the run."""


ReviewCallback = Callable[[ReviewRequest], Sequence[Mark]]


class ReviewStrip(Processor):
    """Passes the signal on; puts the recording's reviewed marks on the Events
    rail. Offline only: a person reviews a whole recording."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.20",
            "name": REVIEW_NODE,
            "version": "0.1.0",
            "type": "stream",
            "description": "Review a recording by hand: mark spans (bad "
                           "segments, states) on the Events rail.",
            "availability": "offline",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "channel_types": list(CHANNEL_TYPES),
            "params": [
                {"name": "labels", "type": "string", "default": DEFAULT_LABELS,
                 "primary": True,
                 "description": "Comma-separated labels offered for marks "
                                "(the input's event types are offered too)"},
                {"name": "reject", "type": "string", "default": "BAD",
                 "description": "Marks whose label starts with one of these "
                                "comma-separated words are cut out, with a "
                                "boundary event (empty: nothing is cut)"},
                {"name": "file", "type": "string", "default": "", "path": "save",
                 "extensions": [".tsv"], "advanced": True,
                 "description": "Where the marks are saved (empty: next to "
                                "each recording, or in its BIDS derivatives)"},
            ],
            **credits(REVIEW_NODE),
        }
    )  # fmt: skip

    def __init__(self) -> None:
        self.marks: tuple[Event, ...] = ()  # set by the batch run from the review

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        lo, hi = block.sample_index, block.stop_index
        inside = tuple(e for e in self.marks if lo <= e.sample_index < hi)
        return TransformResult(block, {}, inside)


def _words(text: object) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def rejected_spans(events: Sequence[Event], params: Params) -> list[tuple[int, int]]:
    """The sample spans ``[start, stop)`` to cut: marks whose label starts with
    a ``reject`` word, overlapping ones merged."""
    words = tuple(_words(params["reject"]))
    spans = sorted((e.sample_index, e.sample_index + e.duration) for e in events
                   if words and e.duration and e.label.startswith(words))  # fmt: skip
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def cut_blocks(
    blocks: Sequence[Block], spans: Sequence[tuple[int, int]]
) -> tuple[list[Block], tuple[Event, ...]]:
    """``blocks`` without the samples in ``spans``: the samples after a cut keep
    their index (rule 8), and a boundary event marks each cut that has signal
    after it (its duration: the samples cut). Strips after it restart there
    (``Stage.restart``), so nothing leaks across."""
    out: list[Block] = []
    for block in blocks:
        start = block.sample_index
        for lo, hi in spans:
            if hi <= start or lo >= block.stop_index:
                continue
            if lo > start:
                out.append(block.slice(start, lo))
            start = max(start, hi)
        if start < block.stop_index:
            out.append(block.slice(start, block.stop_index))
    first = blocks[0].sample_index if blocks else 0
    stop = blocks[-1].stop_index if blocks else 0
    boundaries = tuple(Event(hi, BOUNDARY, hi - lo, {"origin": "review"})
                       for lo, hi in spans if first < hi < stop)  # fmt: skip
    return out, boundaries


def offered_labels(params: Params, seen: Sequence[str]) -> tuple[str, ...]:
    own = _words(params["labels"])
    return tuple(dict.fromkeys([*own, *sorted(set(seen) - {BOUNDARY})]))


_SAFE = re.compile(r"[^A-Za-z0-9]")


def review_path(graph: Any, node_id: str, params: Params) -> Path:
    """The sidecar of the strip ``node_id`` for the graph's input: its ``file``
    parameter, else the recording's BIDS derivatives (an input in a BIDS
    dataset), else next to the input file."""
    if str(params["file"]):
        return Path(str(params["file"])).expanduser().resolve()
    desc = _SAFE.sub("", node_id) or "review"
    from meegxer.core.bids_out import target_for

    try:
        return target_for(graph, {"desc": desc}).path("events.tsv")
    except ValueError:
        pass
    source = graph.source.impl
    assert isinstance(source, Source)
    files = source.input_files()
    if not files:
        raise ValueError(
            f"{node_id}: this input has no file to keep the marks next to; set "
            "the review strip's 'file'"
        )
    raw = Path(files[0]).resolve()
    stem = raw.name.split(".")[0]
    return raw.with_name(f"{stem}_desc-{desc}_events.tsv")


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def load_review(path: Path) -> Review | None:
    """The review saved at ``path`` (None when there is none)."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    marks = tuple(
        Mark.from_wire({"onset": r["onset"],
                        "duration": r.get("duration") or 0.0,
                        "label": r.get("trial_type") or r.get("value") or ""})
        for r in rows
    )  # fmt: skip
    meta: dict[str, Any] = {}
    if _sidecar(path).is_file():
        meta = json.loads(_sidecar(path).read_text(encoding="utf-8")).get("MeegXer", {})
    return Review(marks, str(meta.get("upstream", "")), str(meta.get("reviewed", "")))


def save_review(path: Path, review: Review, node_id: str) -> None:
    """Write the events file and its JSON sidecar (the marks in onset order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marks = sorted(review.marks, key=lambda m: (m.onset, m.label))
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["onset", "duration", "trial_type"])
        for m in marks:
            out.writerow([f"{m.onset:.6f}", f"{m.duration:.6f}", m.label])
    meta = {
        "onset": {"Description": "Start of the span", "Units": "s"},
        "duration": {"Description": "Length of the span", "Units": "s"},
        "trial_type": {"Description": "Label given in a manual review"},
        "MeegXer": {"strip": node_id, "upstream": review.upstream,
                    "reviewed": review.reviewed},
    }  # fmt: skip
    _sidecar(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def marks_to_events(marks: Sequence[Mark], sfreq: float) -> tuple[Event, ...]:
    """Marks as events on a clock at ``sfreq`` counted from the recording's
    start (rule 8), to the nearest sample."""
    return tuple(
        Event(round(m.onset * sfreq), m.label, round(m.duration * sfreq),
              {"origin": "review"})
        for m in sorted(marks, key=lambda m: (m.onset, m.label))
    )  # fmt: skip
