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

A review produces events only. Cutting the rejected spans out is the Reject
segments strip's job (``meegxer.core.reject_segments``), downstream.
"""

from __future__ import annotations

import csv
import dataclasses
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
from meegxer.core.montage import MONTAGES
from meegxer.core.node import Params, Processor, Source, TransformResult
from meegxer.core.references import credits
from meegxer.core.serialization import State

REVIEW_NODE = "meegxer.core.review"
REVIEWER = "reviewer"  # a display setting left to the reviewer's choice
# the display's starting values the strip may set (the reviewer can change them)
PRESET_CHOICES: dict[str, tuple[str, ...]] = {
    "highpass": ("off", "0.1", "0.3", "0.5", "1", "2", "5"),
    "lowpass": ("off", "15", "30", "35", "40", "45", "70", "100"),
    "notch": ("off", "50", "60"),
    "sensitivity": ("1", "2", "3", "5", "7", "10", "15", "20", "30", "50", "70",
                    "100"),
    "page": ("5", "10", "15", "20", "30", "60"),
}  # fmt: skip
PRESET_UNITS = {"highpass": "Hz", "lowpass": "Hz", "notch": "Hz",
                "sensitivity": "µV/mm", "page": "s"}  # fmt: skip
DEFAULT_LABELS = "BAD"
BOUNDARY = "boundary"  # the event at a cut (EEGLAB's name for it)


@dataclass(frozen=True, slots=True)
class Mark:
    """A labelled span, in seconds from the recording's start."""

    onset: float
    duration: float
    label: str
    point: float | None = None  # onset of the review point it answers (s)

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"onset": self.onset, "duration": self.duration,
                               "label": self.label}  # fmt: skip
        if self.point is not None:
            out["point"] = self.point
        return out

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Mark:
        onset, duration = float(data["onset"]), float(data.get("duration", 0.0))
        label = str(data["label"]).strip()
        if not label:
            raise ValueError("a mark needs a label")
        if onset < 0 or duration < 0 or not math.isfinite(onset + duration):
            raise ValueError(f"mark {label!r}: onset and duration must be >= 0")
        point = data.get("point")
        if point in ("", "n/a"):
            point = None
        return cls(onset, duration, label, None if point is None else float(point))


@dataclass(frozen=True, slots=True)
class Review:
    """The marks of one recording and what they were made on."""

    marks: tuple[Mark, ...]
    upstream: str = ""  # fingerprint of the strip's input
    reviewed: str = ""  # when (ISO 8601, UTC)
    viewed: tuple[tuple[float, float], ...] = ()  # spans looked at (s)
    reviewer: str = ""  # who reviewed (asked once per run)
    bads: tuple[str, ...] = ()  # channels the reviewer marked bad
    crop: tuple[float, float | None] | None = None  # kept span (s; review at import)


Span = tuple[float, float]


def merge_spans(spans: Sequence[Sequence[float]]) -> tuple[Span, ...]:
    """Spans ``(start, stop)`` in seconds, sorted, overlapping ones merged."""
    merged: list[Span] = []
    for lo, hi in sorted((float(a), float(b)) for a, b in spans if b > a):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class ReviewAnswer:
    """What the person reviewing gives back: the marks, and the spans they
    looked at (kept in the sidecar, shown as reviewed next time)."""

    marks: tuple[Mark, ...]
    viewed: tuple[Span, ...] = ()
    bads: tuple[str, ...] = ()  # channels marked bad (labels)


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
    reject: tuple[str, ...] = ()  # what the window's Reject key labels a span
    viewed: tuple[Span, ...] = ()  # looked at in the review before
    bads: tuple[str, ...] = ()  # channels marked bad in the review before
    display: dict[str, Any] = dataclasses.field(default_factory=dict)  # presets
    points: tuple[Mark, ...] = ()  # events to step through (review points)

    def to_wire(self) -> dict[str, Any]:
        return {
            "node": self.node_id, "file": str(self.path), "reason": self.reason,
            "marks": [m.to_wire() for m in self.previous],
            "labels": list(self.labels), "duration_s": self.duration_s,
            "reject": list(self.reject), "viewed": [list(v) for v in self.viewed],
            "bads": list(self.bads), "display": dict(self.display),
            "points": [m.to_wire() for m in self.points],
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


ReviewCallback = Callable[[ReviewRequest], "ReviewAnswer | Sequence[Mark]"]


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
                {"name": "points", "type": "string", "default": "",
                 "description": "Comma-separated event labels to step through "
                                "as review points (e.g. a detector's): each is "
                                "selected in turn, to label and retime"},
                {"name": "reject", "type": "string", "default": "BAD",
                 "description": "Comma-separated labels the Reject key gives "
                                "a span (a Reject segments strip after this "
                                "one cuts them out)"},
                {"name": "file", "type": "string", "default": "", "path": "save",
                 "extensions": [".tsv"], "advanced": True,
                 "description": "Where the marks are saved (empty: next to "
                                "each recording, or in its BIDS derivatives)"},
                # the display the window opens with; the reviewer can change it
                {"name": "montage", "type": "enum", "default": REVIEWER,
                 "choices": [REVIEWER, *MONTAGES], "advanced": True,
                 "description": "Display montage to start with"},
                *({"name": key, "type": "enum", "default": REVIEWER,
                   "choices": [REVIEWER, *values], "advanced": True,
                   "description": f"Display {key} to start with "
                                  f"({PRESET_UNITS[key]})"}
                  for key, values in PRESET_CHOICES.items()),
            ],
            **credits(REVIEW_NODE),
        }
    )  # fmt: skip

    def __init__(self) -> None:
        self.marks: tuple[Event, ...] = ()  # set by the batch run from the review
        self.bads: tuple[str, ...] = ()  # channels the review marked bad
        # who reviewed, when, on what input: kept in a session (spec/session.md)
        self.record: dict[str, str] = {}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        lo, hi = block.sample_index, block.stop_index
        inside = tuple(e for e in self.marks if lo <= e.sample_index < hi)
        return TransformResult(with_bads(block, self.bads), {}, inside)


def with_bads(block: Block, bads: Sequence[str]) -> Block:
    """``block`` with ``bads`` added to its stream's bad channels (those it
    has): later strips leave them out as they do the recording's own."""
    info = block.info
    added = [b for b in bads if b in info.ch_labels and b not in info.bads]
    if not added:
        return block
    return dataclasses.replace(
        block, info=dataclasses.replace(info, bads=(*info.bads, *added))
    )


def display_presets(params: Params) -> dict[str, Any]:
    """The display the review window opens with, where the strip sets it (the
    rest is the reviewer's Settings); the reviewer can change all of it."""
    out: dict[str, Any] = {}
    if params["montage"] != REVIEWER:
        out["montage"] = str(params["montage"])
    for key in PRESET_CHOICES:
        value = str(params[key])
        if value != REVIEWER:
            out[key] = 0.0 if value == "off" else float(value)
    return out


def point_words(params: Params) -> tuple[str, ...]:
    return tuple(_words(params["points"]))


def review_points(events: Sequence[Event], params: Params, sfreq: float
                  ) -> tuple[Mark, ...]:  # fmt: skip
    """The events to step through (labels named by ``points``), as spans in
    seconds."""
    words = point_words(params)
    return tuple(Mark(e.sample_index / sfreq, e.duration / sfreq, e.label)
                 for e in sorted(events, key=lambda e: e.sample_index)
                 if words and e.label in words)  # fmt: skip


def _words(text: object) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def rejected_spans(events: Sequence[Event], params: Params) -> list[tuple[int, int]]:
    """The sample spans ``[start, stop)`` to cut: marks whose label starts with
    a ``reject`` word, overlapping ones merged."""
    words = reject_words(params)
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
    blocks: Sequence[Block], spans: Sequence[tuple[int, int]], origin: str = ""
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
    boundaries = tuple(Event(hi, BOUNDARY, hi - lo, {"origin": origin})
                       for lo, hi in spans if first < hi < stop)  # fmt: skip
    return out, boundaries


def reject_words(params: Params) -> tuple[str, ...]:
    return tuple(_words(params["reject"]))


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
                        "label": r.get("trial_type") or r.get("value") or "",
                        "point": r.get("review_point")})
        for r in rows
    )  # fmt: skip
    meta: dict[str, Any] = {}
    if _sidecar(path).is_file():
        meta = json.loads(_sidecar(path).read_text(encoding="utf-8")).get("MeegXer", {})
    viewed = merge_spans(meta.get("viewed", []))
    crop = meta.get("crop")
    return Review(marks, str(meta.get("upstream", "")), str(meta.get("reviewed", "")),
                  viewed, str(meta.get("reviewer", "")),
                  tuple(str(b) for b in meta.get("bads", [])),
                  None if crop is None else (float(crop[0]),
                                             None if crop[1] is None
                                             else float(crop[1])))  # fmt: skip


def save_review(path: Path, review: Review, node_id: str) -> None:
    """Write the events file and its JSON sidecar (the marks in onset order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    marks = sorted(review.marks, key=lambda m: (m.onset, m.label))
    points = any(m.point is not None for m in marks)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["onset", "duration", "trial_type",
                      *(["review_point"] if points else [])])  # fmt: skip
        for m in marks:
            row = [f"{m.onset:.6f}", f"{m.duration:.6f}", m.label]
            if points:
                row.append("n/a" if m.point is None else f"{m.point:.6f}")
            out.writerow(row)
    meta: dict[str, Any] = {
        "onset": {"Description": "Start of the span", "Units": "s"},
        "duration": {"Description": "Length of the span", "Units": "s"},
        "trial_type": {"Description": "Label given in a manual review"},
        "MeegXer": {"strip": node_id, "upstream": review.upstream,
                    "reviewed": review.reviewed, "reviewer": review.reviewer,
                    "bads": list(review.bads),  # channels marked bad
                    # the spans looked at, in seconds from the recording's start
                    "viewed": [[round(a, 6), round(b, 6)] for a, b in review.viewed]},
    }  # fmt: skip
    if review.crop is not None:  # the span kept, in s from the raw file's start
        meta["MeegXer"]["crop"] = list(review.crop)
    if points:
        meta["review_point"] = {"Description": "Onset of the review point (an "
                                "event of the input) the mark answers",
                                "Units": "s"}  # fmt: skip
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
