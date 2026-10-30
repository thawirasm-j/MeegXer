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

Suggestions (step 4b): the events the strip's ``points`` names (a detector's
upstream, or the recording's own) are suggestions. Each gets a verdict,
``accepted``, ``edited`` or ``rejected``, kept with its origin in the marks
file; rejected ones are recorded but not emitted. The strip takes those labels
over: later strips see its verdicts, not the suggestions
(``meegxer.core.events``). With ``decision`` ``verify`` every suggestion needs
a verdict; with ``free`` one left without an answer is rejected (the window
opens each as a mark, so leaving it alone accepts it). A detector's
``bad_channel`` suggestions become bad channels when accepted.
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

from meegxer.core.block import CHANNEL_TYPES, Block, Event, FeatureBlock
from meegxer.core.epoch_review import EpochVerdict
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
BAD_CHANNEL = "bad_channel"  # a channel proposed as bad (payload: channel, reason)
DECISIONS = ("free", "verify")
VERDICTS = ("accepted", "edited", "rejected")


def _number(value: object, name: str) -> float:
    number = float(str(value))
    if number < 0 or not math.isfinite(number):
        raise ValueError(f"{name} must be a number >= 0")
    return number


@dataclass(frozen=True, slots=True)
class Suggestion:
    """An event the review is asked about (a review point): where it is, in
    seconds from the recording's start, and which strip proposed it
    (``input``: the recording's own); ``channel`` for a ``bad_channel``."""

    onset: float
    duration: float
    label: str
    origin: str = ""
    channel: str = ""

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"onset": self.onset, "duration": self.duration,
                               "label": self.label, "origin": self.origin}  # fmt: skip
        if self.channel:
            out["channel"] = self.channel
        return out

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Suggestion:
        return cls(_number(data["onset"], "onset"),
                   _number(data.get("duration", 0.0), "duration"),
                   str(data["label"]), str(data.get("origin") or ""),
                   str(data.get("channel") or ""))  # fmt: skip

    def same(self, other: Suggestion) -> bool:
        """The same suggestion (times to the microsecond)."""
        return (
            (self.label, self.origin, self.channel)
            == (other.label, other.origin, other.channel)
            and _close(self.onset, other.onset)
            and _close(self.duration, other.duration)
        )


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


@dataclass(frozen=True, slots=True)
class Mark:
    """A labelled span, in seconds from the recording's start; for the answer
    to a suggestion, the suggestion and its verdict."""

    onset: float
    duration: float
    label: str
    suggestion: Suggestion | None = None
    verdict: str = ""  # accepted, edited or rejected (a rejected one: not emitted)

    @property
    def point(self) -> float | None:
        """Onset of the suggestion it answers (s)."""
        return None if self.suggestion is None else self.suggestion.onset

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"onset": self.onset, "duration": self.duration,
                               "label": self.label}  # fmt: skip
        if self.suggestion is not None:
            out["suggestion"] = self.suggestion.to_wire()
        if self.verdict:
            out["verdict"] = self.verdict
        return out

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Mark:
        onset, duration = float(data["onset"]), float(data.get("duration", 0.0))
        label = str(data["label"]).strip()
        if not label:
            raise ValueError("a mark needs a label")
        if onset < 0 or duration < 0 or not math.isfinite(onset + duration):
            raise ValueError(f"mark {label!r}: onset and duration must be >= 0")
        given = data.get("suggestion")
        suggestion = Suggestion.from_wire(given) if isinstance(given, dict) else None
        point = data.get("point")  # before 0.2.17: the onset of the point only
        if suggestion is None and point not in (None, "", "n/a"):
            suggestion = Suggestion(_number(point, "point"), 0.0, label)
        verdict = str(data.get("verdict") or "")
        if verdict not in ("", *VERDICTS):
            raise ValueError(f"mark {label!r}: verdict must be one of {VERDICTS}")
        return cls(onset, duration, label, suggestion, verdict)


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
    epochs: tuple[EpochVerdict, ...] = ()  # an epoch review's verdicts
    rejected: tuple[Suggestion, ...] = ()  # suggestions answered no (also: a
    # mark with verdict "rejected" and its suggestion, as the wire gives them)


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
    points: tuple[Suggestion, ...] = ()  # suggestions to step through
    decision: str = "free"  # or "verify": every suggestion needs a verdict
    # an epoch review (meegxer.core.epoch_review): the epochs to review, and
    # their data (not sent: the window fetches it, offline_review_epochs)
    kind: str = "marks"  # or "epochs"
    epochs: tuple[EpochVerdict, ...] = ()
    windows: tuple[FeatureBlock, ...] = dataclasses.field(default=(), compare=False,
                                                          repr=False)  # fmt: skip

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "epochs": [v.to_wire() for v in self.epochs],
            "node": self.node_id, "file": str(self.path), "reason": self.reason,
            "marks": [m.to_wire() for m in self.previous],
            "labels": list(self.labels), "duration_s": self.duration_s,
            "reject": list(self.reject), "viewed": [list(v) for v in self.viewed],
            "bads": list(self.bads), "display": dict(self.display),
            "points": [m.to_wire() for m in self.points],
            "decision": self.decision,
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


ReviewCallback = Callable[
    [ReviewRequest], "ReviewAnswer | Sequence[Mark | EpochVerdict]"
]


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
                 "description": "Comma-separated event labels to review as "
                                "suggestions (e.g. a detector's; bad_channel "
                                "for its proposed bad channels): each gets a "
                                "verdict, and later strips see the verdicts "
                                "instead"},
                {"name": "decision", "type": "enum", "default": "free",
                 "choices": list(DECISIONS),
                 "description": "free: suggestions open as marks to keep, edit "
                                "or delete; verify: yes or no for each "
                                "suggestion, no other marks"},
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

    @staticmethod
    def takes_over(params: Params) -> tuple[str, ...]:
        """The labels it reviews as suggestions: later strips see its verdicts
        for them, not the events from before it (``meegxer.core.events``)."""
        return point_words(params)


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


def review_points(events: Sequence[tuple[Event, str]], params: Params,
                  sfreq: float) -> tuple[Suggestion, ...]:  # fmt: skip
    """The suggestions: the events (with their origins) whose labels
    ``points`` names, as spans in seconds."""
    words = point_words(params)
    return tuple(
        Suggestion(round(e.sample_index / sfreq, 6), round(e.duration / sfreq, 6),
                   e.label, origin, str(e.payload.get("channel") or ""))
        for e, origin in sorted(events, key=lambda p: p[0].sample_index)
        if words and e.label in words
    )  # fmt: skip


class VerdictsMissing(ValueError):
    """A verify review left suggestions without a verdict."""


def settle(
    suggestions: Sequence[Suggestion],
    answer: ReviewAnswer,
    decision: str,
) -> tuple[tuple[Mark, ...], tuple[str, ...]]:
    """The marks to save (each suggestion's verdict among them, rejected ones
    as rows of their own) and the bad channels, from the reviewer's answer.

    A mark answering a suggestion is ``accepted`` when its span and label are
    the suggestion's, else ``edited``; a mark answering a suggestion that is
    no longer there is kept as the reviewer's own. A ``bad_channel``
    suggestion is accepted when its channel is among the answer's bad
    channels. A suggestion with no answer is rejected with ``free``; with
    ``verify`` it is an error (:class:`VerdictsMissing`)."""
    rejections = [
        *answer.rejected,
        *(
            m.suggestion
            for m in answer.marks
            if m.verdict == "rejected" and m.suggestion
        ),
    ]
    own: list[Mark] = []
    answered: list[Mark] = []
    for m in answer.marks:
        given = m.suggestion
        s = next((s for s in suggestions if given and s.same(given)), None)
        if m.verdict == "rejected":
            continue  # a rejection (in rejections)
        if s is None or s.label == BAD_CHANNEL:
            own.append(Mark(m.onset, m.duration, m.label))
            continue
        same = (m.label == s.label and _close(m.onset, s.onset)
                and _close(m.duration, s.duration))  # fmt: skip
        answered.append(Mark(m.onset, m.duration, m.label, s,
                             "accepted" if same else "edited"))  # fmt: skip
    missing = []
    for s in suggestions:
        if s.label == BAD_CHANNEL and s.channel in answer.bads:
            answered.append(Mark(s.onset, 0.0, s.label, s, "accepted"))
            continue
        if any(a.suggestion is s for a in answered):
            continue
        if decision == "verify" and not any(s.same(r) for r in rejections):
            missing.append(s)
            continue
        answered.append(Mark(s.onset, s.duration, s.label, s, "rejected"))
    if missing:
        raise VerdictsMissing(
            f"{len(missing)} suggestion(s) have no verdict (decision verify): "
            + ", ".join(f"{s.label} at {s.onset:g} s" for s in missing[:5])
        )
    return tuple(own + answered), tuple(answer.bads)


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


def _given(row: dict[str, str], key: str) -> str:
    value = row.get(key) or ""
    return "" if value == "n/a" else value


def _row_mark(row: dict[str, str]) -> Mark:
    """A mark from a row of the marks file (its suggestion and verdict when it
    answers one; ``review_point``: the onset of the point, before step 4b)."""
    label = row.get("trial_type") or row.get("value") or ""
    data: dict[str, Any] = {"onset": row["onset"],
                            "duration": row.get("duration") or 0.0,
                            "label": label, "point": _given(row, "review_point"),
                            "verdict": _given(row, "verdict")}  # fmt: skip
    if _given(row, "suggestion_onset"):
        data["suggestion"] = {
            "onset": row["suggestion_onset"],
            "duration": _given(row, "suggestion_duration") or 0.0,
            "label": _given(row, "suggestion_label") or label,
            "origin": _given(row, "suggestion_origin"),
            "channel": _given(row, "channel"),
        }
    return Mark.from_wire(data)


def load_review(path: Path) -> Review | None:
    """The review saved at ``path`` (None when there is none)."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    marks = tuple(_row_mark(r) for r in rows)
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
    points = any(m.suggestion is not None for m in marks)
    columns = ["channel", "suggestion_origin", "suggestion_label",
               "suggestion_onset", "suggestion_duration", "verdict"]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["onset", "duration", "trial_type",
                      *(columns if points else [])])  # fmt: skip
        for m in marks:
            row = [f"{m.onset:.6f}", f"{m.duration:.6f}", m.label]
            s = m.suggestion
            if points and s is None:
                row += ["n/a"] * len(columns)
            elif s is not None:
                row += [
                    s.channel or "n/a",
                    s.origin or "n/a",
                    s.label,
                    f"{s.onset:.6f}",
                    f"{s.duration:.6f}",
                    m.verdict or "n/a",
                ]
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
    if points:  # the suggestion each row answers (docs/REVIEW.md, step 4b)
        meta.update({
            "channel": {"Description": "Channel a bad_channel suggestion names"},
            "suggestion_origin": {"Description": "Strip that proposed the "
                                  "suggestion (input: the recording's own event)"},
            "suggestion_label": {"Description": "Label of the suggestion"},
            "suggestion_onset": {"Description": "Start of the suggestion",
                                 "Units": "s"},
            "suggestion_duration": {"Description": "Length of the suggestion",
                                    "Units": "s"},
            "verdict": {"Description": "The reviewer's verdict on the "
                        "suggestion; rejected rows are not used as events",
                        "Levels": {"accepted": "kept as proposed",
                                   "edited": "kept, retimed or relabelled",
                                   "rejected": "not an event"}},
        })  # fmt: skip
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
        if m.verdict != "rejected" and m.label != BAD_CHANNEL  # not events
    )  # fmt: skip
