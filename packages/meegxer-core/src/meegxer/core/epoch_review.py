"""Epoch review (docs/REVIEW.md, Design v2 step 5): a person accepts, rejects or
relabels each epoch of an Epochs strip.

The strip sits after an Epochs strip. It passes the signal on unchanged, reads
the ``epochs`` rail and writes its own ``epochs`` rail: the epochs kept, with
the labels the reviewer gave. Later strips read the nearest ``epochs`` rail
(docs/RAILS.md rule 7), so they get the reviewed epochs; the Epochs strip's own
rail is left as it was (rule 8: each rail has one producer).

The verdicts live in a sidecar per recording, as for the Review strip: a BIDS
events file (``onset`` and ``duration``: the epoch's span in seconds from the
recording's start; ``trial_type``: the label given; ``event``: the label it was
cut around; ``epoch_status``: ``good`` or ``bad``) with a JSON file naming the
input they were made on. A batch run reuses them when the input (the signal and
the epochs) is the same; otherwise the run asks for a review, offering the
verdicts made before for the epochs at the same place (rule 7). Every epoch is
reviewed: one the reviewer did not touch is kept as it was cut.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from meegxer.core.block import CHANNEL_TYPES, Block, FeatureBlock
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, RailInput, TransformResult
from meegxer.core.references import credits
from meegxer.core.serialization import State

EPOCH_REVIEW_NODE = "meegxer.core.epoch_review"
EPOCHS_RAIL = "epochs"
GOOD, BAD = "good", "bad"


@dataclass(frozen=True, slots=True)
class EpochVerdict:
    """One epoch as reviewed: its span (seconds from the recording's start),
    the label given, kept or rejected, and the label it was cut around."""

    onset: float
    duration: float
    label: str
    keep: bool = True
    event: str = ""

    def to_wire(self) -> dict[str, Any]:
        return {"onset": self.onset, "duration": self.duration, "label": self.label,
                "keep": self.keep, "event": self.event}  # fmt: skip

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> EpochVerdict:
        label = str(data["label"]).strip()
        if not label:
            raise ValueError("an epoch needs a label")
        return cls(float(data["onset"]), float(data.get("duration", 0.0)), label,
                   bool(data.get("keep", True)),
                   str(data.get("event", "")) or label)  # fmt: skip


@dataclass(frozen=True, slots=True)
class EpochReviewRecord:
    """The verdicts of one recording and what they were made on."""

    epochs: tuple[EpochVerdict, ...]
    upstream: str = ""
    reviewed: str = ""
    reviewer: str = ""


class EpochReview(Processor):
    """Passes the signal on; writes the reviewed epochs to its ``epochs`` rail.
    Offline only: a person reviews a whole recording's epochs."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.20",
            "name": EPOCH_REVIEW_NODE,
            "version": "0.1.0",
            "type": "stream",
            "description": "Review epochs by hand: accept, reject or relabel "
                           "each epoch of the Epochs strip before it.",
            "availability": "offline",
            "inputs": [{"name": "in", "type": "signal"},
                       {"name": "epochs", "type": "features", "rail": EPOCHS_RAIL}],
            "outputs": [{"name": "out", "type": "signal"},
                        {"name": "epochs", "type": "features",
                         "rail": EPOCHS_RAIL}],
            "channel_types": list(CHANNEL_TYPES),
            "params": [
                {"name": "labels", "type": "string", "default": "",
                 "primary": True,
                 "description": "Comma-separated labels offered for epochs "
                                "(the epochs' own labels are offered too)"},
                {"name": "file", "type": "string", "default": "", "path": "save",
                 "extensions": [".tsv"], "advanced": True,
                 "description": "Where the verdicts are saved (empty: next to "
                                "each recording, or in its BIDS derivatives)"},
            ],
            **credits(EPOCH_REVIEW_NODE),
        }
    )  # fmt: skip

    def __init__(self) -> None:
        # set by the batch run from the review: epoch start sample -> verdict
        self.verdicts: dict[int, EpochVerdict] = {}
        # who reviewed, when, on what input: kept in a session (spec/session.md)
        self.record: dict[str, str] = {}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return self.transform_rails(block, model, state, params, {"epochs": ()})

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        epochs = rails.get("epochs", ())
        assert isinstance(epochs, tuple)
        windows = [w for w in epochs if isinstance(w, FeatureBlock)]
        kept = reviewed(windows, self.verdicts)
        return TransformResult(block, {}, (), {"epochs": kept})


def epoch_label(window: FeatureBlock) -> str:
    labels = window.coords.get("epoch", ())
    return str(labels[0]) if len(labels) else ""


def reviewed(
    windows: Sequence[FeatureBlock], verdicts: Mapping[int, EpochVerdict]
) -> tuple[FeatureBlock, ...]:
    """``windows`` without the rejected epochs, the others with the label given
    (by the epoch's first sample; one with no verdict stays as it is)."""
    out: list[FeatureBlock] = []
    for w in windows:
        verdict = verdicts.get(w.window[0])
        if verdict is None:
            out.append(w)
        elif verdict.keep:
            coords = {**w.coords, "epoch": (verdict.label,)}
            out.append(dataclasses.replace(w, coords=coords))
    return tuple(out)


def epochs_to_review(
    windows: Sequence[FeatureBlock], sfreq: float, before: Sequence[EpochVerdict] = ()
) -> tuple[EpochVerdict, ...]:
    """The epochs of ``windows`` to review, in time order: each with the
    verdict made before for an epoch at the same span (same label cut around),
    else kept with its own label."""
    previous = {(round(v.onset * sfreq), v.event): v for v in before}
    out: list[EpochVerdict] = []
    for w in sorted(windows, key=lambda w: w.window[0]):
        lo, hi = w.window
        event = epoch_label(w)
        old = previous.get((lo, event))
        out.append(EpochVerdict(lo / sfreq, (hi - lo) / sfreq,
                                old.label if old else event,
                                old.keep if old else True, event))  # fmt: skip
    return tuple(out)


def verdicts_by_sample(
    verdicts: Sequence[EpochVerdict], sfreq: float
) -> dict[int, EpochVerdict]:
    return {round(v.onset * sfreq): v for v in verdicts}


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def load_epoch_review(path: Path) -> EpochReviewRecord | None:
    """The verdicts saved at ``path`` (None when there are none)."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    epochs = tuple(
        EpochVerdict.from_wire({"onset": r["onset"],
                                "duration": r.get("duration") or 0.0,
                                "label": r.get("trial_type") or "",
                                "keep": (r.get("epoch_status") or GOOD) != BAD,
                                "event": r.get("event") or ""})
        for r in rows
    )  # fmt: skip
    meta: dict[str, Any] = {}
    if _sidecar(path).is_file():
        meta = json.loads(_sidecar(path).read_text(encoding="utf-8")).get("MeegXer", {})
    return EpochReviewRecord(epochs, str(meta.get("upstream", "")),
                             str(meta.get("reviewed", "")),
                             str(meta.get("reviewer", "")))  # fmt: skip


def save_epoch_review(path: Path, record: EpochReviewRecord, node_id: str) -> None:
    """Write the events file and its JSON sidecar (the epochs in time order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["onset", "duration", "trial_type", "event", "epoch_status"])
        for v in sorted(record.epochs, key=lambda v: v.onset):
            out.writerow([f"{v.onset:.6f}", f"{v.duration:.6f}", v.label, v.event,
                          GOOD if v.keep else BAD])  # fmt: skip
    meta = {
        "onset": {"Description": "Start of the epoch", "Units": "s"},
        "duration": {"Description": "Length of the epoch", "Units": "s"},
        "trial_type": {"Description": "Label given in an epoch review"},
        "event": {"Description": "Label of the event the epoch was cut around"},
        "epoch_status": {"Description": "Kept or rejected in an epoch review",
                         "Levels": {GOOD: "kept", BAD: "rejected"}},
        "MeegXer": {"strip": node_id, "upstream": record.upstream,
                    "reviewed": record.reviewed, "reviewer": record.reviewer},
    }  # fmt: skip
    _sidecar(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
