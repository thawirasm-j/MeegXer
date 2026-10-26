"""Recording overrides on a pipeline's source: channel names, types, bad flags,
positions (a montage or per channel) and events.

A source node may carry a ``recording`` object in the pipeline file (spec 0.2.3).
The overrides change how the stream is described, never its samples: renamed
labels and bad flags reach every downstream node (and so the fingerprints),
types and positions travel in ``SignalInfo``. Events (from the recording or an
events file, renamed or dropped) are described for clients; routing them through
the pipeline comes with the Events port work.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from meegxer.core.block import CHANNEL_TYPES, Block, Event, Position, SignalInfo
from meegxer.core.node import Source

_EVENT_LABEL_COLUMNS = ("trial_type", "value", "label", "description", "type")


class RecordingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChannelEdit:
    rename: str = ""
    type: str = ""
    bad: bool | None = None
    position: Position | None = None


@dataclass(frozen=True, slots=True)
class RecordingOverrides:
    """``channels`` is keyed by the label the source reports."""

    channels: Mapping[str, ChannelEdit] = field(default_factory=dict)
    montage: str = ""
    events_file: str = ""
    rename_events: Mapping[str, str] = field(default_factory=dict)
    drop_events: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordingOverrides:
        channels: dict[str, ChannelEdit] = {}
        for label, edit in data.get("channels", {}).items():
            ch_type = str(edit.get("type", ""))
            if ch_type and ch_type not in CHANNEL_TYPES:
                raise RecordingError(f"{label}: unknown channel type {ch_type!r}")
            pos = edit.get("position")
            channels[label] = ChannelEdit(
                str(edit.get("rename", "")),
                ch_type,
                edit.get("bad"),
                None if pos is None else (float(pos[0]), float(pos[1]), float(pos[2])),
            )
        events = data.get("events", {})
        return cls(
            channels,
            str(data.get("montage", "")),
            str(events.get("file", "")),
            dict(events.get("rename", {})),
            tuple(events.get("drop", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        channels: dict[str, Any] = {}
        for label, edit in self.channels.items():
            entry: dict[str, Any] = {}
            if edit.rename:
                entry["rename"] = edit.rename
            if edit.type:
                entry["type"] = edit.type
            if edit.bad is not None:
                entry["bad"] = edit.bad
            if edit.position is not None:
                entry["position"] = list(edit.position)
            if entry:
                channels[label] = entry
        if channels:
            out["channels"] = channels
        if self.montage:
            out["montage"] = self.montage
        events: dict[str, Any] = {}
        if self.events_file:
            events["file"] = self.events_file
        if self.rename_events:
            events["rename"] = dict(self.rename_events)
        if self.drop_events:
            events["drop"] = list(self.drop_events)
        if events:
            out["events"] = events
        return out


# -- montages ---------------------------------------------------------------------


def builtin_montages() -> list[str]:
    import mne

    return sorted(mne.channels.get_builtin_montages())


def _electrodes_tsv(path: Path) -> dict[str, Position]:
    """BIDS ``electrodes.tsv`` (name, x, y, z); units guessed from head size."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r.get("x") not in (None, "", "n/a")]  # fmt: skip
    if not rows:
        raise RecordingError(f"{path}: no electrode coordinates")
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    radius = float(np.median(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)))
    scale = 1.0 if radius < 0.5 else 0.01 if radius < 30 else 0.001  # m, cm, mm
    return {r["name"]: (x * scale, y * scale, z * scale)
            for r, (x, y, z) in zip(rows, xyz, strict=True)}  # fmt: skip


def _eeglab_ced(path: Path) -> dict[str, Position]:
    """EEGLAB ``.ced`` channel locations (labels, X, Y, Z with X towards the nose
    and Y towards the left ear), turned to x right, y front, z up."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    keys = {k.strip().casefold(): k for k in (rows[0] if rows else {})}
    if not {"labels", "x", "y", "z"} <= set(keys):
        raise RecordingError(f"{path}: needs labels, X, Y and Z columns")
    named = []
    for row in rows:
        try:
            xyz = [float(row[keys[c]]) for c in ("x", "y", "z")]
        except (TypeError, ValueError):
            continue  # a channel without coordinates
        named.append((row[keys["labels"]].strip(), (-xyz[1], xyz[0], xyz[2])))
    if not named:
        raise RecordingError(f"{path}: no channel coordinates")
    xyz_all = np.array([p for _, p in named])
    radius = float(np.median(np.linalg.norm(xyz_all - xyz_all.mean(axis=0), axis=1)))
    scale = 1.0 if radius < 0.5 else 0.01 if radius < 30 else 0.001  # m, cm, mm
    return {name: (x * scale, y * scale, z * scale) for name, (x, y, z) in named}


@cache
def montage_positions(spec: str) -> dict[str, Position]:
    """Electrode positions of a standard montage (by name) or a montage file
    (.elc, .sfp, .loc, .bvef, ... via MNE, or a BIDS electrodes.tsv)."""
    import mne

    path = Path(spec)
    if not path.suffix:  # a standard montage, including names MNE has renamed
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                montage = mne.channels.make_standard_montage(spec)
        except ValueError as exc:
            raise RecordingError(f"no standard montage named {spec!r}") from exc
    else:
        if not path.is_file():
            raise RecordingError(f"montage file not found: {spec}")
        if path.name.endswith("electrodes.tsv"):
            return _electrodes_tsv(path)
        if path.suffix.lower() == ".ced":
            return _eeglab_ced(path)
        montage = mne.channels.read_custom_montage(path)
    return {name: (float(p[0]), float(p[1]), float(p[2]))
            for name, p in montage.get_positions()["ch_pos"].items()}  # fmt: skip


# How channel labels are often written in files, around a 10-05 name: "EEG Fp1-REF",
# "FP1-LE", "EEG_O1_A2". Bipolar labels ("Fp1-F7") name two sites and never match.
_TYPE_PREFIX = re.compile(r"^(EEG|EOG|ECG|EKG|EMG)[\s_:-]+", re.IGNORECASE)
_REF_SUFFIX = re.compile(r"[\s_-]+(REF|LE|RE|AR|AVG|AVERAGE|LINKED|A1|A2|M1|M2)$",
                         re.IGNORECASE)  # fmt: skip
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def label_candidates(label: str) -> list[str]:
    """Names to try for a channel label, most literal first: the label, then
    without a type prefix or reference suffix, then the 10-20 name for old ones
    (T3/T4/T5/T6)."""
    out = [label.strip()]
    bare = _REF_SUFFIX.sub("", _TYPE_PREFIX.sub("", out[0])).strip()
    if bare and bare not in out:
        out.append(bare)
    for name in list(out):
        new = OLD_NAMES.get(name.upper())
        if new and new not in out:
            out.append(new)
    return out


def _lookup(table: Mapping[str, Position], label: str) -> Position | None:
    """The position of ``label`` in a montage: exact, then case-insensitive, then
    the cleaned-up candidates (see ``label_candidates``)."""
    folded = {k.casefold(): v for k, v in table.items()}
    for name in label_candidates(label):
        if name in table:
            return table[name]
        if name.casefold() in folded:
            return folded[name.casefold()]
    return None


def lookup_positions(
    labels: list[str], montage: str
) -> tuple[dict[str, Position], list[str]]:
    """Look channel labels up in a standard montage or a montage file (like
    EEGLAB's channel-location lookup): ``(positions by label, unmatched labels)``."""
    table = montage_positions(montage)
    found: dict[str, Position] = {}
    for label in labels:
        pos = _lookup(table, label)
        if pos is not None:
            found[label] = pos
    return found, [label for label in labels if label not in found]


# -- events -----------------------------------------------------------------------


def read_events_file(path: str | Path) -> list[dict[str, float | str]]:
    """Events from a BIDS events.tsv or a CSV/TSV with onset (s), optional
    duration (s) and a label column (trial_type, value, label, ...)."""
    p = Path(path)
    if not p.is_file():
        raise RecordingError(f"events file not found: {p}")
    text = p.read_text(encoding="utf-8")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows or "onset" not in rows[0]:
        raise RecordingError(f"{p}: needs an 'onset' column")
    column = next((c for c in _EVENT_LABEL_COLUMNS if c in rows[0]), None)
    events: list[dict[str, float | str]] = []
    for row in rows:
        duration = row.get("duration") or "0"
        events.append({
            "onset": float(row["onset"]),
            "duration": 0.0 if duration in ("n/a", "") else float(duration),
            "label": str(row[column]) if column else "event",
        })  # fmt: skip
    return events


# -- applying overrides -----------------------------------------------------------


def apply_overrides(info: SignalInfo, overrides: RecordingOverrides) -> SignalInfo:
    """The stream description after the overrides (same channels, same order)."""
    unknown = sorted(set(overrides.channels) - set(info.ch_labels))
    if unknown:
        raise RecordingError(f"overrides name channels the input lacks: {unknown}")
    montage = montage_positions(overrides.montage) if overrides.montage else {}
    labels, types, bads, positions = [], [], [], []
    for i, label in enumerate(info.ch_labels):
        edit = overrides.channels.get(label, ChannelEdit())
        new = edit.rename or label
        labels.append(new)
        types.append(edit.type or (info.ch_types[i] if info.ch_types else ""))
        if (label in info.bads) if edit.bad is None else edit.bad:
            bads.append(new)
        old_pos = info.positions[i] if info.positions else None
        positions.append(edit.position or _lookup(montage, new)
                         or _lookup(montage, label) or old_pos)  # fmt: skip
    return SignalInfo(
        tuple(labels),
        info.sfreq,
        info.stream_id,
        tuple(bads),
        tuple(types) if any(types) else (),
        tuple(positions) if any(p is not None for p in positions) else (),
    )


class _RecordingMixin(Source):
    """Mixed into a source class: the same samples, described by the overrides."""

    recording: RecordingOverrides

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._described: tuple[SignalInfo, SignalInfo] | None = None

    @property
    def source_info(self) -> SignalInfo:
        """The description as the source reports it, before the overrides."""
        return super().info  # type: ignore[safe-super]

    @property
    def info(self) -> SignalInfo:
        base = super().info  # type: ignore[safe-super]
        if self._described is None or self._described[0] is not base:
            self._described = (base, apply_overrides(base, self.recording))
        return self._described[1]

    def read(self, max_samples: int) -> Block | None:
        block = super().read(max_samples)  # type: ignore[safe-super]
        if block is None:
            return None
        return dataclasses.replace(block, info=self.info)


@cache
def _recording_class(cls: type[Source]) -> type[Source]:
    return type(cls.__name__, (_RecordingMixin, cls), {"__doc__": cls.__doc__})


def with_recording(
    cls: type[Source], params: Mapping[str, object], overrides: RecordingOverrides
) -> Source:
    """An instance of source ``cls`` whose stream carries the overrides."""
    source = _recording_class(cls)(params)
    assert isinstance(source, _RecordingMixin)
    source.recording = overrides
    return source


def recording_of(source: Source) -> dict[str, Any]:
    """The overrides of a source as pipeline JSON (empty when it has none)."""
    return source.recording.to_dict() if isinstance(source, _RecordingMixin) else {}


def _overrides(source: Source) -> RecordingOverrides:
    return (
        source.recording
        if isinstance(source, _RecordingMixin)
        else RecordingOverrides()
    )


def _raw_events(
    source: Source, overrides: RecordingOverrides
) -> tuple[list[dict[str, float | str]], str]:
    """The recording's events before renaming, and where they came from."""
    if overrides.events_file:
        return read_events_file(overrides.events_file), "file"
    if callable(getattr(source, "annotations", None)):
        events = source.annotations()  # type: ignore[attr-defined]
        reviewed = getattr(source, "import_reviewed", False)  # docs/REVIEW.md
        return events, "import review" if reviewed else "recording"
    return [], "none"


def _renamed(
    events: list[dict[str, float | str]], overrides: RecordingOverrides
) -> list[dict[str, float | str]]:
    return [
        {**e, "label": overrides.rename_events.get(str(e["label"]), str(e["label"]))}
        for e in events
        if str(e["label"]) not in overrides.drop_events
    ]


def recording_events(source: Source) -> tuple[Event, ...]:
    """The recording's events as the pipeline sees them (events file or the
    file's own annotations, renamed and dropped as the overrides say), at
    absolute sample indices: onset (s) x rate from the source's first sample."""
    overrides = _overrides(source)
    sfreq = source.info.sfreq
    events = []
    for e in _renamed(_raw_events(source, overrides)[0], overrides):
        onset = float(e["onset"])
        if onset < 0 or not str(e["label"]):
            continue
        duration = max(0, round(float(e.get("duration", 0.0)) * sfreq))
        events.append(Event(round(onset * sfreq), str(e["label"]), duration))
    return tuple(sorted(events, key=lambda e: e.sample_index))


def describe(source: Source) -> dict[str, Any]:
    """What a client needs to show and edit a recording: channels before and
    after the overrides, the sampling rate and length, and the events."""
    info = source.info
    original = source.source_info if isinstance(source, _RecordingMixin) else info
    overrides = _overrides(source)
    channels = [
        {
            "label": label,
            "original": original.ch_labels[i],
            "type": info.ch_types[i] if info.ch_types else "",
            "bad": label in info.bads,
            "position": list(p)
            if info.positions and (p := info.positions[i])
            else None,
        }
        for i, label in enumerate(info.ch_labels)
    ]
    events, origin = _raw_events(source, overrides)
    labels = sorted({str(e["label"]) for e in events})
    shown = _renamed(events, overrides)
    n = source.n_samples
    return {
        "sfreq": info.sfreq,
        "n_samples": n,
        "duration_s": None if n is None else n / info.sfreq,
        "stream_id": info.stream_id,
        "channels": channels,
        "events": shown,
        "event_labels": labels,
        "events_origin": origin,
        "recording": overrides.to_dict(),
    }
