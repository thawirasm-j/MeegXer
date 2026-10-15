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
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from meegxer.core.block import Block, Position, SignalInfo
from meegxer.core.node import Source

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "stim", "misc")
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
        montage = mne.channels.read_custom_montage(path)
    return {name: (float(p[0]), float(p[1]), float(p[2]))
            for name, p in montage.get_positions()["ch_pos"].items()}  # fmt: skip


def _lookup(table: Mapping[str, Position], label: str) -> Position | None:
    if label in table:
        return table[label]
    folded = {k.casefold(): v for k, v in table.items()}
    return folded.get(label.casefold())


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


def describe(source: Source) -> dict[str, Any]:
    """What a client needs to show and edit a recording: channels before and
    after the overrides, the sampling rate and length, and the events."""
    info = source.info
    original = source.source_info if isinstance(source, _RecordingMixin) else info
    overrides = (
        source.recording
        if isinstance(source, _RecordingMixin)
        else RecordingOverrides()
    )
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
    if overrides.events_file:
        events, origin = read_events_file(overrides.events_file), "file"
    elif callable(getattr(source, "annotations", None)):
        events, origin = source.annotations(), "recording"  # type: ignore[attr-defined]
    else:
        events, origin = [], "none"
    labels = sorted({str(e["label"]) for e in events})
    shown = [
        {**e, "label": overrides.rename_events.get(str(e["label"]), str(e["label"]))}
        for e in events
        if str(e["label"]) not in overrides.drop_events
    ]
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
