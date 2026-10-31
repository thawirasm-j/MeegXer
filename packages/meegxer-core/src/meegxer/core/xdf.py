"""XDF files (Lab Streaming Layer, LabRecorder) read with pyxdf (M10.5).

An XDF file holds every stream recorded together: one signal stream becomes the
recording, its marker streams become its events (``docs/RAILS.md``).

- **The signal:** the stream named by ``stream``, else the first stream of type
  EEG, else the first numeric stream with a sampling rate. Its samples are
  indexed in the order they came (rule 8: counted on the source's clock), at
  its nominal rate.
- **Units:** each channel's ``unit`` in the stream's description (microvolts,
  µV, uV, millivolts, mV, volts, V) turns it into volts. A channel without
  one is taken to be in ``unit`` (µV by default, as the LSL input's), and the
  notes say so.
- **Markers:** every stream without a sampling rate (or those named in
  ``markers``), each marker on the signal sample nearest its timestamp (both
  streams on one clock: pyxdf's clock synchronisation and dejittering), as
  the LSL input places live markers. Markers outside the recording are left
  out (the notes count them).
- **Gaps:** pyxdf dejitters each stretch of the signal between breaks
  (timestamps jumping by more than 1 s or 500 sample periods, its defaults).
  At each break (samples were lost) a ``BAD_gap`` mark spans the first sample
  after it, so an epoch across it is left out (the Epochs strip rejects
  ``BAD`` marks). Shorter losses are not told apart from the timestamps'
  jitter, and dejittering absorbs them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mne

GAP = "BAD_gap"
# units as XDF files state them -> volts per unit
UNITS = {"microvolts": 1e-6, "microvolt": 1e-6, "uv": 1e-6, "µv": 1e-6,
         "μv": 1e-6, "millivolts": 1e-3, "millivolt": 1e-3, "mv": 1e-3,
         "volts": 1.0, "volt": 1.0, "v": 1.0}  # fmt: skip
ASSUMED_UNITS = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}
# LSL channel types -> MNE's (others: misc)
TYPES = {"eeg": "eeg", "eog": "eog", "ecg": "ecg", "ekg": "ecg", "emg": "emg",
         "resp": "resp", "misc": "misc", "stim": "stim", "trigger": "stim"}  # fmt: skip
NUMERIC = ("float32", "double64", "int8", "int16", "int32", "int64")


@dataclass
class XdfRecording:
    """The recording read from an XDF file, and what was decided reading it."""

    raw: mne.io.BaseRaw
    stream: str
    marker_streams: tuple[str, ...]
    notes: list[str] = field(default_factory=list)


def _first(info: Mapping[str, Any], key: str, default: str = "") -> str:
    value = info.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return default if value is None else str(value)


def _channels(info: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    desc = info.get("desc")
    desc = desc[0] if isinstance(desc, list) and desc else desc
    if not isinstance(desc, Mapping):
        return []
    chans = desc.get("channels")
    chans = chans[0] if isinstance(chans, list) and chans else chans
    if not isinstance(chans, Mapping):
        return []
    found = chans.get("channel") or []
    return [c for c in found if isinstance(c, Mapping)]


def _name(stream: Mapping[str, Any]) -> str:
    return _first(stream["info"], "name")


def _is_signal(stream: Mapping[str, Any]) -> bool:
    info = stream["info"]
    return (float(_first(info, "nominal_srate", "0")) > 0
            and _first(info, "channel_format") in NUMERIC
            and len(stream["time_stamps"]) > 0)  # fmt: skip


def _is_marker(stream: Mapping[str, Any]) -> bool:
    return float(_first(stream["info"], "nominal_srate", "0")) == 0


def _words(text: str) -> list[str]:
    return [w.strip() for w in str(text).split(",") if w.strip()]


def pick_signal(streams: Sequence[Mapping[str, Any]], name: str = "") -> int:
    """The signal stream: by name, else the first EEG one, else the first
    numeric stream with a rate."""
    signals = [i for i, s in enumerate(streams) if _is_signal(s)]
    if name:
        named = [i for i in signals if _name(streams[i]) == name]
        if not named:
            have = ", ".join(_name(streams[i]) for i in signals) or "none"
            raise ValueError(f"no signal stream {name!r} in the file (it has: {have})")
        return named[0]
    eeg = [i for i in signals if _first(streams[i]["info"], "type").lower() == "eeg"]
    if eeg:
        return eeg[0]
    if signals:
        return signals[0]
    raise ValueError("the XDF file has no signal stream (none with a sampling rate)")


def nearest_samples(stamps: np.ndarray, at: np.ndarray, sfreq: float) -> np.ndarray:
    """The index of the signal sample nearest each timestamp in ``at``; -1 for
    one more than half a sample period outside the recording."""
    stamps = np.asarray(stamps, dtype=float)
    at = np.asarray(at, dtype=float)
    right = np.clip(np.searchsorted(stamps, at), 1, len(stamps) - 1)
    left = right - 1
    nearer = np.where(at - stamps[left] <= stamps[right] - at, left, right)
    if len(stamps) == 1:
        nearer = np.zeros(len(at), dtype=int)
    half = 0.5 / sfreq
    outside = (at < stamps[0] - half) | (at > stamps[-1] + half)
    return np.where(outside, -1, nearer).astype(int)


def gaps(stamps: np.ndarray, sfreq: float) -> list[tuple[int, float]]:
    """Where samples were lost: (index of the first sample after the gap, the
    seconds missing), for each jump over one and a half sample periods."""
    step = np.diff(np.asarray(stamps, dtype=float))
    jumps = np.flatnonzero(step > 1.5 / sfreq)
    return [(int(k) + 1, float(step[k]) - 1 / sfreq) for k in jumps]


def read_xdf(
    path: str | Path, stream: str = "", markers: str = "", unit: str = "uV"
) -> XdfRecording:
    """Read ``path``: its signal stream as an MNE Raw in volts, with its
    markers and ``BAD_gap`` marks as annotations (seconds from its first
    sample)."""
    import mne
    import pyxdf

    if unit not in ASSUMED_UNITS:
        raise ValueError(f"unit must be one of {sorted(ASSUMED_UNITS)}, not {unit!r}")
    streams, _ = pyxdf.load_xdf(str(path), verbose=False)
    k = pick_signal(streams, stream)
    signal = streams[k]
    info = signal["info"]
    sfreq = float(_first(info, "nominal_srate"))
    data = np.asarray(signal["time_series"], dtype=float).T  # channels x samples
    stamps = np.asarray(signal["time_stamps"], dtype=float)
    described = _channels(info)
    n = data.shape[0]
    labels = [_first(c, "label") for c in described][:n] if len(described) == n else []
    if not all(labels) or len(set(labels)) != n:
        labels = [f"Ch{i + 1}" for i in range(n)]
    notes: list[str] = []
    keep = [i for i, lb in enumerate(labels) if lb.upper() != "SAMPLE_INDEX"]
    kind = _first(info, "type").lower()
    types, scales, assumed = [], [], 0
    for i in keep:
        chan = described[i] if len(described) == n else {}
        ctype = _first(chan, "type").lower() or kind
        types.append(TYPES.get(ctype, "misc"))
        text = _first(chan, "unit").strip()
        if not text:
            assumed += 1
            scales.append(ASSUMED_UNITS[unit])
        elif text.lower() in UNITS:
            scales.append(UNITS[text.lower()])
        else:
            raise ValueError(f"channel {labels[i]!r} is in {text!r}: not a unit "
                             "MeegXer reads (µV, mV or V)")  # fmt: skip
    if assumed:
        notes.append(f"{assumed} of {len(keep)} channels state no unit: taken to "
                     f"be in {unit}")  # fmt: skip
    data = data[keep] * np.asarray(scales)[:, np.newaxis]
    raw = mne.io.RawArray(
        data, mne.create_info([labels[i] for i in keep], sfreq, types),
        verbose="error",
    )  # fmt: skip
    onsets, durations, descriptions = [], [], []
    for after, missing in gaps(stamps, sfreq):
        onsets.append(after / sfreq)
        durations.append(1 / sfreq)
        descriptions.append(GAP)
        notes.append(f"{missing:.3f} s of samples lost at {after / sfreq:.3f} s "
                     f"(marked {GAP})")  # fmt: skip
    wanted = _words(markers)
    chosen = [s for i, s in enumerate(streams) if i != k and _is_marker(s)
              and (not wanted or _name(s) in wanted)]  # fmt: skip
    missing_streams = set(wanted) - {_name(s) for s in chosen}
    if missing_streams:
        raise ValueError(f"no marker stream {sorted(missing_streams)} in the file")
    outside = 0
    for s in chosen:
        at = np.asarray(s["time_stamps"], dtype=float)
        if not len(at):
            continue
        index = nearest_samples(stamps, at, sfreq)
        for idx, value in zip(index, s["time_series"], strict=True):
            label = str(value[0] if len(value) else "").strip()
            if idx < 0 or not label:
                outside += idx < 0
                continue
            onsets.append(int(idx) / sfreq)
            durations.append(0.0)
            descriptions.append(label)
    if outside:
        notes.append(f"{outside} markers outside the recording left out")
    if onsets:
        raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    return XdfRecording(raw, _name(signal), tuple(_name(s) for s in chosen), notes)
