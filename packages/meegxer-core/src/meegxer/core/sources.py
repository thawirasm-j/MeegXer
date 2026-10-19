"""Built-in sources: a synthetic EEG generator and file/BIDS readers.

Signals are in volts (SI, as in MNE). Timestamps of file sources start at 0.0;
the synthetic source starts at ``start_time``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy import signal

from meegxer.core.block import Block, FloatArray, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Source
from meegxer.core.references import credits

if TYPE_CHECKING:
    import mne
    import mne_bids

STANDARD_LABELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
)  # fmt: skip
_POSTERIOR = {"P3", "P4", "P7", "P8", "Pz", "O1", "O2", "Oz"}
_OCCIPITAL = {"O1", "O2", "Oz"}

# Paul Kellet / J.O. Smith pinking filter (-10 dB/decade).
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


def _source_manifest(
    name: str, params: list[dict[str, object]], version: str = "0.1.0"
) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": version,
            "type": "stream",
            "role": "source",
            "inputs": [],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": params,
            **credits(name),
        }
    )


# Synthetic artefacts (all off by default). Each comes from a seeded schedule of
# events in fixed segments of absolute time, so the output never depends on how
# the stream is cut into blocks, and every event is known exactly (the ground
# truth markers). Defaults and where they come from:
#   blinks     17 per minute at rest (Bentivoglio et al. 1997, Mov Disord
#              12:1028); 150 uV, 300 ms, largest at Fp1/Fp2: typical textbook
#              values, not from one study
#   saccades   20 uV per degree of eye rotation (EOG sensitivity, ~5-20 uV/deg
#              reported), horizontal: opposite polarity at F7/F8
#   EMG        bursts of 20-80 Hz noise, largest at temporal and frontal sites
#              (Goncharova et al. 2003, Clin Neurophysiol 114:1580: frontalis
#              ~20-30 Hz, temporalis ~40-80 Hz)
#   ECG        cardiac field artefact, 70 bpm, 5 uV, left-lateral: heuristic
#   pops, drift, DC offset: heuristic
_SEGMENT_S = 10.0
_BLINK, _SACCADE, _EMG, _POP = 1, 2, 3, 4
_KINDS = {"blink": _BLINK, "saccade": _SACCADE, "emg": _EMG, "pop": _POP}


def _weights(
    labels: tuple[str, ...], table: dict[str, float], default: float
) -> FloatArray:
    """Per-channel weights by label (case-insensitive), as a column."""
    lower = {k.lower(): v for k, v in table.items()}
    return np.array([lower.get(lb.lower(), default) for lb in labels])[:, None]


_BLINK_W = {
    "Fp1": 1.0,
    "Fp2": 1.0,
    "Fpz": 1.0,
    "AF3": 0.7,
    "AF4": 0.7,
    "F3": 0.45,
    "F4": 0.45,
    "Fz": 0.45,
    "F7": 0.35,
    "F8": 0.35,
    "C3": 0.12,
    "C4": 0.12,
    "Cz": 0.12,
    "T7": 0.1,
    "T8": 0.1,
}
_SACCADE_W = {
    "F7": 1.0,
    "F8": -1.0,
    "Fp1": 0.4,
    "Fp2": -0.4,
    "F3": 0.3,
    "F4": -0.3,
    "T7": 0.5,
    "T8": -0.5,
    "C3": 0.1,
    "C4": -0.1,
}
_EMG_W = {
    "T7": 1.0,
    "T8": 1.0,
    "F7": 0.8,
    "F8": 0.8,
    "Fp1": 0.5,
    "Fp2": 0.5,
    "P7": 0.5,
    "P8": 0.5,
    "O1": 0.3,
    "O2": 0.3,
}
_ECG_W = {
    "T7": 1.0,
    "P7": 1.0,
    "O1": 0.8,
    "F7": 0.6,
    "C3": 0.5,
    "P3": 0.6,
    "Fz": 0.3,
    "Cz": 0.3,
    "Pz": 0.4,
    "Oz": 0.5,
    "T8": -0.3,
    "P8": -0.3,
}


class SyntheticSource(Source):
    """Synthetic EEG made of parts that can each be switched on or off.

    - ``neural`` (on): pink (1/f) background plus posterior alpha bursts
    - ``line`` (on), ``ssvep`` (on, occipital), ``sine`` (a pure sine)
    - ``dc``: a constant offset per channel
    - artefacts (off): ``blinks``, ``saccades``, ``slow_eye``, ``ecg``,
      ``emg``, ``pops``, ``drift``
    - ``markers`` (on): every artefact event is in the recording's events
      (``blink``, ``saccade``, ``ecg_r``, ``emg``, ``pop``), as ground truth for
      scoring a cleaning method (finite durations only)

    Every component is a function of the absolute sample index, and the white
    noise is drawn sample-major from one generator, so the output does not depend
    on how the stream is cut into blocks.
    """

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.synthetic",
        [
            {"name": "n_channels", "type": "int", "default": 16, "min": 1,
             "max": 256, "primary": True,
             "description": "10-20 labels first, then E21, E22, ..."},
            {"name": "sfreq", "type": "float", "default": 500.0, "min": 1.0,
             "unit": "Hz", "primary": True},
            {"name": "duration", "type": "float", "default": 60.0, "min": 0.0,
             "unit": "s", "description": "0 streams forever"},
            {"name": "seed", "type": "int", "default": 0, "min": 0, "advanced": True},
            *(
                {"name": name, "type": "bool", "default": default,
                 "description": description}
                for name, default, description in (
                    ("neural", True, "Pink (1/f) background and posterior alpha"),
                    ("line", True, "Line noise on every channel"),
                    ("ssvep", True, "Occipital SSVEP sine"),
                    ("sine", False, "A pure sine on every channel"),
                    ("dc", False, "A constant offset per channel"),
                    ("blinks", False, "Eye blinks, frontal"),
                    ("saccades", False, "Horizontal saccades, F7/F8 opposite"),
                    ("slow_eye", False, "Slow horizontal eye movements"),
                    ("ecg", False, "Cardiac field artefact, left-lateral"),
                    ("emg", False, "Muscle bursts, 20-80 Hz, temporal/frontal"),
                    ("pops", False, "Electrode pops on single channels"),
                    ("drift", False, "Slow drift per channel"),
                    ("markers", True,
                     "Artefact events in the recording's events (ground truth)"),
                )
            ),
            *(
                {"name": name, "type": "float", "default": default, "min": 0.0,
                 "unit": unit, "advanced": True}
                for name, default, unit in (
                    ("noise_uv", 10.0, "uV"),
                    ("alpha_uv", 15.0, "uV"),
                    ("alpha_hz", 10.0, "Hz"),
                    ("line_uv", 5.0, "uV"),
                    ("line_hz", 50.0, "Hz"),
                    ("ssvep_uv", 3.0, "uV"),
                    ("ssvep_hz", 12.0, "Hz"),
                    ("sine_uv", 20.0, "uV"),
                    ("sine_hz", 10.0, "Hz"),
                    ("dc_mv", 10.0, "mV"),
                    ("blink_rate", 17.0, "/min"),
                    ("blink_uv", 150.0, "uV"),
                    ("blink_ms", 300.0, "ms"),
                    ("saccade_rate", 20.0, "/min"),
                    ("saccade_deg", 10.0, "deg"),
                    ("saccade_uv_per_deg", 20.0, "uV"),
                    ("slow_eye_uv", 40.0, "uV"),
                    ("ecg_bpm", 70.0, "/min"),
                    ("ecg_uv", 5.0, "uV"),
                    ("emg_rate", 4.0, "/min"),
                    ("emg_uv", 20.0, "uV"),
                    ("emg_s", 1.0, "s"),
                    ("pop_rate", 0.5, "/min"),
                    ("pop_uv", 200.0, "uV"),
                    ("drift_uv", 30.0, "uV"),
                )
            ),
            {"name": "start_time", "type": "float", "default": 0.0, "unit": "s",
             "advanced": True},
        ],
        version="0.2.0",
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        p = self.params
        n_ch = int(p["n_channels"])
        extra = tuple(f"E{k}" for k in range(len(STANDARD_LABELS) + 1, n_ch + 1))
        labels = (STANDARD_LABELS + extra)[:n_ch]
        self._info = SignalInfo(labels, float(p["sfreq"]), "synthetic")
        duration = float(p["duration"])
        self._total = round(duration * self._info.sfreq) if duration > 0 else None
        self._pos = 0
        seed = int(p["seed"])
        self._seed = seed
        self._white = np.random.default_rng([seed, 0])
        phases = np.random.default_rng([seed, 1]).uniform(0, 2 * np.pi, (2, n_ch))
        self._burst_phase, self._alpha_phase = phases[:, :, None]
        labels = self._info.ch_labels
        self._alpha_weight = np.array(
            [1.0 if lab in _POSTERIOR else 0.3 for lab in labels]
        )[:, None]
        self._ssvep_weight = np.array(
            [1.0 if lab in _OCCIPITAL else 0.0 for lab in labels]
        )[:, None]
        self._pink_sos = signal.tf2sos(_PINK_B, _PINK_A)
        impulse = np.zeros(1 << 16)
        impulse[0] = 1.0
        self._pink_gain = float(
            np.sqrt(np.sum(signal.sosfilt(self._pink_sos, impulse) ** 2))
        )
        self._pink_zi = np.zeros((self._pink_sos.shape[0], n_ch, 2))
        # artefacts: fixed per seed
        extra_rng = np.random.default_rng([seed, 2])
        self._dc = extra_rng.uniform(-1.0, 1.0, (n_ch, 1))
        self._drift = (extra_rng.uniform(0.01, 0.05, (n_ch, 2)),
                       extra_rng.uniform(0, 2 * np.pi, (n_ch, 2)))  # fmt: skip
        self._slow_eye = (extra_rng.uniform(0.05, 0.3, 3),
                          extra_rng.uniform(0, 2 * np.pi, 3))  # fmt: skip
        self._blink_w = _weights(labels, _BLINK_W, 0.04)
        self._saccade_w = _weights(labels, _SACCADE_W, 0.0)
        self._emg_w = _weights(labels, _EMG_W, 0.1)
        self._ecg_w = _weights(labels, _ECG_W, 0.2)
        self._emg_cache: dict[tuple[int, int], FloatArray] = {}

    @property
    def info(self) -> SignalInfo:
        return self._info

    @property
    def n_samples(self) -> int | None:
        return self._total

    def read(self, max_samples: int) -> Block | None:
        n = max_samples
        if self._total is not None:
            if self._pos >= self._total:
                return None
            n = min(n, self._total - self._pos)
        data = self._generate(self._pos, n)
        sfreq = self._info.sfreq
        block = Block(
            data,
            self._info,
            self._pos,
            float(self.params["start_time"]) + self._pos / sfreq,
        )
        self._pos += n
        return block

    def _generate(self, start: int, n: int) -> FloatArray:
        p = self.params
        uv = 1e-6
        t = (start + np.arange(n))[None, :] / self._info.sfreq
        white = self._white.standard_normal((n, self._info.n_channels)).T
        pink, self._pink_zi = signal.sosfilt(
            self._pink_sos, white, axis=-1, zi=self._pink_zi
        )
        out = np.zeros((self._info.n_channels, n))
        if p["neural"]:
            out += float(p["noise_uv"]) * uv * pink / self._pink_gain
            envelope = np.maximum(0.0, np.sin(2 * np.pi * 0.15 * t + self._burst_phase))
            alpha = np.sin(2 * np.pi * float(p["alpha_hz"]) * t + self._alpha_phase)
            out += float(p["alpha_uv"]) * uv * self._alpha_weight * envelope**2 * alpha
        if p["line"]:
            out += (
                float(p["line_uv"]) * uv * np.sin(2 * np.pi * float(p["line_hz"]) * t)
            )
        if p["ssvep"]:
            ssvep = np.sin(2 * np.pi * float(p["ssvep_hz"]) * t)
            out += float(p["ssvep_uv"]) * uv * self._ssvep_weight * ssvep
        if p["sine"]:
            out += (
                float(p["sine_uv"]) * uv * np.sin(2 * np.pi * float(p["sine_hz"]) * t)
            )
        if p["dc"]:
            out += float(p["dc_mv"]) * 1e-3 * self._dc
        if p["drift"]:
            freqs, phases = self._drift
            wave = np.sin(2 * np.pi * freqs[:, :, None] * t[None] + phases[:, :, None])
            out += float(p["drift_uv"]) * uv * wave.sum(axis=1) / 2
        if p["slow_eye"]:
            freqs, phases = self._slow_eye
            gaze = np.sin(2 * np.pi * freqs[:, None] * t + phases[:, None]).sum(axis=0)
            out += float(p["slow_eye_uv"]) * uv * self._saccade_w * gaze / 3
        t0, t1 = start / self._info.sfreq, (start + n) / self._info.sfreq
        tt = t[0]
        if p["blinks"]:
            out += uv * self._blink_w * self._blinks(tt, t0, t1)
        if p["saccades"]:
            out += uv * self._saccade_w * self._gaze(tt, t0, t1)
        if p["ecg"]:
            out += float(p["ecg_uv"]) * uv * self._ecg_w * self._ecg(tt, t0, t1)
        if p["emg"]:
            out += uv * self._emg(start, n)
        if p["pops"]:
            out += uv * self._pops(tt, t0, t1)
        result: FloatArray = np.ascontiguousarray(out, dtype=np.float64)
        return result

    # -- artefact events ------------------------------------------------------

    def _onsets(
        self, kind: int, rate: float, t0: float, t1: float
    ) -> list[tuple[float, int, int]]:
        """Events of ``kind`` starting in [t0, t1): (onset s, segment, index)."""
        found = []
        for k in range(max(0, int(t0 // _SEGMENT_S)), int(t1 // _SEGMENT_S) + 1):
            rng = np.random.default_rng([self._seed, 100 + kind, k])
            count = rng.poisson(max(rate, 0.0) / 60 * _SEGMENT_S)
            times = np.sort(k * _SEGMENT_S + rng.uniform(0, _SEGMENT_S, count))
            found += [(float(x), k, i) for i, x in enumerate(times) if t0 <= x < t1]
        return found

    def _event_rng(self, kind: int, k: int, i: int) -> np.random.Generator:
        return np.random.default_rng([self._seed, 200 + kind, k, i])

    def _blinks(self, tt: FloatArray, t0: float, t1: float) -> FloatArray:
        p = self.params
        width = float(p["blink_ms"]) / 1000
        out = np.zeros_like(tt)
        for onset, _, _ in self._onsets(_BLINK, float(p["blink_rate"]), t0 - width, t1):
            x = (tt - onset) / width
            inside = (x >= 0) & (x < 1)
            out[inside] += (
                float(p["blink_uv"]) * 0.5 * (1 - np.cos(2 * np.pi * x[inside]))
            )
        return out

    def _gaze(self, tt: FloatArray, t0: float, t1: float) -> FloatArray:
        """Horizontal gaze in microvolts of EOG: steps of 40 ms at each saccade
        to a new angle, held until the next."""
        p = self.params
        ramp, rate = 0.04, float(p["saccade_rate"])
        events = self._onsets(_SACCADE, rate, t0 - ramp, t1)
        before = self._last_angle(t0 - ramp, rate)
        value = np.full_like(tt, before)
        previous = before
        for onset, k, i in events:
            angle = self._angle(k, i)
            value += (angle - previous) * np.clip((tt - onset) / ramp, 0.0, 1.0)
            previous = angle
        return float(p["saccade_uv_per_deg"]) * value

    def _angle(self, k: int, i: int) -> float:
        limit = float(self.params["saccade_deg"])
        return float(self._event_rng(_SACCADE, k, i).uniform(-limit, limit))

    def _last_angle(self, t: float, rate: float) -> float:
        """The gaze angle reached by the last saccade before ``t``."""
        segment = int(t // _SEGMENT_S)
        for k in range(segment, max(-1, segment - 100), -1):
            events = self._onsets(
                _SACCADE, rate, k * _SEGMENT_S, min(t, (k + 1) * _SEGMENT_S)
            )
            if events:
                _, kk, ii = events[-1]
                return self._angle(kk, ii)
        return 0.0

    def _beats(self, t0: float, t1: float) -> list[float]:
        period = 60.0 / max(float(self.params["ecg_bpm"]), 1.0)
        beats = []
        for k in range(max(0, int(t0 // period) - 1), int(t1 // period) + 2):
            jitter = np.random.default_rng([self._seed, 300, k]).uniform(-0.05, 0.05)
            beat = (k + 0.5 + jitter) * period
            if t0 <= beat < t1:
                beats.append(float(beat))
        return beats

    def _ecg(self, tt: FloatArray, t0: float, t1: float) -> FloatArray:
        """A PQRST-like complex (unit R peak) at every beat."""
        out = np.zeros_like(tt)
        waves = ((-0.04, 0.01, -0.12), (0.0, 0.012, 1.0), (0.04, 0.01, -0.25),
                 (0.25, 0.05, 0.3), (-0.16, 0.025, 0.12))  # fmt: skip
        before, after = 0.3, 0.5  # each complex lives in this window only
        for beat in self._beats(t0 - after, t1 + before):
            x = tt - beat
            inside = (x >= -before) & (x < after)
            for shift, sd, amp in waves:
                out[inside] += amp * np.exp(-0.5 * ((x[inside] - shift) / sd) ** 2)
        return out

    def _emg(self, start: int, n: int) -> FloatArray:
        p = self.params
        sfreq = self._info.sfreq
        length = max(1, round(float(p["emg_s"]) * sfreq))
        t0, t1 = start / sfreq, (start + n) / sfreq
        out = np.zeros((self._info.n_channels, n))
        high = min(80.0, 0.45 * sfreq)
        if high <= 20.0:
            return out
        for onset, k, i in self._onsets(
            _EMG, float(p["emg_rate"]), t0 - length / sfreq, t1
        ):
            burst = self._emg_cache.get((k, i))
            if burst is None:
                rng = self._event_rng(_EMG, k, i)
                noise = rng.standard_normal((self._info.n_channels, length))
                sos = signal.butter(4, [20.0, high], "bandpass", fs=sfreq, output="sos")
                burst = signal.sosfiltfilt(sos, noise, axis=-1)
                burst *= np.hanning(length) / max(float(np.std(burst)), 1e-12)
                burst *= float(p["emg_uv"]) * self._emg_w
                if len(self._emg_cache) > 64:
                    self._emg_cache.clear()
                self._emg_cache[(k, i)] = burst
            first = round(onset * sfreq)
            lo, hi = max(first, start), min(first + length, start + n)
            if hi > lo:
                out[:, lo - start : hi - start] += burst[:, lo - first : hi - first]
        return out

    def _pops(self, tt: FloatArray, t0: float, t1: float) -> FloatArray:
        """A step that decays (tau 0.3 s) on one channel per pop."""
        p = self.params
        tau, span = 0.3, 1.5
        out = np.zeros((self._info.n_channels, len(tt)))
        for onset, k, i in self._onsets(_POP, float(p["pop_rate"]), t0 - span, t1):
            channel = int(self._event_rng(_POP, k, i).integers(self._info.n_channels))
            x = tt - onset
            inside = (x >= 0) & (x < span)
            out[channel, inside] += float(p["pop_uv"]) * np.exp(-x[inside] / tau)
        return out

    def annotations(self) -> list[dict[str, float | str]]:
        """The artefact events (ground truth), for finite durations with
        ``markers`` on: onset and duration in seconds, label."""
        p = self.params
        if not p["markers"] or self._total is None:
            return []
        end = self._total / self._info.sfreq
        events: list[dict[str, float | str]] = []
        if p["blinks"]:
            width = float(p["blink_ms"]) / 1000
            events += [
                {"onset": x, "duration": width, "label": "blink"}
                for x, _, _ in self._onsets(_BLINK, float(p["blink_rate"]), 0, end)
            ]
        if p["saccades"]:
            events += [
                {"onset": x, "duration": 0.04, "label": "saccade"}
                for x, _, _ in self._onsets(_SACCADE, float(p["saccade_rate"]), 0, end)
            ]
        if p["ecg"]:
            events += [
                {"onset": x, "duration": 0.0, "label": "ecg_r"}
                for x in self._beats(0, end)
            ]
        if p["emg"]:
            events += [
                {"onset": x, "duration": float(p["emg_s"]), "label": "emg"}
                for x, _, _ in self._onsets(_EMG, float(p["emg_rate"]), 0, end)
            ]
        if p["pops"]:
            events += [
                {"onset": x, "duration": 1.5, "label": "pop"}
                for x, _, _ in self._onsets(_POP, float(p["pop_rate"]), 0, end)
            ]
        return sorted(events, key=lambda e: float(e["onset"]))


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _location(loc: FloatArray) -> tuple[float, float, float] | None:
    xyz = np.asarray(loc[:3], dtype=float)
    if not np.all(np.isfinite(xyz)) or not np.any(xyz):
        return None
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


class _RawSource(Source):
    """Serves an in-memory MNE Raw block by block."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._pos = 0
        self._annotations: list[dict[str, float | str]] = []

    def _load(self) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _files(self) -> list[Path]:
        raise NotImplementedError

    @cached_property
    def _raw(self) -> tuple[FloatArray, SignalInfo]:
        import mne  # lazy: only file sources need it

        raw = self._load()
        # data channels plus the non-EEG ones nodes skip by default (channel_types);
        # trigger channels stay out: events come from the annotations
        raw.pick(mne.pick_types(raw.info, meg=True, eeg=True, seeg=True, ecog=True,
                                dbs=True, fnirs=True, eog=True, ecg=True, emg=True,
                                misc=True, resp=True, exclude=[]))  # fmt: skip
        data: FloatArray = np.asarray(raw.get_data(), dtype=np.float64)
        info = SignalInfo(
            tuple(raw.ch_names),
            float(raw.info["sfreq"]),
            self._stream_id(),
            tuple(b for b in raw.info["bads"] if b in raw.ch_names),
            tuple(raw.get_channel_types()),
            tuple(_location(ch["loc"]) for ch in raw.info["chs"]),
        )
        self._annotations = [
            {"onset": float(a["onset"] - raw.first_time),
             "duration": float(a["duration"]), "label": str(a["description"])}
            for a in raw.annotations
        ]  # fmt: skip
        return data, info

    def annotations(self) -> list[dict[str, float | str]]:
        """Events stored in the recording: onset (s from the first sample),
        duration (s) and label."""
        _ = self._raw
        return list(self._annotations)

    def _stream_id(self) -> str:
        return self._files()[0].name

    @property
    def info(self) -> SignalInfo:
        return self._raw[1]

    @property
    def n_samples(self) -> int:
        return int(self._raw[0].shape[1])

    def read(self, max_samples: int) -> Block | None:
        data, info = self._raw
        if self._pos >= data.shape[1]:
            return None
        stop = min(self._pos + max_samples, data.shape[1])
        block = Block(
            data[:, self._pos : stop], info, self._pos, self._pos / info.sfreq
        )
        self._pos = stop
        return block

    def content_id(self) -> str:
        return _hash_files(self._files())

    def input_files(self) -> list[Path]:
        return self._files()


class FileSource(_RawSource):
    """EDF/BDF, EEGLAB .set, BrainVision, FIF: anything ``mne.io.read_raw`` reads."""

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.file",
        [
            {
                "name": "path",
                "type": "string",
                "default": "",
                "primary": True,
                "path": "file",
                "extensions": [".set", ".edf", ".bdf", ".vhdr", ".fif", ".gdf", ".cnt"],
            }
        ],
    )

    _COMPANIONS: ClassVar[dict[str, tuple[str, ...]]] = {
        ".set": (".fdt",),
        ".vhdr": (".eeg", ".vmrk"),
    }

    def _path(self) -> Path:
        path = Path(str(self.params["path"]))
        if not str(self.params["path"]) or not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
        return path

    def _files(self) -> list[Path]:
        path = self._path()
        companions = self._COMPANIONS.get(path.suffix.lower(), ())
        extra = [path.with_suffix(ext) for ext in companions]
        return [path, *[p for p in extra if p.is_file()]]

    def _load(self) -> mne.io.BaseRaw:
        import mne

        return mne.io.read_raw(self._path(), preload=True, verbose="ERROR")


class BidsSource(_RawSource):
    """One recording from a BIDS dataset, read with mne-bids."""

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.bids",
        [
            {
                "name": "root",
                "type": "string",
                "default": "",
                "path": "folder",
                "description": "The BIDS dataset (its top folder)",
            },
            {"name": "subject", "type": "string", "default": "", "primary": True},
            {"name": "task", "type": "string", "default": "", "primary": True},
            {"name": "session", "type": "string", "default": ""},
            {"name": "run", "type": "string", "default": ""},
            {"name": "datatype", "type": "string", "default": "eeg"},
        ],
    )

    def _bids_path(self) -> mne_bids.BIDSPath:
        import mne_bids

        p = self.params
        return mne_bids.BIDSPath(
            root=str(p["root"]),
            subject=str(p["subject"]) or None,
            task=str(p["task"]) or None,
            session=str(p["session"]) or None,
            run=str(p["run"]) or None,
            datatype=str(p["datatype"]),
        )

    def _files(self) -> list[Path]:
        fpath = self._bids_path().fpath
        return [Path(fpath)]

    def _load(self) -> mne.io.BaseRaw:
        import mne_bids

        raw: mne.io.BaseRaw = mne_bids.read_raw_bids(self._bids_path(), verbose="ERROR")
        raw.load_data()
        return raw
