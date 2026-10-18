"""IIR and FIR filters: causal (online and offline) and zero-phase (offline only).

Causal filters carry the per-channel ``sosfilt`` delay line as state, so filtering
block by block gives the same result as one pass. The first block initialises the
state to the filter's steady-state response to that block's first sample, which
avoids the start-up step a zero state would produce.

Zero-phase filters run the same design forwards and backwards (``sosfiltfilt``)
over the whole recording. They have no phase distortion or group delay but are
non-causal, so they are offline-only. Running the filter twice squares the
magnitude response: attenuation doubles in dB and a Butterworth edge is -6 dB
instead of -3 dB.

FIR filters are designed by MNE (``mne.filter.create_filter``, windowed sinc with
automatic transition bands and length; Widmann et al. 2015). Live they use the
minimum-phase version of the design (causal, little delay, carried ``lfilter``
state); offline, with phase auto, the linear-phase design applied without delay
(MNE's ``phase="zero"``), so the waveform keeps its shape.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import signal

from meegxer.core.block import Block, Filter
from meegxer.core.manifest import Manifest
from meegxer.core.node import Mode, Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest

SosArray = npt.NDArray[np.float64]


def filters_of(kind: str, low: float, high: float) -> tuple[Filter, ...]:
    """What a high-/low-/band-pass or band-stop leaves on the stream's info
    (``SignalInfo.filters``): a band-pass is a high-pass and a low-pass."""
    if kind == "highpass":
        return (("highpass", low, 0.0),)
    if kind == "lowpass":
        return (("lowpass", 0.0, high),)
    if kind == "bandpass":
        return (("highpass", low, 0.0), ("lowpass", 0.0, high))
    return (("bandstop", low, high),)


def notch_filters(freq: float, harmonics: int, sfreq: float) -> tuple[Filter, ...]:
    """The notches a notch cascade cuts (its harmonics below Nyquist)."""
    return tuple(("notch", freq * k, 0.0) for k in range(1, harmonics + 1)
                 if freq * k < sfreq / 2)  # fmt: skip


def _tagged(block: Block, filters: tuple[Filter, ...]) -> Block:
    return dataclasses.replace(block, info=block.info.filtered(*filters))


def _apply_sos(
    sos: SosArray, block: Block, state: State | None,
    filters: tuple[Filter, ...] = (),
) -> TransformResult:  # fmt: skip
    data = np.asarray(block.data)
    if data.shape[-1] == 0:  # an upstream delay (spec 0.2.5) can emit empty blocks
        # no "zi" yet means the first real sample still seeds it
        return TransformResult(_tagged(block, filters), state or {})
    zi: npt.NDArray[np.float64]
    if state is None or "zi" not in state:
        zi_unit = signal.sosfilt_zi(sos)  # (n_sections, 2)
        zi = zi_unit[:, None, :] * data[None, :, :1]
    else:
        stored = state["zi"]
        assert isinstance(stored, np.ndarray)
        zi = stored
    out, zf = signal.sosfilt(sos, data, axis=-1, zi=zi)
    result = Block(out, block.info.filtered(*filters), block.sample_index,
                   block.timestamp)  # fmt: skip
    return TransformResult(result, {"zi": zf})


def _check_band(name: str, freq: float, sfreq: float) -> float:
    nyquist = sfreq / 2
    if not 0 < freq < nyquist:
        raise ValueError(
            f"{name}={freq} Hz must lie between 0 and Nyquist ({nyquist} Hz)"
        )
    return freq


@lru_cache(maxsize=256)
def _butter_sos(
    kind: str, order: int, low: float, high: float, sfreq: float
) -> SosArray:
    """Designs are pure functions of the parameters, so they are memoised; the
    real-time scheduler would otherwise redesign the filter on every block."""
    btype = cast(Literal["highpass", "lowpass", "bandpass", "bandstop"], kind)
    cutoff: float | list[float]
    if kind == "highpass":
        cutoff = _check_band("low", low, sfreq)
    elif kind == "lowpass":
        cutoff = _check_band("high", high, sfreq)
    else:
        cutoff = [_check_band("low", low, sfreq), _check_band("high", high, sfreq)]
        if low >= high:
            raise ValueError(f"low ({low} Hz) must be below high ({high} Hz)")
    sos: SosArray = signal.butter(order, cutoff, btype=btype, fs=sfreq, output="sos")
    return sos


@lru_cache(maxsize=256)
def _notch_sos(freq: float, quality: float, harmonics: int, sfreq: float) -> SosArray:
    base = _check_band("freq", freq, sfreq)
    sections = []
    for k in range(1, harmonics + 1):
        f = base * k
        if f >= sfreq / 2:
            break
        b, a = signal.iirnotch(f, quality, fs=sfreq)
        sections.append(signal.tf2sos(b, a))
    sos: SosArray = np.vstack(sections)
    return sos


class IIRFilter(Processor):
    """Butterworth high-pass, low-pass, band-pass or band-stop."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.iir_filter",
        "0.1.0",
        [
            {"name": "kind", "type": "enum", "default": "bandpass", "primary": True,
             "choices": ["highpass", "lowpass", "bandpass", "bandstop"]},
            {"name": "low", "type": "float", "default": 1.0, "min": 0.0, "unit": "Hz",
             "primary": True,
             "description": "Lower edge (high-pass, band-pass, band-stop)"},
            {"name": "high", "type": "float", "default": 40.0, "min": 0.0,
             "unit": "Hz", "primary": True,
             "description": "Upper edge (low-, band-pass, band-stop)"},
            {"name": "order", "type": "int", "default": 4, "min": 1, "max": 10,
             "advanced": True,
             "description": "Butterworth order (doubled for band filters)"},
            {"name": "phase", "type": "enum", "default": "auto",
             "choices": ["auto", "causal", "zero"],
             "description": "auto: causal live, zero-phase offline (zero: offline)"},
        ],
        description="Butterworth IIR filter: causal with carried state live, "
        "zero-phase offline (phase auto).",
    )  # fmt: skip

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        phase = params["phase"]
        if mode == "online":
            if phase == "zero":
                raise ValueError(
                    "zero-phase filtering needs the whole recording; "
                    "use phase auto or causal live"
                )
            return self
        return self if phase == "causal" else IIRFilterZeroPhase()

    @staticmethod
    def design(params: Params, sfreq: float) -> SosArray:
        sos = _butter_sos(
            str(params["kind"]),
            int(params["order"]),
            float(params["low"]),
            float(params["high"]),
            sfreq,
        )
        return sos.copy()  # callers may not mutate the memoised design

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos(self.design(params, block.info.sfreq), block, state,
                          IIRFilter.filters(params))  # fmt: skip

    @staticmethod
    def filters(params: Params) -> tuple[Filter, ...]:
        return filters_of(str(params["kind"]), float(params["low"]),
                          float(params["high"]))  # fmt: skip


class NotchFilter(Processor):
    """Cascade of IIR notches at the line frequency and its harmonics."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.notch",
        "0.1.0",
        [
            {"name": "freq", "type": "float", "default": 50.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "quality", "type": "float", "default": 30.0, "min": 0.1,
             "advanced": True,
             "description": "Q factor: centre frequency / -3 dB bandwidth"},
            {"name": "harmonics", "type": "int", "default": 1, "min": 1, "max": 10,
             "advanced": True,
             "description": "Number of notches (freq, 2*freq, ...) below Nyquist"},
        ],
        description="Causal IIR notch (line noise) with carried state.",
    )  # fmt: skip

    @staticmethod
    def design(params: Params, sfreq: float) -> SosArray:
        return _notch_sos(
            float(params["freq"]),
            float(params["quality"]),
            int(params["harmonics"]),
            sfreq,
        )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos(self.design(params, block.info.sfreq), block, state,
                          NotchFilter.filters(params, block.info.sfreq))  # fmt: skip

    @staticmethod
    def filters(params: Params, sfreq: float) -> tuple[Filter, ...]:
        return notch_filters(float(params["freq"]), int(params["harmonics"]), sfreq)


def _apply_sos_zero_phase(
    sos: SosArray, block: Block, filters: tuple[Filter, ...] = ()
) -> TransformResult:
    data = np.asarray(block.data)
    try:
        out = signal.sosfiltfilt(sos, data, axis=-1)
    except ValueError as exc:  # input shorter than the edge padding
        raise ValueError(
            f"zero-phase filtering needs a longer recording ({block.n_samples} "
            f"samples): {exc}"
        ) from exc
    return TransformResult(
        Block(out, block.info.filtered(*filters), block.sample_index,
              block.timestamp), {}
    )  # fmt: skip


def _zero_phase_manifest(causal: Manifest, name: str, description: str) -> Manifest:
    return stream_manifest(
        name,
        causal.version,
        [
            {
                k: v
                for k, v in {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min,
                    "max": p.max,
                    "choices": list(p.choices) if p.choices else None,
                    "unit": p.unit or None,
                    "description": p.description or None,
                    "advanced": p.advanced or None,
                    "primary": p.primary or None,
                }.items()
                if v is not None
            }
            for p in causal.params
            if p.name != "phase"  # the zero-phase node is the resolved variant
        ],
        description=description,
        availability="offline",
    )


class IIRFilterZeroPhase(Processor):
    """Forward-backward Butterworth filter (offline only, no phase shift)."""

    manifest: ClassVar[Manifest] = _zero_phase_manifest(
        IIRFilter.manifest,
        "meegxer.nodes.iir_filter_zerophase",
        "Zero-phase (forward-backward) Butterworth IIR filter; offline only.",
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos_zero_phase(IIRFilter.design(params, block.info.sfreq),
                                     block, IIRFilter.filters(params))  # fmt: skip


class NotchFilterZeroPhase(Processor):
    """Forward-backward notch cascade (offline only, no phase shift)."""

    manifest: ClassVar[Manifest] = _zero_phase_manifest(
        NotchFilter.manifest,
        "meegxer.nodes.notch_zerophase",
        "Zero-phase (forward-backward) IIR notch; offline only.",
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        sos = NotchFilter.design(params, block.info.sfreq)
        return _apply_sos_zero_phase(
            sos, block, NotchFilter.filters(params, block.info.sfreq)
        )


@lru_cache(maxsize=64)
def _fir_taps(
    kind: str, low: float, high: float, sfreq: float, phase: str
) -> npt.NDArray[np.float64]:
    """FIR taps from MNE (memoised: designs are pure functions of the params)."""
    import mne

    l_freq, h_freq = _fir_edges(kind, low, high, sfreq)
    h = mne.filter.create_filter(None, sfreq, l_freq, h_freq, method="fir",
                                 phase=phase, fir_design="firwin",
                                 verbose="error")  # fmt: skip
    return np.asarray(h, dtype=np.float64)


def _fir_edges(
    kind: str, low: float, high: float, sfreq: float
) -> tuple[float | None, float | None]:
    """MNE's (l_freq, h_freq) for a kind (band-stop: l_freq above h_freq)."""
    if kind == "highpass":
        return _check_band("low", low, sfreq), None
    if kind == "lowpass":
        return None, _check_band("high", high, sfreq)
    lo, hi = _check_band("low", low, sfreq), _check_band("high", high, sfreq)
    if lo >= hi:
        raise ValueError(f"low ({low} Hz) must be below high ({high} Hz)")
    return (lo, hi) if kind == "bandpass" else (hi, lo)


class FIRFilter(Processor):
    """Windowed-sinc FIR high-pass, low-pass, band-pass or band-stop (MNE)."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.fir_filter",
        "0.1.0",
        [
            {"name": "kind", "type": "enum", "default": "bandpass", "primary": True,
             "choices": ["highpass", "lowpass", "bandpass", "bandstop"]},
            {"name": "low", "type": "float", "default": 1.0, "min": 0.0, "unit": "Hz",
             "primary": True,
             "description": "Lower edge (high-pass, band-pass, band-stop)"},
            {"name": "high", "type": "float", "default": 40.0, "min": 0.0,
             "unit": "Hz", "primary": True,
             "description": "Upper edge (low-, band-pass, band-stop)"},
            {"name": "phase", "type": "enum", "default": "auto",
             "choices": ["auto", "causal", "zero"],
             "description": "auto: minimum-phase (causal) live, zero-phase offline "
                            "(zero: offline)"},
        ],
        description="FIR filter designed by MNE: minimum-phase with carried state "
        "live, zero-phase offline (phase auto).",
    )  # fmt: skip

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        phase = params["phase"]
        if mode == "online":
            if phase == "zero":
                raise ValueError(
                    "zero-phase filtering needs the whole recording; "
                    "use phase auto or causal live"
                )
            return self
        return self if phase == "causal" else FIRFilterZeroPhase()

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        h = _fir_taps(str(params["kind"]), float(params["low"]),
                      float(params["high"]), block.info.sfreq, "minimum")  # fmt: skip
        tags = filters_of(str(params["kind"]), float(params["low"]),
                          float(params["high"]))  # fmt: skip
        data = np.asarray(block.data)
        if data.shape[-1] == 0:
            return TransformResult(_tagged(block, tags), state or {})
        if state is None or "zi" not in state:
            zi = signal.lfilter_zi(h, [1.0])[None, :] * data[:, :1]
        else:
            zi = np.asarray(state["zi"])
        out, zf = signal.lfilter(h, [1.0], data, axis=-1, zi=zi)
        result = Block(out, block.info.filtered(*tags), block.sample_index,
                       block.timestamp, block.bypassed)  # fmt: skip
        return TransformResult(result, {"zi": zf})


class FIRFilterZeroPhase(Processor):
    """Linear-phase FIR applied without delay (offline only)."""

    manifest: ClassVar[Manifest] = _zero_phase_manifest(
        FIRFilter.manifest,
        "meegxer.nodes.fir_filter_zerophase",
        "Zero-phase FIR filter designed by MNE (linear phase, delay removed); "
        "offline only.",
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        import mne

        kind, low, high = (
            str(params["kind"]),
            float(params["low"]),
            float(params["high"]),
        )
        l_freq, h_freq = _fir_edges(kind, low, high, block.info.sfreq)
        out = mne.filter.filter_data(
            np.asarray(block.data, dtype=np.float64), block.info.sfreq, l_freq,
            h_freq, method="fir", phase="zero", fir_design="firwin",
            verbose="error",
        )  # fmt: skip
        info = block.info.filtered(*filters_of(kind, low, high))
        return TransformResult(
            Block(out, info, block.sample_index, block.timestamp, block.bypassed), {}
        )
