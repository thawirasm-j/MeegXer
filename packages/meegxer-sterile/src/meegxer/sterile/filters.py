"""IIR filters: causal (online and offline) and zero-phase (offline only).

Causal filters carry the per-channel ``sosfilt`` delay line as state, so filtering
block by block gives the same result as one pass. The first block initialises the
state to the filter's steady-state response to that block's first sample, which
avoids the start-up step a zero state would produce.

Zero-phase filters run the same design forwards and backwards (``sosfiltfilt``)
over the whole recording. They have no phase distortion or group delay but are
non-causal, so they are offline-only. Running the filter twice squares the
magnitude response: attenuation doubles in dB and a Butterworth edge is -6 dB
instead of -3 dB.
"""

from __future__ import annotations

from typing import ClassVar, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import signal

from meegxer.core.block import Block
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.sterile._manifest import stream_manifest

SosArray = npt.NDArray[np.float64]


def _apply_sos(sos: SosArray, block: Block, state: State | None) -> TransformResult:
    data = np.asarray(block.data)
    zi: npt.NDArray[np.float64]
    if state is None:
        zi_unit = signal.sosfilt_zi(sos)  # (n_sections, 2)
        zi = zi_unit[:, None, :] * data[None, :, :1]
    else:
        stored = state["zi"]
        assert isinstance(stored, np.ndarray)
        zi = stored
    out, zf = signal.sosfilt(sos, data, axis=-1, zi=zi)
    result = Block(out, block.info, block.sample_index, block.timestamp)
    return TransformResult(result, {"zi": zf})


def _check_band(name: str, freq: float, sfreq: float) -> float:
    nyquist = sfreq / 2
    if not 0 < freq < nyquist:
        raise ValueError(
            f"{name}={freq} Hz must lie between 0 and Nyquist ({nyquist} Hz)"
        )
    return freq


class IIRFilter(Processor):
    """Butterworth high-pass, low-pass, band-pass or band-stop."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.sterile.iir_filter",
        "0.1.0",
        [
            {"name": "kind", "type": "enum", "default": "bandpass",
             "choices": ["highpass", "lowpass", "bandpass", "bandstop"]},
            {"name": "low", "type": "float", "default": 1.0, "min": 0.0, "unit": "Hz",
             "description": "Lower edge (high-pass, band-pass, band-stop)"},
            {"name": "high", "type": "float", "default": 40.0, "min": 0.0,
             "unit": "Hz", "description": "Upper edge (low-, band-pass, band-stop)"},
            {"name": "order", "type": "int", "default": 4, "min": 1, "max": 10,
             "advanced": True,
             "description": "Butterworth order (doubled for band filters)"},
        ],
        description="Causal Butterworth IIR filter with carried state.",
    )  # fmt: skip

    @staticmethod
    def design(params: Params, sfreq: float) -> SosArray:
        kind = cast(
            Literal["highpass", "lowpass", "bandpass", "bandstop"], str(params["kind"])
        )
        order = int(params["order"])
        low, high = float(params["low"]), float(params["high"])
        cutoff: float | list[float]
        if kind == "highpass":
            cutoff = _check_band("low", low, sfreq)
        elif kind == "lowpass":
            cutoff = _check_band("high", high, sfreq)
        else:
            cutoff = [_check_band("low", low, sfreq), _check_band("high", high, sfreq)]
            if low >= high:
                raise ValueError(f"low ({low} Hz) must be below high ({high} Hz)")
        sos: SosArray = signal.butter(order, cutoff, btype=kind, fs=sfreq, output="sos")
        return sos

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos(self.design(params, block.info.sfreq), block, state)


class NotchFilter(Processor):
    """Cascade of IIR notches at the line frequency and its harmonics."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.sterile.notch",
        "0.1.0",
        [
            {"name": "freq", "type": "float", "default": 50.0, "min": 0.0,
             "unit": "Hz"},
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
        base = _check_band("freq", float(params["freq"]), sfreq)
        sections = []
        for k in range(1, int(params["harmonics"]) + 1):
            f = base * k
            if f >= sfreq / 2:
                break
            b, a = signal.iirnotch(f, float(params["quality"]), fs=sfreq)
            sections.append(signal.tf2sos(b, a))
        sos: SosArray = np.vstack(sections)
        return sos

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos(self.design(params, block.info.sfreq), block, state)


def _apply_sos_zero_phase(sos: SosArray, block: Block) -> TransformResult:
    data = np.asarray(block.data)
    try:
        out = signal.sosfiltfilt(sos, data, axis=-1)
    except ValueError as exc:  # input shorter than the edge padding
        raise ValueError(
            f"zero-phase filtering needs a longer recording ({block.n_samples} "
            f"samples): {exc}"
        ) from exc
    return TransformResult(
        Block(out, block.info, block.sample_index, block.timestamp), {}
    )


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
                }.items()
                if v is not None
            }
            for p in causal.params
        ],
        description=description,
        availability="offline",
    )


class IIRFilterZeroPhase(Processor):
    """Forward-backward Butterworth filter (offline only, no phase shift)."""

    manifest: ClassVar[Manifest] = _zero_phase_manifest(
        IIRFilter.manifest,
        "meegxer.sterile.iir_filter_zerophase",
        "Zero-phase (forward-backward) Butterworth IIR filter; offline only.",
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return _apply_sos_zero_phase(IIRFilter.design(params, block.info.sfreq), block)


class NotchFilterZeroPhase(Processor):
    """Forward-backward notch cascade (offline only, no phase shift)."""

    manifest: ClassVar[Manifest] = _zero_phase_manifest(
        NotchFilter.manifest,
        "meegxer.sterile.notch_zerophase",
        "Zero-phase (forward-backward) IIR notch; offline only.",
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        sos = NotchFilter.design(params, block.info.sfreq)
        return _apply_sos_zero_phase(sos, block)
