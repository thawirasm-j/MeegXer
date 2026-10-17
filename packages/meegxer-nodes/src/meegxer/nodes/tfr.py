"""Time-frequency features: Morlet wavelet power.

Each window is convolved with MNE's Morlet wavelets (zero-mean, as
``tfr_array_morlet`` uses them) and the power |x * w|^2 is averaged over the
samples where the whole wavelet lies inside the window, so no zero padding enters
the estimate. A wavelet longer than the window is refused with a hint.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scipy import signal as sps

from meegxer.core.block import SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params
from meegxer.nodes.features import WindowedFeatures, analysis_manifest


def frequencies(params: Params) -> np.ndarray:
    fmin, fmax, fstep = (float(params[k]) for k in ("fmin", "fmax", "fstep"))
    if not 0 < fmin <= fmax or fstep <= 0:
        raise ValueError("Morlet frequencies need 0 < fmin <= fmax and fstep > 0")
    return np.arange(fmin, fmax + fstep / 2, fstep)


class MorletPower(WindowedFeatures):
    """Morlet wavelet power per channel, frequency and window (V^2, or dB)."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.morlet",
        [
            {"name": "fmin", "type": "float", "default": 4.0, "min": 0.1,
             "unit": "Hz", "primary": True},
            {"name": "fmax", "type": "float", "default": 40.0, "min": 0.1,
             "unit": "Hz", "primary": True},
            {"name": "fstep", "type": "float", "default": 1.0, "min": 0.01,
             "unit": "Hz", "advanced": True},
            {"name": "n_cycles", "type": "float", "default": 7.0, "min": 1.0,
             "description": "Cycles per wavelet: more gives finer frequency and "
                            "coarser time resolution"},
            {"name": "db", "type": "bool", "default": False,
             "description": "Report 10*log10(power) instead of V^2"},
        ],
        "Wavelet (Morlet) power per channel and frequency.",
        window=4.0,
        step=0.5,
    )  # fmt: skip

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        from mne.time_frequency import morlet

        freqs = frequencies(params)
        wavelets = morlet(info.sfreq, freqs, n_cycles=float(params["n_cycles"]),
                          zero_mean=True)  # fmt: skip
        longest = max(len(w) for w in wavelets)
        if longest > x.shape[1]:
            raise ValueError(
                f"the {freqs[0]:g} Hz wavelet spans {longest / info.sfreq:.2f} s, "
                f"longer than the {x.shape[1] / info.sfreq:.2f} s window; lengthen "
                "the window, raise fmin or lower the cycles"
            )
        power = np.empty((x.shape[0], len(freqs)))
        for j, w in enumerate(wavelets):
            conv = sps.fftconvolve(x, w[None, :], mode="valid", axes=1)
            power[:, j] = np.mean(np.abs(conv) ** 2, axis=1)
        if params["db"]:
            power = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
        rounded = tuple(round(float(f), 6) for f in freqs)
        coords = {"channel": info.ch_labels, "frequency": rounded}
        return power, ("channel", "frequency"), coords
