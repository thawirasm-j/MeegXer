"""SSVEP frequency detection by canonical correlation analysis (CCA).

For each stimulus frequency f, the window's good channels are correlated with
sine and cosine references at f, 2f, ... (harmonics below Nyquist); the score is
the largest canonical correlation (Lin et al. 2006). Filter-bank CCA (Chen et al.
2015) repeats this on sub-bands [m * low, high] for m = 1..N (zero-phase
Chebyshev type I filters on the window) and scores sum_m (m^-1.25 + 0.25) * r_m^2.
No calibration is needed. The target with the highest score is the detected one.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
from scipy import signal as sps

from meegxer.core.block import SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params
from meegxer.nodes.features import WindowedFeatures, analysis_manifest

FBCCA_A, FBCCA_B = 1.25, 0.25  # sub-band weights m^-a + b (Chen et al. 2015)


def parse_frequencies(text: str) -> list[float]:
    try:
        freqs = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise ValueError(
            f"bad stimulus frequencies {text!r}; use e.g. 8, 10, 12"
        ) from None
    if not freqs or any(f <= 0 for f in freqs):
        raise ValueError("give at least one stimulus frequency above 0 Hz")
    return freqs


def _basis(x: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the columns of ``x`` (samples x k) after centring;
    rank-deficient directions (e.g. after an average reference) are dropped."""
    u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    if not s.size or s[0] == 0:
        return u[:, :0]
    return u[:, s > s[0] * 1e-10]


def canonical_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Largest canonical correlation between the columns of x and y."""
    qx, qy = _basis(x), _basis(y)
    if not qx.shape[1] or not qy.shape[1]:
        return 0.0
    return float(min(1.0, np.linalg.svd(qx.T @ qy, compute_uv=False)[0]))


def references(freq: float, harmonics: int, n: int, sfreq: float) -> np.ndarray:
    """Sine/cosine references (samples x 2k) at freq .. k*freq, below Nyquist."""
    t = np.arange(n) / sfreq
    cols = []
    for h in range(1, harmonics + 1):
        if h * freq >= sfreq / 2:
            break
        cols += [np.sin(2 * np.pi * h * freq * t), np.cos(2 * np.pi * h * freq * t)]
    return np.column_stack(cols) if cols else np.zeros((n, 0))


@lru_cache(maxsize=64)
def subband(sfreq: float, low: float, high: float) -> np.ndarray:
    """Chebyshev type I sub-band filter (SOS) as in Chen et al.'s FBCCA code:
    3 dB pass ripple bound, 40 dB stop, 0.5 dB ripple, 2 Hz / 10 Hz transitions."""
    nyq = sfreq / 2
    if low >= 0.9 * nyq:
        raise ValueError(
            f"sub-band from {low:g} Hz is too close to Nyquist ({nyq:g} Hz)"
        )
    stop_low = max(low - 2.0, low / 2)
    if high + 10 >= nyq:  # no room for an upper edge: high-pass only
        order, edge = sps.cheb1ord(low, stop_low, 3, 40, fs=sfreq)
        return sps.cheby1(order, 0.5, edge, btype="highpass", fs=sfreq, output="sos")
    order, edges = sps.cheb1ord([low, high], [stop_low, high + 10], 3, 40, fs=sfreq)
    return sps.cheby1(order, 0.5, edges, btype="bandpass", fs=sfreq, output="sos")


class SSVEPCCA(WindowedFeatures):
    """One score per stimulus frequency and window: CCA or filter-bank CCA."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.ssvep_cca",
        [
            {"name": "frequencies", "type": "string", "default": "8, 10, 12, 15",
             "primary": True,
             "description": "Stimulus frequencies in Hz, comma-separated"},
            {"name": "method", "type": "enum", "default": "fbcca",
             "choices": ["cca", "fbcca"], "primary": True,
             "description": "Standard CCA, or filter-bank CCA (uses the "
                            "harmonics' sub-bands; usually more accurate)"},
            {"name": "harmonics", "type": "int", "default": 3, "min": 1, "max": 10,
             "description": "Reference harmonics per frequency"},
            {"name": "subbands", "type": "int", "default": 5, "min": 1, "max": 10,
             "advanced": True, "description": "Filter-bank CCA sub-bands"},
            {"name": "subband_low", "type": "float", "default": 8.0, "min": 2.5,
             "unit": "Hz", "advanced": True,
             "description": "Lower edge of the first sub-band, and the spacing "
                            "between sub-bands"},
            {"name": "subband_high", "type": "float", "default": 88.0, "min": 5.0,
             "unit": "Hz", "advanced": True,
             "description": "Upper edge of every sub-band"},
        ],
        "SSVEP target scores by canonical correlation (CCA, filter-bank CCA).",
        window=2.0,
        step=0.5,
    )  # fmt: skip

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        freqs = parse_frequencies(str(params["frequencies"]))
        harmonics = int(params["harmonics"])
        good = [i for i, ch in enumerate(info.ch_labels) if ch not in set(info.bads)]
        data = x[good].T  # samples x channels
        refs = [references(f, harmonics, data.shape[0], info.sfreq) for f in freqs]
        if params["method"] == "cca":
            scores = [canonical_correlation(data, y) for y in refs]
        else:
            low, high = float(params["subband_low"]), float(params["subband_high"])
            scores = [0.0] * len(freqs)
            for m in range(1, int(params["subbands"]) + 1):
                if m * low >= min(high, 0.9 * info.sfreq / 2):
                    break  # higher sub-bands lie beyond the top edge or Nyquist
                sos = subband(info.sfreq, m * low, high)
                band = sps.sosfiltfilt(sos, data, axis=0)
                weight = m**-FBCCA_A + FBCCA_B
                for k, y in enumerate(refs):
                    scores[k] += weight * canonical_correlation(band, y) ** 2
        coords = {"frequency": tuple(round(f, 6) for f in freqs)}
        return np.asarray(scores, dtype=float), ("frequency",), coords
