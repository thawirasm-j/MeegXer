"""Spectral parameterization (specparam, formerly FOOOF): each window's spectrum
split into an aperiodic (1/f-like) part and Gaussian peaks above it.

The feature layout is fixed, so peaks are reported per named band: the strongest
peak whose centre falls in the band (NaN when the band has none). Power values are
specparam's: log10 of V^2/Hz for the offset, log10 power above the aperiodic fit
for a peak; a peak's bandwidth is twice its Gaussian standard deviation (Hz).
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params
from meegxer.nodes.features import (
    SPECTRUM_PARAMS,
    WindowedFeatures,
    analysis_manifest,
    parse_bands,
    spectrum,
)


def measures(aperiodic: str, bands: list[tuple[str, float, float]]) -> tuple[str, ...]:
    head = (
        ["offset", "knee", "exponent"]
        if aperiodic == "knee"
        else ["offset", "exponent"]
    )
    peaks = [f"{name}_{field}" for name, _, _ in bands
             for field in ("frequency", "power", "bandwidth")]  # fmt: skip
    return (*head, *peaks, "r_squared", "error")


class SpecParam(WindowedFeatures):
    """Aperiodic offset, exponent (and knee), the strongest peak per band, and the
    fit quality (R², mean absolute error), per channel and window."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.specparam",
        [
            *SPECTRUM_PARAMS,
            {"name": "fmin", "type": "float", "default": 1.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "fmax", "type": "float", "default": 40.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "aperiodic", "type": "enum", "default": "fixed",
             "choices": ["fixed", "knee"], "primary": True,
             "description": "Aperiodic mode: a straight line in log-log, or with "
                            "a knee (for wide frequency ranges)"},
            {"name": "bands", "type": "string", "default": "theta:4-8, alpha:8-13, "
             "beta:13-30", "description": "Report the strongest peak in each "
             "band (name:low-high in Hz, comma-separated)"},
            {"name": "max_n_peaks", "type": "int", "default": 6, "min": 0,
             "max": 20, "advanced": True},
            {"name": "peak_width_min", "type": "float", "default": 1.0, "min": 0.0,
             "unit": "Hz", "advanced": True},
            {"name": "peak_width_max", "type": "float", "default": 12.0,
             "min": 0.0, "unit": "Hz", "advanced": True},
            {"name": "min_peak_height", "type": "float", "default": 0.0,
             "min": 0.0, "advanced": True,
             "description": "Smallest peak kept, in log10 power above the "
                            "aperiodic fit"},
            {"name": "peak_threshold", "type": "float", "default": 2.0,
             "min": 0.0, "advanced": True,
             "description": "Smallest peak kept, in standard deviations of the "
                            "flattened spectrum"},
        ],
        "Aperiodic (1/f) and periodic (peak) parameters of the spectrum.",
        window=4.0,
        step=1.0,
    )  # fmt: skip

    @classmethod
    def warm_up(cls) -> None:
        import specparam  # noqa: F401

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        from specparam import SpectralModel

        aperiodic = str(params["aperiodic"])
        bands = parse_bands(str(params["bands"]))
        fmin, fmax = float(params["fmin"]), float(params["fmax"])
        freqs, psd = spectrum(x, info.sfreq, params)
        keep = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
        if keep.sum() < 4:
            raise ValueError("specparam needs at least 4 frequencies in its range; "
                             "lengthen the window or widen the range")  # fmt: skip
        names = measures(aperiodic, bands)
        rows = np.full((x.shape[0], len(names)), np.nan)
        for i, power in enumerate(np.maximum(psd[:, keep], np.finfo(float).tiny)):
            model = SpectralModel(
                aperiodic_mode=aperiodic,
                peak_width_limits=(float(params["peak_width_min"]),
                                   float(params["peak_width_max"])),
                max_n_peaks=int(params["max_n_peaks"]),
                min_peak_height=float(params["min_peak_height"]),
                peak_threshold=float(params["peak_threshold"]),
                verbose=False,
            )  # fmt: skip
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    model.fit(freqs[keep], power)
                except Exception:  # a failed fit leaves NaNs
                    continue
            if not model.results.has_model:
                continue
            rows[i] = _row(model, aperiodic, bands)
        coords = {"channel": info.ch_labels, "measure": names}
        return rows, ("channel", "measure"), coords


def _row(
    model: Any, aperiodic: str, bands: list[tuple[str, float, float]]
) -> list[float]:
    row = [float(v) for v in np.atleast_1d(model.get_params("aperiodic"))]
    peaks = np.asarray(model.get_params("periodic"), dtype=float).reshape(-1, 3)
    for _, lo, hi in bands:
        inside = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] < hi)]
        if len(inside):
            row.extend(float(v) for v in inside[np.argmax(inside[:, 1])])
        else:
            row.extend([np.nan] * 3)
    row.append(float(model.get_metrics("gof")))
    row.append(float(model.get_metrics("error")))
    return row
