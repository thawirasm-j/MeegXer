"""Which frequencies a stream's filters have changed (``SignalInfo.filters``).

A 1/f (aperiodic) fit should see only the frequencies the filters left alone:
below a high-pass cutoff and above a low-pass one the spectrum bends with the
filter's roll-off, and a notch carves a dip. The margins are wide enough for
the filters MeegXer builds (Butterworth order 4, MNE's FIR, IIR notches at
Q 30): below 1.5 x a high-pass cutoff, above 0.9 x a low-pass cutoff, and
within 2 Hz of a notch or a band-stop's edges count as filtered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from meegxer.core.block import Filter

HIGHPASS_MARGIN = 1.5  # x the cutoff
LOWPASS_MARGIN = 0.9  # x the cutoff
NOTCH_HALF_WIDTH = 2.0  # Hz either side


def filtered_frequencies(
    freqs: npt.ArrayLike, filters: Sequence[Filter]
) -> npt.NDArray[np.bool_]:
    """True where ``freqs`` (Hz) lie in a band one of ``filters`` changed."""
    f = np.asarray(freqs, dtype=float)
    out = np.zeros(f.shape, dtype=bool)
    for kind, low, high in filters:
        if kind == "highpass":
            out |= f < HIGHPASS_MARGIN * low
        elif kind == "lowpass":
            out |= f > LOWPASS_MARGIN * high
        elif kind == "notch":
            out |= np.abs(f - low) <= NOTCH_HALF_WIDTH
        elif kind == "bandstop":
            out |= (f >= low - NOTCH_HALF_WIDTH) & (f <= high + NOTCH_HALF_WIDTH)
    return out


def filters_wire(filters: Sequence[Filter]) -> list[list[object]]:
    """``filters`` for JSON (protocols, status)."""
    return [[kind, low, high] for kind, low, high in filters]


def filters_from_wire(wire: object) -> tuple[Filter, ...]:
    """Inverse of :func:`filters_wire` (empty for anything missing)."""
    if not isinstance(wire, list):
        return ()
    return tuple((str(k), float(lo), float(hi)) for k, lo, hi in wire)
