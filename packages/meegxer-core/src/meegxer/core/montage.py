"""Display montages and display filters (docs/REVIEW.md, Design v2).

What a reviewer sees, never what the pipeline processes: the offline views ask
for a page through a montage and a filter, and the core computes it from the
probe's raw samples (``ProbeStore.window``). A montage is a matrix over the
channels:

- ``as recorded``: the channels as they are.
- ``average``: each EEG channel minus the mean of the good EEG channels
  (monopolar against the average reference; the default for review). Other
  channels (EOG, ECG, …) stay as they are.
- ``longitudinal bipolar`` (the "double banana") and ``transverse bipolar``:
  differences of neighbouring 10-20 electrodes, those the recording has
  (``T3``/``T4``/``T5``/``T6`` are taken for ``T7``/``T8``/``P7``/``P8``).
  Other channels follow as they are.
- ``laplacian``: each EEG channel minus the mean of its four nearest good EEG
  neighbours (Hjorth's local average), by electrode position.

Bad channels are left out of every reference (the average, a Laplacian's
neighbours); a bipolar pair with a bad electrode is still shown.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from meegxer.core.block import SignalInfo

MONTAGES = ("as recorded", "average", "longitudinal bipolar", "transverse bipolar",
            "laplacian")  # fmt: skip
DEFAULT_MONTAGE = "average"

LONGITUDINAL = (
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fz", "Cz"), ("Cz", "Pz"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
)  # fmt: skip
TRANSVERSE = (
    ("Fp1", "Fp2"),
    ("F7", "F3"), ("F3", "Fz"), ("Fz", "F4"), ("F4", "F8"),
    ("T7", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T8"),
    ("P7", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "P8"),
    ("O1", "O2"),
)  # fmt: skip
ALIASES = {"t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8"}  # old 10-20 names
NEIGHBOURS = 4  # for the Laplacian

Matrix = npt.NDArray[np.float64]


def _eeg(info: SignalInfo) -> list[bool]:
    """Which channels are EEG (every channel when the types are not known)."""
    if not info.ch_types:
        return [True] * info.n_channels
    return [t == "eeg" for t in info.ch_types]


def montage_matrix(info: SignalInfo, name: str) -> tuple[Matrix, tuple[str, ...]]:
    """The rows (one per shown channel) that make montage ``name`` from the
    channels of ``info``, and the shown channels' labels."""
    n = info.n_channels
    eeg = _eeg(info)
    good = [e and label not in info.bads for e, label in zip(eeg, info.ch_labels,
                                                             strict=True)]  # fmt: skip
    others = [i for i in range(n) if not eeg[i]]
    if name == "as recorded":
        return np.eye(n), info.ch_labels
    if name == "average":
        m = np.eye(n)
        if any(good):
            ref = np.where(good, 1.0 / sum(good), 0.0)
            for i in range(n):
                if eeg[i]:
                    m[i] -= ref
        return m, info.ch_labels
    if name in ("longitudinal bipolar", "transverse bipolar"):
        where = {ALIASES.get(lab.lower(), lab.lower()): i
                 for i, lab in enumerate(info.ch_labels) if eeg[i]}  # fmt: skip
        pairs = LONGITUDINAL if name.startswith("longitudinal") else TRANSVERSE
        rows: list[Matrix] = []
        labels: list[str] = []
        for a, b in pairs:
            ia, ib = where.get(a.lower()), where.get(b.lower())
            if ia is None or ib is None:
                continue
            row = np.zeros(n)
            row[ia], row[ib] = 1.0, -1.0
            rows.append(row)
            labels.append(f"{info.ch_labels[ia]}-{info.ch_labels[ib]}")
        if not rows:
            raise ValueError(f"this recording has none of the electrode pairs of "
                             f"the {name} montage (10-20 names)")  # fmt: skip
        for i in others:
            rows.append(np.eye(n)[i])
            labels.append(info.ch_labels[i])
        return np.array(rows), tuple(labels)
    if name == "laplacian":
        return _laplacian(info, eeg, good), info.ch_labels
    raise ValueError(f"unknown montage {name!r}; one of {', '.join(MONTAGES)}")


def _laplacian(info: SignalInfo, eeg: list[bool], good: list[bool]) -> Matrix:
    n = info.n_channels
    pos = info.positions or (None,) * n
    placed = [i for i in range(n) if good[i] and pos[i] is not None]
    if len(placed) <= NEIGHBOURS:
        raise ValueError("the Laplacian montage needs electrode positions for at "
                         f"least {NEIGHBOURS + 1} good EEG channels (set a "
                         "montage on the input)")  # fmt: skip
    xyz = np.array([pos[i] for i in placed], dtype=float)
    m = np.eye(n)
    for i in range(n):
        p = pos[i]
        if not eeg[i] or p is None:
            continue
        dist = np.linalg.norm(xyz - np.asarray(p, dtype=float), axis=1)
        order = [placed[k] for k in np.argsort(dist) if placed[k] != i]
        for j in order[:NEIGHBOURS]:
            m[i, j] -= 1.0 / NEIGHBOURS
    return m


def filter_pad(sfreq: float, highpass: float) -> int:
    """Samples read on each side of a page so the display filter settles."""
    seconds = 1.0 if highpass <= 0 else min(10.0, max(1.0, 3.0 / highpass))
    return round(seconds * sfreq)


def check_filter(sfreq: float, highpass: float, lowpass: float, notch: float) -> None:
    nyquist = sfreq / 2
    for name, f in (("highpass", highpass), ("lowpass", lowpass), ("notch", notch)):
        if f < 0 or f >= nyquist:
            raise ValueError(f"display {name} must be 0 (off) or below "
                             f"{nyquist:g} Hz, got {f:g}")  # fmt: skip
    if highpass and lowpass and highpass >= lowpass:
        raise ValueError("the display highpass must be below the lowpass")


def display_filter(
    data: Matrix, sfreq: float, highpass: float, lowpass: float, notch: float
) -> Matrix:
    """``data`` (channels x samples, no gaps) through zero-phase display
    filters: 2nd-order Butterworth high- and low-pass, and a notch (0: off)."""
    from scipy import signal

    out = np.asarray(data, dtype=float)
    if out.shape[1] < 16:  # too short to filter
        return out
    nyquist = sfreq / 2
    stages = []
    if highpass > 0:
        stages.append(signal.butter(2, highpass / nyquist, "highpass", output="sos"))
    if lowpass > 0:
        stages.append(signal.butter(2, lowpass / nyquist, "lowpass", output="sos"))
    if notch > 0:
        b, a = signal.iirnotch(notch, 30.0, sfreq)
        stages.append(signal.tf2sos(b, a))
    for sos in stages:
        pad = min(out.shape[1] - 1, 3 * (2 * len(sos) + 1))
        out = signal.sosfiltfilt(sos, out, axis=1, padlen=pad)
    return out
