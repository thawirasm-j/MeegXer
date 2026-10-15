"""Shared test recording for the M6 ICA/ICLabel checks (acceptance test 6).

Deterministic EEG-like data built from known sources, saved as
``tests/data/m6_ica.npz``: 19 channels (10-20), 250 Hz, 60 s, in volts, band-limited
to 1-100 Hz and average-referenced, as ICLabel expects. Each source has a dipole
topography on a spherical head (unit sphere, infinite-medium dipole potential):

- 11 brain sources: deep radial dipoles with bursty 1/f activity (slowly modulated,
  so super-Gaussian and separable), alpha on the posterior ones and beta on the
  central ones;
- eye blinks and horizontal eye movements: dipoles in front of the head;
- 2 muscle sources: shallow temporal dipoles with bursty broadband (20-100 Hz) noise;
- heart: a distant dipole below the head with QRS-like pulses at 72 bpm;
- line noise: 50 Hz on a smooth topography;
- channel noise: drifts and pops on one electrode (O2).

The file also stores the sources' kinds and the true mixing matrix. Re-running this
script gives the same file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

LABELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
          "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2")  # fmt: skip
SFREQ = 250.0
SECONDS = 60.0


def positions() -> np.ndarray:
    """Electrode positions (m) from MNE's colin27 10-20 montage, head frame."""
    import mne

    montage = mne.channels.make_standard_montage("colin27_1020")
    pos = montage.get_positions()["ch_pos"]
    return np.array([pos[label] for label in LABELS])


def dipole(unit: np.ndarray, where: np.ndarray, moment: np.ndarray) -> np.ndarray:
    d = unit - where
    return np.asarray((d @ moment) / np.linalg.norm(d, axis=1) ** 3)


def band(x: np.ndarray, lo: float, hi: float, slope: float = 0.0) -> np.ndarray:
    """Band-limit (FFT mask) with an optional 1/f**slope amplitude; unit std."""
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(n, 1 / SFREQ)
    keep = (f >= lo) & (f <= hi)
    gain = np.zeros_like(f)
    gain[keep] = f[keep] ** -slope
    y = np.fft.irfft(spec * gain, n, axis=-1)
    return y / y.std(axis=-1, keepdims=True)


def make() -> dict[str, Any]:
    rng = np.random.default_rng(20261016)
    n = int(SFREQ * SECONDS)
    t = np.arange(n) / SFREQ
    pos = positions()
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    topo, src, kinds = [], [], []

    def add(kind: str, pattern: np.ndarray, activity: np.ndarray, uv: float) -> None:
        pattern = pattern / np.abs(pattern).max()
        topo.append(pattern)
        src.append(activity / activity.std() * uv * 1e-6)
        kinds.append(kind)

    # brain: radial dipoles at depth 0.55 under scattered scalp points
    centres = ["Fz", "F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
    for label in centres:
        u = unit[LABELS.index(label)]
        act = band(rng.standard_normal(n), 1, 45, slope=1.0)
        if label in ("P3", "Pz", "P4", "O1", "O2"):
            bursts = band(rng.standard_normal(n), 0.1, 0.5)  # slow alpha envelope
            act += 1.5 * np.clip(bursts, 0, None) * band(rng.standard_normal(n), 9, 11)
        elif label in ("C3", "Cz", "C4"):
            act += 0.6 * band(rng.standard_normal(n), 18, 24)
        act *= 0.2 + np.abs(band(rng.standard_normal(n), 0.05, 0.5))  # bursty
        add("brain", dipole(unit, 0.55 * u, u), band(act, 1, 100), 8.0)

    eyes = np.array([0.0, 1.15, -0.35])  # in front of the forehead, below Fp
    blinks = np.zeros(n)
    for onset in np.arange(1.5, SECONDS - 1, 3.7) + rng.uniform(-0.8, 0.8, 16):
        blinks += np.exp(-0.5 * ((t - onset) / 0.09) ** 2)
    add("eye", dipole(unit, eyes, np.array([0.0, 0.3, 1.0])), band(blinks, 1, 100),
        60.0)  # fmt: skip
    gaze = np.zeros(n)
    level = 0.0
    for onset in np.arange(2.3, SECONDS, 2.9):
        level = rng.uniform(-1, 1) - 0.3 * level
        gaze[t >= onset] = level
    add("eye", dipole(unit, eyes, np.array([1.0, 0.0, 0.0])), band(gaze, 1, 100),
        25.0)  # fmt: skip

    for label in ("T7", "T8"):
        u = unit[LABELS.index(label)]
        envelope = np.clip(band(rng.standard_normal(n), 0.05, 0.4), 0.3, None) - 0.3
        act = envelope * band(rng.standard_normal(n), 20, 100)
        tangent = np.cross(u, [0.0, 0.0, 1.0])
        add("muscle", dipole(unit, 0.92 * u, tangent + 0.3 * u), band(act, 1, 100),
            12.0)  # fmt: skip

    beats = np.zeros(n)
    for onset in np.arange(0.4, SECONDS, 60 / 72):
        beats += np.exp(-0.5 * ((t - onset) / 0.012) ** 2)  # R wave
        beats -= 0.25 * np.exp(-0.5 * ((t - onset - 0.04) / 0.02) ** 2)  # S wave
        beats += 0.3 * np.exp(-0.5 * ((t - onset - 0.25) / 0.05) ** 2)  # T wave
    heart = np.array([0.4, -0.3, -2.2])
    add("heart", dipole(unit, heart, np.array([0.6, -0.5, 1.0])), band(beats, 1, 100),
        6.0)  # fmt: skip

    line = np.sin(2 * np.pi * 50 * t) * (1 + 0.2 * band(rng.standard_normal(n), 0, 1))
    add("line_noise", dipole(unit, np.array([-0.3, -0.4, 1.6]),
        np.array([0.2, 0.3, 1.0])), line, 4.0)  # fmt: skip

    bad = np.zeros(len(LABELS))
    bad[LABELS.index("O2")] = 1.0
    drift = band(rng.standard_normal(n), 1, 3) + 0.5 * band(
        rng.standard_normal(n), 1, 100
    )
    for onset in (13.0, 31.0, 47.0):
        pop = t >= onset
        drift[pop] += 6 * np.exp(-(t[pop] - onset) / 0.15)
    add("channel_noise", bad, band(drift, 1, 100), 20.0)

    mixing = np.array(topo).T  # (channels, sources)
    sources = np.array(src)
    sensor = band(rng.standard_normal((len(LABELS), n)), 1, 100) * 0.5e-6
    x = mixing @ sources + sensor
    x -= x.mean(axis=0, keepdims=True)  # average reference
    mixing = mixing - mixing.mean(axis=0, keepdims=True)
    return {
        "data": x.astype(np.float32),
        "labels": np.array(LABELS),
        "sfreq": np.array(SFREQ),
        "positions": pos,
        "kinds": np.array(kinds),
        "mixing": mixing,
    }


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "tests" / "data" / "m6_ica.npz"
    rec = make()
    np.savez_compressed(out, **rec)
    kinds = ", ".join(sorted(set(rec["kinds"].tolist())))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {len(rec['kinds'])} "
          f"sources ({kinds})")  # fmt: skip


if __name__ == "__main__":
    main()
