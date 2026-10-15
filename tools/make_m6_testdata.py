"""Shared test recording for the M6 concordance checks (acceptance test 6).

Deterministic EEG-like data with artifacts, saved as ``tests/data/m6_eeg.npz``:
8 channels (10-20 labels), 128 Hz, 30 s, in volts. Background is 1/f noise above
1 Hz (as after a high-pass) plus a posterior alpha rhythm; artifacts are eye blinks
(frontal, slow, large), muscle bursts (temporal, broadband) and an electrode pop.
Re-running this script gives the same file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABELS = ("Fp1", "Fp2", "C3", "C4", "T7", "T8", "O1", "O2")
SFREQ = 128.0
SECONDS = 30.0


def make() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(20261015)
    n = int(SFREQ * SECONDS)
    t = np.arange(n) / SFREQ
    spec = np.fft.rfft(rng.standard_normal((len(LABELS), n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / SFREQ)
    spec[:, 1:] /= np.sqrt(f[1:])  # 1/f power
    spec[:, f < 1.0] = 0  # high-passed at 1 Hz, as ASR expects
    x = np.fft.irfft(spec, n, axis=1)
    x /= x.std(axis=1, keepdims=True)
    x *= 10e-6  # 10 uV background
    alpha = 8e-6 * np.sin(2 * np.pi * 10 * t)
    x[6:8] += alpha  # O1, O2
    notes = []
    for onset in (5.0, 12.5, 21.0):  # blinks: frontal, ~300 ms, 150 uV
        k = np.exp(-0.5 * ((t - onset) / 0.08) ** 2) * 150e-6
        x[0] += k
        x[1] += 0.9 * k
        notes.append(f"blink {onset} s")
    for onset in (9.0, 17.0):  # muscle: temporal, 1 s, broadband 40 uV
        mask = (t >= onset) & (t < onset + 1.0)
        burst = rng.standard_normal((2, mask.sum())) * 40e-6
        x[4:6, mask] += burst
        notes.append(f"muscle {onset} s")
    pop = (t >= 25.0) & (t < 25.4)  # electrode pop on C3
    x[2, pop] += 200e-6 * np.exp(-(t[pop] - 25.0) / 0.1)
    notes.append("pop C3 25 s")
    return x, notes


def main() -> None:
    x, notes = make()
    out = Path(__file__).resolve().parents[1] / "tests" / "data" / "m6_eeg.npz"
    np.savez_compressed(out, data=x.astype(np.float32), labels=np.array(LABELS),
                        sfreq=SFREQ, notes=np.array(notes))  # fmt: skip
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {', '.join(notes)}")


if __name__ == "__main__":
    main()
