"""References for the M6 feature checks (acceptance test 6).

Runs GNU Octave on ``tests/data/m6_eeg.npz`` and stores the results in
``tests/data/m6_ref_features.npz``, so CI needs no Octave.

    uv run python tools/make_m6_feature_references.py

- PSD: for every 2 s window (1 s step) of each channel, ``pwelch`` with a 1 s
  Hamming window, 50 % overlap, as EEGLAB's ``spectopo`` calls MATLAB's pwelch. Octave's
  pwelch removes the mean by default, which MATLAB's does not, so the reference
  passes ``'no-strip'``.
- Hilbert envelope: ``abs(hilbert(x))`` over the whole recording of the signal
  band-passed to 8-13 Hz (4th-order Butterworth, forward-backward), stored with it.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "m6_eeg.npz"
WINDOW, STEP, SEGMENT = 2.0, 1.0, 1.0

SCRIPT = r"""
pkg load signal;
function A = readbin(name, shape)
  fid = fopen(name); A = fread(fid, shape, "float64"); fclose(fid);
end
X = readbin(fullfile(TMP, "x.bin"), [NCH, Inf]);
B = readbin(fullfile(TMP, "alpha.bin"), [NCH, Inf]);
n = WIN; step = STEP; seg = SEG;
starts = 0:step:(columns(X) - n);
P = zeros(numel(starts), NCH, seg / 2 + 1);
for w = 1:numel(starts)
  for c = 1:NCH
    x = X(c, starts(w) + (1:n));
    P(w, c, :) = pwelch(x, hamming(seg), 0.5, seg, SRATE, "onesided", [], "no-strip");
  end
end
E = abs(hilbert(B'))';
fid = fopen(fullfile(TMP, "out.bin"), "w");
fwrite(fid, P, "float64"); fwrite(fid, E, "float64"); fclose(fid);
"""


def alpha_band(x: np.ndarray, sfreq: float) -> np.ndarray:
    sos = sps.butter(4, [8.0, 13.0], btype="bandpass", fs=sfreq, output="sos")
    return np.asarray(sps.sosfiltfilt(sos, x, axis=1))


def main() -> None:
    rec = np.load(DATA)
    x = rec["data"].astype(np.float64)
    sfreq = float(rec["sfreq"])
    n_ch = x.shape[0]
    alpha = alpha_band(x, sfreq).astype(np.float32).astype(np.float64)  # as stored
    win, step, seg = (round(v * sfreq) for v in (WINDOW, STEP, SEGMENT))
    n_win = (x.shape[1] - win) // step + 1
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        np.asfortranarray(x).T.tofile(t / "x.bin")  # Octave reads column by column
        np.asfortranarray(alpha).T.tofile(t / "alpha.bin")
        head = (f'TMP = "{t}"; NCH = {n_ch}; SRATE = {sfreq}; WIN = {win}; '
                f"STEP = {step}; SEG = {seg};\n")  # fmt: skip
        subprocess.run(["octave-cli", "--no-gui", "--quiet", "--norc", "--eval",
                        head + SCRIPT], check=True)  # fmt: skip
        flat = np.fromfile(t / "out.bin", dtype=np.float64)
    n_psd = n_win * n_ch * (seg // 2 + 1)
    psd = flat[:n_psd].reshape((n_win, n_ch, seg // 2 + 1), order="F")
    envelope = flat[n_psd:].reshape(x.shape, order="F")
    arrays: dict[str, Any] = {
        "psd": psd,
        "freqs": np.arange(seg // 2 + 1) * sfreq / seg,
        "window_starts": np.arange(n_win) * step,
        "alpha": alpha.astype(np.float32),
        "envelope": envelope.astype(np.float32),
        "settings": np.array(f"window {WINDOW} s, step {STEP} s, segment {SEGMENT} s, "
                             "hamming, overlap 0.5; alpha 8-13 Hz butter(4) filtfilt"),
    }  # fmt: skip
    dest = ROOT / "tests" / "data" / "m6_ref_features.npz"
    np.savez_compressed(dest, **arrays)
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB): {n_win} PSD windows")


if __name__ == "__main__":
    main()
