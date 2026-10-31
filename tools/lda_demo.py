"""The LDA checks of docs/TESTING.md §L: a synthetic motor-imagery recording
and a CSP + LDA pipeline for it.

``uv run python tools/lda_demo.py OUT_DIR`` writes

- ``OUT_DIR/mi.edf``: 40 trials of imagined left- or right-hand movement, 6 s
  apart (cues ``left``/``right`` at each trial's start, imagery from 0.5 to 4
  s): a 10 Hz mu rhythm on C3 and C4 that weakens over the hemisphere opposite
  the imagined hand (event-related desynchronisation), over background noise;
- ``OUT_DIR/csp_lda.json``: File input → Band-pass 8-30 Hz → CSP (2
  components, 1 s windows every 0.25 s) → LDA → *Save as .set*
  (``OUT_DIR/mi_lda.set``, with the Events rail: the cues and LDA's
  ``pred_left``/``pred_right``).

Run it with ``uv run meegxer run OUT_DIR/csp_lda.json`` or open it in the GUI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import numpy as np

CHANNELS = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
SFREQ = 160.0
TRIALS, PERIOD = 40, 6.0


def recording(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(TRIALS * PERIOD * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.standard_normal((len(CHANNELS), n)) * 5e-6
    mu = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
    gain = {ch: np.ones(n) for ch in ("C3", "C4")}
    onsets, labels = [], []
    for k in range(TRIALS):
        label = "left" if rng.random() < 0.5 else "right"
        onset = k * PERIOD + 1.0
        onsets.append(onset)
        labels.append(label)
        span = slice(int((onset + 0.5) * SFREQ), int((onset + 4.0) * SFREQ))
        gain["C4" if label == "left" else "C3"][span] = 0.3  # contralateral ERD
    for ch, g in gain.items():
        data[CHANNELS.index(ch)] += 1e-5 * g * mu
    raw = mne.io.RawArray(data, mne.create_info(CHANNELS, SFREQ, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_montage("standard_1020", verbose="error")
    raw.set_annotations(mne.Annotations(onsets, [0.0] * TRIALS, labels))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")


def pipeline(out: Path) -> dict[str, object]:
    from meegxer.core.manifest import SPEC_VERSION

    trial = {"classes": "left, right", "tmin": 0.5, "tmax": 4.0}
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file",
         "params": {"path": str((out / "mi.edf").resolve())}},
        {"id": "bandpass1", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "bandpass", "low": 8.0, "high": 30.0}},
        {"id": "csp1", "node": "meegxer.nodes.csp",
         "params": {**trial, "n_components": 2, "window": 1.0, "step": 0.25}},
        {"id": "lda1", "node": "meegxer.nodes.lda", "params": trial},
        {"id": "out", "node": "meegxer.core.set_writer", "input": "bandpass1",
         "params": {"path": str((out / "mi_lda.set").resolve())}},
    ]}  # fmt: skip


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    recording(out / "mi.edf")
    path = out / "csp_lda.json"
    path.write_text(json.dumps(pipeline(out), indent=2) + "\n")
    print(f"wrote {out / 'mi.edf'} and {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
