"""The SSVEP → keypress demo of docs/TESTING.md §N (M11's exit criterion):
a synthetic SSVEP recording and the SSVEP BCI template's pipeline reading it.

``uv run python tools/ssvep_demo.py OUT_DIR`` writes

- ``OUT_DIR/ssvep.edf``: 24 trials, each 5 s looking at one of four targets
  flickering at 8, 10, 12 or 15 Hz (cue ``target_8`` … at its start), then
  2 s of rest: the SSVEP (fundamental and 2nd harmonic) on the occipital
  channels over background noise, 250 Hz;
- ``OUT_DIR/ssvep_keys.json``: the SSVEP BCI template with this recording as
  its input instead of the LSL amplifier: SSVEP detection (filter-bank CCA,
  detections on) → the keyboard strip (``ssvep_8=left``, ``ssvep_10=right``,
  ``ssvep_12=up``, ``ssvep_15=down``), and *Save as .set*
  (``OUT_DIR/ssvep_out.set``: the cues and the detections).

Play it live: ``uv run meegxer daemon --pipeline OUT_DIR/ssvep_keys.json
--start``, then ``uv run meegxer arm keys`` (or the GUI: *Open…*, live, *Run*,
the keyboard strip's *Arm*).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import numpy as np

CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Cz"]
SFREQ = 250.0
TARGETS = (8.0, 10.0, 12.0, 15.0)
TRIALS, LOOK, REST = 24, 5.0, 2.0


def recording(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    period = LOOK + REST
    n = int(TRIALS * period * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.standard_normal((len(CHANNELS), n)) * 5e-6
    order = rng.permutation(np.repeat(TARGETS, TRIALS // len(TARGETS)))
    onsets = []
    for k, freq in enumerate(order):
        onset = k * period + 1.0
        onsets.append(onset)
        span = slice(int(onset * SFREQ), int((onset + LOOK) * SFREQ))
        for ch, name in enumerate(CHANNELS):
            if name == "Cz":
                continue  # far from the visual cortex
            phase = rng.uniform(0, 2 * np.pi)
            wave = (np.sin(2 * np.pi * freq * t[span] + phase)
                    + 0.5 * np.sin(4 * np.pi * freq * t[span] + 2 * phase))  # fmt: skip
            data[ch, span] += 2e-6 * wave
    raw = mne.io.RawArray(data, mne.create_info(CHANNELS, SFREQ, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_montage("standard_1020", verbose="error")
    labels = [f"target_{f:g}" for f in order]
    raw.set_annotations(mne.Annotations(onsets, [LOOK] * TRIALS, labels))
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")


def pipeline(out: Path) -> dict[str, object]:
    from meegxer.core.templates import template_pipeline

    spec = template_pipeline("ssvep_bci")
    nodes: list[dict[str, object]] = spec["nodes"]
    nodes[0] = {"id": "input", "node": "meegxer.core.file",
                "params": {"path": str((out / "ssvep.edf").resolve())}}  # fmt: skip
    nodes.append({"id": "out", "node": "meegxer.core.set_writer", "input": "input",
                  "params": {"path": str((out / "ssvep_out.set").resolve()),
                             "overwrite": True}})  # fmt: skip
    return spec


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    recording(out / "ssvep.edf")
    path = out / "ssvep_keys.json"
    path.write_text(json.dumps(pipeline(out), indent=2) + "\n")
    print(f"wrote {out / 'ssvep.edf'} and {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
