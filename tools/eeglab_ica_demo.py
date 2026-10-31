"""The EEGLAB decomposition check of docs/TESTING.md §G (M10.6).

``uv run python tools/eeglab_ica_demo.py OUT_DIR`` writes two EEGLAB files of
the ICA test recording (19 channels, 60 s, tests/data/m6_ica.npz) carrying
EEGLAB's own decomposition of it (tests/data/m6_ref_ica.npz):

- ``ica_demo.set``: with EEGLAB's ICLabel classes (``etc.ic_classification``);
- ``ica_demo_unclassified.set``: without them (MeegXer computes ICLabel's).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np

from meegxer.core.eeglab_ica import Decomposition, write_decomposition

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"


def main(out: Path) -> None:
    rec, ref = np.load(DATA / "m6_ica.npz"), np.load(DATA / "m6_ref_ica.npz")
    labels = tuple(str(s) for s in rec["labels"])
    info = mne.create_info(list(labels), float(rec["sfreq"]), "eeg")
    raw = mne.io.RawArray(rec["data"].astype(np.float64), info, verbose="error")
    montage = mne.channels.make_dig_montage(
        dict(zip(labels, rec["positions"], strict=True)), coord_frame="head"
    )
    raw.set_montage(montage, verbose="error")
    out.mkdir(parents=True, exist_ok=True)
    w = ref["icaweights"]
    for name, proba in (("ica_demo.set", ref["probabilities"]),
                        ("ica_demo_unclassified.set", None)):  # fmt: skip
        path = out / name
        raw.export(path, fmt="eeglab", overwrite=True, verbose="error")
        dec = Decomposition(labels, w, np.linalg.pinv(w), proba)
        write_decomposition(path, dec, labels)
        print(f"wrote {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
