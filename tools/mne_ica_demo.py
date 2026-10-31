"""The MNE ICA check of docs/TESTING.md §H (M10.7).

``uv run python tools/mne_ica_demo.py OUT_DIR`` writes the ICA test
recording (19 channels, 60 s, tests/data/m6_ica.npz) as ``ica_demo_raw.fif``
and MNE's own ICA of it (Picard, 15 components, on average-referenced data,
component 1 marked excluded) as ``ica_demo-ica.fif``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import mne
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"


def main(out: Path) -> None:
    rec = np.load(DATA / "m6_ica.npz")
    labels = [str(s) for s in rec["labels"]]
    info = mne.create_info(labels, float(rec["sfreq"]), "eeg")
    raw = mne.io.RawArray(rec["data"].astype(np.float64), info, verbose="error")
    montage = mne.channels.make_dig_montage(
        dict(zip(labels, rec["positions"], strict=True)), coord_frame="head"
    )
    raw.set_montage(montage, verbose="error")
    raw.set_eeg_reference(projection=True, verbose="error")
    ica = mne.preprocessing.ICA(n_components=15, method="picard", max_iter=500,
                                random_state=0)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose="error")
    ica.exclude = [0]
    out.mkdir(parents=True, exist_ok=True)
    raw.save(out / "ica_demo_raw.fif", overwrite=True, verbose="error")
    ica.save(out / "ica_demo-ica.fif", overwrite=True, verbose="error")
    print(f"wrote {out / 'ica_demo_raw.fif'} and {out / 'ica_demo-ica.fif'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
