"""The review checks of docs/TESTING.md §J (review report) and §K (two
reviewers).

``uv run python tools/review_demo.py OUT_DIR`` writes two small BIDS datasets
(2 subjects, 60 s, Fz Cz Pz Oz and an EOG channel with blinks), each with a
pipeline next to it that runs every subject:

- ``OUT_DIR/j/review.json`` (§J): a blink detector, then a Review strip
  asking about its blinks (``points: blink``), then a BIDS writer;
- ``OUT_DIR/k/two_reviewers.json`` (§K): the same, with *Reviewers* 2.

Open one in the GUI (*Open…*) and *Run offline*: each recording waits for its
review. The QC reports and ``qc.html`` go to ``…/derivatives/meegxer``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mne
import mne_bids
import numpy as np

CHANNELS = ["Fz", "Cz", "Pz", "Oz", "EOG"]
BLINKS = {"01": (5.0, 17.0, 33.0, 48.0), "02": (8.0, 21.0, 40.0, 52.0)}


def dataset(root: Path) -> None:
    info = mne.create_info(CHANNELS, 250.0, ["eeg"] * 4 + ["eog"])
    for i, (subject, blinks) in enumerate(BLINKS.items()):
        rng = np.random.default_rng(i)
        data = rng.standard_normal((5, 250 * 60)) * 1e-5
        for t in blinks:  # a 300 µV, 240 ms blink on the EOG, a tenth of it on Fz
            k = int(t * 250)
            bump = 3e-4 * np.hanning(60)
            data[4, k : k + 60] += bump
            data[0, k : k + 60] += bump / 10
        raw = mne.io.RawArray(data, info, verbose="error")
        raw.set_montage("colin27_1020", on_missing="ignore", verbose="error")
        path = mne_bids.BIDSPath(root=root, subject=subject, task="rest",
                                 datatype="eeg")  # fmt: skip
        mne_bids.write_raw_bids(raw, path, format="EDF", allow_preload=True,
                                overwrite=True, verbose="error")  # fmt: skip


def pipeline(root: Path, reviewers: int) -> dict[str, object]:
    from meegxer.core.manifest import SPEC_VERSION

    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.bids",
         "params": {"root": str(root.resolve()), "subject": "", "task": "rest"}},
        {"id": "blinks1", "node": "meegxer.nodes.detect_blinks",
         "params": {"eog": "EOG"}},
        {"id": "review1", "node": "meegxer.core.review",
         "params": {"points": "blink", "reviewers": reviewers}},
        {"id": "out", "node": "meegxer.core.bids_writer"},
    ]}  # fmt: skip


def main(out: Path) -> None:
    for name, file, reviewers in (("j", "review.json", 1),
                                  ("k", "two_reviewers.json", 2)):  # fmt: skip
        root = out / name / "ds"
        dataset(root)
        path = out / name / file
        path.write_text(json.dumps(pipeline(root, reviewers), indent=2) + "\n")
        print(f"wrote {path} (dataset {root})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
