"""The XDF check of docs/TESTING.md §F (F4, F5).

Write the pipeline: ``uv run python tools/xdf_check.py write XDF OUT_DIR``
writes ``OUT_DIR/xdf_epochs.json``: the XDF file as the input, an Epochs strip
on ``Marker`` (-200 to 800 ms), the recording as ``out.set`` (its markers as
events) and the epochs as ``out-epo.fif``.

Run it: ``uv run meegxer run OUT_DIR/xdf_epochs.json``.

Check what it wrote: ``uv run python tools/xdf_check.py check XDF OUT_DIR``
prints the number of epochs, whether any spans a ``BAD_gap`` of the XDF file,
and how many events ``out.set`` holds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def write(xdf: Path, out: Path) -> None:
    from meegxer.core.manifest import SPEC_VERSION

    out.mkdir(parents=True, exist_ok=True)
    pipeline = {
        "spec_version": SPEC_VERSION,
        "nodes": [
            {"id": "input", "node": "meegxer.core.file",
             "params": {"path": str(xdf.resolve())}},
            {"id": "ep", "node": "meegxer.nodes.epochs",
             "params": {"events": "Marker", "tmin": -0.2, "tmax": 0.8}},
            {"id": "set", "node": "meegxer.core.set_writer",
             "params": {"path": str((out / "out.set").resolve()),
                        "overwrite": True}},
            {"id": "epochs", "node": "meegxer.core.epochs_writer",
             "input": "ep:epochs",
             "params": {"path": str((out / "out-epo.fif").resolve()),
                        "overwrite": True}},
        ],
    }  # fmt: skip
    path = out / "xdf_epochs.json"
    path.write_text(json.dumps(pipeline, indent=2) + "\n")
    print(f"wrote {path}\nnext: uv run meegxer run {path}")


def check(xdf: Path, out: Path) -> int:
    import mne

    from meegxer.core.xdf import GAP, read_xdf

    raw = read_xdf(xdf).raw
    sfreq = raw.info["sfreq"]
    gaps = [(o, o + d) for o, d, lb in zip(raw.annotations.onset,
            raw.annotations.duration, raw.annotations.description, strict=True)
            if lb == GAP]  # fmt: skip
    epochs = mne.read_epochs(out / "out-epo.fif", verbose="error")
    starts = epochs.events[:, 0] / sfreq + epochs.tmin
    stops = epochs.events[:, 0] / sfreq + epochs.tmax
    across = [k for k, (a, b) in enumerate(zip(starts, stops, strict=True))
              if any(a < g1 and g0 < b for g0, g1 in gaps)]  # fmt: skip
    written = mne.io.read_raw_eeglab(out / "out.set", verbose="error")
    print(f"F4: {len(epochs)} epochs; gaps at "
          + ", ".join(f"{g0:.2f} s" for g0, _ in gaps)
          + f"; epochs across a gap: {len(across)}"
          + (" (PASS)" if not across else " (FAIL)"))  # fmt: skip
    print(
        f"F5: out.set holds {len(written.annotations)} events "
        f"({len(raw.annotations)} in the XDF file, BAD_gap included)"
    )
    return 1 if across else 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("write", "check"):
        sys.exit(__doc__)
    xdf_file, folder = Path(sys.argv[2]), Path(sys.argv[3])
    if sys.argv[1] == "write":
        write(xdf_file, folder)
    else:
        sys.exit(check(xdf_file, folder))
