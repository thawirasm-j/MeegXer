"""The Python API check of docs/TESTING.md §H (M10.7).

``uv run python tools/api_check.py [RECORDING [EVENT]] [--plot]`` runs a
band-pass (1 to 30 Hz) and an Epochs strip on EVENT (-200 to 800 ms) over
RECORDING with ``meegxer.api.run``: once from an ``mne.io.Raw`` read with MNE,
once from the file itself. It prints what came back, and with ``--plot`` shows
the cleaned recording and the ERP in MNE's own windows.

- Without RECORDING: the ERP CORE flanker sample (``meegxer new erp … --sample``
  uses it; downloaded the first time, unless §C already did).
- Without EVENT: the recording's most frequent event label (the others are
  listed).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any


def erp_core() -> tuple[Any, Path]:
    """The ERP CORE flanker recording, where MeegXer's samples download it."""
    import mne

    from meegxer.core.samples import _downloads

    folder = Path(mne.datasets.erp_core.data_path(path=_downloads(),
                                                  update_path=False))  # fmt: skip
    path = sorted(folder.rglob("*Flankers*_eeg.fif"))[0]
    return mne.io.read_raw_fif(path, preload=True, verbose="error"), path


def main(recording: Path | None, event: str | None, plot: bool) -> None:
    import mne

    from meegxer.api import run
    from meegxer.core.manifest import SPEC_VERSION

    if recording is None:
        raw, recording = erp_core()
        print(f"recording: the ERP CORE flanker sample ({recording})")
    else:
        if not recording.is_file():
            sys.exit(f"no such file: {recording}")
        raw = mne.io.read_raw(recording, preload=True, verbose="error")
    labels = Counter(str(d) for d in raw.annotations.description)
    if not labels:
        sys.exit(f"{recording.name} has no events: give a recording with events")
    if event is None:
        event = labels.most_common(1)[0][0]
    others = ", ".join(f"{lb} ({n})" for lb, n in labels.most_common(8))
    print(f"events: {others}{', …' if len(labels) > 8 else ''}; using {event!r}")
    if event not in labels:
        sys.exit(f"{event!r} is not one of the recording's event labels")
    pipeline = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file"},
        {"id": "bp1", "node": "meegxer.nodes.iir_filter",
         "params": {"low": 1.0, "high": 30.0}},
        {"id": "ep1", "node": "meegxer.nodes.epochs",
         "params": {"events": event, "tmin": -0.2, "tmax": 0.8}},
    ]}  # fmt: skip
    clean = run(pipeline, raw)
    print(f"H1 Raw: {len(clean.ch_names)} channels, {clean.times[-1]:.1f} s, "
          f"{len(clean.annotations)} annotations "
          f"(the input: {len(raw.annotations)})")  # fmt: skip
    epochs = run(pipeline, raw, output="ep1")
    if epochs is None:
        print(f"H2 Epochs: none on {event!r}")
        return
    print(f"H2 Epochs: {len(epochs)} on {event!r}, {epochs.tmin * 1000:.0f} to "
          f"{epochs.tmax * 1000:.0f} ms")  # fmt: skip
    again = run(pipeline, recording, output="ep1")
    same = again is not None and len(again) == len(epochs)
    count = 0 if again is None else len(again)
    print(f"H3 from the file: {count} epochs" + (" (the same)" if same else ""))
    if plot:
        clean.plot(block=False)
        epochs.average().plot()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plot"]
    if len(args) > 2 or any(a.startswith("-") for a in args):
        sys.exit(__doc__)
    main(Path(args[0]) if args else None, args[1] if len(args) > 1 else None,
         "--plot" in sys.argv)  # fmt: skip
