"""The Python API check of docs/TESTING.md §H (M10.7).

``uv run python tools/api_check.py RECORDING EVENT`` runs a band-pass (1 to
30 Hz) and an Epochs strip on EVENT (-200 to 800 ms) over RECORDING with
``meegxer.api.run``: once from an ``mne.io.Raw`` read with MNE, once from the
file itself. It prints what came back, and with ``--plot`` shows the cleaned
recording and the ERP in MNE's own windows.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(recording: Path, event: str, plot: bool) -> None:
    import mne

    from meegxer.api import run
    from meegxer.core.manifest import SPEC_VERSION

    pipeline = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file"},
        {"id": "bp1", "node": "meegxer.nodes.iir_filter",
         "params": {"low": 1.0, "high": 30.0}},
        {"id": "ep1", "node": "meegxer.nodes.epochs",
         "params": {"events": event, "tmin": -0.2, "tmax": 0.8}},
    ]}  # fmt: skip
    raw = mne.io.read_raw(recording, preload=True, verbose="error")
    clean = run(pipeline, raw)
    print(f"H1 Raw: {len(clean.ch_names)} channels, {clean.times[-1]:.1f} s, "
          f"{len(clean.annotations)} annotations "
          f"(the input: {len(raw.annotations)})")  # fmt: skip
    epochs = run(pipeline, raw, output="ep1")
    if epochs is None:
        print(f"H2 Epochs: none (no {event!r} event in the recording)")
        return
    print(f"H2 Epochs: {len(epochs)} on {event!r}, {epochs.tmin * 1000:.0f} to "
          f"{epochs.tmax * 1000:.0f} ms")  # fmt: skip
    again = run(pipeline, recording, output="ep1")
    same = len(again) == len(epochs)
    print(f"H3 from the file: {len(again)} epochs" + (" (the same)" if same else ""))
    if plot:
        clean.plot(block=False)
        epochs.average().plot()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plot"]
    if len(args) != 2:
        sys.exit(__doc__)
    main(Path(args[0]), args[1], "--plot" in sys.argv)
