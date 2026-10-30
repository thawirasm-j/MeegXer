"""Write the live ERP check of docs/TESTING.md §E: a synthetic input played
live with markers (``go`` and ``stop`` taking turns every 1.2 s for ten
minutes) and an Epochs strip, to open in the GUI.

Usage: ``uv run python tools/live_erp_demo.py OUT_DIR`` (writes
``live_erp.json`` and ``events.tsv`` there).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from meegxer.core.manifest import SPEC_VERSION


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events.tsv"
    rows = [f"{1.0 + 1.2 * k:.1f}\t0\t{'go' if k % 2 == 0 else 'stop'}"
            for k in range(500)]  # fmt: skip
    events.write_text("onset\tduration\ttrial_type\n" + "\n".join(rows) + "\n")
    pipeline = {
        "spec_version": SPEC_VERSION,
        "nodes": [
            {"id": "input", "node": "meegxer.core.synthetic",
             "params": {"duration": 610.0},
             "recording": {"events": {"file": str(events.resolve())}}},
            {"id": "epochs1", "node": "meegxer.nodes.epochs",
             "params": {"tmin": -0.2, "tmax": 0.8}},
        ],
    }  # fmt: skip
    path = out / "live_erp.json"
    path.write_text(json.dumps(pipeline, indent=2) + "\n")
    print(f"wrote {path} and {events}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
