"""Check that every planned third-party dependency installs and imports.

Each distribution is installed into a fresh virtualenv on its own, so one broken
package does not hide the status of the others. Usage:

    python tools/smoke_deps.py --python 3.14 [--venv .smoke-venv]

Exits non-zero if any *required* dependency fails; *candidate* dependencies (not
yet chosen, see docs/WORKPLAN.md) are reported but never fail the run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# (distribution, import name, required). Required = already declared by a package;
# candidate = planned in docs but not yet declared.
DEPENDENCIES: list[tuple[str, str, bool]] = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("pyzmq", "zmq", True),
    ("pylsl", "pylsl", True),
    ("pyxdf", "pyxdf", True),
    ("mne", "mne", True),
    ("PySide6", "PySide6.QtCore", True),
    ("pyqtgraph", "pyqtgraph", True),
    ("typer", "typer", True),
    ("docker", "docker", False),
    ("jsonschema", "jsonschema", True),
    ("mne-bids", "mne_bids", True),
    ("eeglabio", "eeglabio", True),
    ("edfio", "edfio", True),
    ("python-picard", "picard", True),
    ("mne-icalabel", "mne_icalabel", True),
    ("onnxruntime", "onnxruntime", True),
    ("mne-connectivity", "mne_connectivity", True),
    ("antropy", "antropy", True),
    ("specparam>=2.0.0rc7", "specparam", True),  # 2.0 is in release candidates
    ("numba", "numba", True),
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:160] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--venv", default=".smoke-venv")
    args = parser.parse_args()

    venv = Path(args.venv)
    if venv.exists():
        shutil.rmtree(venv)
    created = _run(["uv", "venv", "--python", args.python, str(venv)])
    if created.returncode != 0:
        print(created.stderr)
        return 2
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    failures_required = 0
    print("| dependency | required | install | import | note |\n|---|---|---|---|---|")
    for dist, module, required in DEPENDENCIES:
        installed = _run(["uv", "pip", "install", "--python", str(py), dist])
        if installed.returncode != 0:
            ok_install, ok_import, note = "FAIL", "-", _last_line(installed.stderr)
        else:
            imported = _run([str(py), "-c", f"import {module}"])
            ok_install = "ok"
            ok_import = "ok" if imported.returncode == 0 else "FAIL"
            note = "" if imported.returncode == 0 else _last_line(imported.stderr)
        if required and "FAIL" in (ok_install, ok_import):
            failures_required += 1
        print(
            f"| {dist} | {'yes' if required else 'candidate'} | {ok_install} "
            f"| {ok_import} | {note.replace('|', '/')} |"
        )
    return 1 if failures_required else 0


if __name__ == "__main__":
    sys.exit(main())
