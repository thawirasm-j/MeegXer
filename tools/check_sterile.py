"""Verify the sterile build contains no plugin runtime (CLAUDE.md rule 10).

Builds wheels for the sterile package set (core + sterile + gui + cli, and the
``meegxer`` bundle that installs them), then fails if
any of them ships ``meegxer/runtime`` files, declares a dependency on the runtime or
Docker, or pulls either in transitively when the set is resolved. Usage:

    uv run python tools/check_sterile.py [--out-dir dist/sterile] [--no-resolve]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

STERILE_PACKAGES = [
    "meegxer-core", "meegxer-sterile", "meegxer-gui", "meegxer-cli",
    "meegxer",  # the bundle that installs the four
]  # fmt: skip
FORBIDDEN_DISTS = {"meegxer-runtime", "docker"}
FORBIDDEN_PATH_PREFIX = "meegxer/runtime/"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return normalize(match.group(1)) if match else ""


def check_wheels(wheels: list[Path]) -> list[str]:
    """Return a list of violations found in the given wheel files."""
    problems: list[str] = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            problems += [
                f"{wheel.name}: ships forbidden file {name}"
                for name in names
                if name.startswith(FORBIDDEN_PATH_PREFIX)
            ]
            metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                problems.append(f"{wheel.name}: expected one METADATA file")
                continue
            metadata = Parser().parsestr(zf.read(metadata_files[0]).decode())
        for requirement in metadata.get_all("Requires-Dist") or []:
            if requirement_name(requirement) in FORBIDDEN_DISTS:
                problems.append(f"{wheel.name}: depends on {requirement}")
    return problems


def check_resolution(wheel_dir: Path) -> list[str]:
    """Resolve the sterile set (exactly the wheels just built) against PyPI plus
    the local wheels."""
    pins = [
        f"{w.name.split('-')[0]}=={w.name.split('-')[1]}"
        for w in sorted(wheel_dir.glob("*.whl"))
    ]
    result = subprocess.run(
        ["uv", "pip", "compile", "--quiet", "--find-links", str(wheel_dir), "-"],
        input="\n".join(pins) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"resolution failed:\n{result.stderr}"]
    resolved = {
        requirement_name(line.split("==")[0])
        for line in result.stdout.splitlines()
        if line and not line.startswith((" ", "#"))
    }
    return [
        f"sterile set resolves to {dist}" for dist in sorted(resolved & FORBIDDEN_DISTS)
    ]


def build_wheels(out_dir: Path) -> list[Path]:
    for package in STERILE_PACKAGES:
        subprocess.run(
            ["uv", "build", "--package", package, "--wheel", "--out-dir", str(out_dir)],
            check=True,
        )
    return sorted(out_dir.glob("*.whl"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-resolve", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.out_dir or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        wheels = build_wheels(out_dir)
        built = {normalize(w.name.split("-")[0]) for w in wheels}
        problems = [
            f"missing wheel for {p}"
            for p in STERILE_PACKAGES
            if normalize(p) not in built
        ]
        problems += check_wheels(wheels)
        if not args.no_resolve:
            problems += check_resolution(out_dir)

    for problem in problems:
        print(f"STERILE BUILD VIOLATION: {problem}", file=sys.stderr)
    if not problems:
        print(f"Sterile build OK: {', '.join(STERILE_PACKAGES)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
