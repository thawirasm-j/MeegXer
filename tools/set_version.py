"""Stamp one release version on every package before building (release workflow).

    python tools/set_version.py 0.1.0a1        # or v0.1.0a1, as the git tag reads

All packages share one version. In the repository they stay at 0.0.0 with unpinned
dependencies on each other (uv resolves them from the workspace); for a release
this sets ``version`` in each package's pyproject.toml and pins every dependency
on a sibling package to exactly that version, so a published ``meegxer-gui``
always installs the ``meegxer-core`` it was built and tested with.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# PEP 440 public versions without epochs or local parts, e.g. 0.1.0, 0.1.0a1, 1.2rc3
VERSION = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def packages() -> dict[str, Path]:
    found = {}
    for path in sorted((ROOT / "packages").glob("*/pyproject.toml")):
        name = re.search(r'(?m)^name = "([^"]+)"', path.read_text(encoding="utf-8"))
        assert name, path
        found[name.group(1)] = path
    return found


def stamp(text: str, version: str, siblings: set[str]) -> str:
    text, n = re.subn(
        r'(?m)^version = "[^"]*"', f'version = "{version}"', text, count=1
    )
    if n != 1:
        raise ValueError("no version line")
    names = "|".join(re.escape(s) for s in sorted(siblings, key=len, reverse=True))
    # a sibling requirement inside the dependencies list, with or without a specifier
    pattern = re.compile(rf'(?m)^(\s*)"({names})(\[[^\]]*\])?[^"]*"(,?)$')
    return pattern.sub(lambda m: f'{m[1]}"{m[2]}{m[3] or ""}=={version}"{m[4]}', text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="e.g. 0.1.0a1 or v0.1.0a1")
    parser.add_argument(
        "--check", action="store_true", help="only validate and list the packages"
    )
    args = parser.parse_args()
    version = args.version.removeprefix("v")
    if not VERSION.match(version):
        print(f"not a PEP 440 release version: {args.version!r}", file=sys.stderr)
        return 2
    found = packages()
    for name, path in found.items():
        if not args.check:
            text = path.read_text(encoding="utf-8")
            path.write_text(stamp(text, version, set(found) - {name}), encoding="utf-8")
        print(f"{name} {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
