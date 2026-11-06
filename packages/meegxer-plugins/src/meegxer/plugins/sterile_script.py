"""A custom script in Sterile mode (``docs/STERILE.md``): the container's entry
point ``python -m meegxer.plugins.sterile_script``.

Nothing from the host is mounted, so the script's code arrives on stdin: the
first frame is ``{"type": "script", "name": NAME, "folder": BOOL, "files":
[{"name": PATH, "data": ARRAY, "sha256": HEX}, ...]}`` with each file's bytes
as an array: one file, or every file of a script folder (by its path in the
folder). Each is checked against its SHA-256, written to the container's
private ``/tmp`` (the only writable place), and then served exactly as
outside Sterile mode (``python -m meegxer.plugins.script FILE|FOLDER``): the
rest of stdin is the plugin protocol.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path, PurePosixPath

from meegxer.plugins.wire import WireError, read_message

# a script folder holds at most 100 MB (meegxer.plugins.script), plus headroom
MAX_SOURCE = 128 * 1024 * 1024


def _relative(name: str) -> PurePosixPath:
    """A file's path in the script folder: relative, without ``..``."""
    path = PurePosixPath(name)
    bad = any(part in ("", ".", "..") for part in path.parts)
    if not name or path.is_absolute() or bad or "\\" in name or ":" in name:
        raise WireError(f"not a path inside the script folder: {name!r}")
    return path


def receive(stream: object) -> Path:
    """Read the script frame from ``stream`` and write its files; refuse a
    frame that is not one, a path that leaves the folder, or bytes that do not
    match their SHA-256. Returns what to serve: the file, or the folder."""
    message = read_message(stream, MAX_SOURCE)  # type: ignore[arg-type]
    header = message.header
    files = header.get("files")
    if header.get("type") != "script" or not isinstance(files, list) or not files:
        raise WireError("the first frame must be the script (type 'script')")
    folder = bool(header.get("folder"))
    name = Path(str(header.get("name") or "script")).name  # no folders
    if not folder and (len(files) != 1 or not name.endswith(".py")):
        raise WireError(f"not a Python script: {name!r}")
    root = Path(tempfile.mkdtemp(prefix="meegxer-script-"))
    base = root / name if folder else root
    for item in files:
        key = str(item.get("data", ""))
        if key not in message.arrays:
            raise WireError(f"no bytes for {item.get('name')!r}")
        data = message.arrays[key].tobytes()
        if hashlib.sha256(data).hexdigest() != item.get("sha256"):
            raise WireError(
                f"{item.get('name')!r}: the bytes do not match their SHA-256"
            )
        rel = _relative(str(item.get("name", ""))) if folder else PurePosixPath(name)
        path = base.joinpath(*rel.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return base / name if not folder else base


def main() -> int:
    try:
        path = receive(sys.stdin.buffer)
    except (WireError, EOFError) as exc:
        print(f"meegxer sterile script: {exc}", file=sys.stderr)
        return 2
    from meegxer.plugins.script import main as serve

    return serve([str(path)])


if __name__ == "__main__":
    sys.exit(main())
