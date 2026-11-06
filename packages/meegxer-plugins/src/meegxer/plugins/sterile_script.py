"""A custom script in Sterile mode (``docs/STERILE.md``): the container's entry
point ``python -m meegxer.plugins.sterile_script``.

Nothing from the host is mounted, so the script's code arrives on stdin: the
first frame is ``{"type": "script", "name": FILE, "sha256": HEX}`` with the
file's bytes as the array ``source``. It is checked against its SHA-256,
written to the container's private ``/tmp`` (the only writable place), and
then served exactly as outside Sterile mode (``python -m
meegxer.plugins.script FILE``): the rest of stdin is the plugin protocol.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

from meegxer.plugins.wire import WireError, read_message

MAX_SOURCE = 64 * 1024 * 1024  # a script, not a recording


def receive(stream: object) -> Path:
    """Read the script frame from ``stream`` and write the file; refuse a
    frame that is not one, or whose bytes do not match their SHA-256."""
    message = read_message(stream, MAX_SOURCE)  # type: ignore[arg-type]
    header = message.header
    if header.get("type") != "script" or "source" not in message.arrays:
        raise WireError("the first frame must be the script (type 'script')")
    source = message.arrays["source"].tobytes()
    if hashlib.sha256(source).hexdigest() != header.get("sha256"):
        raise WireError("the script's bytes do not match their SHA-256")
    name = Path(str(header.get("name") or "script.py")).name  # no folders
    if not name.endswith(".py"):
        raise WireError(f"not a Python script: {name!r}")
    folder = Path(tempfile.mkdtemp(prefix="meegxer-script-"))
    path = folder / name
    path.write_bytes(source)
    return path


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
