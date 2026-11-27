"""The command-line tools' output streams."""

from __future__ import annotations

import io
import sys


def never_fail_to_print() -> None:
    """Make stdout and stderr replace a character their encoding cannot write
    instead of raising. Windows writes a redirected or piped stdout in the
    code page (cp1252), so text from a tool's own output (pip's progress bar
    ``━`` during ``meegxer sterile build > build.log``) raised
    UnicodeEncodeError; a console, and every other platform, writes UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
