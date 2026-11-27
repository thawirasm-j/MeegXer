"""The command-line tools never fail to print (Windows check, error 2):
Windows writes a redirected or piped stdout in its code page (cp1252), and
``meegxer sterile build > build.log`` raised UnicodeEncodeError at pip's
progress bar (``━``)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

PRINT = "print('build ' + chr(0x2501) * 3 + ' done')"


def child(code: str) -> subprocess.CompletedProcess[bytes]:
    """``code`` in a child Python whose stdout is a cp1252 pipe."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          env=env, check=False)  # fmt: skip


def test_a_character_the_code_page_lacks_is_replaced_not_raised() -> None:
    assert b"UnicodeEncodeError" in child(PRINT).stderr  # as on Windows
    done = child("from meegxer.core.console import never_fail_to_print\n"
                 f"never_fail_to_print()\n{PRINT}")  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == b"build ??? done"


def test_both_tools_set_it_up(monkeypatch: pytest.MonkeyPatch) -> None:
    from meegxer.cli import app
    from meegxer.core import console
    from meegxer.plugins.cli import main

    calls: list[str] = []
    monkeypatch.setattr(console, "never_fail_to_print", lambda: calls.append("x"))
    assert CliRunner().invoke(app, ["templates"]).exit_code == 0
    assert main(["list"]) == 0
    assert calls == ["x", "x"]
