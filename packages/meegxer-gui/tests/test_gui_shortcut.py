"""``meegxer-gui --create-shortcut`` (M11.6): the Start menu and desktop on
Windows, the application menu on Linux, ~/Applications on macOS."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from meegxer.gui import shortcut
from meegxer.gui.app import main


def test_linux_desktop_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    made = shortcut.create_shortcuts("Linux", tmp_path)
    entry = tmp_path / ".local/share/applications/meegxer.desktop"
    assert made == [entry]
    text = entry.read_text()
    assert "Name=MeegXer" in text and "not a medical device" in text
    assert f'Exec="{sys.executable}" -m meegxer.gui.app' in text
    assert "Terminal=false" in text
    assert shortcut.remove_shortcuts("Linux", tmp_path) == [entry]
    assert not entry.exists() and shortcut.remove_shortcuts("Linux", tmp_path) == []


def test_macos_app_bundle(tmp_path: Path) -> None:
    [bundle] = shortcut.create_shortcuts("Darwin", tmp_path)
    assert bundle == tmp_path / "Applications/MeegXer.app"
    launcher = bundle / "Contents/MacOS/MeegXer"
    assert f'exec "{sys.executable}" -m meegxer.gui.app' in launcher.read_text()
    if os.name == "posix":  # Windows has no execute bit (chmod: read-only only)
        assert launcher.stat().st_mode & 0o111  # executable
    assert "<string>MeegXer</string>" in (bundle / "Contents/Info.plist").read_text()
    assert shortcut.remove_shortcuts("Darwin", tmp_path) == [bundle]
    assert not bundle.exists()


def test_windows_start_menu_and_desktop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("MEEGXER_DESKTOP_DIR", str(tmp_path / "Desktop"))
    scripts: list[str] = []

    def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        assert cmd[:3] == ["powershell", "-NoProfile", "-Command"]
        scripts.append(cmd[3])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    made = shortcut.create_shortcuts("Windows", tmp_path)
    start = tmp_path / "Roaming/Microsoft/Windows/Start Menu/Programs/MeegXer.lnk"
    assert made == [start, tmp_path / "Desktop/MeegXer.lnk"]
    assert len(scripts) == 2 and f"CreateShortcut('{start}')" in scripts[0]
    assert "$s.Arguments = '-m meegxer.gui.app'" in scripts[0]
    assert "WScript.Shell" in scripts[1] and "$s.Save()" in scripts[1]


def test_the_flag_makes_it_and_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    made: list[str] = []

    def create() -> list[Path]:
        made.append("x")
        return [tmp_path / "a.desktop"]

    monkeypatch.setattr(shortcut, "create_shortcuts", create)
    assert main(["--create-shortcut"]) == 0  # no window, no daemon
    assert made == ["x"] and "shortcut" in capsys.readouterr().out
