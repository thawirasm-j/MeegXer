"""``meegxer-gui --create-shortcut`` (M11.6): start MeegXer like any other app.

- **Windows:** ``MeegXer.lnk`` in the Start menu (``%APPDATA%\\Microsoft\\Windows\\
  Start Menu\\Programs``) and on the desktop, written by PowerShell's
  ``WScript.Shell`` (no extra package). It runs ``pythonw.exe -m
  meegxer.gui.app`` (no console window).
- **Linux:** ``meegxer.desktop`` in ``$XDG_DATA_HOME/applications`` (default
  ``~/.local/share/applications``): the desktop's application menu.
- **macOS:** ``~/Applications/MeegXer.app``, a small launcher bundle.

Each runs the Python that ran this command, so the shortcut starts the
MeegXer it was made from. ``--remove-shortcut`` removes them again.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

NAME = "MeegXer"
COMMENT = (
    "A user-friendly mixer for EEG processing (research software, not a medical device)"
)
MODULE = "meegxer.gui.app"


def _python(system: str) -> Path:
    """The interpreter the shortcut runs: this one (``pythonw`` on Windows)."""
    exe = Path(sys.executable)
    if system == "Windows":
        windowed = exe.with_name("pythonw.exe")
        if windowed.exists():
            return windowed
    return exe


def shortcut_paths(system: str | None = None, home: Path | None = None) -> list[Path]:
    """Where the shortcuts go on ``system`` (``platform.system()`` names)."""
    system = system or platform.system()
    home = home or Path.home()
    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
        start = appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        return [start / f"{NAME}.lnk", _desktop(home) / f"{NAME}.lnk"]
    if system == "Darwin":
        return [home / "Applications" / f"{NAME}.app"]
    data = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return [data / "applications" / "meegxer.desktop"]


def _desktop(home: Path) -> Path:
    """The Windows desktop folder (it may be redirected, e.g. to OneDrive)."""
    found = os.environ.get("MEEGXER_DESKTOP_DIR")  # tests, odd setups
    if found:
        return Path(found)
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, check=True, timeout=30,
        )  # fmt: skip
        if out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return home / "Desktop"


def desktop_entry(python: Path) -> str:
    """The Linux ``.desktop`` file (freedesktop.org Desktop Entry)."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={NAME}\n"
        "GenericName=EEG mixer\n"
        f"Comment={COMMENT}\n"
        f'Exec="{python}" -m {MODULE}\n'
        "Terminal=false\n"
        "Categories=Science;Education;\n"
        f"StartupWMClass={NAME}\n"
    )


def _powershell_link(link: Path, python: Path) -> str:
    def quoted(text: object) -> str:
        return "'" + str(text).replace("'", "''") + "'"

    return (
        f"New-Item -ItemType Directory -Force -Path {quoted(link.parent)} | Out-Null; "
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
        f"{quoted(link)}); "
        f"$s.TargetPath = {quoted(python)}; "
        f"$s.Arguments = {quoted('-m ' + MODULE)}; "
        f"$s.WorkingDirectory = {quoted(Path.home())}; "
        f"$s.Description = {quoted(COMMENT)}; "
        "$s.Save()"
    )


def _mac_app(bundle: Path, python: Path) -> None:
    macos = bundle / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    launcher = macos / NAME
    launcher.write_text(f'#!/bin/sh\nexec "{python}" -m {MODULE} "$@"\n')
    launcher.chmod(0o755)
    (bundle / "Contents" / "Info.plist").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        f"<key>CFBundleName</key><string>{NAME}</string>\n"
        f"<key>CFBundleDisplayName</key><string>{NAME}</string>\n"
        f"<key>CFBundleExecutable</key><string>{NAME}</string>\n"
        "<key>CFBundleIdentifier</key><string>io.github.meegxer</string>\n"
        "<key>CFBundlePackageType</key><string>APPL</string>\n"
        "</dict></plist>\n"
    )


def create_shortcuts(system: str | None = None, home: Path | None = None) -> list[Path]:
    """Make the shortcuts; returns where they are."""
    system = system or platform.system()
    python = _python(system)
    paths = shortcut_paths(system, home)
    if system == "Windows":
        for link in paths:
            script = _powershell_link(link, python)
            subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           check=True, capture_output=True, timeout=60)  # fmt: skip
    elif system == "Darwin":
        _mac_app(paths[0], python)
    else:
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(desktop_entry(python), encoding="utf-8")
        paths[0].chmod(0o755)
    return paths


def remove_shortcuts(system: str | None = None, home: Path | None = None) -> list[Path]:
    """Remove the shortcuts; returns those that were there."""
    removed = []
    for path in shortcut_paths(system, home):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
        elif path.exists():
            path.unlink()
            removed.append(path)
    return removed
