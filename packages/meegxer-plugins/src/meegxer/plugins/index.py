"""Installed plugins: a small JSON index plus the registry node provider.

``meegxer-plugin add IMAGE`` starts the image once, reads its manifest and records
it together with the image ID, so later runs use exactly that image.

Local plugins (``add --local COMMAND`` or ``add --script FILE``) run outside any
sandbox with the user's permissions (rule 9). They are recorded only after the
user accepts ``LOCAL_CODE_NOTICE``; the entry keeps that consent, and sessions
record the SHA-256 of the code files so a changed script is noticed. The
development switch ``MEEGXER_ALLOW_UNSANDBOXED=1`` still loads local entries
without recorded consent. The core finds
the plugins through the ``meegxer.node_providers`` entry point (``node_classes``),
which only exists when this package is installed (rule 9).

Index location: ``$MEEGXER_PLUGIN_INDEX``, else
``<user config dir>/meegxer/plugins.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meegxer.core.notices import LOCAL_CODE_NOTICE_VERSION
from meegxer.plugins.adapter import PluginProcessor, plugin_class, probe_manifest
from meegxer.plugins.worker import (
    UNSANDBOXED_ENV,
    ContainerLauncher,
    Launcher,
    LocalLauncher,
    find_engine,
)

INDEX_ENV = "MEEGXER_PLUGIN_INDEX"


def index_path() -> Path:
    if os.environ.get(INDEX_ENV):
        return Path(os.environ[INDEX_ENV])
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "meegxer" / "plugins.json"


def load_index(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or index_path()
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("plugins", [])
    return entries if isinstance(entries, list) else []


def save_index(entries: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plugins": entries}, indent=2) + "\n", encoding="utf-8")
    return path


def image_id(image: str, engine: str | None = None) -> str:
    result = subprocess.run(
        [find_engine(engine), "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip()


def repo_digests(image: str, engine: str | None = None) -> list[str]:
    """Registry digests of a pulled image (empty for a locally built one)."""
    result = subprocess.run(
        [find_engine(engine), "image", "inspect", "--format",
         "{{json .RepoDigests}}", image],
        capture_output=True, text=True, check=False, timeout=60,
    )  # fmt: skip
    try:
        digests = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return []
    return [str(d) for d in digests or []]


class ConsentRequired(PermissionError):
    """A local plugin needs the user to accept the local-code notice first."""


def code_files(command: list[str]) -> list[Path]:
    """The files a local command runs: its arguments that are existing files
    (the interpreter itself excluded)."""
    files = []
    for arg in command[1:]:
        path = Path(arg)
        if path.suffix and path.is_file():
            files.append(path.resolve())
    return files


def code_sha256(command: list[str]) -> str:
    """One SHA-256 over the code files of a local command (in name order), so
    a session can tell whether the script changed since it was saved."""
    digest = hashlib.sha256()
    for path in sorted(code_files(command), key=lambda p: p.name):
        digest.update(path.name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def origin_of(entry: dict[str, Any]) -> dict[str, Any]:
    """Where an index entry's plugin comes from, as sessions record it."""
    if "command" in entry:
        command = list(entry["command"])
        origin: dict[str, Any] = {"command": command, "unsandboxed": True}
        if code_files(command):
            origin["sha256"] = code_sha256(command)
        if entry.get("script"):
            origin["script"] = Path(entry["script"]).name
        return origin
    origin = {"image": entry["image"]}
    for key in ("image_id", "repo_digests", "engine"):
        if entry.get(key):
            origin[key] = entry[key]
    return origin


def launcher_for_entry(entry: dict[str, Any]) -> Launcher:
    if "command" in entry:
        if not entry.get("unsandboxed"):
            raise ValueError("local plugin entries must be marked 'unsandboxed'")
        return LocalLauncher(
            tuple(entry["command"]),
            allow=bool(entry.get("consent")),
            env=entry.get("env", {}),
        )
    return ContainerLauncher(
        entry.get("image_id") or entry["image"], entry.get("engine")
    )


def add_image(
    image: str, engine: str | None = None, path: Path | None = None
) -> dict[str, Any]:
    launcher = ContainerLauncher(image, engine)
    manifest = probe_manifest(launcher)
    plugin_class(manifest, launcher)  # validate before recording
    entry = {
        "image": image,
        "engine": engine,
        "image_id": image_id(image, engine),
        "repo_digests": repo_digests(image, engine),
        "manifest": manifest,
    }
    entries = [e for e in load_index(path) if e["manifest"]["name"] != manifest["name"]]
    save_index([*entries, entry], path)
    return entry


def script_command(script: Path) -> list[str]:
    """The command that runs a custom script (``process(raw, ...)``) as a plugin."""
    return [sys.executable, "-m", "meegxer.plugins.script", str(script.resolve())]


def add_local(
    command: list[str],
    path: Path | None = None,
    *,
    accept_risk: bool = False,
    script: Path | None = None,
) -> dict[str, Any]:
    """Register a local plugin, which runs unsandboxed with the user's
    permissions. Refused unless the user accepted ``LOCAL_CODE_NOTICE``."""
    if not accept_risk:
        raise ConsentRequired(
            "local plugins run outside any sandbox; accept the notice first"
        )
    launcher = LocalLauncher(tuple(command), allow=True)
    manifest = probe_manifest(launcher)
    plugin_class(manifest, launcher)
    entry: dict[str, Any] = {
        "command": command,
        "unsandboxed": True,
        "consent": {
            "accepted": datetime.now(UTC).isoformat(timespec="seconds"),
            "notice": LOCAL_CODE_NOTICE_VERSION,
        },
        "manifest": manifest,
    }
    if script is not None:
        entry["script"] = str(script.resolve())
    entries = [e for e in load_index(path) if e["manifest"]["name"] != manifest["name"]]
    save_index([*entries, entry], path)
    return entry


def remove(name: str, path: Path | None = None) -> bool:
    entries = load_index(path)
    kept = [e for e in entries if e["manifest"]["name"] != name]
    save_index(kept, path)
    return len(kept) != len(entries)


def node_classes() -> list[type[PluginProcessor]]:
    """Entry point for ``meegxer.node_providers``: one class per installed plugin.

    Local entries load once the user has accepted the notice for them (or with
    MEEGXER_ALLOW_UNSANDBOXED=1, for development). A broken entry is reported on
    stderr and skipped rather than breaking the registry.
    """
    classes = []
    for entry in load_index():
        try:
            allowed = entry.get("consent") or os.environ.get(UNSANDBOXED_ENV) == "1"
            if "command" in entry and not allowed:
                continue
            classes.append(
                plugin_class(
                    entry["manifest"],
                    launcher_for_entry(entry),
                    origin=origin_of(entry),
                )
            )
        except Exception as exc:
            name = entry.get("manifest", {}).get("name", "?")
            print(f"meegxer: skipping plugin {name}: {exc}", file=sys.stderr)
    return classes
