"""Installed plugins: a small JSON index plus the registry node provider.

``meegxer-plugin add IMAGE`` starts the image once, reads its manifest and records
it together with the image ID, so later runs use exactly that image. The core finds
the plugins through the ``meegxer.node_providers`` entry point (``node_classes``),
which only exists when the runtime is installed (research builds, rule 10).

Index location: ``$MEEGXER_PLUGIN_INDEX``, else
``<user config dir>/meegxer/plugins.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from meegxer.runtime.adapter import PluginProcessor, plugin_class, probe_manifest
from meegxer.runtime.worker import (
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


def launcher_for_entry(entry: dict[str, Any]) -> Launcher:
    if "command" in entry:
        if not entry.get("unsandboxed"):
            raise ValueError("local plugin entries must be marked 'unsandboxed'")
        return LocalLauncher(tuple(entry["command"]), env=entry.get("env", {}))
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
        "manifest": manifest,
    }
    entries = [e for e in load_index(path) if e["manifest"]["name"] != manifest["name"]]
    save_index([*entries, entry], path)
    return entry


def add_local(command: list[str], path: Path | None = None) -> dict[str, Any]:
    """Register an unsandboxed local plugin (development only)."""
    launcher = LocalLauncher(tuple(command), allow=True)
    manifest = probe_manifest(launcher)
    plugin_class(manifest, launcher)
    entry = {"command": command, "unsandboxed": True, "manifest": manifest}
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

    Unsandboxed entries are skipped unless MEEGXER_ALLOW_UNSANDBOXED=1. A broken
    entry is reported on stderr and skipped rather than breaking the registry.
    """
    classes = []
    for entry in load_index():
        try:
            if "command" in entry and os.environ.get(UNSANDBOXED_ENV) != "1":
                continue
            classes.append(plugin_class(entry["manifest"], launcher_for_entry(entry)))
        except Exception as exc:
            name = entry.get("manifest", {}).get("name", "?")
            print(f"meegxer: skipping plugin {name}: {exc}", file=sys.stderr)
    return classes
