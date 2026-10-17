"""User settings (``settings.json``) and remembered state (``state.json``).

Both live in the MeegXer config folder (``config_dir``): ``%APPDATA%\\meegxer`` on
Windows, ``$XDG_CONFIG_HOME/meegxer`` (default ``~/.config/meegxer``) elsewhere,
or ``MEEGXER_CONFIG_DIR``. Settings are preferences the user chooses, checked
against ``schema/settings.schema.json``; state is what the GUI remembers between
launches (window size, last band, last folder) and can be cleared on its own.

Neither changes what a pipeline computes, so neither is written into pipelines,
sessions or provenance. A value is taken from, in order: a command-line flag, an
environment variable, ``settings.json``, the default (``DEFAULTS``). A broken or
partly invalid file never stops MeegXer: the bad parts are reported and replaced
by their defaults. Pure Python: the daemon and the CLI read it too (rule 1).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

CONFIG_ENV = "MEEGXER_CONFIG_DIR"
VERSION = 1
log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "version": VERSION,
    "general": {
        "start_mode": "live",
        "advanced": False,
        "reopen_last": True,
        "daemon": "tcp://127.0.0.1:5555",  # control.DEFAULT_CONTROL (a test checks)
        "confirm_quit_running": True,
    },
    "appearance": {"theme": "dark", "scale": 100},
    "visualization": {
        "sequential": "viridis",
        "diverging": "blue-white-red",
        "peak": "plasma",
        "flip_on_light": False,
        "span": 10,
        "time_axis": "before_now",
        "spectrum": {
            "band": [8.0, 13.0],
            "relative": False,
            "aperiodic": False,
            "scheme": "heat",
            "range": 20,
            "smoothing": 2,
        },
    },
    "storage": {"offline_dir": "", "cache_gb": 5.0, "output_dir": "", "data_dir": ""},
}


def config_dir() -> Path:
    if os.environ.get(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV])
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "meegxer"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def state_path() -> Path:
    return config_dir() / "state.json"


@cache
def schema() -> dict[str, Any]:
    text = resources.files("meegxer.core").joinpath("schema/settings.schema.json")
    data: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    return data


def defaults() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _drop(data: dict[str, Any], path: list[str | int]) -> None:
    node: Any = data
    for key in path[:-1]:
        node = node[key]
    if isinstance(node, dict):
        node.pop(path[-1], None)


def clean(data: Any) -> tuple[dict[str, Any], list[str]]:
    """``data`` with every part the schema rejects removed, and what was removed."""
    if not isinstance(data, dict):
        return {}, ["settings must be a JSON object"]
    data = copy.deepcopy(data)
    problems: list[str] = []
    validator = jsonschema.Draft202012Validator(schema())
    for _ in range(100):  # each pass removes at least one bad part
        errors = sorted(validator.iter_errors(data), key=lambda e: len(e.path))
        if not errors:
            break
        error = errors[0]
        path = list(error.path)
        if error.validator == "additionalProperties":
            known = set(error.schema.get("properties", {}))
            parent: Any = data
            for key in path:
                parent = parent[key]
            for extra in sorted(set(parent) - known):
                problems.append(f"unknown setting {'.'.join(map(str, [*path, extra]))}")
                parent.pop(extra)
            continue
        # a bad value inside a list or object: drop the whole setting it belongs to
        while path and isinstance(path[-1], int):
            path.pop()
        if not path:
            return {}, [*problems, f"settings: {error.message}"]
        problems.append(f"{'.'.join(map(str, path))}: {error.message}")
        _drop(data, path)
    return data, problems


def read_settings(path: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """The settings (every key present: defaults fill what the file lacks) and
    the problems found in the file."""
    path = path or settings_path()
    if not path.is_file():
        return defaults(), []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return defaults(), [f"{path} could not be read ({exc}); using defaults"]
    good, problems = clean(raw)
    return _merge(DEFAULTS, good), [f"{path}: {p}" for p in problems]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    settings, problems = read_settings(path)
    for problem in problems:
        log.warning("%s", problem)
    return settings


def save_settings(settings: dict[str, Any], path: Path | None = None) -> Path:
    """Write the settings (refused if the schema rejects them)."""
    jsonschema.validate(settings, schema())
    path = path or settings_path()
    _write_json(path, settings)
    return path


def get(settings: dict[str, Any], key: str) -> Any:
    """A dotted key, e.g. ``get(s, "storage.cache_gb")``."""
    node: Any = settings
    for part in key.split("."):
        node = node[part]
    return node


def put(settings: dict[str, Any], key: str, value: Any) -> None:
    node: Any = settings
    *parents, last = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[last] = value


def resolve(key: str, env: str | None = None, path: Path | None = None) -> Any:
    """A setting as MeegXer uses it: ``env`` (an environment variable) wins
    over ``settings.json``, which wins over the default."""
    if env and os.environ.get(env):
        return os.environ[env]
    return get(load_settings(path), key)


def load_state(path: Path | None = None) -> dict[str, Any]:
    """What the GUI remembered (empty when there is nothing, or it is broken)."""
    path = path or state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict[str, Any], path: Path | None = None) -> Path:
    path = path or state_path()
    _write_json(path, state)
    return path


def clear_state(path: Path | None = None) -> None:
    (path or state_path()).unlink(missing_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
