"""Run a custom script as a plugin: ``python -m meegxer.plugins.script FILE.py``.

A custom script is an ordinary MNE-Python script with one function::

    def process(raw, low: float = 1.0, high: float = 40.0):
        \"\"\"Band-pass the recording.

        Parameters
        ----------
        low : float
            Low cut-off (Hz).
        high : float
            High cut-off (Hz).
        \"\"\"
        return raw.filter(low, high)

``process`` receives the whole recording as an ``mne.io.Raw`` (the channels the
strip processes, in volts) and returns an ``mne.io.Raw`` or an array with the same
channels and length. Its settings come from the signature: every parameter after
``raw`` needs a default; its type comes from the annotation or the default (bool,
int, float, str, or ``Literal[...]`` for a choice). The numpy-style docstring gives
descriptions, and a trailing ``(unit)`` becomes the unit. Optional module-level
``NAME`` (default ``script.<file name>``), ``__version__`` (default 0.1.0) and the
module docstring (the strip description). Annotations the script adds to the
returned recording come out as events.

This tier runs offline only (the whole recording at once). Live scripts
(fit/transform) come later. The runner speaks spec/plugin-protocol.md through the
bundled SDK, in its own process, with the user's permissions (rule 9).
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import typing
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np

from meegxer.plugins._sdk import Plugin, run

_TYPES: dict[type, str] = {bool: "bool", int: "int", float: "float", str: "string"}
_UNIT = re.compile(r"\(([^()\s]{1,8})\)\.?\s*$")


class ScriptError(ValueError):
    """The script does not follow the custom-script conventions."""


def load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"meegxer_script_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ScriptError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "process", None)):
        raise ScriptError(f"{path.name} has no function process(raw, ...)")
    return module


def _docstring_params(doc: str) -> dict[str, str]:
    """numpy-style ``Parameters`` section: name -> description."""
    lines = inspect.cleandoc(doc).splitlines()
    found: dict[str, str] = {}
    name = None
    inside = False
    for i, line in enumerate(lines):
        underline = i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1].strip())
        if underline:  # a section heading
            inside = line.strip() == "Parameters"
            name = None
            continue
        if not inside or re.fullmatch(r"-{3,}", line.strip()):
            continue
        if line and not line[0].isspace():
            name = line.split(":")[0].strip()
            found[name] = ""
        elif name and line.strip():
            found[name] = (found[name] + " " + line.strip()).strip()
    return found


def _param(name: str, p: inspect.Parameter, hint: Any, text: str) -> dict[str, Any]:
    if p.default is inspect.Parameter.empty:
        raise ScriptError(f"setting {name!r} needs a default value")
    spec: dict[str, Any] = {"name": name, "default": p.default}
    if typing.get_origin(hint) is typing.Literal:
        spec["type"] = "enum"
        spec["choices"] = [str(c) for c in typing.get_args(hint)]
        spec["default"] = str(p.default)
    else:
        kind = hint if isinstance(hint, type) else type(p.default)
        if kind not in _TYPES:
            raise ScriptError(
                f"setting {name!r}: use bool, int, float, str or Literal[...]"
            )
        spec["type"] = _TYPES[kind]
        if kind is float:
            spec["default"] = float(p.default)
    if text:
        unit = _UNIT.search(text)
        if unit and spec["type"] in ("int", "float"):
            spec["unit"] = unit.group(1)
            text = text[: unit.start()].rstrip()
        spec["description"] = text.rstrip(".")
    return spec


def script_manifest(module: ModuleType, path: Path) -> dict[str, Any]:
    """The plugin manifest of a custom script, from its signature and docstrings."""
    process = module.process
    signature = inspect.signature(process)
    params = list(signature.parameters.items())
    if not params:
        raise ScriptError("process needs the recording as its first argument")
    hints = typing.get_type_hints(process)
    texts = _docstring_params(process.__doc__ or "")
    stem = re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "script"
    summary = (inspect.getdoc(module) or inspect.getdoc(process) or "").split("\n\n")
    return {
        "spec_version": "0.2.5",
        "name": str(getattr(module, "NAME", f"script.{stem}")),
        "version": str(getattr(module, "__version__", "0.1.0")),
        "type": "stream",
        "availability": "offline",
        "description": summary[0].replace("\n", " ").strip()
        or f"Custom script {path.name}",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [
            _param(name, p, hints.get(name), texts.get(name, ""))
            for name, p in params[1:]
        ],
    }


class ScriptPlugin(Plugin):
    """Wraps ``process(raw, ...)``: the whole recording in, the same shape out."""

    manifest: ClassVar[dict[str, Any]] = {}

    def __init__(self, module: ModuleType, manifest: dict[str, Any]) -> None:
        self.module = module
        self.manifest = manifest  # type: ignore[misc]

    def transform(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        model: dict[str, Any] | None,
        state: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        import mne

        labels = list(info["ch_labels"])
        sfreq = float(info["sfreq"])
        raw_info = mne.create_info(labels, sfreq, "eeg")
        raw_info["bads"] = [b for b in info.get("bads", []) if b in labels]
        raw = mne.io.RawArray(data, raw_info, verbose="error")
        before = len(raw.annotations)
        result = self.module.process(raw, **params)
        out = result if result is not None else raw
        if isinstance(out, mne.io.BaseRaw):
            if list(out.ch_names) != labels:
                raise ScriptError("process must keep the channels (names and order)")
            if out.info["sfreq"] != sfreq:
                raise ScriptError("process must keep the sampling rate")
            events = [
                {
                    "sample_index": int(info["sample_index"])
                    + round(float(a["onset"]) * sfreq),
                    "label": str(a["description"]),
                    "duration": round(float(a["duration"]) * sfreq),  # samples
                }
                for a in list(out.annotations)[before:]
            ]
            array = out.get_data()
        else:
            events, array = [], np.asarray(out, dtype=np.float64)
        if array.shape != data.shape:
            raise ScriptError(
                f"process must return {data.shape[0]} channels x {data.shape[1]} "
                f"samples, got {array.shape}"
            )
        return array, state or {}, events


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m meegxer.plugins.script FILE.py", file=sys.stderr)
        return 2
    path = Path(args[0])
    module = load_script(path)
    run(ScriptPlugin(module, script_manifest(module, path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
