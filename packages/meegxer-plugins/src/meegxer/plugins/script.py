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
module docstring (the strip description). Credit, shown with the strip and kept in
session and BIDS provenance: ``__author__`` (a string or a list), ``__license__``
and ``REFERENCES``, a list of citations (strings, or dicts with ``citation`` and an
optional ``doi`` or ``url``). Annotations the script adds to the
returned recording come out as events.

That tier runs offline only (the whole recording at once). A live script instead
defines ``transform``, called block by block, live and offline alike::

    def transform(data, state, low: float = 1.0):
        ...                      # data: (channels, samples) in volts
        return data, state       # state: a dict kept between blocks, or None

``state`` is None on the first block; keep everything that must survive between
blocks in it (the host saves it and restores it after a restart) and return a new
one rather than changing it in place. ``transform`` may also take ``info`` (channel
labels, sampling rate, bad channels, sample index) and ``model``. An optional
``fit(raw, ...)`` makes the strip need calibration: it gets the calibration
recording as an ``mne.io.Raw`` plus the settings it names, and returns the model
(a dict) that ``transform`` then receives. Output covers the same samples as the
input (no look-ahead). A live script must give the same result block by block as
in one pass (acceptance test 1); ``meegxer-plugin check --script`` tests that.

The runner speaks spec/plugin-protocol.md through the bundled SDK, in its own
process, with the user's permissions (rule 9).
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

_ENTRY = ("process", "transform")
# transform parameters that are not settings
_RESERVED = {"state", "model", "info"}
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
    process, transform = (callable(getattr(module, f, None)) for f in _ENTRY)
    if process == transform:
        raise ScriptError(
            f"{path.name} needs one function: process(raw, ...) for offline use, "
            "or transform(data, state, ...) for live and offline use"
            + (" (it has both)" if process else "")
        )
    if process and callable(getattr(module, "fit", None)):
        raise ScriptError("fit goes with transform (live scripts), not process")
    return module


def is_live(module: ModuleType) -> bool:
    return callable(getattr(module, "transform", None))


def _raw(data: np.ndarray, info: dict[str, Any]) -> Any:
    """The data as an mne.io.Raw (EEG channels, bad channels marked)."""
    import mne

    labels = list(info["ch_labels"])
    raw_info = mne.create_info(labels, float(info["sfreq"]), "eeg")
    raw_info["bads"] = [b for b in info.get("bads", []) if b in labels]
    return mne.io.RawArray(np.array(data, dtype=np.float64), raw_info, verbose="error")


def _accepted(fn: Any, names: Any) -> dict[str, Any]:
    """The subset of ``names`` (a mapping) that ``fn`` takes as parameters."""
    wanted = inspect.signature(fn).parameters
    return {k: v for k, v in names.items() if k in wanted}


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
    live = is_live(module)
    entry = module.transform if live else module.process
    params = list(inspect.signature(entry).parameters.items())
    first = "the block (data)" if live else "the recording"
    if not params:
        raise ScriptError(f"{entry.__name__} needs {first} as its first argument")
    if live and "state" not in dict(params):
        raise ScriptError("transform needs a state parameter: transform(data, state)")
    settings = [(n, p) for n, p in params[1:] if not (live and n in _RESERVED)]
    hints = typing.get_type_hints(entry)
    texts = _docstring_params(entry.__doc__ or "")
    stem = re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "script"
    summary = (inspect.getdoc(module) or inspect.getdoc(entry) or "").split("\n\n")
    manifest: dict[str, Any] = {
        "spec_version": "0.2.5",
        "name": str(getattr(module, "NAME", f"script.{stem}")),
        "version": str(getattr(module, "__version__", "0.1.0")),
        "type": "stream",
        "availability": "both" if live else "offline",
        "description": summary[0].replace("\n", " ").strip()
        or f"Custom script {path.name}",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [
            _param(name, p, hints.get(name), texts.get(name, ""))
            for name, p in settings
        ],
    }
    if live and callable(getattr(module, "fit", None)):
        manifest["requires_fit"] = True
    manifest.update(_credits(module))
    return manifest


def _credits(module: ModuleType) -> dict[str, Any]:
    """``__author__``, ``__license__`` and ``REFERENCES`` as manifest fields."""
    out: dict[str, Any] = {}
    authors = getattr(module, "__author__", None)
    if authors:
        out["authors"] = [authors] if isinstance(authors, str) else list(authors)
    if getattr(module, "__license__", None):
        out["license"] = str(module.__license__)
    references = getattr(module, "REFERENCES", None) or []
    if isinstance(references, str | dict):
        references = [references]
    refs = []
    for ref in references:
        if isinstance(ref, str):
            refs.append({"citation": ref})
        elif isinstance(ref, dict) and ref.get("citation"):
            refs.append(
                {k: str(v) for k, v in ref.items() if k in ("citation", "doi", "url")}
            )
        else:
            raise ScriptError(
                "REFERENCES: give each as a string or a dict with 'citation'"
            )
    for ref in refs:
        if ref.get("doi", "").startswith(("https://doi.org/", "doi:")):
            ref["doi"] = (
                ref["doi"].removeprefix("https://doi.org/").removeprefix("doi:")
            )
    if refs:
        out["references"] = refs
    return out


class ScriptPlugin(Plugin):
    """Wraps ``process(raw, ...)`` (the whole recording in, the same shape out) or
    ``transform(data, state, ...)`` with an optional ``fit(raw, ...)``."""

    manifest: ClassVar[dict[str, Any]] = {}

    def __init__(self, module: ModuleType, manifest: dict[str, Any]) -> None:
        self.module = module
        self.manifest = manifest  # type: ignore[misc]

    def fit(
        self, data: np.ndarray, info: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        model = self.module.fit(_raw(data, info), **_accepted(self.module.fit, params))
        if not isinstance(model, dict):
            raise ScriptError("fit must return the model as a dict")
        return model

    def transform(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        model: dict[str, Any] | None,
        state: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        if is_live(self.module):
            return self._live(data, info, model, state, params)
        return self._process(data, info, state, params)

    def _live(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        model: dict[str, Any] | None,
        state: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        extra = _accepted(self.module.transform, {"info": info, "model": model})
        got = self.module.transform(data, state, **params, **extra)
        out, new_state = got if isinstance(got, tuple) else (got, state)
        array = np.asarray(out, dtype=np.float64)
        if array.shape != data.shape:
            raise ScriptError(
                f"transform must return {data.shape[0]} channels x {data.shape[1]} "
                f"samples (the block it got), got {array.shape}"
            )
        if new_state is not None and not isinstance(new_state, dict):
            raise ScriptError("transform must return its state as a dict (or None)")
        return array, new_state or {}, []

    def _process(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        state: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        import mne

        labels = list(info["ch_labels"])
        sfreq = float(info["sfreq"])
        raw = _raw(data, info)
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
