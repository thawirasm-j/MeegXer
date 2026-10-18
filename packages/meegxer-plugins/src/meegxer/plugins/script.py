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
optional ``doi`` or ``url``). ``CATEGORY`` (e.g. ``"filters"``, ``"artifacts"``,
``"spectral"``) lists the strip under that heading of its group in the GUI's menu.
Annotations the script adds to the returned recording come out as events.

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

A feature-extraction script defines ``features``, called once per analysis
window, live and offline alike::

    def features(data, info, fmin: float = 8.0, fmax: float = 13.0):
        ...                      # data: (channels, samples) of one window
        return {"power": p, "peak": f}   # each: one value per channel

It returns a dict of feature name -> one value per channel (or a channels x
features array). The runner cuts the stream into windows and keeps the unfinished
one between blocks, so the result never depends on how the stream is cut. Every
such strip gets ``window`` (default 2 s) and ``step`` (0.5 s) settings; declare
them in the signature to choose other defaults (``features`` then receives them
too). ``info`` is optional, as with ``transform``.

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

_ENTRY = ("process", "transform", "features")
# transform/features parameters that are not settings
_RESERVED = {"state", "model", "info"}
# the windowing settings every features script gets (seconds)
_WINDOWING = {"window": 2.0, "step": 0.5}
_WINDOW_TEXT = {
    "window": "Length of each analysis window",
    "step": "Time from one window to the next",
}
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
    found = [f for f in _ENTRY if callable(getattr(module, f, None))]
    if len(found) != 1:
        raise ScriptError(
            f"{path.name} needs one function: process(raw, ...) for offline use, "
            "transform(data, state, ...) for live and offline use, or "
            "features(data, ...) for feature extraction"
            + (f" (it has {' and '.join(found)})" if found else "")
        )
    if found[0] != "transform" and callable(getattr(module, "fit", None)):
        raise ScriptError(f"fit goes with transform (live scripts), not {found[0]}")
    return module


def entry_kind(module: ModuleType) -> str:
    """Which convention the script follows: process, transform or features."""
    return next(f for f in _ENTRY if callable(getattr(module, f, None)))


def is_live(module: ModuleType) -> bool:
    return entry_kind(module) == "transform"


def _raw(data: np.ndarray, info: dict[str, Any]) -> Any:
    """The data as an mne.io.Raw (EEG channels, bad channels marked)."""
    import mne

    labels = list(info["ch_labels"])
    sfreq = float(info["sfreq"])
    raw_info = mne.create_info(labels, sfreq, "eeg")
    raw_info["bads"] = [b for b in info.get("bads", []) if b in labels]
    raw = mne.io.RawArray(np.array(data, dtype=np.float64), raw_info, verbose="error")
    events = info.get("events") or []
    if events:  # labelled trials (FIT_EVENTS), timed from the data's start
        start = int(info.get("sample_index", 0))
        raw.set_annotations(mne.Annotations(
            [(int(e["sample_index"]) - start) / sfreq for e in events],
            [int(e.get("duration", 0)) / sfreq for e in events],
            [str(e["label"]) for e in events],
        ))  # fmt: skip
    return raw


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
    kind = entry_kind(module)
    live = kind == "transform"
    entry = getattr(module, kind)
    params = list(inspect.signature(entry).parameters.items())
    first = "the recording" if kind == "process" else "the data"
    if not params:
        raise ScriptError(f"{entry.__name__} needs {first} as its first argument")
    if live and "state" not in dict(params):
        raise ScriptError("transform needs a state parameter: transform(data, state)")
    if kind == "features" and "state" in dict(params):
        raise ScriptError(
            "features gets one whole window and keeps no state; "
            "use transform(data, state) for a stream processor"
        )
    settings = [(n, p) for n, p in params[1:]
                if not (kind != "process" and n in _RESERVED)]  # fmt: skip
    hints = typing.get_type_hints(entry)
    texts = _docstring_params(entry.__doc__ or "")
    windowing = []
    if kind == "features":
        declared = dict(settings)
        for name, default in _WINDOWING.items():
            if name not in declared:
                windowing.append({"name": name, "type": "float",
                                  "default": default, "unit": "s",
                                  "description": _WINDOW_TEXT[name]})  # fmt: skip
        for name, text in _WINDOW_TEXT.items():
            texts[name] = texts.get(name) or text
    stem = re.sub(r"[^a-z0-9_]+", "_", path.stem.lower()).strip("_") or "script"
    summary = (inspect.getdoc(module) or inspect.getdoc(entry) or "").split("\n\n")
    manifest: dict[str, Any] = {
        "spec_version": "0.2.5",
        "name": str(getattr(module, "NAME", f"script.{stem}")),
        "version": str(getattr(module, "__version__", "0.1.0")),
        "type": "analysis" if kind == "features" else "stream",
        "availability": "offline" if kind == "process" else "both",
        "description": summary[0].replace("\n", " ").strip()
        or f"Custom script {path.name}",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [
            {"name": "out", "type": "features" if kind == "features" else "signal"}
        ],
        "params": [
            _param(name, p, hints.get(name), texts.get(name, ""))
            for name, p in settings
        ]
        + windowing,
    }
    if kind == "features":
        for spec in manifest["params"]:
            if spec["name"] in _WINDOWING:
                if spec["type"] not in ("int", "float"):
                    raise ScriptError(f"{spec['name']} is a time in seconds")
                spec.setdefault("unit", "s")
                if spec["type"] == "float":
                    spec.setdefault("min", 0.01)
    if live and callable(getattr(module, "fit", None)):
        manifest["requires_fit"] = True
        if getattr(module, "FIT_EVENTS", False):  # fit(raw) gets raw.annotations
            manifest["fit_events"] = True
    manifest.update(_credits(module))
    category = getattr(module, "CATEGORY", None)
    if category:
        text = re.sub(r"[^a-z0-9]+", "-", str(category).lower()).strip("-")
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", text):
            raise ScriptError(f"CATEGORY {category!r}: use a short word, e.g. filters")
        manifest["category"] = text
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


def _script_units(module: ModuleType, names: list[str]) -> dict[str, Any]:
    """A feature script's ``UNITS``: one unit for every feature (``"V^2"``), or a
    unit per feature name (``{"alpha": "V^2", "ratio": "1"}``); absent: none."""
    units = getattr(module, "UNITS", None)
    if isinstance(units, str):
        return {"unit": units}
    if isinstance(units, dict):
        return {"unit_dim": "feature",
                "units": [str(units.get(name, "")) for name in names]}  # fmt: skip
    return {}


class ScriptPlugin(Plugin):
    """Wraps ``process(raw, ...)`` (the whole recording in, the same shape out),
    ``transform(data, state, ...)`` with an optional ``fit(raw, ...)``, or
    ``features(data, ...)`` (one value per channel and feature, per window)."""

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
        kind = entry_kind(self.module)
        if kind == "transform":
            return self._live(data, info, model, state, params)
        if kind == "features":
            return self._windows(data, info, state, params)
        return self._process(data, info, state, params)

    def _windows(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        state: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Buffer the stream and compute each complete window (the same windows
        whatever the block size; spec/plugin-protocol.md, Feature results)."""
        sfreq = float(info["sfreq"])
        n = max(1, round(float(params["window"]) * sfreq))
        step = max(1, round(float(params["step"]) * sfreq))
        index = int(info["sample_index"])
        if not state:  # times come from the first block, so splits agree
            state = {"buffer": np.zeros((data.shape[0], 0)), "start": index,
                     "next": index, "t_ref": float(info.get("timestamp", 0.0)),
                     "i_ref": index}  # fmt: skip
        t_ref, i_ref = float(state["t_ref"]), int(state["i_ref"])
        buffer = np.asarray(state["buffer"], dtype=np.float64)
        start, nxt = int(state["start"]), int(state["next"])
        if index < start:  # a skipped stretch (step > window)
            data = data[:, start - index :]
        buffer = np.concatenate([buffer, data], axis=1)
        wanted = _accepted(self.module.features, params)
        extra = _accepted(self.module.features, {"info": info})
        windows = []
        while nxt + n <= start + buffer.shape[1]:
            x = buffer[:, nxt - start : nxt - start + n]
            values, names = _feature_table(
                self.module.features(x.copy(), **wanted, **extra), x.shape[0]
            )
            windows.append({"window": [nxt, nxt + n],
                            "timestamp": t_ref + (nxt - i_ref) / sfreq,
                            "dims": ["channel", "feature"],
                            "coords": {"channel": list(info["ch_labels"]),
                                       "feature": names},
                            "data": values,
                            **_script_units(self.module, names)})  # fmt: skip
            nxt += step
        keep = min(max(nxt - start, 0), buffer.shape[1])
        buffer, start = buffer[:, keep:], start + keep
        start = max(start, nxt)  # the next window starts beyond what we hold
        new_state = {"buffer": buffer, "start": start, "next": nxt,
                     "t_ref": t_ref, "i_ref": i_ref}  # fmt: skip
        return windows, new_state, []

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


def _feature_table(got: Any, n_channels: int) -> tuple[np.ndarray, list[str]]:
    """``features``' return value as a channels x features array and names."""
    if isinstance(got, dict):
        names = [str(k) for k in got]
        columns = []
        for name, v in got.items():
            col = np.asarray(v, dtype=np.float64).reshape(-1)
            if col.size != n_channels:
                raise ScriptError(
                    f"feature {name!r}: give one value per channel ({n_channels}), "
                    f"got {col.size}"
                )
            columns.append(col)
        return np.stack(columns, axis=1), names
    values = np.asarray(got, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != n_channels:
        raise ScriptError(
            "features must return a dict of name -> one value per channel, or a "
            f"{n_channels} channels x features array, got shape {values.shape}"
        )
    return values, [f"feature_{i}" for i in range(values.shape[1])]


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
