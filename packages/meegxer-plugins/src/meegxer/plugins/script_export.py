"""A custom Python script in a standalone MNE script (M11.7 step 2;
``meegxer.core.mne_export``): its source inlined, as a module of its own, and
called as MeegXer calls it.

``process(raw, ...)``, ``process_epochs(epochs, ...)`` and ``features(data,
...)`` (on MeegXer's windows) are called as the script runner calls them
(``meegxer.plugins.script``); a live script, ``transform(data, state, ...)``,
once on the whole recording (the same as block by block: acceptance test 1),
its ``fit(raw, ...)`` fitted again first as MeegXer fits it offline. A script
folder (helpers, data files) has no standalone code yet. MeegXer sends a
plugin its data in single precision; the standalone script keeps double, so
the two agree to single precision.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from meegxer.core.mne_export import ExportCode

_SUPPORTED = ("process", "process_epochs", "transform", "features")


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _arguments(fn: ast.FunctionDef) -> list[str]:
    return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]


def script_export_code(path: Path, params: dict[str, Any]) -> ExportCode | None:
    """The standalone code of the custom script at ``path`` with these
    settings, or None when it has none yet (see the module docstring)."""
    if not path.is_file() or path.suffix != ".py":
        return None  # a script folder
    source = path.read_text(encoding="utf-8")
    functions = _functions(ast.parse(source))
    found = [f for f in ("process", "transform", "features", "process_epochs")
             if f in functions]  # fmt: skip
    if len(found) != 1 or found[0] not in _SUPPORTED:
        return None
    kind = found[0]
    fitted = kind == "transform" and "fit" in functions
    fit_events = any(  # FIT_EVENTS = True: fit(raw) gets the events
        isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(getattr(target, "id", "") == "FIT_EVENTS" for target in node.targets)
        for node in ast.parse(source).body
    )  # fmt: skip
    module = "script_" + re.sub(r"\W", "_", path.stem)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    safe = "'''" not in source and not source.endswith("\\")
    text = f"r'''{source}'''" if safe else repr(source)  # as written, if it can
    definition = (
        f"# {path.name} (SHA-256 {sha[:16]}…), the custom script, as it was\n"
        f"{module} = types.ModuleType({module!r})\n"
        f"exec(compile({text}, {path.name!r}, 'exec'), {module}.__dict__)"
    )
    settings = "".join(f", {k}={v!r}" for k, v in params.items())
    window, step = params.get("window", 2.0), params.get("step", 0.5)
    windows = (f"feats = windowed(raw, picks, {window!r}, {step!r}, "
               "lambda x, sfreq, names: run_features(x, sfreq, names, "
               f"raw.info['bads'], {module}.features{settings}), "
               f"script_unit({module}))")  # fmt: skip
    call = {
        "features": windows,
        "process": f"raw = run_process(raw, picks, {module}.process{settings})",
        "transform": f"raw = run_transform(raw, picks, {module}.transform"
        + (f", fit={module}.fit, fit_events={fit_events!r}" if fitted else "")
        + f"{settings})",
        "process_epochs": "epochs = run_process_epochs(epochs, "
        f"{module}.process_epochs{settings})",
    }[kind]
    shown = ", ".join(f"{k}={v!r}" for k, v in params.items())
    from meegxer.plugins.requires import declared

    return ExportCode(
        packages=tuple(str(r) for r in declared(path)),  # its REQUIRES
        comment=f"custom script {path.name}: {kind}({shown})"
        + (", fitted on the recording" if fitted else ""),
        lines=(call,),
        imports=("import types",),
        helpers=("windowed", "run_features")
        if kind == "features"
        else ("run_transform", "events_of")
        if fitted
        else (f"run_{kind}",),
        definitions=(definition,),
    )
