"""Session files: the unit of reproducibility (``spec/session.md``).

A session is a pipeline document plus what pins a run: node versions and
provenance (built-in, or an external plugin's image), fitted models in readable
JSON, input and output file hashes and the software environment. MeegXer writes
it; people can read, edit and write it too. Only ``meegxer_session`` and
``pipeline`` are required.

Loading never uses a model silently on other data (rule 7): a model whose
fingerprint does not match the pipeline it is loaded into stays unused (offline
runs refit, live runs wait for calibration), and a node whose version or plugin
image differs is refused unless explicitly allowed (then its model is dropped).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from meegxer.core.block import SignalInfo
from meegxer.core.epoch_review import EPOCH_REVIEW_NODE, EpochReview
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.node import Sink, Source
from meegxer.core.registry import Registry, canonical_name
from meegxer.core.review import REVIEW_NODE, ReviewStrip
from meegxer.core.serialization import State, hash_state
from meegxer.core.stage import FittedModel, Stage

SESSION_FORMAT = "1.0"
PATH_PARAMS = ("path", "root")  # parameters holding file or folder locations
PACKAGES = (
    "meegxer-core", "meegxer-nodes", "meegxer-gui", "meegxer-cli",
    "meegxer-plugins", "numpy", "scipy", "mne", "mne-bids", "eeglabio", "edfio",
    "python-picard", "mne-icalabel", "onnxruntime", "mne-connectivity", "antropy",
    "pylsl",
)  # fmt: skip
_ARRAY = "__array__"
_FLOAT = "__float__"
_NONFINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
# what a share-safe session holds instead of a path: <input>.edf, <output-2>.set
_PLACEHOLDER = re.compile(r"<[a-z]+(?:-[a-z]+)?(?:-\d+)?>(?:\.[A-Za-z0-9.]+)?")


class SessionError(ValueError):
    """A session cannot be loaded as it is (the message says why)."""


@cache
def session_schema() -> dict[str, Any]:
    text = resources.files("meegxer.core").joinpath("schema/session.schema.json")
    schema: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    return schema


def is_session(data: object) -> bool:
    return isinstance(data, Mapping) and "meegxer_session" in data


# -- readable state ------------------------------------------------------------------


def _finite_or_name(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    if isinstance(x, list):
        return [_finite_or_name(v) for v in x]
    return x


def _from_names(x: Any) -> Any:
    if isinstance(x, str):
        return _NONFINITE[x]
    if isinstance(x, list):
        return [_from_names(v) for v in x]
    return x


def state_to_json(value: Any) -> Any:
    """A model or state as a readable JSON tree (``spec/session.md``)."""
    if isinstance(value, np.ndarray):
        entry: dict[str, Any] = {_ARRAY: value.dtype.name, "shape": list(value.shape)}
        if value.dtype.kind == "c":
            entry["real"] = _finite_or_name(value.real.tolist())
            entry["imag"] = _finite_or_name(value.imag.tolist())
        else:
            entry["data"] = _finite_or_name(value.tolist())
        return entry
    if isinstance(value, np.generic):
        return state_to_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {_FLOAT: _finite_or_name(value)}
    if isinstance(value, dict):
        return {str(k): state_to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [state_to_json(v) for v in value]
    return value


def state_from_json(value: Any) -> Any:
    """Inverse of :func:`state_to_json` (bit-exact for the values it wrote)."""
    if isinstance(value, dict):
        if _ARRAY in value:
            dtype = np.dtype(value[_ARRAY])
            shape = tuple(value["shape"])
            if dtype.kind == "c":
                real = np.array(_from_names(value["real"]), dtype=float)
                imag = np.array(_from_names(value["imag"]), dtype=float)
                return (real + 1j * imag).astype(dtype).reshape(shape)
            return np.array(_from_names(value["data"]), dtype=dtype).reshape(shape)
        if _FLOAT in value:
            return float(_NONFINITE[value[_FLOAT]])
        return {k: state_from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [state_from_json(v) for v in value]
    return value


def _info_to_json(info: SignalInfo) -> dict[str, Any]:
    entry: dict[str, Any] = {"channels": list(info.ch_labels), "sfreq": info.sfreq}
    if info.stream_id:
        entry["stream_id"] = info.stream_id
    entry["bads"] = list(info.bads)
    if info.ch_types:
        entry["types"] = list(info.ch_types)
    if info.positions:
        entry["positions"] = [list(p) if p is not None else None
                              for p in info.positions]  # fmt: skip
    return entry


def _info_from_json(entry: Mapping[str, Any]) -> SignalInfo:
    return SignalInfo(
        tuple(entry["channels"]),
        float(entry["sfreq"]),
        str(entry.get("stream_id", "")),
        tuple(entry.get("bads", ())),
        tuple(entry.get("types", ())),
        tuple(tuple(p) if p is not None else None for p in entry.get("positions", ())),
    )


# -- files and environment ------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def environment() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "meegxer": packages.get("meegxer-core", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


# -- building ------------------------------------------------------------------------


def build_session(
    graph: Graph,
    *,
    name: str | None = None,
    description: str | None = None,
    outputs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """The session of ``graph`` as it is now (its fitted models included), with
    absolute paths; :func:`write_session` makes them relative to the file."""
    session: dict[str, Any] = {"meegxer_session": SESSION_FORMAT,
                               "spec_version": SPEC_VERSION}  # fmt: skip
    if name:
        session["name"] = name
    if description:
        session["description"] = description
    session["pipeline"] = graph.to_dict()
    nodes: dict[str, Any] = {}
    models: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for node in graph.nodes:
        impl_cls = type(node.impl.processor if isinstance(node.impl, Stage)
                        else node.impl)  # fmt: skip
        entry: dict[str, Any] = {"node": node.manifest.name,
                                 "version": node.manifest.version}  # fmt: skip
        provenance = impl_cls.provenance() if hasattr(impl_cls, "provenance") else {}
        if provenance:
            entry["provenance"] = provenance
        if node.manifest.references:  # what to cite, so the session stands alone
            entry["references"] = [r.as_dict() for r in node.manifest.references]
        processor = node.impl.processor if isinstance(node.impl, Stage) else None
        record = getattr(processor, "record", None)  # a review: who reviewed
        if isinstance(processor, ReviewStrip | EpochReview) and record:
            entry["review"] = dict(record)
        nodes[node.id] = entry
        if isinstance(node.impl, Stage) and node.impl.model is not None:
            model = node.impl.model
            models[node.id] = {
                "fitted_on": _info_to_json(model.input_info),
                "fingerprint": model.fingerprint,
                "content_hash": "sha256:" + model.content_hash,
                "data": state_to_json(model.data),
            }
        if isinstance(node.impl, Source):
            files = node.impl.input_files()
            if files:
                inputs[node.id] = [
                    {"path": str(Path(f).resolve()), "sha256": sha256_file(Path(f))}
                    for f in files
                ]
    session["nodes"] = nodes
    if models:
        session["models"] = models
    if inputs:
        session["inputs"] = inputs
    written = [Path(p) for p in outputs]
    if written:
        session["outputs"] = [{"path": str(p.resolve()), "sha256": sha256_file(p)}
                              for p in written]  # fmt: skip
    session["environment"] = environment()
    return session


def sink_outputs(graph: Graph) -> list[Path]:
    """The files the graph's sinks wrote (those that exist; not a file a sink
    kept because it existed)."""
    paths = []
    for node in graph.nodes:
        if (isinstance(node.impl, Sink) and "path" in node.impl.params
                and not node.impl.skipped):  # fmt: skip
            path = Path(str(node.impl.params["path"]))
            if path.is_file():
                paths.append(path)
    return paths


# -- loading -------------------------------------------------------------------------


@dataclass
class LoadedSession:
    graph: Graph
    warnings: list[str] = field(default_factory=list)
    dropped_models: list[str] = field(default_factory=list)
    unpinned_models: list[str] = field(default_factory=list)  # no fingerprint


def load_session(
    session: Mapping[str, Any],
    registry: Registry,
    *,
    allow_version_mismatch: bool = False,
    strict: bool = False,
) -> LoadedSession:
    """The session's pipeline as a graph with its models installed (paths as
    given: :func:`read_session` resolves them against the file)."""
    try:
        jsonschema.validate(dict(session), session_schema())
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise SessionError(f"session {location}: {exc.message}") from exc
    major = str(session["meegxer_session"]).split(".")[0]
    if major != SESSION_FORMAT.split(".")[0]:
        raise SessionError(
            f"session format {session['meegxer_session']} is not supported "
            f"(this MeegXer reads {SESSION_FORMAT.split('.')[0]}.x)"
        )
    pipeline, notes = _without_shared_files(session["pipeline"])
    try:
        graph = Graph.from_dict(pipeline, registry)
    except LookupError as exc:  # UnknownNodeError: a node is not installed
        raise SessionError(_missing_node_hint(session, exc)) from exc
    except GraphError as exc:
        raise SessionError(f"pipeline: {exc}") from exc
    loaded = LoadedSession(graph, warnings=notes)
    mismatched = _check_nodes(session, graph, allow_version_mismatch, loaded)
    _install_models(session, graph, mismatched, loaded)
    _check_inputs(session, graph, strict, loaded)
    _check_environment(session, loaded)
    return loaded


def _missing_node_hint(session: Mapping[str, Any], exc: LookupError) -> str:
    message = str(exc.args[0]) if exc.args else str(exc)
    for entry in session.get("nodes", {}).values():
        image = entry.get("provenance", {}).get("image")
        if image and repr(entry["node"]) in message:
            return (f"{message}; it is an external plugin: install it with "
                    f"`meegxer-plugin add {image}`")  # fmt: skip
    return message


def _check_nodes(
    session: Mapping[str, Any], graph: Graph, allow: bool, loaded: LoadedSession
) -> set[str]:
    mismatched = set()
    for node_id, recorded in session.get("nodes", {}).items():
        if node_id not in graph:
            loaded.warnings.append(f"nodes: {node_id!r} is not in the pipeline")
            continue
        node = graph[node_id]
        problems = []
        if canonical_name(recorded["node"]) != node.manifest.name:
            problems.append(f"is {node.manifest.name}, recorded {recorded['node']}")
        elif recorded["version"] != node.manifest.version:
            problems.append(
                f"{node.manifest.name} is version {node.manifest.version}, "
                f"recorded {recorded['version']}"
            )
        impl = node.impl.processor if isinstance(node.impl, Stage) else node.impl
        installed = type(impl).provenance() if hasattr(impl, "provenance") else {}
        want = recorded.get("provenance", {}).get("image_id")
        if want and installed.get("image_id") != want:
            problems.append(
                f"plugin image is {installed.get('image_id') or 'not an image'}, "
                f"recorded {want} ({recorded['provenance'].get('image', '?')})"
            )
        code = recorded.get("provenance", {}).get("sha256")  # a local plugin's files
        if code and installed.get("sha256") != code:
            problems.append(
                "plugin code changed since the session was saved "
                f"(sha256 {str(installed.get('sha256') or 'missing')[:12]}, "
                f"recorded {code[:12]})"
            )
        if not problems:
            continue
        text = f"node {node_id!r}: " + "; ".join(problems)
        if not allow:
            raise SessionError(text + " (allow the mismatch to run it anyway; "
                               "its model is then refitted)")  # fmt: skip
        loaded.warnings.append(text)
        mismatched.add(node_id)
    return mismatched


def _install_models(
    session: Mapping[str, Any],
    graph: Graph,
    mismatched: set[str],
    loaded: LoadedSession,
) -> None:
    for node_id, entry in session.get("models", {}).items():
        if node_id not in graph or not isinstance(graph[node_id].impl, Stage):
            raise SessionError(
                f"models: {node_id!r} is not a processor of the pipeline"
            )
        stage = graph[node_id].impl
        assert isinstance(stage, Stage)
        if not stage.manifest.requires_fit:
            raise SessionError(f"models: {node_id!r} does not use a fitted model")
        data: State = state_from_json(entry["data"])
        content = hash_state(data)
        if "content_hash" in entry and entry["content_hash"] != "sha256:" + content:
            raise SessionError(
                f"models: {node_id!r} was changed after it was saved (its "
                "content_hash does not match; remove content_hash to use the "
                "edited model)"
            )
        if node_id in mismatched:
            loaded.dropped_models.append(node_id)
            continue
        if "fitted_on" not in entry:
            loaded.warnings.append(
                f"models: {node_id!r} has no fitted_on (its channels and rate), "
                "so it cannot be checked against the data and is not used"
            )
            loaded.dropped_models.append(node_id)
            continue
        if "fingerprint" not in entry:
            loaded.unpinned_models.append(node_id)
        stage.model = FittedModel(
            data=data,
            fingerprint=str(entry.get("fingerprint", "")),
            content_hash=content,
            input_info=_info_from_json(entry["fitted_on"]),
        )


def _check_inputs(
    session: Mapping[str, Any], graph: Graph, strict: bool, loaded: LoadedSession
) -> None:
    for node_id, files in session.get("inputs", {}).items():
        for entry in files:
            if "sha256" not in entry or is_placeholder(entry["path"]):
                continue  # a shared session names no files: nothing to compare
            path = Path(entry["path"])
            if not path.is_file():
                message = f"inputs: {node_id!r} file {path} is missing"
            elif sha256_file(path) != entry["sha256"]:
                message = (
                    f"inputs: {node_id!r} file {path} differs from the recorded one"
                )
            else:
                continue
            if strict:
                raise SessionError(message)
            loaded.warnings.append(message)


def _check_environment(session: Mapping[str, Any], loaded: LoadedSession) -> None:
    recorded = session.get("environment", {}).get("packages", {})
    now = environment()["packages"]
    changed = [
        f"{name} {version} -> {now.get(name, 'not installed')}"
        for name, version in recorded.items()
        if now.get(name) != version
    ]
    if changed:
        loaded.warnings.append(
            "environment differs (results may change in the last bits): "
            + ", ".join(changed)
        )


def adopt_models(loaded: LoadedSession, input_fps: Mapping[str, str]) -> list[str]:
    """Bind the models written without a fingerprint (by hand) to this input:
    the channels and rate they were fitted on are still checked on every block."""
    adopted = []
    for node_id in loaded.unpinned_models:
        stage = loaded.graph[node_id].impl
        assert isinstance(stage, Stage) and stage.model is not None
        stage.model = dataclasses.replace(
            stage.model, fingerprint=stage.fingerprint(input_fps.get(node_id, ""))
        )
        adopted.append(node_id)
        loaded.warnings.append(
            f"models: {node_id!r} has no fingerprint; used for this input "
            "(its channels and rate are still checked)"
        )
    loaded.unpinned_models = []
    return adopted


def model_status(graph: Graph, input_fps: Mapping[str, str]) -> dict[str, str]:
    """``reused`` or ``needs refit`` for every node with a loaded model, given the
    input fingerprint of each node (as the scheduler that runs it computes it)."""
    status = {}
    for node in graph.nodes:
        if isinstance(node.impl, Stage) and node.impl.model is not None:
            fp = node.impl.fingerprint(input_fps.get(node.id, ""))
            status[node.id] = ("reused" if node.impl.model.fingerprint == fp
                               else "needs refit")  # fmt: skip
    return status


# -- sharing (spec/session.md, Sharing) ----------------------------------------------


def is_placeholder(path: object) -> bool:
    """Whether ``path`` is a share-safe placeholder (``<input>.edf``), not a file."""
    return isinstance(path, str) and _PLACEHOLDER.fullmatch(path) is not None


def _suffix(path: str) -> str:
    name = re.split(r"[/\\]", path)[-1]
    return "".join(Path(name).suffixes[-1:])


def share_safe(session: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``session`` that names no file or folder: every path becomes a
    placeholder that keeps only the extension (``<input>.set``, ``<output-1>.edf``,
    ``<montage>.ced``), as does the stream a model was fitted on (its
    fingerprint, a one-way hash, is what reloading compares). Hashes, models
    and the free text (``name``,
    ``description``) are kept; :func:`free_text` lists the text to check.
    A shared session reruns on a recording given with ``-i`` (and ``-o``)."""
    out = copy.deepcopy(dict(session))
    places: dict[str, str] = {}
    count = {"output": 0}

    def key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def output(path: str) -> str:
        if key(path) not in places:
            count["output"] += 1
            places[key(path)] = f"<output-{count['output']}>{_suffix(path)}"
        return places[key(path)]

    nodes = out.get("pipeline", {}).get("nodes", [])
    for index, node in enumerate(nodes):
        params = node.get("params", {})
        path, root = params.get("path"), params.get("root")
        if index == 0:  # the source (first by construction)
            if isinstance(path, str) and path and not is_placeholder(path):
                params["path"] = places[key(path)] = f"<input>{_suffix(path)}"
            if isinstance(root, str) and root and not is_placeholder(root):
                params["root"] = "<input-root>"
        else:
            if isinstance(path, str) and path and not is_placeholder(path):
                params["path"] = output(path)
            if isinstance(root, str) and root:
                params["root"] = ""  # BIDS writers: the default derivatives folder
            reviews = (REVIEW_NODE, EPOCH_REVIEW_NODE)
            if node.get("node") in reviews and params.get("file"):
                params["file"] = ""  # the marks: next to each recording
        recording = node.get("recording", {})
        events = recording.get("events", {})
        if isinstance(events.get("file"), str) and events["file"]:
            events["file"] = f"<events>{_suffix(events['file'])}"
        montage = recording.get("montage")
        if isinstance(montage, str) and _is_file_montage(montage):
            recording["montage"] = f"<montage>{_suffix(montage)}"
    for files in out.get("inputs", {}).values():
        for entry in files:
            path = entry["path"]
            entry["path"] = places.get(key(path)) or f"<input>{_suffix(path)}"
    for entry in out.get("outputs", []):
        entry["path"] = output(entry["path"])
    for model in out.get("models", {}).values():
        stream = model.get("fitted_on", {}).get("stream_id")
        if stream and not is_placeholder(stream):  # a file source: its file name
            model["fitted_on"]["stream_id"] = f"<input>{_suffix(stream)}"
    for entry in out.get("nodes", {}).values():
        provenance = entry.get("provenance", {})
        if isinstance(provenance.get("command"), list):  # a local plugin's command
            provenance["command"] = [re.split(r"[/\\]", str(a))[-1]
                                     for a in provenance["command"]]  # fmt: skip
        entry.get("review", {}).pop("reviewer", None)  # it names a person
    return out


def free_text(session: Mapping[str, Any]) -> dict[str, str]:
    """The text a person wrote into ``session`` (``name``, ``description``):
    sharing keeps it, so it is shown for checking before a session leaves."""
    return {k: str(session[k]) for k in ("name", "description") if session.get(k)}


def _is_file_montage(montage: str) -> bool:
    return "/" in montage or "\\" in montage or "." in montage  # not a built-in


def _without_shared_files(
    pipeline: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """The pipeline without the recording files a shared session left out
    (placeholders), and a note for each."""
    out = copy.deepcopy(dict(pipeline))
    notes = []
    for node in out.get("nodes", []):
        recording = node.get("recording", {})
        if is_placeholder(recording.get("montage")):
            notes.append(f"{node['id']}: the montage file was left out when this "
                         "session was shared; positions come from the recording "
                         "(set recording.montage to use a file)")  # fmt: skip
            del recording["montage"]
        events = recording.get("events", {})
        if is_placeholder(events.get("file")):
            notes.append(
                f"{node['id']}: the events file was left out when this "
                "session was shared; the recording's own events are used"
            )
            del events["file"]
    return out, notes


def shared_input_hint(graph: Graph) -> str | None:
    """Why ``graph`` cannot run as it is: its source names a placeholder (a
    shared session), so the recording has to be given; None otherwise."""
    params = graph.source.impl.params
    if is_placeholder(params.get("path")):
        return (
            "this session was shared without file names: give the recording "
            "with -i FILE"
        )
    if is_placeholder(params.get("root")):
        return (
            "this session was shared without file names: give the BIDS "
            "dataset with --bids ROOT"
        )
    return None


def drop_shared_sinks(graph: Graph) -> list[str]:
    """Remove the sinks whose path a shared session left out; their ids."""
    dropped = [n.id for n in graph.nodes if isinstance(n.impl, Sink)
               and is_placeholder(n.impl.params.get("path"))]  # fmt: skip
    for node_id in dropped:
        graph.remove(node_id)
    return dropped


# -- files on disk -------------------------------------------------------------------


def _map_paths(session: dict[str, Any], fix: Any) -> dict[str, Any]:
    """Apply ``fix`` to every path a session holds (``spec/session.md``);
    placeholders (a shared session) are not paths and stay as they are."""
    out = copy.deepcopy(session)

    def keep_placeholder(p: str) -> str:
        return p if is_placeholder(p) else str(fix(p))

    for node in out.get("pipeline", {}).get("nodes", []):
        params = node.get("params", {})
        for key in PATH_PARAMS:
            if isinstance(params.get(key), str) and params[key]:
                params[key] = keep_placeholder(params[key])
        recording = node.get("recording", {})
        events = recording.get("events", {})
        if isinstance(events.get("file"), str) and events["file"]:
            events["file"] = keep_placeholder(events["file"])
        montage = recording.get("montage")
        if isinstance(montage, str) and _is_file_montage(montage):
            recording["montage"] = keep_placeholder(montage)  # a file, not a name
    for files in out.get("inputs", {}).values():
        for entry in files:
            entry["path"] = keep_placeholder(entry["path"])
    for entry in out.get("outputs", []):
        entry["path"] = keep_placeholder(entry["path"])
    return out


def read_session(path: str | Path) -> dict[str, Any]:
    """A session file with its relative paths resolved against its folder. A
    plain pipeline file is read as a session holding only that pipeline."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_session(data):
        return {"meegxer_session": SESSION_FORMAT, "pipeline": data}
    base = path.resolve().parent

    def resolve(p: str) -> str:
        return p if Path(p).is_absolute() else str(base / p)

    return _map_paths(data, resolve)


def write_session(path: str | Path, session: dict[str, Any]) -> Path:
    """Write ``session``, with paths inside the file's folder made relative."""
    path = Path(path)
    base = path.resolve().parent

    def relative(p: str) -> str:
        absolute = Path(p).resolve() if Path(p).is_absolute() else None
        if absolute is None:
            return p
        try:
            return absolute.relative_to(base).as_posix()
        except ValueError:
            return p  # outside the session folder: kept absolute

    text = json.dumps(_map_paths(session, relative), indent=1, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def default_session_path(output: str | Path) -> Path:
    """``out.set`` -> ``out.meegxer.json`` (next to the output)."""
    output = Path(output)
    return output.with_name(output.stem + ".meegxer.json")


__all__ = [
    "SESSION_FORMAT", "LoadedSession", "SessionError", "adopt_models", "build_session",
    "default_session_path", "drop_shared_sinks", "environment", "free_text",
    "is_placeholder", "is_session", "load_session", "model_status", "read_session",
    "session_schema", "sha256_file", "share_safe", "shared_input_hint", "sink_outputs",
    "state_from_json", "state_to_json", "write_session",
]  # fmt: skip
