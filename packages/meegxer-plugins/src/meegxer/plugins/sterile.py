"""Sterile mode (rule 10): the built-in strips run in an isolated worker.

One worker per run (a container with no network and a read-only filesystem;
``docs/STERILE.md``) holds every built-in processor of the pipeline. On the host
each of them is replaced by a ``SterileProcessor``, a proxy that hands every
call (fit, transform, flush, ...) to the worker over the plugin wire format
(``spec/plugin-protocol.md`` → *Sterile worker*) and returns its answer. The
schedulers, stages, rails, calibration and sessions are unchanged: they call
the proxy as they would call the node, and the host keeps model and state, so
a worker that dies is restarted and goes on where it stopped.

What stays on the host: sources, probes and sinks (trusted I/O), the review
strips (they apply a reviewer's own marks: ``ReviewStrip``, ``EpochReview``,
``RejectSegments``) and container plugins (already isolated in their own
containers). A local plugin or custom script, which would run unisolated, is
refused.

Both ends are the same MeegXer (the worker image is built from the installed
packages), so values travel exactly: arrays keep their dtype (float64 stays
float64, so the output is the local run's, bit for bit), and blocks, events and
feature windows keep their types. The host rebuilds only the types it knows.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import math
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from meegxer.core import __version__
from meegxer.core.block import (
    Block,
    Event,
    EventBlock,
    FeatureBlock,
    SignalInfo,
    Windows,
)
from meegxer.core.epoch_review import EpochReview
from meegxer.core.node import Mode, Params, Processor, TransformResult
from meegxer.core.reject_segments import RejectSegments
from meegxer.core.review import ReviewStrip
from meegxer.core.serialization import State, hash_state
from meegxer.core.stage import Stage
from meegxer.plugins.adapter import PluginProcessor
from meegxer.plugins.wire import DTYPES, Message
from meegxer.plugins.worker import (
    ContainerLauncher,
    Launcher,
    LocalLauncher,
    PluginError,
    PluginWorker,
)

WORKER_NAME = "meegxer.sterile_worker"
SCRIPT_ENTRY = "meegxer.plugins.sterile_script"  # a custom script's container
# the largest frame either side accepts: a whole recording travels in one
# transform offline (both ends are MeegXer, so not a plugin's 256 MiB)
MAX_FRAME = 64 * 1024**3
# methods the worker runs (the processor contract, meegxer.core.node)
METHODS = frozenset({
    "fit", "fit_labelled", "fit_rails", "fit_features", "transform",
    "transform_rails", "transform_features", "flush", "describe_model",
})  # fmt: skip
# the call argument that is the model, per method (sent once, by its id)
MODEL_ARG = {"transform": 1, "transform_rails": 1, "transform_features": 1,
             "flush": 1, "describe_model": 0}  # fmt: skip
# strips that apply a reviewer's own marks: they stay on the host
HOST_PROCESSORS = (ReviewStrip, EpochReview, RejectSegments)
# the types values may be rebuilt as (both directions)
TYPES: dict[str, type] = {
    t.__name__: t
    for t in (
        Block,
        FeatureBlock,
        SignalInfo,
        Event,
        EventBlock,
        Windows,
        TransformResult,
    )
}
# what a worker's error may be raised as on the host (else RuntimeError)
ERRORS: dict[str, type[Exception]] = {
    e.__name__: e
    for e in (
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        NotImplementedError,
        ArithmeticError,
        ZeroDivisionError,
        np.linalg.LinAlgError,
    )
}


# the MeegXer packages the worker runs (copied into its image, and hashed)
CODE_PACKAGES = ("meegxer.core", "meegxer.nodes", "meegxer.plugins")


class SterileError(RuntimeError):
    """Sterile mode cannot run this pipeline (or its worker failed)."""


def package_dir(name: str) -> Path:
    """Where the installed package ``name`` (``meegxer.core``) lives."""
    return Path(next(iter(importlib.import_module(name).__path__)))


def code_files(name: str) -> list[Path]:
    """The files of the installed package ``name`` (no compiled caches)."""
    root = package_dir(name)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )  # fmt: skip


@functools.cache
def code_hash() -> str:
    """A SHA-256 over the code the worker runs (``CODE_PACKAGES``): the host
    and its worker must agree on it, so an image built from other code (an
    edited checkout, another version) is refused even under the same version."""
    digest = hashlib.sha256()
    for name in CODE_PACKAGES:
        root = package_dir(name)
        for path in code_files(name):
            digest.update(f"{name}/{path.relative_to(root).as_posix()}\0".encode())
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


# -- values on the wire ----------------------------------------------------------


def to_wire(value: object, arrays: dict[str, npt.NDArray[Any]]) -> Any:
    """``value`` as a JSON tree, its arrays moved to ``arrays`` (exact dtype)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__f": repr(value)}
    if isinstance(value, np.ndarray):
        name = f"a{len(arrays)}"
        kind = "bool" if value.dtype == np.bool_ else value.dtype.name
        if kind in DTYPES:
            arrays[name] = value
            return {"__a": name}
        # another dtype (complex, int16, ...): its bytes, rebuilt as it was
        arrays[name] = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
        return {"__a": name, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return to_wire(value.item(), arrays)
    if isinstance(value, TransformResult):  # a NamedTuple: by its fields
        values = value._asdict().items()
        return {"__o": "TransformResult",
                "f": {k: to_wire(v, arrays) for k, v in values}}  # fmt: skip
    if isinstance(value, tuple):
        return {"__t": [to_wire(v, arrays) for v in value]}
    if isinstance(value, list):
        return [to_wire(v, arrays) for v in value]
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("only str-keyed mappings travel to the Sterile worker")
        return {"__d": {k: to_wire(v, arrays) for k, v in value.items()}}
    if is_dataclass(value) and type(value).__name__ in TYPES:
        return {"__o": type(value).__name__,
                "f": {f.name: to_wire(getattr(value, f.name), arrays)
                      for f in fields(value)}}  # fmt: skip
    raise TypeError(f"{type(value).__name__} cannot travel to the Sterile worker")


def from_wire(tree: Any, arrays: Mapping[str, npt.NDArray[Any]]) -> Any:
    """The value ``to_wire`` sent (only the types in ``TYPES`` are rebuilt)."""
    if isinstance(tree, list):
        return [from_wire(v, arrays) for v in tree]
    if not isinstance(tree, dict):
        return tree
    if "__a" in tree:
        array = arrays[tree["__a"]]
        if "dtype" in tree:
            return array.view(np.dtype(tree["dtype"])).reshape(tree["shape"]).copy()
        return array  # read_message made it (its own, writable copy)
    if "__f" in tree:
        return float(tree["__f"])
    if "__t" in tree:
        return tuple(from_wire(v, arrays) for v in tree["__t"])
    if "__d" in tree:
        return {k: from_wire(v, arrays) for k, v in tree["__d"].items()}
    if "__o" in tree:
        kind = TYPES.get(tree["__o"])
        if kind is None:
            raise PluginError(
                f"the Sterile worker sent an unknown type {tree['__o']!r}"
            )
        values = {k: from_wire(v, arrays) for k, v in tree["f"].items()}
        if kind is TransformResult and "rails" in values:
            values["rails"] = MappingProxyType(values["rails"])
        return kind(**values)
    raise PluginError(f"the Sterile worker sent a value it should not: {tree!r}")


def impl_key(processor: Processor) -> str:
    """Where the worker finds ``processor``'s class: ``module:qualname``."""
    cls = type(processor)
    return f"{cls.__module__}:{cls.__qualname__}"


def raise_remote(reply: Message) -> None:
    """A node's error in the worker (a ``raised`` reply), raised on the host as
    the same kind of error (a ValueError stays a ValueError) with its message,
    as the local run would have raised it."""
    if reply.type != "raised":
        return
    kind = ERRORS.get(str(reply.header.get("exception", "")), RuntimeError)
    raise kind(str(reply.header.get("message", "the Sterile worker failed")))


# -- the worker process ----------------------------------------------------------


class SterileWorker(PluginWorker):
    """The run's one worker: every proxy of the pipeline shares it."""

    max_payload = MAX_FRAME

    def __init__(self, launcher: Launcher, *, timeout: float = 3600.0) -> None:
        super().__init__(launcher, timeout=timeout, start_timeout=300.0)

    def start(self) -> dict[str, Any]:
        manifest = super().start()
        if manifest.get("name") != WORKER_NAME:
            self.close()
            raise SterileError(f"{self.launcher.describe()} is not a Sterile worker")
        if manifest.get("version") != __version__:
            self.close()
            raise SterileError(
                f"the Sterile worker is MeegXer {manifest.get('version')}, this is "
                f"{__version__}: build its image again (meegxer sterile build)"
            )
        if manifest.get("code") != code_hash():
            self.close()
            raise SterileError(
                "the Sterile worker runs other code than this MeegXer (an image "
                "built before a change): build its image again (meegxer sterile "
                "build)"
            )
        return manifest

    def call(
        self,
        processor: Processor,
        method: str,
        args: tuple[Any, ...],
        model: tuple[str, State] | None = None,
    ) -> Any:
        """``processor.method(*args)`` in the worker; ``model``: the model
        argument (its id and value), sent once and then named by its id."""
        arrays: dict[str, npt.NDArray[Any]] = {}
        header: dict[str, Any] = {
            "type": "call", "impl": impl_key(processor), "method": method,
            "args": to_wire(list(args), arrays), "model_id": None,
        }  # fmt: skip
        models = None
        if model is not None:
            model_id, value = model
            model_arrays: dict[str, npt.NDArray[Any]] = {}
            models = {model_id: (to_wire(value, model_arrays), model_arrays)}
            header["model_id"] = model_id
            header["model_arg"] = MODEL_ARG[method]
        try:
            reply = self.request(header, arrays, models=models)
        except PluginError as exc:  # the worker broke, not the node
            raise SterileError(f"the Sterile worker failed: {exc}") from None
        raise_remote(reply)
        if reply.type != "return":
            raise SterileError(f"the Sterile worker sent {reply.type!r}")
        return from_wire(reply.header.get("value"), reply.arrays)


# -- the host proxy --------------------------------------------------------------


class SterileProcessor(Processor):
    """A built-in processor whose calls run in the Sterile worker."""

    def __init__(self, target: Processor, worker: SterileWorker) -> None:
        self.target = target
        self.worker = worker
        self.manifest = target.manifest  # type: ignore[misc]
        self._model: tuple[State, str] | None = None

    def _model_ref(self, model: State) -> tuple[str, State]:
        if self._model is None or self._model[0] is not model:
            self._model = (model, hash_state(model))  # once per model
        return self._model[1], model

    def _call(self, method: str, *args: Any) -> Any:
        index = MODEL_ARG.get(method)
        if index is None:
            return self.worker.call(self.target, method, args)
        model = args[index]
        rest = (*args[:index], None, *args[index + 1 :])
        return self.worker.call(self.target, method, rest, self._model_ref(model))

    def fit(self, calibration: Block, params: Params) -> State:
        return self._call("fit", calibration, params)  # type: ignore[no-any-return]

    def fit_labelled(self, calibration: Block, events: Any, params: Params) -> State:
        return self._call("fit_labelled", calibration, tuple(events), params)  # type: ignore[no-any-return]

    def fit_rails(self, calibration: Block, params: Params, rails: Any) -> State:
        return self._call("fit_rails", calibration, params, dict(rails))  # type: ignore[no-any-return]

    def fit_features(self, windows: Windows, events: Any, params: Params) -> State:
        return self._call("fit_features", windows, tuple(events), params)  # type: ignore[no-any-return]

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return self._call("transform", block, model, state, params)  # type: ignore[no-any-return]

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Any,
    ) -> TransformResult:
        return self._call("transform_rails", block, model, state, params, dict(rails))  # type: ignore[no-any-return]

    def transform_features(
        self, windows: Windows, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return self._call("transform_features", windows, model, state, params)  # type: ignore[no-any-return]

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        return self._call("flush", info, model, state, params)  # type: ignore[no-any-return]

    def describe_model(self, model: State, params: Params) -> dict[str, str]:
        return self._call("describe_model", model, params)  # type: ignore[no-any-return]

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        variant = self.target.for_mode(params, mode)  # which class: no data seen
        return (
            self if variant is self.target else SterileProcessor(variant, self.worker)
        )


# -- isolating a pipeline --------------------------------------------------------


def worker_launcher(image: str | None = None, *, local: bool = False) -> Launcher:
    """The Sterile worker: the image (default ``meegxer-sterile:VERSION``), or
    a local process for tests and development (``local``; refused unless
    ``MEEGXER_ALLOW_UNSANDBOXED=1``, as unsandboxed plugins are)."""
    if local:
        return LocalLauncher((sys.executable, "-m", "meegxer.plugins.sterile_worker"),
                             env={"PYTHONUNBUFFERED": "1"})  # fmt: skip
    # a whole recording, its copies and an ICA fit: more memory than a plugin
    return ContainerLauncher(image or IMAGE, memory="8g")


IMAGE = f"meegxer-sterile:{__version__}"  # what meegxer sterile build makes
IMAGE_ENV = "MEEGXER_STERILE_IMAGE"  # another tag (CI's meegxer-sterile:ci)
# development and tests: the worker as a local process, without a container
# (refused unless MEEGXER_ALLOW_UNSANDBOXED=1 too)
LOCAL_ENV = "MEEGXER_STERILE_LOCAL"


@dataclass(frozen=True)
class SterileScriptLauncher:
    """A custom script in Sterile mode: a container of the Sterile image (the
    same isolation as the built-in strips) running ``python -m
    meegxer.plugins.sterile_script``, which takes the script's code from the
    first frame on stdin (``preamble``): nothing is mounted. A script folder
    is sent whole (its code and data files). ``local`` runs it as a local
    process instead (tests and development; needs
    ``MEEGXER_ALLOW_UNSANDBOXED=1``)."""

    name: str  # the script's file name, or its folder's name
    files: tuple[tuple[str, bytes], ...]  # by path in the folder: read at start
    folder: bool = False  # a script folder (else one file, ``name``)
    image: str = IMAGE
    local: bool = False
    memory: str = "8g"

    @property
    def sandboxed(self) -> bool:
        return not self.local

    def _container(self) -> ContainerLauncher:
        return ContainerLauncher(self.image, memory=self.memory)

    def command(self, name: str) -> list[str]:
        if self.local:
            return LocalLauncher((sys.executable, "-m", SCRIPT_ENTRY)).command(name)
        base = self._container().command(name)  # [..., image]
        return [*base[:-1], "--entrypoint", "python", base[-1], "-m", SCRIPT_ENTRY]

    def kill(self, name: str) -> None:
        if not self.local:
            self._container().kill(name)

    def environment(self) -> dict[str, str] | None:
        return None

    def describe(self) -> str:
        where = "a local process" if self.local else f"container {self.image}"
        return f"custom script {self.name} (Sterile, {where})"

    def preamble(self) -> list[tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]]:
        """The frame the worker reads before the plugin protocol: the code
        (each file with its SHA-256; ``sterile_script.receive``)."""
        header = {"type": "script", "name": self.name, "folder": self.folder,
                  "files": [{"name": name, "data": f"f{i}",
                             "sha256": hashlib.sha256(data).hexdigest()}
                            for i, (name, data) in enumerate(self.files)]}  # fmt: skip
        arrays = {f"f{i}": np.frombuffer(data, dtype=np.uint8)
                  for i, (_, data) in enumerate(self.files)}  # fmt: skip
        return [(header, arrays)]


def script_files(path: Path) -> tuple[tuple[str, bytes], ...]:
    """What a script strip sends into its container: the file, or every file
    of the script folder (``meegxer.plugins.script.folder_files``), as they
    are on disk now."""
    from meegxer.plugins.script import folder_files

    if path.is_dir():
        return tuple((name, file.read_bytes()) for name, file in folder_files(path))
    return ((path.name, path.read_bytes()),)


def script_path(processor: PluginProcessor) -> Path | None:
    """The file (or script folder) a custom script strip runs (``python -m
    meegxer.plugins.script PATH``), or None for another local plugin."""
    command = list(type(processor).origin.get("command") or [])
    if len(command) >= 4 and command[1:3] == ["-m", "meegxer.plugins.script"]:
        return Path(command[-1])
    return None


class Isolation:
    """A pipeline running in Sterile mode: its worker, closed with ``close``,
    which also gives each stage its local processor back (and each custom
    script its local process)."""

    def __init__(
        self,
        worker: SterileWorker,
        proxied: list[str],
        undo: list[Callable[[], None]] | None = None,
    ) -> None:
        self.worker = worker
        self.proxied = proxied  # the node ids that run in the Sterile image
        self._undo = undo or []

    def close(self) -> None:
        for step in self._undo:
            step()
        self._undo = []
        self.worker.close()

    def __enter__(self) -> Isolation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def isolate(
    graph: Any,
    *,
    image: str | None = None,
    local: bool = False,
    timeout: float = 3600.0,
) -> Isolation:
    """Run ``graph``'s built-in processors in one Sterile worker (the
    ``meegxer.isolation`` entry point; ``meegxer.core.sterile.sterilize``),
    and each custom script in a container of the same image, its code sent
    in when the run starts. Refuses another local plugin (a program of its
    own, which the image cannot run)."""
    stages = [(node.id, node.impl) for node in graph.nodes
              if isinstance(node.impl, Stage)]  # fmt: skip
    scripts: list[tuple[str, PluginProcessor, Path]] = []
    for node_id, stage in stages:
        processor = stage.processor
        if isinstance(processor, PluginProcessor) and not isinstance(
            processor.launcher, ContainerLauncher
        ):
            path = script_path(processor)
            if path is None:
                raise SterileError(
                    f"{node_id}: a local plugin runs as a program of its own, "
                    "which Sterile mode cannot isolate (custom scripts can)"
                )
            scripts.append((node_id, processor, path))
    local = local or os.environ.get(LOCAL_ENV) == "1"
    image = image or os.environ.get(IMAGE_ENV) or None
    packages: dict[str, str] = {}  # what the scripts declare (REQUIRES)
    for node_id, _, path in scripts:
        from meegxer.plugins.requires import installed
        from meegxer.plugins.script import ScriptError

        try:
            packages.update(installed(path))
        except ScriptError as exc:
            raise SterileError(f"{node_id}: {exc}") from None
    if not local:  # missing or stale: say so now, not after a container start
        from meegxer.plugins.sterile_image import check_image

        check_image(image or IMAGE, requires=packages)
    worker = SterileWorker(worker_launcher(image, local=local), timeout=timeout)
    proxied = []
    undo: list[Callable[[], None]] = []
    for node_id, stage in stages:
        processor = stage.processor
        if isinstance(processor, (PluginProcessor, SterileProcessor, *HOST_PROCESSORS)):
            continue
        stage.processor = SterileProcessor(processor, worker)
        undo.append(functools.partial(setattr, stage, "processor", processor))
        proxied.append(node_id)
    for node_id, processor, path in scripts:
        try:
            files = script_files(path)  # the code as it is when the run starts
        except (OSError, ValueError) as exc:
            raise SterileError(f"{node_id}: cannot read the script: {exc}") from None
        processor.close()  # a local process from before: not used again
        processor.launcher = SterileScriptLauncher(  # type: ignore[misc]
            path.name, files, path.is_dir(), image or IMAGE, local
        )

        def back(p: PluginProcessor = processor) -> None:
            p.close()
            del p.launcher  # the class's: the local process again

        undo.append(back)
        proxied.append(node_id)
    return Isolation(worker, proxied, undo)


# -- the worker side (python -m meegxer.plugins.sterile_worker) -------------------


def resolve(key: str) -> type[Processor]:
    """The processor class ``module:qualname`` names: MeegXer's own only."""
    module_name, _, qualname = key.partition(":")
    if not module_name.startswith("meegxer.") or not qualname:
        raise ValueError(f"not a MeegXer processor: {key!r}")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not (isinstance(target, type) and issubclass(target, Processor)):
        raise ValueError(f"not a processor: {key!r}")
    if issubclass(target, (PluginProcessor, *HOST_PROCESSORS)):
        raise ValueError(f"{key!r} does not run in the Sterile worker")
    return target


class Server:
    """Answers the host's calls (one request at a time)."""

    def __init__(self) -> None:
        self.processors: dict[str, Processor] = {}
        self.models: dict[str, State] = {}

    def processor(self, key: str) -> Processor:
        if key not in self.processors:
            self.processors[key] = resolve(key)()
        return self.processors[key]

    def handle(self, message: Message) -> tuple[dict[str, Any], dict[str, Any]]:
        header = message.header
        kind = header.get("type")
        if kind == "hello":
            manifest = {"name": WORKER_NAME, "version": __version__,
                        "code": code_hash()}  # fmt: skip
            return {"type": "hello", "manifest": manifest,
                    "sdk": f"meegxer-sterile/{__version__}"}, {}  # fmt: skip
        if kind == "load_model":
            self.models[str(header["model_id"])] = from_wire(
                header["model"], message.arrays
            )
            return {"type": "ok"}, {}
        if kind == "shutdown":
            return {"type": "ok"}, {}
        if kind != "call" or header.get("method") not in METHODS:
            return {"type": "error", "error": "bad_request",
                    "message": f"unknown request {kind!r}"}, {}  # fmt: skip
        args = list(from_wire(header["args"], message.arrays))
        if header.get("model_id") is not None:
            model = self.models.get(str(header["model_id"]))
            if model is None:
                return {"type": "error", "error": "unknown_model",
                        "message": "load the model first"}, {}  # fmt: skip
            args[int(header["model_arg"])] = model
        method: Callable[..., Any] = getattr(
            self.processor(str(header["impl"])), str(header["method"])
        )
        try:
            value = method(*args)
        except Exception as exc:  # the node's own error: the host raises it
            name = type(exc).__name__
            return {"type": "raised",
                    "exception": name if name in ERRORS else "RuntimeError",
                    "message": str(exc) or name}, {}  # fmt: skip
        arrays: dict[str, npt.NDArray[Any]] = {}
        return {"type": "return", "value": to_wire(value, arrays)}, arrays


def serve() -> int:
    """The worker's main loop: requests on stdin, replies on the real stdout.
    Anything a library prints goes to stderr (the host's log), never into a
    reply."""
    from meegxer.plugins.wire import read_message, write_message

    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    requests = sys.stdin.buffer
    server = Server()
    while True:
        try:
            message = read_message(requests, MAX_FRAME)
        except EOFError:
            return 0
        header, arrays = server.handle(message)
        write_message(replies, {**header, "id": message.header.get("id")}, arrays)
        if message.header.get("type") == "shutdown":
            return 0
