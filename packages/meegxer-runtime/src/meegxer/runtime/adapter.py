"""Expose a plugin as an ordinary Processor node.

The adapter class is created from the plugin's handshake manifest, so the rest of
MeegXer (graph, schedulers, conformance) cannot tell a plugin from a built-in. Each
node instance owns one worker; the host keeps model and state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State, hash_state
from meegxer.runtime.wire import tree_from_wire, tree_to_wire
from meegxer.runtime.worker import Launcher, PluginError, PluginWorker

RESERVED_PREFIXES = ("meegxer.",)


def _info_dict(info: SignalInfo) -> dict[str, Any]:
    return {
        "ch_labels": list(info.ch_labels),
        "sfreq": info.sfreq,
        "stream_id": info.stream_id,
        "bads": list(info.bads),
    }


class PluginProcessor(Processor):
    """Base class of generated plugin adapters."""

    launcher: ClassVar[Launcher]
    worker_options: ClassVar[dict[str, Any]] = {}

    def __init__(self) -> None:
        self._worker: PluginWorker | None = None
        self._model_ids: dict[int, tuple[State, str]] = {}

    @property
    def worker(self) -> PluginWorker:
        if self._worker is None:
            self._worker = PluginWorker(self.launcher, **self.worker_options)
        return self._worker

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _model_id(self, model: State) -> str:
        cached = self._model_ids.get(id(model))
        if cached is not None and cached[0] is model:
            return cached[1]
        model_id = hash_state(model)
        self._model_ids = {id(model): (model, model_id)}  # keep only the current model
        return model_id

    def fit(self, calibration: Block, params: Params) -> State:
        reply = self.worker.request(
            {
                "type": "fit",
                "params": dict(params),
                "info": _info_dict(calibration.info),
                "sample_index": calibration.sample_index,
                "timestamp": calibration.timestamp,
            },
            {"data": calibration.data.astype(np.float32)},
        )
        if reply.type != "model":
            raise PluginError(f"expected a model reply, got {reply.type!r}")
        model = tree_from_wire(reply.header.get("model", {}), reply.arrays)
        if not isinstance(model, dict):
            raise PluginError("model must be an object")
        return model

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        arrays = {"data": block.data.astype(np.float32)}
        state_tree = None if state is None else tree_to_wire(state, arrays, "s")
        models: dict[str, tuple[Any, dict[str, Any]]] = {}
        model_id = None
        if self.manifest.requires_fit:
            model_id = self._model_id(model)
            model_arrays: dict[str, Any] = {}
            models[model_id] = (tree_to_wire(model, model_arrays, "m"), model_arrays)
        reply = self.worker.request(
            {
                "type": "transform",
                "params": dict(params),
                "info": _info_dict(block.info),
                "sample_index": block.sample_index,
                "timestamp": block.timestamp,
                "model_id": model_id,
                "state": state_tree,
            },
            arrays,
            models=models,
        )
        if reply.type != "result" or "data" not in reply.arrays:
            raise PluginError(f"expected a result with data, got {reply.header}")
        info = block.info
        if isinstance(reply.header.get("info"), dict):
            new = reply.header["info"]
            info = SignalInfo(
                tuple(new["ch_labels"]),
                float(new.get("sfreq", info.sfreq)),
                new.get("stream_id", info.stream_id),
                tuple(new.get("bads", ())),
            )
        out = Block(
            reply.arrays["data"].astype(np.float64),
            info,
            block.sample_index,
            block.timestamp,
        )
        new_state = tree_from_wire(reply.header.get("state") or {}, reply.arrays)
        events = tuple(Event(**e) for e in reply.header.get("events", []))
        return TransformResult(out, new_state, events)


def plugin_class(
    manifest: dict[str, Any],
    launcher: Launcher,
    *,
    worker_options: dict[str, Any] | None = None,
) -> type[PluginProcessor]:
    """Build the node class for a plugin from its (handshake) manifest."""
    parsed = Manifest.from_dict(manifest)
    if parsed.role != "processor":
        raise PluginError(f"plugins must be processors, not {parsed.role}s")
    if parsed.name.startswith(RESERVED_PREFIXES):
        raise PluginError(f"plugin name {parsed.name!r} uses a reserved prefix")
    attrs: dict[str, Any] = {
        "manifest": parsed,
        "launcher": launcher,
        "worker_options": dict(worker_options or {}),
        "__doc__": f"Plugin {parsed.name} {parsed.version} ({launcher.describe()}).",
    }
    name = "Plugin_" + parsed.name.replace(".", "_")
    return type(name, (PluginProcessor,), attrs)


def probe_manifest(launcher: Launcher, **worker_options: Any) -> dict[str, Any]:
    """Start a plugin once to read its manifest."""
    worker = PluginWorker(launcher, **worker_options)
    try:
        return worker.start()
    finally:
        worker.close()


PluginFactory = Callable[[], type[PluginProcessor]]
