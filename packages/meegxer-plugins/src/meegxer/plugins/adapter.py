"""Expose a plugin as an ordinary Processor node.

The adapter class is created from the plugin's handshake manifest, so the rest of
MeegXer (graph, schedulers, conformance) cannot tell a plugin from a built-in. Each
node instance owns one worker; the host keeps model and state.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import (
    Block,
    Event,
    FeatureBlock,
    SignalInfo,
    units_from_wire,
    units_wire,
)
from meegxer.core.manifest import Manifest
from meegxer.core.node import (
    Params,
    Processor,
    RailInput,
    RailOutput,
    TransformResult,
)
from meegxer.core.passband import filters_from_wire, filters_wire
from meegxer.core.ports import PortType
from meegxer.core.serialization import State, hash_state
from meegxer.plugins.wire import tree_from_wire, tree_to_wire
from meegxer.plugins.worker import Launcher, PluginError, PluginWorker

RESERVED_PREFIXES = ("meegxer.",)
# what a plugin's events may hold (rule 9; red team M3): they reach files
# (events.tsv, .set), the GUI and the action sinks
MAX_LABEL = 256
MAX_PAYLOAD_ITEMS = 32
EVENT_SLACK = 16  # events per reply beyond one per sample given
_EVENT_KEYS = {"sample_index", "label", "duration", "payload"}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _info_dict(info: SignalInfo) -> dict[str, Any]:
    return {
        "ch_labels": list(info.ch_labels),
        "sfreq": info.sfreq,
        "stream_id": info.stream_id,
        "bads": list(info.bads),
        **({"filters": filters_wire(info.filters)} if info.filters else {}),
    }


def _event_dict(e: Event) -> dict[str, Any]:
    return {"sample_index": e.sample_index, "label": e.label,
            "duration": e.duration, "payload": dict(e.payload)}  # fmt: skip


def _windows_to_wire(
    windows: Sequence[FeatureBlock], arrays: dict[str, Any], prefix: str
) -> list[dict[str, Any]]:
    wire = []
    for w in windows:
        name = f"{prefix}{len(arrays)}"
        arrays[name] = np.asarray(w.data, dtype=np.float64)
        wire.append({"window": list(w.window), "dims": list(w.dims),
                     "coords": {k: list(v) for k, v in w.coords.items()},
                     "data": name, "timestamp": w.timestamp,
                     **units_wire(w.unit, w.unit_dim, w.units)})  # fmt: skip
    return wire


def _windows_from_wire(
    wire: Any, arrays: Mapping[str, Any], info: SignalInfo, block: Block | None
) -> tuple[FeatureBlock, ...]:
    windows = []
    for w in wire:
        start, stop = (int(v) for v in w["window"])
        if w.get("timestamp") is not None:
            timestamp = float(w["timestamp"])
        elif block is not None:
            timestamp = block.timestamp + (start - block.sample_index) / info.sfreq
        else:
            timestamp = 0.0
        coords = {str(k): tuple(v) for k, v in (w.get("coords") or {}).items()}
        data = np.asarray(arrays[w["data"]], dtype=np.float64)
        windows.append(FeatureBlock(data, tuple(w["dims"]), (start, stop),
                                    timestamp, coords, info.stream_id,
                                    *units_from_wire(w)))  # fmt: skip
    return tuple(windows)


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plain(payload: object) -> bool:
    """An event payload a plugin may send: a few plain values by name."""
    if not isinstance(payload, dict) or len(payload) > MAX_PAYLOAD_ITEMS:
        return False
    for key, value in payload.items():
        if not isinstance(key, str) or len(key) > 64:
            return False
        if isinstance(value, str):
            if len(value) > MAX_LABEL:
                return False
        elif isinstance(value, float):
            if not math.isfinite(value):
                return False
        elif value is not None and not isinstance(value, (bool, int)):
            return False
    return True


def _info_from(new: Mapping[str, Any], info: SignalInfo) -> SignalInfo:
    return SignalInfo(
        tuple(new["ch_labels"]),
        float(new.get("sfreq", info.sfreq)),
        new.get("stream_id", info.stream_id),
        tuple(new.get("bads", ())),
        # a result's filters (spec 0.2.19), else the input's: a plugin that
        # filters lists what the stream has been through
        filters=filters_from_wire(new["filters"]) if "filters" in new else info.filters,
    )


# a local command's code hash, as last found: (checked at, files' stat,
# sha256, big). Code up to HASH_ALWAYS_BYTES (any script) is hashed at every
# check: an edit can keep the size, and file times are only as fine as the
# system's clock tick. A bigger script folder (data files) is looked at again
# at most once a second (a live strip asks on every block), and hashed again
# only when a file's size or modification time changed.
_CODE: dict[tuple[str, ...], tuple[float, tuple[Any, ...], str, bool]] = {}
RECHECK_SECONDS = 1.0
HASH_ALWAYS_BYTES = 4 * 1024 * 1024


def current_code(command: tuple[str, ...], recorded: str) -> str:
    """The SHA-256 of ``command``'s code files as they are now (a custom
    script, or a script folder's files); ``recorded`` when they cannot be
    read."""
    import time

    from meegxer.plugins.index import code_entries, code_sha256

    now = time.monotonic()
    known = _CODE.get(command)
    if known is not None and known[3] and now - known[0] < RECHECK_SECONDS:
        return known[2]
    try:
        entries = code_entries(list(command))
        stat = tuple((name, p.stat().st_mtime_ns, p.stat().st_size)
                     for name, p in entries)  # fmt: skip
        big = sum(size for _, _, size in stat) > HASH_ALWAYS_BYTES
        if big and known is not None and known[1] == stat:
            digest = known[2]
        else:
            digest = code_sha256(list(command)) if entries else recorded
    except (OSError, ValueError):  # gone, or a folder that grew too big
        return recorded
    _CODE[command] = (now, stat, digest, big)
    return digest


class PluginProcessor(Processor):
    """Base class of generated plugin adapters."""

    launcher: ClassVar[Launcher]
    worker_options: ClassVar[dict[str, Any]] = {}
    origin: ClassVar[dict[str, Any]] = {}  # image or command (sessions)

    @classmethod
    def provenance(cls) -> dict[str, Any]:
        """Where the plugin comes from (sessions; ``Stage.fingerprint``): for a
        local script, ``sha256`` is its code as it is on disk now, so an edited
        script is noticed while MeegXer runs (its cached outputs and fitted
        models no longer match)."""
        origin = dict(cls.origin)
        command = origin.get("command")
        if command and origin.get("sha256"):
            origin["sha256"] = current_code(tuple(command), origin["sha256"])
        script = origin.get("octave_script")
        if script and origin.get("sha256"):  # one small file: hashed each time
            with contextlib.suppress(OSError):  # gone: the recorded one
                origin["sha256"] = hashlib.sha256(Path(script).read_bytes()).hexdigest()
        return origin

    def __init__(self) -> None:
        self._worker: PluginWorker | None = None
        self._model_ids: dict[int, tuple[State, str]] = {}
        self._given: tuple[int, float] = (0, 0.0)  # input's end (index, sfreq)
        self._span = 0  # samples in the last block given
        self._budget: float | None = None  # a live run's, per block

    def go_live(self, budget: float | None) -> None:
        """In a live run, each block within ``budget`` seconds, no restart
        after a timeout (the run passes the strip over; red team M4)."""
        self._budget = budget

    def _within(self, what: str, first: int, n: int, sfreq: float) -> None:
        """Refuse output a plugin was not given the input for (rule 9; red
        team M2): starting before sample 0, or ending past the input's end
        (at the output's rate, by the exact ratio of the rates; one sample
        of rounding). A lagging plugin may start earlier than its block."""
        stop, in_sfreq = self._given
        limit = math.ceil(stop * sfreq / in_sfreq) + 1 if in_sfreq > 0 else 0
        if first < 0 or first + n > limit:
            raise PluginError(
                f"{self.manifest.name}: its {what} covers samples {first} to "
                f"{first + n}, but it was given samples up to {limit - 1} "
                f"(at {sfreq:g} Hz)"
            )

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
        return self.fit_labelled(calibration, (), params)

    def fit_rails(
        self, calibration: Block, params: Params, rails: Mapping[str, RailInput]
    ) -> State:
        return self.fit_labelled(calibration, (), params, rails)

    def _rails_to_wire(
        self, rails: Mapping[str, RailInput], arrays: dict[str, Any]
    ) -> dict[str, Any]:
        """The rails the extra input ports read, for a request (spec 0.2.17)."""
        wire: dict[str, Any] = {}
        for port in self.manifest.rail_inputs:
            data = rails[port.name]
            if port.type is PortType.SIGNAL:
                assert isinstance(data, Block)
                name = f"r{len(arrays)}"
                arrays[name] = data.data.astype(np.float32)
                wire[port.name] = {"type": "signal", "data": name,
                                   "info": _info_dict(data.info),
                                   "sample_index": data.sample_index}  # fmt: skip
            elif port.type is PortType.FEATURES:
                assert isinstance(data, tuple)
                windows = [w for w in data if isinstance(w, FeatureBlock)]
                wire[port.name] = {
                    "type": "features",
                    "windows": _windows_to_wire(windows, arrays, "rw"),
                }
            else:
                assert isinstance(data, tuple)
                events = [_event_dict(e) for e in data if isinstance(e, Event)]
                wire[port.name] = {"type": "events", "events": events}
        return wire

    def _rails_from_wire(
        self, reply: Any, info: SignalInfo, block: Block | None
    ) -> dict[str, RailOutput]:
        """What the extra output ports wrote, from a result (spec 0.2.17)."""
        got = reply.header.get("rails") or {}
        rails: dict[str, RailOutput] = {}
        for port in self.manifest.rail_outputs:
            item = got.get(port.name)
            if item is None:
                continue
            if port.type is PortType.FEATURES:
                rails[port.name] = self._windows(
                    item.get("windows", []), reply.arrays, info, block
                )
                continue
            rail_info = _info_from(item["info"], info) if item.get("info") else info
            data = np.asarray(reply.arrays[item["data"]], dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(rail_info.n_channels, -1)
            first = int(item.get("sample_index",
                                 block.sample_index if block else 0))  # fmt: skip
            timestamp = (block.timestamp + (first - block.sample_index) / info.sfreq
                         if block is not None else 0.0)  # fmt: skip
            self._within(f"rail {port.name}", first, data.shape[-1], rail_info.sfreq)
            rails[port.name] = Block(data, rail_info, first, timestamp)
        return rails

    def fit_labelled(
        self,
        calibration: Block,
        events: Sequence[Event],
        params: Params,
        rails: Mapping[str, RailInput] | None = None,
    ) -> State:
        header: dict[str, Any] = {
            "type": "fit",
            "params": dict(params),
            "info": _info_dict(calibration.info),
            "sample_index": calibration.sample_index,
            "timestamp": calibration.timestamp,
        }
        if self.manifest.fit_events:  # spec 0.2.16
            header["events"] = [_event_dict(e) for e in events]
        arrays: dict[str, Any] = {"data": calibration.data.astype(np.float32)}
        if rails is not None:  # spec 0.2.17
            header["rails"] = self._rails_to_wire(rails, arrays)
        reply = self.worker.request(header, arrays)
        if reply.type != "model":
            raise PluginError(f"expected a model reply, got {reply.type!r}")
        model = tree_from_wire(reply.header.get("model", {}), reply.arrays)
        if not isinstance(model, dict):
            raise PluginError("model must be an object")
        return model

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        result = self._exchange("transform", block.info, model, state, params, block)
        assert result is not None
        return result

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        result = self._exchange(
            "transform", block.info, model, state, params, block, rails
        )
        assert result is not None
        return result

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        """End of stream: ask a lagging plugin for what it still holds. Plugins
        built for protocol 0.1 refuse the message: they hold nothing."""
        try:
            result = self._exchange("flush", info, model, state, params, None)
        except PluginError as exc:
            if "bad_request" in str(exc):
                return None
            raise
        if result is None:
            return None
        if not isinstance(result.block, Block):  # feature windows: any left?
            return result if result.block else None
        return result if result.block.n_samples else None

    def _events(self, wire: Any, sfreq: float) -> tuple[Event, ...]:
        """A reply's events, each checked (red team M3): at most one per
        sample given (plus ``EVENT_SLACK``); a label of text without control
        characters, up to ``MAX_LABEL``; a sample index within the input given
        (at the higher of the two rates); a duration of 0 or more; a payload
        of up to ``MAX_PAYLOAD_ITEMS`` plain values."""
        name = self.manifest.name
        if wire is None:
            return ()
        if not isinstance(wire, list):
            raise PluginError(f"{name}: events must be a list")
        if len(wire) > self._span + EVENT_SLACK:
            raise PluginError(f"{name}: {len(wire)} events for a block of "
                              f"{self._span} samples")  # fmt: skip
        stop, in_sfreq = self._given
        ratio = max(1.0, sfreq / in_sfreq) if in_sfreq > 0 else 1.0
        limit = math.ceil(stop * ratio) + 1
        events = []
        for e in wire:
            if not isinstance(e, dict) or not {"sample_index", "label"} <= set(e):
                raise PluginError(f"{name}: an event needs a sample_index and a "
                                  "label")  # fmt: skip
            if set(e) - _EVENT_KEYS:
                raise PluginError(f"{name}: an event has unknown fields "
                                  f"{sorted(set(e) - _EVENT_KEYS)}")  # fmt: skip
            index, label = e["sample_index"], e["label"]
            duration, payload = e.get("duration", 0), e.get("payload") or {}
            if not isinstance(label, str) or not label or len(label) > MAX_LABEL:
                raise PluginError(f"{name}: an event label must be text of 1 to "
                                  f"{MAX_LABEL} characters")  # fmt: skip
            if _CONTROL.search(label):
                raise PluginError(f"{name}: an event label holds a control "
                                  f"character: {label!r}")  # fmt: skip
            if not _whole(index) or not 0 <= index < limit:
                raise PluginError(f"{name}: an event at sample {index!r}, but it "
                                  f"was given samples up to {limit - 1}")  # fmt: skip
            if not _whole(duration) or duration < 0:
                raise PluginError(f"{name}: an event's duration must be a whole "
                                  "number of samples >= 0, got "
                                  f"{duration!r}")  # fmt: skip
            if not _plain(payload):
                raise PluginError(f"{name}: an event's payload must be up to "
                                  f"{MAX_PAYLOAD_ITEMS} plain values (text, "
                                  "numbers, true/false) by name")  # fmt: skip
            events.append(Event(index, label, duration, payload))
        return tuple(events)

    def _windows(
        self,
        wire: Any,
        arrays: Mapping[str, Any],
        info: SignalInfo,
        block: Block | None,
    ) -> tuple[FeatureBlock, ...]:
        """Feature windows from a reply, each within the input given (M2)."""
        windows = _windows_from_wire(wire, arrays, info, block)
        for w in windows:
            start, stop = w.window
            if stop < start:
                raise PluginError(f"{self.manifest.name}: a window ends before "
                                  f"it starts ({start}, {stop})")  # fmt: skip
            self._within("window", start, stop - start, info.sfreq)
        return windows

    def _features(
        self, reply: Any, info: SignalInfo, block: Block | None
    ) -> TransformResult:
        """An analysis plugin's result: its completed feature windows (0.2.14)."""
        if reply.type != "result" or not isinstance(reply.header.get("features"), list):
            raise PluginError(f"expected a result with features, got {reply.header}")
        windows = self._windows(reply.header["features"], reply.arrays, info, block)
        new_state = tree_from_wire(reply.header.get("state") or {}, reply.arrays)
        events = self._events(reply.header.get("events"), info.sfreq)
        rails = self._rails_from_wire(reply, info, block)
        return TransformResult(windows, new_state, events, rails)

    def _exchange(
        self,
        kind: str,
        info: SignalInfo,
        model: State,
        state: State | None,
        params: Params,
        block: Block | None,
        rails: Mapping[str, RailInput] | None = None,
    ) -> TransformResult | None:
        arrays: dict[str, Any] = {}
        if block is not None:
            arrays["data"] = block.data.astype(np.float32)
            self._given = (block.sample_index + block.n_samples, block.info.sfreq)
            self._span = block.n_samples
        rails_wire = self._rails_to_wire(rails, arrays) if rails is not None else None
        state_tree = None if state is None else tree_to_wire(state, arrays, "s")
        models: dict[str, tuple[Any, dict[str, Any]]] = {}
        model_id = None
        if self.manifest.requires_fit:
            model_id = self._model_id(model)
            model_arrays: dict[str, Any] = {}
            models[model_id] = (tree_to_wire(model, model_arrays, "m"), model_arrays)
        header: dict[str, Any] = {
            "type": kind,
            "params": dict(params),
            "info": _info_dict(info),
            "model_id": model_id,
            "state": state_tree,
        }
        if block is not None:
            header.update(sample_index=block.sample_index, timestamp=block.timestamp)
        if rails_wire is not None:
            header["rails"] = rails_wire
        reply = self.worker.request(header, arrays, models=models,
                                    timeout=self._budget,
                                    retry_timeouts=self._budget is None)  # fmt: skip
        if self.manifest.kind == "analysis":
            return self._features(reply, info, block)
        if reply.type != "result" or "data" not in reply.arrays:
            raise PluginError(f"expected a result with data, got {reply.header}")
        in_info = info
        if isinstance(reply.header.get("info"), dict):
            info = _info_from(reply.header["info"], info)
        data = np.asarray(reply.arrays["data"], dtype=np.float64)
        if data.ndim == 1:  # an empty result may lose its channel axis
            data = data.reshape(info.n_channels, -1)
        first = int(reply.header.get("sample_index",
                                     block.sample_index if block else 0))  # fmt: skip
        if block is not None:
            timestamp = block.timestamp + (first - block.sample_index) / info.sfreq
        else:
            timestamp = 0.0
        self._within("output", first, data.shape[-1], info.sfreq)
        out = Block(data, info, first, timestamp)
        new_state = tree_from_wire(reply.header.get("state") or {}, reply.arrays)
        events = self._events(reply.header.get("events"), info.sfreq)
        rails = self._rails_from_wire(reply, in_info, block)
        return TransformResult(out, new_state, events, rails)


def plugin_class(
    manifest: dict[str, Any],
    launcher: Launcher,
    *,
    worker_options: dict[str, Any] | None = None,
    origin: dict[str, Any] | None = None,
) -> type[PluginProcessor]:
    """Build the node class for a plugin from its (handshake) manifest."""
    parsed = Manifest.from_dict(manifest)
    if parsed.role != "processor":
        raise PluginError(f"plugins must be processors, not {parsed.role}s")
    if parsed.name.startswith(RESERVED_PREFIXES):
        raise PluginError(f"plugin name {parsed.name!r} uses a reserved prefix")
    if parsed.inputs[0].type.value != "signal":  # spec 0.2.23: built-ins only
        raise PluginError(
            "plugins take Signal input (a Features input, e.g. a classifier "
            "reading a feature strip, is built-in only)"
        )
    attrs: dict[str, Any] = {
        "manifest": parsed,
        "launcher": launcher,
        "worker_options": dict(worker_options or {}),
        "origin": dict(origin or {}),
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
