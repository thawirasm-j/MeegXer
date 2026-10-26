"""Control API: JSON-RPC 2.0 over ZeroMQ REQ/REP (rule 3).

The daemon owns the pipeline and the real-time scheduler; clients (GUI, CLI,
scripts) only talk to it through these methods. See ``spec/control-api.md``.
The control and display sockets bind to localhost by default and accept only
clients with the daemon's token (``meegxer.core.auth``). The token is not
encrypted on the wire, so binding to other interfaces is only safe on a trusted
network.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import zmq

from meegxer.core import __version__
from meegxer.core.auth import (
    Guard,
    login,
    missing_token_hint,
    read_token,
    remove_token,
    write_token,
)
from meegxer.core.bids_batch import is_all_subjects, recordings_of
from meegxer.core.display import DisplayPublisher, ZmqContext
from meegxer.core.epoch_review import EpochReview
from meegxer.core.graph import Graph
from meegxer.core.import_review import open_import_review, save_import_review
from meegxer.core.manifest import SPEC_VERSION, Manifest
from meegxer.core.node import Source
from meegxer.core.offline import OfflineJob
from meegxer.core.ports import Port
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.recording import builtin_montages, describe, lookup_positions
from meegxer.core.registry import Registry, node_origin
from meegxer.core.review import Mark, ReviewStrip
from meegxer.core.session import (
    build_session,
    load_session,
    sink_outputs,
    state_to_json,
)
from meegxer.core.sinks import discard as discard_spool
from meegxer.core.sinks import free_name, recover, unfinished_recordings
from meegxer.core.sources import BidsSource
from meegxer.core.stage import Stage

DEFAULT_CONTROL = "tcp://127.0.0.1:5555"
DEFAULT_DISPLAY = "tcp://127.0.0.1:5556"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SERVER_ERROR = -32000


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _port_dict(port: Port) -> dict[str, str]:
    out = {"name": port.name, "type": port.type.value}
    if port.rail:
        out["rail"] = port.rail
    if port.space:
        out["space"] = port.space
    return out


def manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "version": manifest.version,
        "role": manifest.role,
        "type": manifest.kind,
        "availability": manifest.availability,
        "requires_fit": manifest.requires_fit,
        "requires_positions": manifest.requires_positions,
        "fit_events": manifest.fit_events,
        "latency_samples": manifest.latency_samples,
        "description": manifest.description,
        "authors": list(manifest.authors),
        "references": [r.as_dict() for r in manifest.references],
        "license": manifest.license,
        "category": manifest.category,
        "channel_types": list(manifest.channel_types or ("eeg",)),
        "live_params": sorted(manifest.live_params),
        "inputs": [_port_dict(p) for p in manifest.inputs],
        "outputs": [_port_dict(p) for p in manifest.outputs],
        "params": [
            {
                "name": p.name,
                "type": p.type,
                "default": p.default,
                "min": p.min,
                "max": p.max,
                "choices": list(p.choices),
                "unit": p.unit,
                "description": p.description,
                "advanced": p.advanced,
                "primary": p.primary,
                "step": p.step,
                "path": p.path,
                "extensions": list(p.extensions),
            }
            for p in manifest.params
        ],
    }


class Daemon:
    """Serves the control API and runs one pipeline at a time."""

    def __init__(
        self,
        control_endpoint: str = DEFAULT_CONTROL,
        display_endpoint: str = DEFAULT_DISPLAY,
        *,
        registry: Registry | None = None,
        context: ZmqContext | None = None,
    ) -> None:
        self._own_registry = registry is None  # rescan_nodes rebuilds only our own
        self.registry = registry or Registry.from_entry_points()
        # its own context: the token check (ZAP) is one handler per context
        self._own_context = context is None
        self._context = context or zmq.Context()
        self._guard = Guard(self._context)
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        Guard.protect(self._socket)
        self._socket.bind(control_endpoint)
        self.control_endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.display = DisplayPublisher(display_endpoint, context=self._context,
                                        secure=True)  # fmt: skip
        self.display_endpoint = self.display.endpoint
        self.token_file = write_token(self.control_endpoint, self._guard.token)
        self._pipeline: dict[str, Any] | None = None
        self._graph: Graph | None = None
        self._scheduler: RealtimeScheduler | None = None
        self._offline: OfflineJob | None = None
        # models the next offline run starts from: the previous offline run's, or
        # those of the session loaded last (fingerprints decide if they are used)
        self._offline_models: Graph | None = None
        self._stop = threading.Event()
        self._methods: dict[str, Callable[..., Any]] = {
            "ping": self.ping,
            "list_nodes": self.list_nodes,
            "rescan_nodes": self.rescan_nodes,
            "load_pipeline": self.load_pipeline,
            "get_pipeline": self.get_pipeline,
            "get_session": self.get_session,
            "get_model": self.get_model,
            "load_session": self.load_session,
            "start": self.start,
            "stop": self.stop,
            "status": self.status,
            "set_params": self.set_params,
            "calibrate": self.calibrate,
            "clear_calibration": self.clear_calibration,
            "set_pass_uncalibrated": self.set_pass_uncalibrated,
            "set_strip_params": self.set_strip_params,
            "describe_source": self.describe_source,
            "import_review_open": self.import_review_open,
            "save_import_review": self.save_import_review,
            "list_montages": self.list_montages,
            "lookup_positions": self.lookup_positions,
            "run_offline": self.run_offline,
            "offline_window": self.offline_window,
            "offline_features": self.offline_features,
            "offline_events": self.offline_events,
            "submit_review": self.submit_review,
            "offline_review_epochs": self.offline_review_epochs,
            "cancel_review": self.cancel_review,
            "unfinished_recordings": self.unfinished_recordings,
            "recover_recording": self.recover_recording,
            "shutdown": self.shutdown,
        }

    # -- methods ---------------------------------------------------------------------

    def ping(self) -> dict[str, str]:
        return {"version": __version__, "spec_version": SPEC_VERSION,
                "display": self.display_endpoint}  # fmt: skip

    def list_nodes(self) -> list[dict[str, Any]]:
        return [
            {**manifest_to_dict(cls.manifest), "origin": node_origin(cls)}
            for cls in (self.registry.get(n) for n in self.registry.names())
        ]

    def rescan_nodes(self) -> list[dict[str, Any]]:
        """Find nodes installed since the daemon started (e.g. a plugin just
        added); a running pipeline keeps the classes it was built with."""
        if self._own_registry:
            self.registry = Registry.from_entry_points()
        return self.list_nodes()

    def load_pipeline(self, pipeline: dict[str, Any]) -> dict[str, Any]:
        if self._scheduler is not None and self._scheduler.running:
            raise RuntimeError("stop the running pipeline before loading another")
        graph = Graph.from_dict(pipeline, self.registry)
        RealtimeScheduler(graph)  # validates: no offline-only nodes
        if self._graph is not None:
            # the GUI loads the pipeline again on every start: keep calibration
            self._carry_models(self._graph, graph)
            self._close_processors(self._graph)
        self._pipeline, self._graph, self._scheduler = graph.to_dict(), graph, None
        return self._pipeline

    def get_session(
        self,
        name: str | None = None,
        pipeline: dict[str, Any] | None = None,
        offline: bool = False,
    ) -> dict[str, Any]:
        """A session (``spec/session.md``) of the loaded pipeline with the models
        calibrated live, or with ``offline`` of the last offline run (its fitted
        models and the files it wrote). With ``pipeline`` (e.g. the GUI's), that
        pipeline is saved, with the models of nodes it shares. Paths are
        absolute (the client writes the file and makes them relative to it)."""
        source = self._offline.graph if offline and self._offline else self._graph
        if offline and (self._offline is None or self._offline.state != "done"):
            raise RuntimeError("no finished offline run to save")
        if source is None:
            raise RuntimeError("no pipeline loaded")
        graph = source
        if pipeline is not None:
            graph = Graph.from_dict(pipeline, self.registry)
            self._carry_models(source, graph)
        outputs = sink_outputs(graph) if offline else ()
        return build_session(graph, name=name, outputs=outputs)

    def get_model(self, node: str, offline: bool = False) -> dict[str, Any]:
        """One node's fitted model as readable JSON (``spec/session.md``), or
        null without one: of the loaded pipeline, or with ``offline`` of the
        last offline run (e.g. ICA's unmixing, for the IC head plot)."""
        graph = self._offline.graph if offline and self._offline else self._graph
        if graph is None:
            raise RuntimeError("no pipeline loaded")
        if node not in graph:
            raise ValueError(f"no node {node!r}")
        impl = graph[node].impl
        model = impl.model if isinstance(impl, Stage) else None
        data = None if model is None else state_to_json(model.data)
        return {"node": node, "model": data}

    def load_session(
        self, session: dict[str, Any], allow_version_mismatch: bool = False
    ) -> dict[str, Any]:
        """Load a session's pipeline with its fitted models (paths must be
        absolute: the client resolves them against the session file). A model
        is used live only if its fingerprint matches this input (rule 7)."""
        if self._scheduler is not None and self._scheduler.running:
            raise RuntimeError("stop the running pipeline before loading another")
        loaded = load_session(
            session, self.registry, allow_version_mismatch=bool(allow_version_mismatch)
        )
        warnings = list(loaded.warnings)
        for node_id in loaded.unpinned_models:
            warnings.append(f"models: {node_id!r} has no fingerprint; not used live")
        if self._graph is not None:
            self._close_processors(self._graph)
        graph = loaded.graph
        # an offline-only pipeline loads too; start() refuses it with the reason
        self._pipeline, self._graph, self._scheduler = graph.to_dict(), graph, None
        self._offline_models = graph
        models = [
            n.id
            for n in graph.nodes
            if isinstance(n.impl, Stage) and n.impl.model is not None
        ]
        return {
            "pipeline": self._pipeline,
            "models": models,
            "dropped_models": loaded.dropped_models,
            "warnings": warnings,
        }

    def get_pipeline(self) -> dict[str, Any] | None:
        if self._graph is not None:
            return self._graph.to_dict()
        return self._pipeline

    def start(self) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("no pipeline loaded")
        if self._scheduler is not None and self._scheduler.running:
            raise RuntimeError("pipeline is already running")
        if self._scheduler is not None:  # restart: fresh sources and sinks
            old = self._graph
            self._graph = Graph.from_dict(self.get_pipeline() or {}, self.registry)
            if old is not None:
                self._carry_models(old, self._graph)
                self._close_processors(old)
        assert self._graph is not None
        self._scheduler = RealtimeScheduler(self._graph, display=self.display)
        self._scheduler.start()
        return self.status()

    @staticmethod
    def _carry_models(old: Graph, new: Graph) -> None:
        """Fitted models survive a restart by default (drop them explicitly with
        ``clear_calibration``). Fingerprints still mark them needs-refit if the
        source or anything upstream changed, and a channel or rate mismatch is
        refused, so a kept model is never used silently on different data."""
        for node in new.nodes:
            if node.id not in old or not isinstance(node.impl, Stage):
                continue
            previous = old[node.id].impl
            if (
                isinstance(previous, Stage)
                and previous.model is not None
                and previous.manifest.name == node.impl.manifest.name
                and previous.manifest.version == node.impl.manifest.version
            ):
                node.impl.model = previous.model
            if (
                isinstance(previous, Stage)
                and isinstance(previous.processor, ReviewStrip | EpochReview)
                and type(node.impl.processor) is type(previous.processor)
            ):
                node.impl.processor.record = previous.processor.record  # who reviewed

    @staticmethod
    def _close_processors(graph: Graph) -> None:
        for node in graph.nodes:
            if isinstance(node.impl, Stage):
                node.impl.processor.close()

    def stop(self) -> dict[str, Any]:
        if self._scheduler is not None:
            self._scheduler.stop()
        return self.status()

    def status(self) -> dict[str, Any]:
        result = self._live_status()
        if self._offline is not None:
            result["offline"] = self._offline.status()
        return result

    def _live_status(self) -> dict[str, Any]:
        if self._scheduler is not None:
            return self._scheduler.status()
        nodes = []
        for n in self._graph.nodes if self._graph else []:
            entry: dict[str, Any] = {
                "id": n.id, "node": n.manifest.name, "role": n.role.value,
                "status": "stopped", "params": dict(n.impl.params),
            }  # fmt: skip
            if isinstance(n.impl, Stage) and n.impl.manifest.requires_fit:
                entry["calibrated"] = n.impl.model is not None
                entry["pass_uncalibrated"] = n.impl.pass_uncalibrated
            nodes.append(entry)
        return {"running": False, "nodes": nodes}

    def set_params(self, node: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._graph is None:
            raise RuntimeError("no pipeline loaded")
        if node not in self._graph:
            raise KeyError(f"no node {node!r}")
        if self._scheduler is not None:
            return self._scheduler.set_params(node, params)
        return RealtimeScheduler(self._graph).set_params(node, params)

    def calibrate(self, node: str, seconds: float) -> dict[str, Any]:
        if self._scheduler is None or not self._scheduler.running:
            raise RuntimeError("start the pipeline before calibrating")
        self._scheduler.calibrate(node, float(seconds))
        return self.status()

    def clear_calibration(self, node: str | None = None) -> dict[str, Any]:
        """Drop fitted models: one node, or all nodes when ``node`` is omitted."""
        if self._graph is None:
            raise RuntimeError("no pipeline loaded")
        if node is not None and node not in self._graph:
            raise KeyError(f"no node {node!r}")
        scheduler = self._scheduler or RealtimeScheduler(self._graph)
        cleared = scheduler.clear_calibration(node)
        return {"cleared": cleared, "status": self.status()}

    def set_pass_uncalibrated(self, node: str, enabled: bool) -> dict[str, Any]:
        """Per-node opt-in to pass data through while it has no valid model."""
        if self._graph is None:
            raise RuntimeError("no pipeline loaded")
        if node not in self._graph:
            raise KeyError(f"no node {node!r}")
        scheduler = self._scheduler or RealtimeScheduler(self._graph)
        scheduler.set_pass_uncalibrated(node, bool(enabled))
        return self.status()

    def set_strip_params(self, node: str, params: list[str]) -> dict[str, Any]:
        """Display only: which parameters the node's strip shows in advanced mode."""
        if self._graph is None:
            raise RuntimeError("no pipeline loaded")
        if node not in self._graph:
            raise RuntimeError(f"unknown node {node!r}")
        self._graph.set_strip_params(node, params)
        return self._graph.to_dict()

    def _raw_source(self, node: dict[str, Any]) -> Source:
        """The input a source entry names, as the raw file has it: without the
        pipeline's recording settings or the review at import."""
        names = {p.name for p in self.registry.get(node["node"]).manifest.params}
        if "import_review" not in names:
            raise ValueError("this input has no file to review at import")
        params = {**node.get("params", {}), "import_review": False}
        entry = {"id": "input", "node": node["node"], "params": params}
        source = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [entry]},
                                 self.registry)["input"].impl  # fmt: skip
        if not isinstance(source, Source) or is_all_subjects(source):
            raise RuntimeError("review at import needs one recording file")
        return source

    def import_review_open(self, node: dict[str, Any]) -> dict[str, Any]:
        """The review window's request to review a recording at import
        (docs/REVIEW.md): the raw file's events and bad channels, or the
        review saved before (its marks, bad channels, crop)."""
        return open_import_review(self._raw_source(node))

    def save_import_review(
        self,
        node: dict[str, Any],
        marks: list[dict[str, Any]],
        bads: list[str] | None = None,
        crop: list[float | None] | None = None,
        viewed: list[list[float]] | None = None,
        reviewer: str = "",
    ) -> dict[str, Any]:
        """Save the review at import of a source entry's recording: every
        pipeline reading it then uses its events, bad channels and crop."""
        span = None
        if crop is not None:
            span = (float(crop[0] or 0.0), None if crop[1] is None else float(crop[1]))
        path = save_import_review(
            self._raw_source(node), [Mark.from_wire(m) for m in marks],
            [str(b) for b in bads or ()], span,
            [(float(a), float(b)) for a, b in viewed or ()], str(reviewer),
        )  # fmt: skip
        return {"file": str(path)}

    def describe_source(self, node: dict[str, Any]) -> dict[str, Any]:
        """Describe the recording a source entry (with its ``recording``
        overrides) would produce: channels before and after the overrides,
        sampling rate, length and events. Opens the file or stream to read it.
        A BIDS input naming no subject (every subject) shows its first."""

        def build(entry: dict[str, Any]) -> object:
            return Graph.from_dict(
                {"spec_version": SPEC_VERSION, "nodes": [{**entry, "id": "input"}]},
                self.registry,
            )["input"].impl

        source = build(node)
        if is_all_subjects(source):
            assert isinstance(source, BidsSource)
            first = recordings_of(source)
            if not first:
                raise RuntimeError("no matching EEG recordings in the dataset")
            source = build({**node, "params": {**node.get("params", {}), **first[0]}})
        if not isinstance(source, Source):
            raise RuntimeError(f"{node.get('node')} is not a source")
        try:
            return describe(source)
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()

    def list_montages(self) -> list[str]:
        return builtin_montages()

    def lookup_positions(self, labels: list[str], montage: str) -> dict[str, Any]:
        """Positions for channel labels from a standard montage or montage file:
        ``{positions: {label: [x, y, z]}, unmatched: [label]}``."""
        found, unmatched = lookup_positions(list(labels), str(montage))
        return {"positions": {k: list(v) for k, v in found.items()},
                "unmatched": unmatched}  # fmt: skip

    def unfinished_recordings(self) -> list[dict[str, Any]]:
        """Recordings a crashed run left in their spools (``sinks.py``)."""
        return unfinished_recordings()

    def recover_recording(self, path: str, discard: bool = False) -> dict[str, Any]:
        """Write an unfinished recording (to its own name, or ``<name>_recovered``
        when that exists), or with ``discard`` delete it: ``{path}``, the file
        written, or ``null``. Only recordings ``unfinished_recordings`` lists."""
        if path not in {r["path"] for r in unfinished_recordings()}:
            raise ValueError(f"no unfinished recording at {path}")
        target = Path(path)
        if discard:
            discard_spool(target)
            return {"path": None}
        output = free_name(target, "recovered") if target.exists() else target
        return {"path": str(recover(target, output))}

    def run_offline(
        self, pipeline: dict[str, Any], reviewer: str = ""
    ) -> dict[str, Any]:
        """Run ``pipeline`` over its whole (finite) input in the background; its
        probes become scrollable envelopes (``offline_window``), its sinks write
        files. Independent of the live pipeline, but not while it runs.
        ``reviewer``: who reviews at its review strips (saved with each review,
        docs/REVIEW.md)."""
        if self._scheduler is not None and self._scheduler.running:
            raise RuntimeError("stop the live pipeline before an offline run")
        if self._offline is not None and self._offline.busy:
            raise RuntimeError("an offline run is already in progress")
        graph = Graph.from_dict(pipeline, self.registry)
        if is_all_subjects(graph.source.impl):  # models fitted per recording
            registry = self.registry
            job = OfflineJob(graph, rebuild=lambda: Graph.from_dict(pipeline, registry),
                             reviewer=str(reviewer))  # fmt: skip
            self._offline_models = None
        else:
            if self._offline_models is not None:
                self._carry_models(self._offline_models, graph)
            job = OfflineJob(graph, reviewer=str(reviewer))
            self._offline_models = graph
        if self._offline is not None:
            self._offline.discard()  # the previous run's files
        self._offline = job
        job.start()
        return job.status()

    def submit_review(
        self,
        marks: list[dict[str, Any]],
        viewed: list[list[float]] | None = None,
        bads: list[str] | None = None,
        epochs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """The marks for the review strip the offline run waits at (state
        ``review``), the spans looked at and the channels marked bad, or the
        verdicts of an epoch review strip (``epochs``): saved to its sidecar,
        and the run goes on."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        self._offline.submit_review(marks, viewed, bads, epochs)
        return self._offline.status()

    def offline_review_epochs(
        self,
        indices: list[int] | None = None,
        reduce: str = "none",
        max_samples: int = 1000,
    ) -> dict[str, Any]:
        """The epochs an epoch review strip waits on (the request's
        ``epochs``, by index; none given: all): each channel's samples (in the
        unit given, usually volts) in at most ``max_samples`` points per
        epoch, or their mean with ``reduce="mean"`` (the ERP)."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        if reduce not in ("none", "mean"):
            raise ValueError("reduce must be 'none' or 'mean'")
        chosen = None if indices is None else [int(i) for i in indices]
        return self._offline.review_epochs(chosen, reduce, int(max_samples))

    def cancel_review(self) -> dict[str, Any]:
        """Stop the offline run waiting at a review strip (nothing saved)."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        self._offline.cancel_review()
        self._offline.join(5)
        return self._offline.status()

    def offline_window(
        self,
        probe: str,
        start: int,
        n_samples: int,
        max_bins: int = 2000,
        display: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Min/max envelope of samples ``[start, start + n_samples)`` of one probe
        of the current offline run, in at most ``max_bins`` bins; with
        ``display`` (``montage``, ``highpass``, ``lowpass``, ``notch`` in Hz,
        ``bads``) through a display montage and filter, from the raw samples
        (``meegxer.core.montage``)."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        options: dict[str, Any] = {}
        for key, value in (display or {}).items():
            if key == "montage":
                options[key] = str(value)
            elif key in ("highpass", "lowpass", "notch"):
                options[key] = float(value)
            elif key == "bads":
                options[key] = tuple(str(b) for b in value)
            else:
                raise ValueError(f"unknown display option {key!r}")
        return self._offline.window(probe, int(start), int(n_samples), int(max_bins),
                                    **options)  # fmt: skip

    def offline_events(self, node: str | None = None) -> dict[str, Any]:
        """The events of the current offline run so far (seconds from the
        recording's start): the input's and every strip's (a review's marks),
        or those ``node`` sees."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        return {"events": self._offline.events(node)}

    def offline_features(
        self,
        probe: str,
        start: int = 0,
        stop: int = 2**62,
        max_windows: int = 2000,
        reduce: str = "none",
    ) -> dict[str, Any]:
        """Feature windows of one Features probe of the current offline run that
        end in ``(start, stop]`` (samples): at most ``max_windows`` of them, or
        their mean with ``reduce="mean"``."""
        if self._offline is None:
            raise RuntimeError("no offline run")
        if reduce not in ("none", "mean"):
            raise ValueError("reduce must be 'none' or 'mean'")
        return self._offline.feature_windows(probe, int(start), int(stop),
                                             int(max_windows), reduce)  # fmt: skip

    def shutdown(self) -> str:
        self._stop.set()
        return "bye"

    # -- transport -------------------------------------------------------------------

    def handle(self, raw: bytes) -> dict[str, Any]:
        request_id: Any = None
        try:
            try:
                request = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RpcError(PARSE_ERROR, f"parse error: {exc}") from exc
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                raise RpcError(
                    INVALID_REQUEST, "expected a JSON-RPC 2.0 request object"
                )
            request_id = request.get("id")
            method = self._methods.get(request.get("method", ""))
            if method is None:
                raise RpcError(
                    METHOD_NOT_FOUND, f"unknown method {request.get('method')!r}"
                )
            params = request.get("params", {})
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "params must be an object")
            try:
                result = method(**params)
            except TypeError as exc:
                raise RpcError(INVALID_PARAMS, str(exc)) from exc
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except RpcError as exc:
            error = {"code": exc.code, "message": exc.message}
        except Exception as exc:
            error = {"code": SERVER_ERROR, "message": f"{type(exc).__name__}: {exc}"}
        return {"jsonrpc": "2.0", "id": request_id, "error": error}

    def serve_forever(self, poll_ms: int = 100) -> None:
        try:
            while not self._stop.is_set():
                if not self._socket.poll(poll_ms):
                    continue
                reply = self.handle(self._socket.recv())
                self._socket.send(json.dumps(reply, default=str).encode())
        finally:
            self.close()

    def request_shutdown(self) -> None:
        self._stop.set()

    def warm_up_nodes(self) -> threading.Thread:
        """Let every installed node load its slow dependencies in the background
        (e.g. antropy compiles for ~9 s on import), so the first block of a
        pipeline using it does not stall the real-time thread."""

        def run() -> None:
            for name in self.registry.names():
                warm_up = getattr(self.registry.get(name), "warm_up", None)
                if callable(warm_up):
                    try:
                        warm_up()
                    except Exception as exc:  # used nodes report it when they run
                        print(f"meegxer-daemon: warm-up of {name} failed: {exc}",
                              file=sys.stderr, flush=True)  # fmt: skip

        thread = threading.Thread(target=run, name="warm-up", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.stop()
        if self._graph is not None:
            self._close_processors(self._graph)
        if self._offline is not None:
            self._offline.discard()
        self._socket.close(linger=0)
        self.display.close()
        self._guard.stop()
        remove_token(self.control_endpoint, self._guard.token)
        if self._own_context:
            self._context.destroy(linger=0)


class ControlClient:
    """Synchronous JSON-RPC client; recovers from timeouts (lazy pirate).

    Presents the daemon's token: ``token``, else ``MEEGXER_TOKEN``, else the
    daemon's token file, read again on every reconnect (a restarted daemon has a
    new one)."""

    def __init__(
        self,
        endpoint: str = DEFAULT_CONTROL,
        timeout: float = 5.0,
        *,
        context: ZmqContext | None = None,
        token: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._context = context or zmq.Context.instance()
        self._given_token = token
        self.token: str | None = None
        self._socket = self._connect()
        self._next_id = 1

    def _connect(self) -> zmq.Socket[bytes]:
        self.token = self._given_token or read_token(self.endpoint)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        login(socket, self.token)
        socket.connect(self.endpoint)
        return socket

    def call(self, method: str, **params: Any) -> Any:
        request = {"jsonrpc": "2.0", "method": method, "params": params,
                   "id": self._next_id}  # fmt: skip
        self._next_id += 1
        self._socket.send(json.dumps(request).encode())
        if not self._socket.poll(int(self.timeout * 1000)):
            missing = self.token is None
            self._socket.close(linger=0)
            self._socket = self._connect()
            hint = f" ({missing_token_hint(self.endpoint)})" if missing else ""
            raise TimeoutError(
                f"no reply from {self.endpoint} within {self.timeout} s{hint}"
            )
        reply = json.loads(self._socket.recv())
        if "error" in reply:
            raise RpcError(reply["error"]["code"], reply["error"]["message"])
        return reply["result"]

    def close(self) -> None:
        self._socket.close(linger=0)
