"""Control API: JSON-RPC 2.0 over ZeroMQ REQ/REP (rule 3).

The daemon owns the pipeline and the real-time scheduler; clients (GUI, CLI,
scripts) only talk to it through these methods. See ``spec/control-api.md``.
The control socket binds to localhost by default and has no authentication, so
binding it to other interfaces is only safe on a trusted network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import zmq

from meegxer.core import __version__
from meegxer.core.display import DisplayPublisher, ZmqContext
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION, Manifest
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
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


def manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "version": manifest.version,
        "role": manifest.role,
        "type": manifest.kind,
        "availability": manifest.availability,
        "requires_fit": manifest.requires_fit,
        "description": manifest.description,
        "live_params": sorted(manifest.live_params),
        "inputs": [{"name": p.name, "type": p.type.value} for p in manifest.inputs],
        "outputs": [{"name": p.name, "type": p.type.value} for p in manifest.outputs],
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
        self.registry = registry or Registry.from_entry_points()
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(control_endpoint)
        self.control_endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.display = DisplayPublisher(display_endpoint, context=self._context)
        self.display_endpoint = self.display.endpoint
        self._pipeline: dict[str, Any] | None = None
        self._graph: Graph | None = None
        self._scheduler: RealtimeScheduler | None = None
        self._stop = threading.Event()
        self._methods: dict[str, Callable[..., Any]] = {
            "ping": self.ping,
            "list_nodes": self.list_nodes,
            "load_pipeline": self.load_pipeline,
            "get_pipeline": self.get_pipeline,
            "start": self.start,
            "stop": self.stop,
            "status": self.status,
            "set_params": self.set_params,
            "calibrate": self.calibrate,
            "clear_calibration": self.clear_calibration,
            "set_pass_uncalibrated": self.set_pass_uncalibrated,
            "set_strip_params": self.set_strip_params,
            "shutdown": self.shutdown,
        }

    # -- methods ---------------------------------------------------------------------

    def ping(self) -> dict[str, str]:
        return {"version": __version__, "spec_version": SPEC_VERSION,
                "display": self.display_endpoint}  # fmt: skip

    def list_nodes(self) -> list[dict[str, Any]]:
        return [
            manifest_to_dict(self.registry.get(n).manifest)
            for n in self.registry.names()
        ]

    def load_pipeline(self, pipeline: dict[str, Any]) -> dict[str, Any]:
        if self._scheduler is not None and self._scheduler.running:
            raise RuntimeError("stop the running pipeline before loading another")
        graph = Graph.from_dict(pipeline, self.registry)
        RealtimeScheduler(graph)  # validates: no offline-only nodes
        if self._graph is not None:
            self._close_processors(self._graph)
        self._pipeline, self._graph, self._scheduler = graph.to_dict(), graph, None
        return self._pipeline

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

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.stop()
        if self._graph is not None:
            self._close_processors(self._graph)
        self._socket.close(linger=0)
        self.display.close()


class ControlClient:
    """Synchronous JSON-RPC client; recovers from timeouts (lazy pirate)."""

    def __init__(
        self,
        endpoint: str = DEFAULT_CONTROL,
        timeout: float = 5.0,
        *,
        context: ZmqContext | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._context = context or zmq.Context.instance()
        self._socket = self._connect()
        self._next_id = 1

    def _connect(self) -> zmq.Socket[bytes]:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def call(self, method: str, **params: Any) -> Any:
        request = {"jsonrpc": "2.0", "method": method, "params": params,
                   "id": self._next_id}  # fmt: skip
        self._next_id += 1
        self._socket.send(json.dumps(request).encode())
        if not self._socket.poll(int(self.timeout * 1000)):
            self._socket.close(linger=0)
            self._socket = self._connect()
            raise TimeoutError(f"no reply from {self.endpoint} within {self.timeout} s")
        reply = json.loads(self._socket.recv())
        if "error" in reply:
            raise RpcError(reply["error"]["code"], reply["error"]["message"])
        return reply["result"]

    def close(self) -> None:
        self._socket.close(linger=0)
