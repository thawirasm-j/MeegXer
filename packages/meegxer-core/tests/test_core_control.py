import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import zmq

from meegxer.core.control import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_ERROR,
    ControlClient,
    Daemon,
    RpcError,
)

PIPELINE: dict[str, Any] = {
    "spec_version": "0.2.0",
    "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 0, "n_channels": 4, "sfreq": 200}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "main", "node": "meegxer.core.probe"},
    ],
}  # fmt: skip


@pytest.fixture
def daemon() -> Iterator[tuple[Daemon, ControlClient]]:
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    thread = threading.Thread(target=d.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(d.control_endpoint, timeout=10)
    yield d, client
    try:
        client.call("shutdown")
    finally:
        client.close()
        thread.join(5)


def test_ping_and_node_catalogue(daemon: tuple[Daemon, ControlClient]) -> None:
    d, client = daemon
    assert client.call("ping")["display"] == d.display_endpoint
    nodes = {n["name"]: n for n in client.call("list_nodes")}
    iir = nodes["meegxer.nodes.iir_filter"]
    assert iir["availability"] == "both"
    order = next(p for p in iir["params"] if p["name"] == "order")
    assert order["advanced"] is True and order["max"] == 10
    assert nodes["meegxer.nodes.iir_filter_zerophase"]["availability"] == "offline"
    assert nodes["meegxer.core.lsl_in"]["availability"] == "online"


def test_lifecycle_and_live_params(daemon: tuple[Daemon, ControlClient]) -> None:
    _, client = daemon
    with pytest.raises(RpcError, match="no pipeline"):
        client.call("start")
    loaded = client.call("load_pipeline", pipeline=PIPELINE)
    assert [n["id"] for n in loaded["nodes"]] == ["src", "hp", "main"]
    assert client.call("status")["running"] is False
    # parameters can be set before starting too
    assert client.call("set_params", node="hp", params={"low": 0.5})["low"] == 0.5

    client.call("start")
    with pytest.raises(RpcError, match="already running"):
        client.call("start")
    with pytest.raises(RpcError, match="stop the running"):
        client.call("load_pipeline", pipeline=PIPELINE)
    updated = client.call("set_params", node="hp", params={"low": 2.0})
    assert updated["low"] == 2.0
    with pytest.raises(RpcError):
        client.call("set_params", node="hp", params={"low": -1})
    with pytest.raises(RpcError, match="no node"):
        client.call("set_params", node="nope", params={})

    status = client.call("status")
    assert status["running"] is True
    assert {n["id"]: n["params"] for n in status["nodes"]}["hp"]["low"] == 2.0
    client.call("stop")
    assert client.call("status")["running"] is False
    assert client.call("get_pipeline")["nodes"][1]["params"]["low"] == 2.0
    client.call("start")  # restart rebuilds sources and sinks, keeps params
    client.call("stop")


def test_offline_pipeline_rejected(daemon: tuple[Daemon, ControlClient]) -> None:
    _, client = daemon
    offline = json.loads(json.dumps(PIPELINE))
    offline["nodes"][1]["node"] = "meegxer.nodes.iir_filter_zerophase"
    with pytest.raises(RpcError, match="offline-only") as info:
        client.call("load_pipeline", pipeline=offline)
    assert info.value.code == SERVER_ERROR


def test_protocol_errors(daemon: tuple[Daemon, ControlClient]) -> None:
    d, client = daemon
    with pytest.raises(RpcError) as unknown:
        client.call("explode")
    assert unknown.value.code == METHOD_NOT_FOUND
    with pytest.raises(RpcError) as bad:
        client.call("calibrate", nonsense=1)
    assert bad.value.code == INVALID_PARAMS
    with pytest.raises(RpcError, match="start the pipeline"):
        client.call("calibrate", node="hp", seconds=1.0)

    raw: zmq.Socket[bytes] = zmq.Context.instance().socket(zmq.REQ)
    raw.connect(d.control_endpoint)
    raw.send(b"{not json")
    assert raw.poll(5000)
    assert json.loads(raw.recv())["error"]["code"] == PARSE_ERROR
    raw.close(linger=0)


def test_client_timeout_recovers() -> None:
    client = ControlClient("tcp://127.0.0.1:1", timeout=0.2)
    with pytest.raises(TimeoutError):
        client.call("ping")
    with pytest.raises(TimeoutError):  # socket was rebuilt, still usable
        client.call("ping")
    client.close()


def test_warm_up_preloads_every_node_and_survives_failures() -> None:
    """The daemon preloads slow node dependencies off the real-time thread
    (antropy compiles for ~9 s on import); one failing node does not stop the rest."""
    from meegxer.core.manifest import Manifest
    from meegxer.core.node import Processor, TransformResult
    from meegxer.core.registry import Registry
    from meegxer.core.sources import SyntheticSource

    warmed: list[str] = []

    def node(name: str, fails: bool = False) -> type[Processor]:
        class Node(Processor):
            manifest = Manifest.from_dict({
                "spec_version": "0.2.0", "name": name, "version": "0.1.0",
                "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
                "outputs": [{"name": "out", "type": "signal"}], "params": [],
            })  # fmt: skip

            @classmethod
            def warm_up(cls) -> None:
                warmed.append(name)
                if fails:
                    raise RuntimeError("missing library")

            def transform(self, block, model, state, params):  # type: ignore[no-untyped-def]
                return TransformResult(block, {})

        return Node

    registry = Registry([SyntheticSource, node("t.a", fails=True), node("t.b")])
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0", registry=registry)
    try:
        d.warm_up_nodes().join(10)
        assert warmed == ["t.a", "t.b"]
    finally:
        d.close()


def test_reloading_the_pipeline_keeps_fitted_models() -> None:
    """The GUI's Start (and its restart after an edit) loads the pipeline again;
    calibrated models must survive that like a daemon restart (fingerprints
    still mark them needs-refit if the input changed, rule 7)."""
    import time

    import numpy as np

    from meegxer.core.block import Block
    from meegxer.core.manifest import SPEC_VERSION, Manifest
    from meegxer.core.node import Params, Processor, TransformResult
    from meegxer.core.registry import Registry
    from meegxer.core.serialization import State
    from meegxer.core.stage import Stage

    class Mean(Processor):
        manifest = Manifest.from_dict({
            "spec_version": SPEC_VERSION, "name": "test.mean", "version": "0.1.0",
            "type": "stream", "requires_fit": True,
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
        })  # fmt: skip

        def fit(self, calibration: Block, params: Params) -> State:
            return {"mean": calibration.data.mean(axis=1)}

        def transform(
            self, block: Block, model: State, state: State | None, params: Params
        ) -> TransformResult:
            return TransformResult(block, {})

    registry = Registry.from_entry_points()
    registry.add(Mean)
    pipeline = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"n_channels": 3, "sfreq": 100.0}},
        {"id": "m", "node": "test.mean"},
    ]}  # fmt: skip
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0", registry=registry)
    try:
        d.load_pipeline(pipeline)
        d.start()
        d.calibrate("m", 0.3)

        def model() -> object:
            assert d._graph is not None
            stage = d._graph["m"].impl
            assert isinstance(stage, Stage)
            return stage.model

        for _ in range(100):
            if model() is not None:
                break
            time.sleep(0.05)
        fitted = model()
        assert fitted is not None
        d.stop()
        d.load_pipeline(pipeline)  # what the GUI's Start does
        assert model() is fitted
        assert np.all(np.isfinite(fitted.data["mean"]))  # type: ignore[attr-defined]
    finally:
        d.stop()
        d.close()
