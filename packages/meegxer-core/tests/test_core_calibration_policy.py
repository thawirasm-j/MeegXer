"""Calibration defaults: models are kept until dropped explicitly; an uncalibrated
node blocks unless it is set (per node) to pass data through, marked as bypassed."""

import threading
import time
from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo, concat_blocks
from meegxer.core.control import ControlClient, Daemon, RpcError
from meegxer.core.display import Decimator, DisplayFrame, decode_frame, encode_frame
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, signal_fingerprint


def manifest(name: str, requires_fit: bool) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.0", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "requires_fit": requires_fit,
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip


class Center(Processor):
    manifest: ClassVar[Manifest] = manifest("center", requires_fit=True)

    def fit(self, calibration: Block, params: Params) -> State:
        return {"mean": calibration.data.mean(axis=1)}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        mean = model["mean"]
        assert isinstance(mean, np.ndarray)
        out = Block(block.data - mean[:, None], block.info, block.sample_index,
                    block.timestamp)  # fmt: skip
        return TransformResult(out, {})


class Double(Processor):
    manifest: ClassVar[Manifest] = manifest("double", requires_fit=False)

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        out = Block(block.data * 2, block.info, block.sample_index, block.timestamp)
        return TransformResult(out, {})


INFO = SignalInfo(("a", "b"), 100.0)


def test_bypass_mark_survives_slicing_and_downstream_processors() -> None:
    block = Block(np.ones((2, 10)), INFO, 0, 0.0, bypassed=("ica",))
    assert block.slice(2, 5).bypassed == ("ica",)
    assert block.pick(["b"]).bypassed == ("ica",)
    joined = concat_blocks([block.slice(0, 5), Block(np.ones((2, 5)), INFO, 5, 0.05)])
    assert joined.bypassed == ("ica",)  # any bypassed part marks the whole
    out = Stage(Double()).transform(block, signal_fingerprint(INFO)).block
    assert isinstance(out, Block) and out.bypassed == ("ica",)


def test_pass_uncalibrated_only_for_nodes_that_need_fitting() -> None:
    with pytest.raises(ValueError, match="does not need calibration"):
        Stage(Double(), pass_uncalibrated=True)
    stage = Stage(Center())
    assert stage.pass_uncalibrated is False  # default: block
    stage.set_pass_uncalibrated(True)
    stage.fit(Block(np.ones((2, 10)), INFO, 0, 0.0), "fp")
    stage.clear_model()
    assert stage.model is None


def test_pipeline_file_round_trips_the_flag() -> None:
    registry = Registry([SyntheticSource, Center, Double, MemoryProbe])
    data = {"spec_version": "0.2.1", "nodes": [
        {"node": "meegxer.core.synthetic", "params": {"duration": 1}},
        {"id": "c", "node": "test.center", "pass_uncalibrated": True},
        {"id": "d", "node": "test.double"},
    ]}  # fmt: skip
    graph = Graph.from_dict(data, registry)
    stage = graph["c"].impl
    assert isinstance(stage, Stage) and stage.pass_uncalibrated
    dumped = graph.to_dict()
    assert dumped["nodes"][1]["pass_uncalibrated"] is True
    assert "pass_uncalibrated" not in dumped["nodes"][2]
    not_fitted = {"node": "test.double", "pass_uncalibrated": True}
    bad = {"spec_version": "0.2.1", "nodes": [data["nodes"][0], not_fitted]}
    with pytest.raises(GraphError, match="does not need calibration"):
        Graph.from_dict(bad, registry)


def wait_for(predicate: object, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def live(pass_uncalibrated: bool) -> tuple[RealtimeScheduler, MemoryProbe]:
    probe = MemoryProbe()
    graph = Graph()
    graph.add(SyntheticSource({"duration": 6.0, "n_channels": 2, "sfreq": 200.0}))
    graph.add(Stage(Center(), pass_uncalibrated=pass_uncalibrated), id="center")
    graph.add(Stage(Double()), id="double")
    graph.add(probe, id="probe")
    return RealtimeScheduler(graph), probe


def test_default_blocks_until_calibrated() -> None:
    scheduler, probe = live(pass_uncalibrated=False)
    scheduler.start()
    time.sleep(0.3)
    assert probe.blocks == []
    scheduler.calibrate("center", 0.2)
    wait_for(lambda: probe.blocks)
    assert all(not b.bypassed for b in probe.blocks)  # type: ignore[union-attr]
    scheduler.stop()


def test_opt_in_passes_through_marked_then_processes_after_calibration() -> None:
    scheduler, probe = live(pass_uncalibrated=True)
    scheduler.start()
    wait_for(lambda: len(probe.blocks) > 3)
    early = list(probe.blocks)
    assert all(b.bypassed == ("center",) for b in early)  # type: ignore[union-attr]
    scheduler.calibrate("center", 0.2)
    wait_for(lambda: any(not b.bypassed for b in probe.blocks))  # type: ignore[union-attr]
    statuses = {
        n["id"]: n for n in scheduler.submit(scheduler.status).result()["nodes"]
    }
    assert statuses["center"]["pass_uncalibrated"] is True

    assert scheduler.clear_calibration("center") == ["center"]
    count = len(probe.blocks)
    wait_for(lambda: len(probe.blocks) > count + 3)
    assert probe.blocks[-1].bypassed == ("center",)  # type: ignore[union-attr]
    scheduler.set_pass_uncalibrated("center", False)
    time.sleep(0.1)
    count = len(probe.blocks)
    time.sleep(0.3)
    assert len(probe.blocks) == count  # now blocking
    with pytest.raises(GraphError):
        scheduler.clear_calibration("double")
    scheduler.stop()


def test_display_frames_carry_the_bypass_mark() -> None:
    env = Decimator(2).feed(Block(np.ones((2, 4)), INFO, 0, 0.0))
    assert env is not None
    frame = DisplayFrame("p", 0, ("a", "b"), 100.0, env, ("center",))
    assert decode_frame(encode_frame(frame)).bypassed == ("center",)


@pytest.fixture
def daemon() -> Iterator[ControlClient]:
    registry = Registry([SyntheticSource, Center, Double, MemoryProbe])
    d = Daemon("tcp://127.0.0.1:0", "tcp://127.0.0.1:0", registry=registry)
    thread = threading.Thread(target=d.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(d.control_endpoint, timeout=10)
    yield client
    client.call("shutdown")
    client.close()
    thread.join(5)


def node(status: dict[str, object], node_id: str) -> dict[str, object]:
    nodes = status["nodes"]
    assert isinstance(nodes, list)
    return next(n for n in nodes if n["id"] == node_id)


def test_daemon_keeps_models_across_restart_until_cleared(
    daemon: ControlClient,
) -> None:
    daemon.call("load_pipeline", pipeline={"spec_version": "0.2.1", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 0, "n_channels": 2, "sfreq": 200}},
        {"id": "center", "node": "test.center"},
        {"id": "probe", "node": "meegxer.core.probe"},
    ]})  # fmt: skip
    daemon.call("start")
    daemon.call("calibrate", node="center", seconds=0.2)
    wait_for(lambda: node(daemon.call("status"), "center")["status"] == "ready")
    daemon.call("stop")
    assert node(daemon.call("status"), "center")["status"] in ("ready", "stopped")

    daemon.call("start")  # restart: the fitted model is kept
    wait_for(lambda: node(daemon.call("status"), "center")["status"] == "ready", 3)
    daemon.call("stop")

    cleared = daemon.call("clear_calibration")
    assert cleared["cleared"] == ["center"]
    daemon.call("start")
    wait_for(
        lambda: node(daemon.call("status"), "center")["status"] == "uncalibrated", 3
    )
    daemon.call("set_pass_uncalibrated", node="center", enabled=True)
    assert node(daemon.call("status"), "center")["pass_uncalibrated"] is True
    daemon.call("stop")
    pipeline = daemon.call("get_pipeline")
    assert pipeline["nodes"][1]["pass_uncalibrated"] is True
    with pytest.raises(RpcError, match="no node"):
        daemon.call("clear_calibration", node="nope")
