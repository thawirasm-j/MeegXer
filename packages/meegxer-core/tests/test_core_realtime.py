import itertools
import time
from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, SignalInfo
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import PacedSource, RealtimeScheduler
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage


def manifest(name: str, **extra: object) -> Manifest:
    data: dict[str, object] = {
        "spec_version": "0.2.0", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [{"name": "k", "type": "float", "default": 1.0},
                   {"name": "mode", "type": "string", "default": "a"}],
        "live_params": ["k"],
    }  # fmt: skip
    data.update(extra)
    return Manifest.from_dict(data)


class Leaky(Processor):
    """Stateful leaky integrator scaled by k (k is a live parameter)."""

    manifest: ClassVar[Manifest] = manifest("leaky")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        y = state["y"] if state is not None else np.zeros(block.info.n_channels)
        assert isinstance(y, np.ndarray)
        out = np.empty_like(block.data)
        for i in range(block.n_samples):
            y = 0.95 * y + 0.05 * block.data[:, i]
            out[:, i] = y
        result = Block(out * float(params["k"]), block.info, block.sample_index,
                       block.timestamp)  # fmt: skip
        return TransformResult(result, {"y": y})


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


class Slow(Leaky):
    manifest: ClassVar[Manifest] = manifest("slow")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        time.sleep(block.n_samples / block.info.sfreq * 3)
        return super().transform(block, model, state, params)


class Broken(Leaky):
    manifest: ClassVar[Manifest] = manifest("broken")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        raise RuntimeError("boom")


def source(duration: float = 1.0) -> SyntheticSource:
    return SyntheticSource({"duration": duration, "n_channels": 4, "sfreq": 200.0})


def run(graph: Graph, timeout: float = 20.0) -> RealtimeScheduler:
    scheduler = RealtimeScheduler(graph)
    scheduler.start()
    assert scheduler.wait(timeout)
    return scheduler


def test_realtime_output_equals_batch_output() -> None:
    """Rule 4: the same node code gives the same result in both schedulers."""
    probe = MemoryProbe()
    scheduler = run(Graph.chain([source(), Stage(Leaky(), {"k": 2.0}), probe]))
    assert scheduler.metrics.errors == []
    batch_probe = MemoryProbe()
    BatchScheduler().run(
        Graph.chain([source(), Stage(Leaky(), {"k": 2.0}), batch_probe])
    )
    live, offline = probe.signal(), batch_probe.signal()
    assert live.n_samples == offline.n_samples == 200
    np.testing.assert_allclose(live.data, offline.data, rtol=1e-12)
    assert len(probe.blocks) > 10  # it really ran block-wise


def test_paced_source_runs_at_real_time_and_restamps() -> None:
    paced = PacedSource(source(0.5))
    started = time.perf_counter()
    blocks = []
    while (block := paced.read(1000)) is not None:
        if block.n_samples:
            blocks.append(block)
        time.sleep(0.005)
    elapsed = time.perf_counter() - started
    assert 0.4 < elapsed < 2.0
    assert sum(b.n_samples for b in blocks) == 100
    assert min(b.n_samples for b in blocks[:-1]) >= 2  # 10 ms chunks at 200 Hz
    assert blocks[1].timestamp == pytest.approx(
        blocks[0].timestamp + blocks[0].n_samples / 200
    )


def test_offline_only_nodes_refused() -> None:
    class Offline(Leaky):
        manifest: ClassVar[Manifest] = manifest("offline", availability="offline")

    with pytest.raises(GraphError, match="offline-only"):
        RealtimeScheduler(Graph.chain([source(), Stage(Offline())]))


def test_live_param_changes() -> None:
    probe = MemoryProbe()
    stage = Stage(Leaky())
    scheduler = RealtimeScheduler(Graph.chain([source(3.0), stage, probe]))
    scheduler.start()
    time.sleep(0.3)
    assert scheduler.set_params("leaky1", {"k": 3.0})["k"] == 3.0
    kept = stage.state
    assert kept is not None  # live parameter: state carried over
    scheduler.set_params("leaky1", {"mode": "b"})
    assert scheduler.submit(
        lambda: stage.state is None or stage.state is not kept
    ).result()
    with pytest.raises(GraphError):
        scheduler.set_params("synthetic1", {"seed": 2})
    scheduler.stop()
    assert scheduler.status()["running"] is False


def test_calibration_gates_downstream_until_fitted() -> None:
    probe = MemoryProbe()
    scheduler = RealtimeScheduler(Graph.chain([source(5.0), Stage(Center()), probe]))
    scheduler.start()
    time.sleep(0.3)
    assert probe.blocks == []

    def center_status() -> str:
        future = scheduler.submit(
            lambda: scheduler.node_status(scheduler.graph["center1"])
        )
        return str(future.result())

    assert center_status() == "uncalibrated"
    scheduler.calibrate("center1", 0.3)
    assert center_status() == "calibrating"
    deadline = time.monotonic() + 5
    while not probe.blocks and time.monotonic() < deadline:
        time.sleep(0.05)
    assert probe.blocks, "data should flow after calibration"
    assert center_status() == "ready"
    with pytest.raises(GraphError):
        scheduler.calibrate("synthetic1", 1.0)
    scheduler.stop()


def test_input_ending_during_calibration_is_reported() -> None:
    """A 60 s calibration started on a 60 s input never gets its 60 s: say so
    rather than stopping without a reason."""
    scheduler = RealtimeScheduler(Graph.chain([source(1.0), Stage(Center()),
                                               MemoryProbe()]))  # fmt: skip
    scheduler.start()
    scheduler.calibrate("center1", 5.0)
    assert scheduler.wait(20)
    errors = scheduler.status()["errors"]
    assert any("calibration of 'center1' did not finish" in e and "of 5 s" in e
               for e in errors), errors  # fmt: skip
    stage = scheduler.graph["center1"].impl
    assert isinstance(stage, Stage) and stage.model is None


class NeedsTen(Center):
    """Fits only on ten channels or more: says so before collecting data."""

    manifest: ClassVar[Manifest] = manifest("needsten", requires_fit=True)

    def check_fit(self, info: SignalInfo, params: Params) -> None:
        if info.n_channels < 10:
            raise ValueError("needs ten channels")


def test_calibration_that_cannot_succeed_fails_at_once() -> None:
    scheduler = RealtimeScheduler(Graph.chain([source(30.0), Stage(NeedsTen()),
                                               MemoryProbe()]))  # fmt: skip
    scheduler.start()
    scheduler.calibrate("needsten1", 20.0)
    deadline = time.monotonic() + 5
    while not scheduler.status()["errors"] and time.monotonic() < deadline:
        time.sleep(0.05)
    errors = scheduler.status()["errors"]
    assert errors == ["calibration of 'needsten1' cannot start: needs ten channels"]
    assert scheduler.status()["running"]  # the pipeline keeps running
    status = scheduler.submit(
        lambda: scheduler.node_status(scheduler.graph["needsten1"])
    ).result()
    assert str(status) == "uncalibrated"  # not stuck "calibrating"
    scheduler.stop()


def test_overruns_are_counted() -> None:
    scheduler = run(Graph.chain([source(0.5), Stage(Slow()), MemoryProbe()]))
    assert scheduler.metrics.overruns > 0
    assert scheduler.status()["overruns"] == scheduler.metrics.overruns


def test_processor_errors_stop_the_run_and_are_reported() -> None:
    scheduler = run(Graph.chain([source(), Stage(Broken()), MemoryProbe()]))
    assert any("boom" in e for e in scheduler.status()["errors"])


def test_status_reports_timing_and_latency() -> None:
    from meegxer.core.sinks import SetSink

    class NullSink(SetSink):
        def close(self) -> None:
            pass

    scheduler = run(
        Graph.chain([source(), Stage(Leaky()), NullSink({"path": "unused"})])
    )
    status = scheduler.status()
    assert status["samples"] == 200 and status["blocks"] > 10
    nodes = {n["id"]: n for n in status["nodes"]}
    assert nodes["leaky1"]["timing"]["blocks"] == status["blocks"]
    assert 0 <= status["latency"]["mean_ms"] < 150


def test_display_frames_cover_the_probe_contiguously() -> None:
    from meegxer.core.display import DisplayPublisher, DisplaySubscriber

    publisher = DisplayPublisher("tcp://127.0.0.1:0")
    subscriber = DisplaySubscriber(publisher.endpoint, ["main"])
    time.sleep(0.2)  # let the subscription reach the publisher
    graph = Graph()
    graph.add(source(1.5))
    graph.add(Stage(Leaky()))
    graph.add(MemoryProbe(), id="main")
    scheduler = RealtimeScheduler(graph, display=publisher, display_bins_per_second=50)
    scheduler.start()
    frames = []
    while (frame := subscriber.recv(timeout=1.0)) is not None:
        frames.append(frame)
    assert scheduler.wait(5)
    assert len(frames) > 10
    assert [f.seq for f in frames] == list(range(len(frames)))
    bins = [(f.envelope.start_bin, f.envelope.n_bins) for f in frames]
    for (start, n), (nxt, _) in itertools.pairwise(bins):
        assert start + n == nxt  # no missing or repeated bins
    assert frames[0].envelope.bin_size == 4  # 200 Hz / 50 bins per second
    assert sum(n for _, n in bins) == 75  # 1.5 s at 50 bins/s
    subscriber.close()
    publisher.close()
