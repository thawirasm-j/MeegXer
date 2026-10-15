"""Feature windows through the core: tuple outputs, both schedulers, sinks, the
graph's default inputs and the analysis variant of acceptance test 1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.conformance import ConformanceError, check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Params, Processor, Sink, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage


def analysis(name: str) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.0", "name": f"test.{name}", "version": "0.1.0",
        "type": "analysis", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "features"}],
        "params": [{"name": "n", "type": "int", "default": 50}],
    })  # fmt: skip


class WindowMean(Processor):
    """Mean of each channel over non-overlapping windows of n samples."""

    manifest: ClassVar[Manifest] = analysis("window_mean")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        n = int(params["n"])
        if state is None:
            state = {"held": np.zeros((block.info.n_channels, 0)),
                     "start": block.sample_index}  # fmt: skip
        held = np.concatenate([np.asarray(state["held"]), block.data], axis=1)
        start = state["start"]
        assert isinstance(start, int)
        out = []
        while held.shape[1] >= n:
            out.append(FeatureBlock(held[:, :n].mean(axis=1), ("channel",),
                                    (start, start + n), float(start),
                                    {"channel": block.info.ch_labels}))  # fmt: skip
            held, start = held[:, n:], start + n
        return TransformResult(tuple(out), {"held": held, "start": start})


class SplitDependent(WindowMean):
    """Wrong on purpose: one window per call, whatever its size."""

    manifest: ClassVar[Manifest] = analysis("split_dependent")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        fb = FeatureBlock(block.data.mean(axis=1), ("channel",),
                          (block.sample_index, block.stop_index), 0.0)  # fmt: skip
        return TransformResult((fb,), {})


class FeatureCollector(Sink):
    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.collect", "version": "0.1.0",
        "type": "stream", "role": "sink",
        "inputs": [{"name": "in", "type": "features"}], "outputs": [], "params": [],
    })  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self.windows: list[FeatureBlock] = []

    def write(self, block: AnyBlock) -> None:
        assert isinstance(block, FeatureBlock)
        self.windows.append(block)


def source(duration: float = 1.0) -> SyntheticSource:
    return SyntheticSource({"duration": duration, "n_channels": 3, "sfreq": 200.0})


def test_features_branch_off_the_chain_by_port_type() -> None:
    probe, sink = MemoryProbe(), FeatureCollector()
    graph = Graph.chain([source(), Stage(WindowMean()), sink, probe])
    inputs = {node.id: node.input for node in graph.nodes}
    assert inputs["collect1"] == "window_mean1"  # features from features
    assert inputs["probe1"] == "synthetic1"  # signal skips the branch


@pytest.mark.parametrize("block_size", [7, 64, None])
def test_batch_writes_every_window(block_size: int | None) -> None:
    sink = FeatureCollector()
    BatchScheduler(block_size=block_size).run(
        Graph.chain([source(), Stage(WindowMean()), sink])
    )
    assert [w.window for w in sink.windows] == [(i, i + 50) for i in range(0, 200, 50)]
    raw = source().read(200)
    assert raw is not None
    np.testing.assert_allclose(sink.windows[1].data, raw.data[:, 50:100].mean(axis=1))


def test_realtime_routes_windows_to_sinks_like_batch() -> None:
    live, offline = FeatureCollector(), FeatureCollector()
    scheduler = RealtimeScheduler(
        Graph.chain([source(), Stage(WindowMean()), live]), max_block=13
    )
    scheduler.start()
    assert scheduler.wait(20.0)
    assert scheduler.metrics.errors == []
    BatchScheduler().run(Graph.chain([source(), Stage(WindowMean()), offline]))
    assert [w.window for w in live.windows] == [w.window for w in offline.windows]
    for a, b in zip(live.windows, offline.windows, strict=True):
        np.testing.assert_allclose(a.data, b.data, rtol=1e-12)


def test_analysis_conformance_accepts_and_rejects() -> None:
    raw = source().read(200)
    assert raw is not None
    assert check_blockwise_equivalence(WindowMean(), raw) == 0.0
    with pytest.raises(ConformanceError, match="windows differ"):
        check_blockwise_equivalence(SplitDependent(), raw)


def test_analysis_stage_without_its_channel_types_outputs_nothing() -> None:
    info = SignalInfo(("EOG1",), 200.0, "s", ch_types=("eog",))
    block = Block(np.ones((1, 100)), info, 0, 0.0)
    result = Stage(WindowMean()).transform(block, "fp")
    assert result.block == ()
