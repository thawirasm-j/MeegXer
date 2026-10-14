from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler, split_block
from meegxer.core.block import Block, Event
from meegxer.core.cache import DirectoryCache, MemoryCache
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.serialization import State
from meegxer.core.sinks import SetSink
from meegxer.core.sources import FileSource, SyntheticSource
from meegxer.core.stage import Stage


def manifest(name: str, *, requires_fit: bool = False) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.0", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [{"name": "k", "type": "float", "default": 2.0}],
        "requires_fit": requires_fit,
    })  # fmt: skip


class Scale(Processor):
    manifest: ClassVar[Manifest] = manifest("scale")
    calls = 0

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        type(self).calls += 1
        out = Block(block.data * float(params["k"]), block.info, block.sample_index,
                    block.timestamp)  # fmt: skip
        return TransformResult(out, {})


class CumSum(Processor):
    """Stateful: running sum along time; emits an event at every 100th sample."""

    manifest: ClassVar[Manifest] = manifest("cumsum")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        carry = state["carry"] if state is not None else np.zeros(block.info.n_channels)
        assert isinstance(carry, np.ndarray)
        out = np.cumsum(block.data, axis=1) + carry[:, None]
        idx = block.sample_indices()
        events = tuple(Event(int(i), "tick") for i in idx[idx % 100 == 0])
        result = Block(out, block.info, block.sample_index, block.timestamp)
        return TransformResult(result, {"carry": out[:, -1].copy()}, events)


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


def source(duration: float = 2.0) -> SyntheticSource:
    return SyntheticSource({"duration": duration, "n_channels": 3, "sfreq": 250.0})


def raw_signal(duration: float = 2.0) -> Block:
    src = source(duration)
    block = src.read(10**6)
    assert block is not None
    return block


def test_chain_output_matches_manual_computation() -> None:
    graph = Graph.chain([source(), Stage(Scale(), {"k": 3.0}), Stage(CumSum())])
    result = BatchScheduler().run(graph)
    expected = np.cumsum(raw_signal().data * 3.0, axis=1)
    out = result.outputs["cumsum1"]
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data, expected, rtol=1e-12)
    assert [e.sample_index for e in result.events["cumsum1"]] == [0, 100, 200, 300, 400]
    assert result.computed == ["synthetic1", "scale1", "cumsum1"]


@pytest.mark.parametrize("block_size", [1, 7, 128, 500, None])
def test_block_size_does_not_change_output(block_size: int | None) -> None:
    graph = Graph.chain([source(), Stage(CumSum())])
    whole = BatchScheduler().run(graph).outputs["cumsum1"]
    graph = Graph.chain([source(), Stage(CumSum())])
    blocked = BatchScheduler(block_size=block_size).run(graph)
    out = blocked.outputs["cumsum1"]
    assert isinstance(out, Block) and isinstance(whole, Block)
    np.testing.assert_allclose(out.data, whole.data, rtol=1e-12)
    assert len(blocked.events["cumsum1"]) == 5


def test_probes_capture_each_tap_and_intermediates_are_released() -> None:
    r1, main = MemoryProbe(), MemoryProbe()
    graph = Graph.chain([source(), Stage(Scale()), r1, Stage(Scale()), main])
    result = BatchScheduler(block_size=100).run(graph)
    np.testing.assert_allclose(r1.signal().data, raw_signal().data * 2)
    np.testing.assert_allclose(main.signal().data, raw_signal().data * 4)
    assert len(r1.blocks) == 5
    assert set(result.outputs) == {"scale2"}  # scale1 feeds a processor


def test_fit_on_whole_input_or_first_seconds() -> None:
    graph = Graph.chain([source(), Stage(Center())])
    result = BatchScheduler().run(graph)
    out = result.outputs["center1"]
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data.mean(axis=1), 0.0, atol=1e-18)
    assert result.fitted == ["center1"]

    graph = Graph.chain([source(), Stage(Center())])
    first = BatchScheduler(calibration_seconds=0.5).run(graph).outputs["center1"]
    assert isinstance(first, Block)
    np.testing.assert_allclose(first.data[:, :125].mean(axis=1), 0.0, atol=1e-18)


def test_cache_recomputes_only_from_changed_stage() -> None:
    cache = MemoryCache()
    stages = [Stage(Scale()), Stage(CumSum()), Stage(Scale())]
    graph = Graph.chain([source(), *stages])
    first = BatchScheduler(cache=cache).run(graph)
    assert first.cached == []

    second = BatchScheduler(cache=cache, block_size=64).run(graph)
    assert second.cached == ["scale1", "cumsum1", "scale2"]
    a, b = first.outputs["scale2"], second.outputs["scale2"]
    assert isinstance(a, Block) and isinstance(b, Block)
    np.testing.assert_array_equal(a.data, b.data)
    assert second.events["cumsum1"] == first.events["cumsum1"]

    stages[1].set_params(k=5.0)  # change the middle stage only
    third = BatchScheduler(cache=cache).run(graph)
    assert third.cached == ["scale1"]
    assert third.computed == ["synthetic1", "cumsum1", "scale2"]


def test_cache_follows_source_content(tmp_path: Path) -> None:
    path = tmp_path / "rec.set"
    cache = DirectoryCache(tmp_path / "cache")

    def write(seed: int) -> None:
        sink = SetSink({"path": str(path), "overwrite": True})
        block = SyntheticSource({"duration": 1.0, "seed": seed}).read(10**6)
        assert block is not None
        sink.write(block)
        sink.close()

    def run() -> list[str]:
        graph = Graph.chain([FileSource({"path": str(path)}), Stage(Scale())])
        return BatchScheduler(cache=cache).run(graph).cached

    write(0)
    assert run() == []
    assert run() == ["scale1"]
    write(1)  # same path, new content
    assert run() == []


def test_directory_cache_round_trip(tmp_path: Path) -> None:
    from meegxer.core.cache import CachedOutput

    cache = DirectoryCache(tmp_path)
    block = raw_signal()
    events = (Event(3, "x", payload={"a": 1}),)
    cache.put("ab12", CachedOutput(block, events))
    hit = cache.get("ab12")
    assert hit is not None and hit.events == events
    np.testing.assert_array_equal(hit.block.data, block.data)
    assert hit.block.info == block.info
    assert cache.get("cd34") is None
    with pytest.raises(ValueError):
        cache.get("../escape")


def test_sinks_are_written(tmp_path: Path) -> None:
    out = tmp_path / "out.set"
    graph = Graph.chain([source(), Stage(Scale()), SetSink({"path": str(out)})])
    BatchScheduler(block_size=77).run(graph)
    loaded = FileSource({"path": str(out)}).read(10**6)
    assert loaded is not None
    np.testing.assert_allclose(loaded.data, raw_signal().data * 2, atol=1e-11)


def test_infinite_source_refused() -> None:
    graph = Graph.chain([SyntheticSource({"duration": 0.0}), Stage(Scale())])
    with pytest.raises(GraphError, match="finite"):
        BatchScheduler().run(graph)


def test_split_block() -> None:
    block = raw_signal()
    parts = split_block(block, 200)
    assert [p.n_samples for p in parts] == [200, 200, 100]
    assert split_block(block, None) == [block]


class WholeOnly(Scale):
    """Offline-only: records how many blocks it was given."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.whole", "version": "0.1.0",
        "type": "stream", "availability": "offline",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [{"name": "k", "type": "float", "default": 1.0}],
    })  # fmt: skip
    sizes: ClassVar[list[int]] = []

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        type(self).sizes.append(block.n_samples)
        return super().transform(block, model, state, params)


class LiveOnly(Scale):
    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.live", "version": "0.1.0",
        "type": "stream", "availability": "online",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip


def test_offline_nodes_get_the_whole_recording() -> None:
    WholeOnly.sizes.clear()
    probe = MemoryProbe()
    graph = Graph.chain([source(), Stage(WholeOnly()), Stage(Scale()), probe])
    BatchScheduler(block_size=50).run(graph)
    assert WholeOnly.sizes == [500]
    np.testing.assert_allclose(probe.signal().data, raw_signal().data * 2)


def test_online_only_nodes_are_refused_in_batch() -> None:
    graph = Graph.chain([source(), Stage(LiveOnly())])
    with pytest.raises(GraphError, match="online-only"):
        BatchScheduler().run(graph)
