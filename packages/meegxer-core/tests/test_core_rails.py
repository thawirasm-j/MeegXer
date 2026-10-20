"""Rail ports (docs/RAILS.md, spec 0.2.17): extra inputs read rails, extra outputs
write them, cut to each block by sample index in both schedulers (rule 4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest, ManifestError
from meegxer.core.node import Params, Processor, RailInput, TransformResult
from meegxer.core.ports import Port, PortType
from meegxer.core.probes import MemoryProbe
from meegxer.core.rails import RailBuffer, RailError, RailFeed, check_rates
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage

SIGNAL = {"name": "in", "type": "signal"}
OUT = {"name": "out", "type": "signal"}


def manifest(
    name: str, inputs: list[Any], outputs: list[Any], **extra: Any
) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.17", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "inputs": inputs, "outputs": outputs,
        "params": [{"name": "k", "type": "float", "default": 2.0}], **extra,
    })  # fmt: skip


class DelayedWriter(Processor):
    """Passes its input on and writes k x input, 3 samples late, to the 'mixed'
    rail (a delaying writer: its rail lags its input)."""

    manifest: ClassVar[Manifest] = manifest("delayed", [SIGNAL], [
        OUT, {"name": "mix", "type": "signal", "rail": "mixed",
              "space": "component"}])  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        held = state["held"] if state else np.zeros((block.info.n_channels, 0))
        assert isinstance(held, np.ndarray)
        start = int(state["start"]) if state else block.sample_index  # type: ignore[arg-type]
        data = np.hstack([held, block.data * float(params["k"])])
        n = max(0, data.shape[1] - 3)
        rail = Block(data[:, :n], block.info, start, block.timestamp)
        new: State = {"held": data[:, n:], "start": start + n}
        return TransformResult(block, new, (), {"mix": rail})

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        if not state:
            return None
        held = state["held"]
        assert isinstance(held, np.ndarray)
        rail = Block(held, info, int(state["start"]), 0.0)  # type: ignore[arg-type]
        end = rail.stop_index  # the main output ended there already
        empty = Block(np.zeros((info.n_channels, 0)), info, end, 0.0)
        return TransformResult(empty, state, (), {"mix": rail})


class PowerWriter(Processor):
    """Passes its input on and writes one window of mean power per 10 samples to
    the 'power' rail."""

    manifest: ClassVar[Manifest] = manifest("power", [SIGNAL], [
        OUT, {"name": "pow", "type": "features", "rail": "power"}])  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        windows = []
        acc = float(state["acc"]) if state else 0.0  # type: ignore[arg-type]
        for i in range(block.n_samples):
            index = block.sample_index + i
            acc += float(np.mean(block.data[:, i] ** 2))
            if (index + 1) % 10 == 0:
                windows.append(FeatureBlock(np.array([acc / 10]), ("feature",),
                                            (index - 9, index + 1), 0.0))  # fmt: skip
                acc = 0.0
        return TransformResult(block, {"acc": acc}, (), {"pow": tuple(windows)})


READER_INPUTS = [
    SIGNAL, {"name": "mix", "type": "signal", "rail": "mixed"},
    {"name": "pow", "type": "features", "rail": "power"},
    {"name": "cues", "type": "events", "rail": "events"}]  # fmt: skip


class Reader(Processor):
    """Subtracts the 'mixed' rail; emits an event per power window that ended in
    the block and echoes the events it sees."""

    manifest: ClassVar[Manifest] = manifest("reader", READER_INPUTS, [OUT])

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        raise AssertionError("reads rails")

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        mix = rails["mix"]
        assert isinstance(mix, Block)
        assert (mix.sample_index, mix.n_samples) == (
            block.sample_index,
            block.n_samples,
        )
        out = Block(block.data - mix.data, block.info, block.sample_index,
                    block.timestamp)  # fmt: skip
        windows, cues = rails["pow"], rails["cues"]
        assert isinstance(windows, tuple) and isinstance(cues, tuple)
        events = [Event(w.window[1] - 1, "window") for w in windows
                  if isinstance(w, FeatureBlock)]  # fmt: skip
        events += [Event(e.sample_index, f"seen {e.label}") for e in cues
                   if isinstance(e, Event)]  # fmt: skip
        return TransformResult(
            out, {}, tuple(sorted(events, key=lambda e: e.sample_index))
        )


class FitReader(Reader):
    """Fits the mean of the 'mixed' rail over the calibration."""

    manifest: ClassVar[Manifest] = manifest(
        "fitreader", READER_INPUTS, [OUT], requires_fit=True
    )

    def fit_rails(
        self, calibration: Block, params: Params, rails: Mapping[str, RailInput]
    ) -> State:
        mix = rails["mix"]
        assert isinstance(mix, Block)
        assert (mix.sample_index, mix.n_samples) == (
            calibration.sample_index, calibration.n_samples)  # fmt: skip
        return {"mean": float(mix.data.mean())}


def source(seconds: float = 1.0) -> SyntheticSource:
    return SyntheticSource({"duration": seconds, "n_channels": 2, "sfreq": 100.0})


def chain(reader: Processor | None = None, seconds: float = 1.0) -> Graph:
    return Graph.chain([source(seconds), Stage(DelayedWriter()), Stage(PowerWriter()),
                        Stage(reader or Reader()), MemoryProbe()])  # fmt: skip


def test_manifest_rail_ports_are_checked() -> None:
    with pytest.raises(ManifestError, match="must name the rail"):
        manifest("x", [SIGNAL, {"name": "b", "type": "signal"}], [OUT])
    with pytest.raises(ManifestError, match="main port"):
        manifest("x", [{**SIGNAL, "rail": "ics"}], [OUT])
    with pytest.raises(ManifestError, match="Events rail is read by an Events"):
        manifest("x", [SIGNAL], [OUT, {"name": "e", "type": "events",
                                       "rail": "events"}])  # fmt: skip
    with pytest.raises(ManifestError, match="only Signal ports have a space"):
        manifest("x", [SIGNAL], [OUT, {"name": "f", "type": "features",
                                       "rail": "f", "space": "sensor"}])  # fmt: skip
    m = Reader.manifest
    assert [p.rail for p in m.rail_inputs] == ["mixed", "power", "events"]
    assert m.reads_events and DelayedWriter.manifest.rail_outputs[0] == Port(
        "mix", PortType.SIGNAL, "mixed", "component"
    )


def test_graph_binds_rails_to_their_nearest_earlier_writer() -> None:
    with pytest.raises(GraphError, match="reads the 'mixed' rail, which no earlier"):
        Graph.chain([source(), Stage(PowerWriter()), Stage(Reader())])
    graph = chain()
    assert graph.rail_sources("reader1") == {
        "mix": ("delayed1", "mix"),
        "pow": ("power1", "pow"),
        "cues": ("", ""),
    }
    with pytest.raises(GraphError, match="other nodes read from it"):
        graph.remove("delayed1")


def expected(graph: Graph) -> Block:
    src = graph.source.impl
    assert isinstance(src, SyntheticSource)
    return src.read(10_000)  # type: ignore[return-value]


def test_batch_cuts_the_rails_to_each_block() -> None:
    graph = chain()
    whole = expected(chain())
    result = BatchScheduler(block_size=7).run(graph, keep=["reader1"])
    out = result.outputs["reader1"]
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data, -whole.data)  # x - 2x, aligned by index
    windows = [e.sample_index for e in result.events["reader1"]]
    assert windows == [10 * k + 9 for k in range(10)]


def test_realtime_waits_for_the_lagging_rail_and_matches_batch() -> None:
    """Rule 4 with rails: live output equals the batch output, though the 'mixed'
    rail lags its writer's input by 3 samples (wait and align, rule 8)."""
    graph = chain()
    scheduler = RealtimeScheduler(graph, max_block=13)
    scheduler.start()
    assert scheduler.wait(20)
    assert scheduler.status()["errors"] == []
    probe = graph.nodes[-1].impl
    assert isinstance(probe, MemoryProbe)
    live = concat_blocks([b for b in probe.blocks if isinstance(b, Block)])
    batch = BatchScheduler().run(chain(), keep=["reader1"]).outputs["reader1"]
    assert isinstance(batch, Block)
    # live ends 3 samples early: the rail's last samples come only with flush
    assert live.sample_index == 0 and live.n_samples == batch.n_samples - 3
    np.testing.assert_allclose(live.data, batch.data[:, : live.n_samples])


def test_blockwise_equivalence_covers_rails() -> None:
    """Acceptance test 1: the reader with its rails cut per block, and the writers'
    rails, are the same whatever the block sizes."""
    block = expected(chain())
    mix = Block(block.data * 2, block.info, 0, 0.0)
    windows = PowerWriter().transform(block, {}, None, {"k": 2.0}).rails["pow"]
    rails: dict[str, RailInput] = {
        "mix": mix,
        "pow": windows,
        "cues": (Event(5, "a"), Event(55, "b")),
    }
    worst = check_blockwise_equivalence(Reader(), block, rails=rails)
    assert worst < 1e-12
    check_blockwise_equivalence(DelayedWriter(), block)
    check_blockwise_equivalence(PowerWriter(), block)


def test_a_reader_fits_on_its_rails_and_follows_their_writer() -> None:
    graph = chain(FitReader())
    BatchScheduler().run(graph)
    stage = graph["fitreader1"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    whole = expected(chain())
    assert stage.model.data["mean"] == pytest.approx(2 * whole.data.mean())
    before = offline_input_fingerprints(graph)["fitreader1"]
    writer = graph["delayed1"].impl
    assert isinstance(writer, Stage)
    writer.set_params(k=3.0)  # the rail it reads changes
    after = offline_input_fingerprints(graph)["fitreader1"]
    assert after != before and stage.status(after).value == "needs-refit"


def test_feed_waits_for_a_lagging_rail_and_drops_before_a_late_one() -> None:
    info = SignalInfo(("A",), 100.0)
    port = Port("mix", PortType.SIGNAL, "mixed")

    def block(start: int, n: int, value: float = 1.0) -> Block:
        return Block(np.full((1, n), value), info, start, 0.0)

    feed = RailFeed((port,), hold=50)
    feed.push_main(block(0, 10))
    assert feed.ready(lambda a, b: ()) is None  # the rail has nothing yet
    feed.push_rail("mix", block(4, 4, 2.0), 0)  # starts late, at 4
    got = feed.ready(lambda a, b: ())
    assert got is not None
    main, rails = got
    assert (main.sample_index, main.n_samples) == (4, 4)
    mix = rails["mix"]
    assert isinstance(mix, Block) and mix.sample_index == 4
    feed.push_rail("mix", block(8, 20, 2.0), 0)
    got = feed.ready(lambda a, b: ())
    assert got is not None and (got[0].sample_index, got[0].n_samples) == (8, 2)
    feed.push_main(block(10, 100))
    got = feed.ready(lambda a, b: ())  # what the rail covers, up to 28
    assert got is not None and (got[0].sample_index, got[0].n_samples) == (10, 18)
    assert feed.ready(lambda a, b: ()) is None  # the rest waits ...
    feed.push_rail("mix", block(28, 100, 2.0), 0)
    got = feed.ready(lambda a, b: ())  # main and rail each keep `hold` (50)
    assert got is not None and (got[0].sample_index, got[0].n_samples) == (78, 32)


def test_features_buffer_hands_each_window_once() -> None:
    buffer = RailBuffer(PortType.FEATURES)
    windows = tuple(FeatureBlock(np.array([1.0]), ("f",), (k, k + 10), 0.0)
                    for k in range(0, 50, 10))  # fmt: skip
    buffer.add(windows, 50)
    taken = []
    for start, stop in [(0, 15), (15, 30), (30, 50)]:
        got = buffer.take(start, stop)
        assert isinstance(got, tuple)
        taken += [w.window for w in got if isinstance(w, FeatureBlock)]
    assert taken == [w.window for w in windows]


def test_a_node_can_take_a_rail_output_as_its_main_input() -> None:
    """``input: "delayed1:mix"`` (spec 0.2.18): a probe on the rail sees what the
    writer wrote, live as in batch."""
    graph = Graph.chain([source(), Stage(DelayedWriter())])
    probe = MemoryProbe()
    node = graph.add(probe, input="delayed1:mix")
    assert (node.input_node, node.input_port) == ("delayed1", "mix")
    with pytest.raises(GraphError, match="no rail output 'nope'"):
        graph.add(MemoryProbe(), input="delayed1:nope")
    BatchScheduler().run(graph)
    whole = expected(chain())
    batch = concat_blocks([b for b in probe.blocks if isinstance(b, Block)])
    np.testing.assert_allclose(batch.data, 2 * whole.data)  # k x input, all of it
    live_graph = Graph.chain([source(), Stage(DelayedWriter())])
    live_probe = MemoryProbe()
    live_graph.add(live_probe, input="delayed1:mix")
    scheduler = RealtimeScheduler(live_graph, max_block=13)
    scheduler.start()
    assert scheduler.wait(20)
    live = concat_blocks([b for b in live_probe.blocks if isinstance(b, Block)])
    assert live.sample_index == 0
    np.testing.assert_allclose(live.data, batch.data[:, : live.n_samples])


def test_a_signal_rail_at_another_rate_is_refused() -> None:
    rail = Block(np.zeros((1, 10)), SignalInfo(("IC1",), 250.0), 0, 0.0)
    check_rates({"ics": rail, "events": ()}, 250.0)
    with pytest.raises(RailError, match="before the rate changes"):
        check_rates({"ics": rail}, 500.0)
