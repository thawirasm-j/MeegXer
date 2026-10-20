"""Resampling: exact rational ratios, the same samples as scipy's polyphase
resampler whatever the blocks, and events that keep their time (rule 8)."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from scipy import signal

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, SignalInfo, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.registry import Registry
from meegxer.core.serialization import State
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.nodes.resample import Resample, ratio

RATES = [(1000.0, 250.0), (500.0, 256.0), (250.0, 500.0)]


def block(sfreq: float, n: int = 1200, start: int = 0) -> Block:
    data = np.random.default_rng(1).standard_normal((3, n))
    return Block(data, SignalInfo(("Fz", "Cz", "Pz"), sfreq), start, 0.0)


def one_pass(b: Block, sfreq: float) -> Block:
    stage = Stage(Resample(), {"sfreq": sfreq})
    fp = signal_fingerprint(b.info)
    head = stage.transform(b, fp).block
    tail = stage.flush(fp)
    parts = [head] + ([tail.block] if tail is not None else [])
    blocks = [p for p in parts if isinstance(p, Block) and p.n_samples]
    return concat_blocks(blocks)


@pytest.mark.parametrize(("sfreq_in", "sfreq_out"), RATES)
def test_matches_scipy_resample_poly(sfreq_in: float, sfreq_out: float) -> None:
    b = block(sfreq_in)
    out = one_pass(b, sfreq_out)
    up, down = ratio(sfreq_in, sfreq_out)
    expected = signal.resample_poly(b.data, up, down, axis=1)
    assert out.info.sfreq == sfreq_out
    assert out.n_samples == expected.shape[1]
    np.testing.assert_allclose(out.data, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(("sfreq_in", "sfreq_out"), RATES)
@pytest.mark.parametrize("start", [0, 37])
def test_blockwise_equivalence(sfreq_in: float, sfreq_out: float, start: int) -> None:
    b = block(sfreq_in, start=start)
    check_blockwise_equivalence(Resample(), b, {"sfreq": sfreq_out}, atol=0.0)


def test_output_index_counts_the_same_time() -> None:
    out = one_pass(block(1000.0, start=400), 250.0)
    assert out.sample_index == 100  # 400 / 1000 Hz = 100 / 250 Hz
    assert out.info.filters[-1] == ("lowpass", 0.0, 125.0)


def test_same_rate_is_the_identity() -> None:
    b = block(250.0)
    out = Stage(Resample(), {"sfreq": 250.0}).transform(b, signal_fingerprint(b.info))
    assert out.block is b or np.array_equal(out.block.data, b.data)  # type: ignore[union-attr]


def test_unwieldy_ratio_is_refused() -> None:
    with pytest.raises(ValueError, match="simpler ratio"):
        ratio(1000.0, 999.7)


def test_a_rate_change_must_cover_every_channel() -> None:
    info = SignalInfo(("Fz", "VEOG"), 500.0, "t", (), ("eeg", "eog"))
    b = Block(np.zeros((2, 100)), info, 0, 0.0)
    stage = Stage(Resample(), {"sfreq": 250.0}, channel_types=["eeg"])
    with pytest.raises(ValueError, match="changes the rate"):
        stage.transform(b, signal_fingerprint(info))


def _manifest(name: str, **extra: object) -> Manifest:
    data: dict[str, object] = {
        "spec_version": "0.2.16", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}], "params": [],
    }  # fmt: skip
    data.update(extra)
    return Manifest.from_dict(data)


class Labelled(Processor):
    """Remembers the onsets its fit was given."""

    manifest: ClassVar[Manifest] = _manifest(
        "labelled", requires_fit=True, fit_events=True
    )

    def fit(self, calibration: Block, params: Params) -> State:
        raise AssertionError("fits with events")

    def fit_labelled(self, calibration: Block, events: object, params: Params) -> State:
        assert isinstance(events, tuple)
        return {"onsets": np.array([e.sample_index for e in events], dtype=float)}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block, {})


class Detector(Processor):
    """Emits a 'beat' at every 100th sample of its own clock."""

    manifest: ClassVar[Manifest] = _manifest("detector")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        first = -(-block.sample_index // 100) * 100
        beats = tuple(Event(i, "beat") for i in range(first, block.stop_index, 100))
        return TransformResult(block, {}, beats)


def onsets(graph: Graph, node_id: str) -> list[int]:
    stage = graph[node_id].impl
    assert isinstance(stage, Stage) and stage.model is not None
    return sorted(np.asarray(stage.model.data["onsets"]).astype(int).tolist())


def test_events_keep_their_time_across_a_rate_change(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n0.5\t0\tcue\n2.5\t0\tcue\n")
    source: dict[str, Any] = {
        "id": "src", "node": "meegxer.core.synthetic",
        "params": {"duration": 3.0, "n_channels": 2, "sfreq": 100.0},
        "recording": {"events": {"file": str(events)}},
    }  # fmt: skip
    graph = Graph.from_dict({"spec_version": "0.2.16", "nodes": [source]},
                            Registry.from_entry_points())  # fmt: skip
    graph.add(Stage(Detector()), id="before")  # beats at 0, 1, 2 s (100 Hz)
    graph.add(Stage(Resample(), {"sfreq": 50.0}), id="rs")
    graph.add(Stage(Labelled()), id="fit50")
    graph.add(Stage(Detector()), id="after")  # beats at 0, 2 s (50 Hz)
    graph.add(Stage(Resample(), {"sfreq": 200.0}), id="up")
    graph.add(Stage(Labelled()), id="fit200")
    graph.add(MemoryProbe({}), id="tap")
    BatchScheduler(block_size=64).run(graph)
    # the cues at 0.5 s and 2.5 s and the beats, each at its time on each clock
    assert onsets(graph, "fit50") == [0, 25, 50, 100, 125]
    assert onsets(graph, "fit200") == [0, 0, 100, 200, 400, 400, 500]
    probe = graph["tap"].impl
    assert isinstance(probe, MemoryProbe)
    assert probe.signal().n_samples == 600  # 3 s at 200 Hz
