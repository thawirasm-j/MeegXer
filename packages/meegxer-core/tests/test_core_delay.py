"""Delaying processors (look-ahead, e.g. ASR): output lags input by sample index,
the rest comes in flush, and untouched channels stay aligned (rule 8)."""

from typing import ClassVar

import numpy as np

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, SignalInfo, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION, Manifest
from meegxer.core.node import Params, Processor, Source, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.serialization import State
from meegxer.core.stage import Stage, signal_fingerprint

DELAY = 7


def _int(value: object) -> int:
    assert isinstance(value, int | np.integer)
    return int(value)


class Delay(Processor):
    """Outputs its input DELAY samples late (an identity with look-ahead)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": SPEC_VERSION, "name": "test.delay", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        held = (np.empty((block.info.n_channels, 0)) if state is None
                else np.asarray(state["held"]))  # fmt: skip
        first = block.sample_index if state is None else _int(state["first"])
        data = np.hstack([held, block.data])
        out_n = max(0, data.shape[1] - DELAY)
        out = Block(data[:, :out_n], block.info, first, block.timestamp)
        return TransformResult(out, {"held": data[:, out_n:], "first": first + out_n})

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        if state is None:
            return None
        held = np.asarray(state["held"])
        return TransformResult(Block(held, info, _int(state["first"]), 0.0), state)


def recording(n: int = 500) -> Block:
    rng = np.random.default_rng(1)
    info = SignalInfo(("Fz", "VEOG", "Cz"), 100.0, "t", (), ("eeg", "eog", "eeg"))
    return Block(rng.standard_normal((3, n)), info, 0, 0.0)


def test_delayed_output_stays_aligned_with_untouched_channels() -> None:
    rec = recording()
    stage = Stage(Delay())
    fp = signal_fingerprint(rec.info)
    parts = []
    for start in range(0, 500, 33):
        out = stage.transform(rec.slice(start, min(start + 33, 500)), fp).block
        assert isinstance(out, Block)
        assert out.info.ch_labels == rec.info.ch_labels
        parts.append(out)
    tail = stage.flush(fp)
    assert tail is not None and isinstance(tail.block, Block)
    whole = concat_blocks([*parts, tail.block])
    assert (whole.sample_index, whole.n_samples) == (0, 500)
    np.testing.assert_array_equal(whole.data, rec.data)  # identity, aligned


def test_delaying_node_passes_blockwise_equivalence() -> None:
    worst = check_blockwise_equivalence(Delay(), recording(), trials=4, atol=0.0)
    assert worst == 0.0


class Fixed(Source):
    manifest = Manifest.from_dict({
        "spec_version": SPEC_VERSION, "name": "test.fixed", "version": "0.1.0",
        "type": "stream", "role": "source", "inputs": [],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip

    def __init__(self) -> None:
        super().__init__({})
        self._pos = 0
        self.rec = recording()

    @property
    def info(self) -> SignalInfo:
        return self.rec.info

    @property
    def n_samples(self) -> int:
        return self.rec.n_samples

    def read(self, max_samples: int) -> Block | None:
        if self._pos >= self.rec.n_samples:
            return None
        stop = min(self._pos + max_samples, self.rec.n_samples)
        out = self.rec.slice(self._pos, stop)
        self._pos = stop
        return out


def test_batch_runs_flush_delaying_nodes() -> None:
    probe = MemoryProbe({"keep_seconds": 0.0})
    src = Fixed()
    graph = Graph.chain([src, Stage(Delay()), probe])
    BatchScheduler(block_size=40).run(graph)
    out = probe.signal()
    assert (out.sample_index, out.n_samples) == (0, 500)
    np.testing.assert_array_equal(out.data, src.rec.data)
