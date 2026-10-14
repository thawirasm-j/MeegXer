from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.conformance import (
    ConformanceError,
    check_blockwise_equivalence,
    random_block_sizes,
)
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State

MANIFEST = Manifest.from_dict({
    "spec_version": "0.2.0", "name": "test.ema", "version": "0.1.0", "type": "stream",
    "inputs": [{"name": "in", "type": "signal"}],
    "outputs": [{"name": "out", "type": "signal"}],
})  # fmt: skip

SIGNAL = Block(
    np.random.default_rng(1).standard_normal((3, 600)),
    SignalInfo(("a", "b", "c"), 100.0),
    sample_index=1000,
    timestamp=0.0,
)


class Ema(Processor):
    manifest: ClassVar[Manifest] = MANIFEST
    carry_state = True

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        y = state["y"] if state is not None and self.carry_state else None
        y = np.zeros(block.info.n_channels) if y is None else y
        assert isinstance(y, np.ndarray)
        out = np.empty_like(block.data)
        for i in range(block.n_samples):
            y = 0.9 * y + 0.1 * block.data[:, i]
            out[:, i] = y
        events = tuple(
            Event(int(block.sample_index + i), "peak")
            for i in np.flatnonzero(out[0] > 0.4)
        )
        return TransformResult(
            Block(out, block.info, block.sample_index, block.timestamp),
            {"y": y},
            events,
        )


class ForgetfulEma(Ema):
    carry_state = False  # classic block-boundary bug


class DropsSample(Ema):
    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        result = super().transform(block, model, state, params)
        assert isinstance(result.block, Block)
        n = max(result.block.n_samples - 1, 0)
        cut = result.block.slice(
            result.block.sample_index, result.block.sample_index + n
        )
        return TransformResult(cut, result.state, ())


def test_correct_stateful_processor_passes() -> None:
    worst = check_blockwise_equivalence(Ema(), SIGNAL, trials=10)
    assert worst < 1e-12


def test_block_boundary_bug_is_caught() -> None:
    with pytest.raises(ConformanceError, match="deviates"):
        check_blockwise_equivalence(ForgetfulEma(), SIGNAL)


def test_wrong_length_is_caught() -> None:
    with pytest.raises(ConformanceError):
        check_blockwise_equivalence(DropsSample(), SIGNAL)


def test_random_block_sizes() -> None:
    rng = np.random.default_rng(0)
    sizes = random_block_sizes(1000, rng, 50)
    assert sum(sizes) == 1000 and min(sizes) >= 1 and max(sizes) <= 50
    assert 1 in sizes


def test_offline_processors_are_out_of_scope() -> None:
    class Offline(Ema):
        manifest: ClassVar[Manifest] = Manifest.from_dict({
            "spec_version": "0.2.0", "name": "test.offline", "version": "0.1.0",
            "type": "stream", "availability": "offline",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
        })  # fmt: skip

    with pytest.raises(ValueError, match="offline-only"):
        check_blockwise_equivalence(Offline(), SIGNAL)
