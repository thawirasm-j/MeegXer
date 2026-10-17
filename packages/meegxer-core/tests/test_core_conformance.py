from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo
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


class WindowMeans(Processor):
    """Mean per channel over each 100-sample window; NaN on odd windows (a
    missing value, as a feature node reports e.g. an absent spectral peak)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.means", "version": "0.1.0",
        "type": "analysis", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "features"}],
    })  # fmt: skip
    nan_when_split = False

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        held = np.zeros((block.info.n_channels, 0)) if state is None else state["x"]
        start = block.sample_index - held.shape[1]  # type: ignore[union-attr]
        x = np.concatenate([held, block.data], axis=1)
        out = []
        while x.shape[1] >= 100:
            values = x[:, :100].mean(axis=1)
            split = self.nan_when_split and start < block.sample_index
            if (start // 100) % 2 or split:
                values = np.full_like(values, np.nan)
            out.append(FeatureBlock(values, ("channel",), (start, start + 100), 0.0,
                                    {"channel": block.info.ch_labels}))  # fmt: skip
            x, start = x[:, 100:], start + 100
        return TransformResult(tuple(out), {"x": x})


def test_feature_nans_in_the_same_place_count_as_equal() -> None:
    assert check_blockwise_equivalence(WindowMeans(), SIGNAL, trials=5) < 1e-12
    broken = WindowMeans()
    broken.nan_when_split = True  # a NaN that depends on block sizes
    with pytest.raises(ConformanceError, match="deviates"):
        check_blockwise_equivalence(broken, SIGNAL, trials=5)
