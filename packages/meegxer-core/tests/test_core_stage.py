"""Stage status and the fit/transform contract, with small example processors."""

import itertools
from typing import ClassVar

import numpy as np
import pytest

from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo, concat_blocks
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.ports import PortTypeError
from meegxer.core.serialization import State, state_from_bytes, state_to_bytes
from meegxer.core.stage import (
    Stage,
    StageStatus,
    StaleModelError,
    chain_input_fingerprints,
    signal_fingerprint,
)

INFO = SignalInfo(("Fz", "Cz"), sfreq=100.0, stream_id="amp")


def manifest(
    name: str,
    *,
    requires_fit: bool = False,
    params: list[dict[str, object]] | None = None,
    output: str = "signal",
) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": f"test.{name}",
            "version": "0.1.0",
            "type": "stream" if output == "signal" else "analysis",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": output}],
            "params": params or [],
            "requires_fit": requires_fit,
        }
    )


class Gain(Processor):
    manifest: ClassVar[Manifest] = manifest(
        "gain", params=[{"name": "k", "type": "float", "default": 2.0}]
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        out = Block(
            block.data * float(params["k"]),
            block.info,
            block.sample_index,
            block.timestamp,
        )
        return TransformResult(out, {})


class Demean(Processor):
    """Requires fit: subtracts the per-channel mean of the calibration data."""

    manifest: ClassVar[Manifest] = manifest("demean", requires_fit=True)

    def fit(self, calibration: Block, params: Params) -> State:
        return {"mean": calibration.data.mean(axis=1)}

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        mean = model["mean"]
        assert isinstance(mean, np.ndarray)
        out = Block(
            block.data - mean[:, None], block.info, block.sample_index, block.timestamp
        )
        return TransformResult(out, {})


class Smooth(Processor):
    """Stateful causal exponential smoother; emits an event on threshold crossing."""

    manifest: ClassVar[Manifest] = manifest(
        "smooth",
        params=[
            {"name": "alpha", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0}
        ],
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        alpha = float(params["alpha"])
        y = state["y"] if state is not None else block.data[:, :1].copy()
        assert isinstance(y, np.ndarray)
        out = np.empty_like(block.data)
        for i in range(block.n_samples):  # reference implementation, not hot path
            y = (1 - alpha) * y + alpha * block.data[:, i : i + 1]
            out[:, i : i + 1] = y
        events = tuple(
            Event(block.sample_index + int(i), "high")
            for i in np.flatnonzero(out[0] > 0.5)[:1]
        )
        result = Block(out, block.info, block.sample_index, block.timestamp)
        return TransformResult(result, {"y": y}, events)


class WrongOutput(Processor):
    manifest: ClassVar[Manifest] = manifest("wrong")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        fb = FeatureBlock(block.data[:, 0], ("channel",), (0, 1), 0.0)
        return TransformResult(fb, {})


class LeakyState(Processor):
    manifest: ClassVar[Manifest] = manifest("leaky")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block, {"bad": object()})  # type: ignore[dict-item]


def signal(n: int, seed: int = 0, start: int = 0) -> Block:
    data = np.random.default_rng(seed).standard_normal((2, n)) + 3.0
    return Block(data, INFO, start, start / INFO.sfreq)


SOURCE_FP = signal_fingerprint(INFO)


def test_no_fit_stage_is_always_ready() -> None:
    stage = Stage(Gain(), {"k": 3.0})
    assert stage.status(SOURCE_FP) is StageStatus.READY
    out = stage.transform(signal(10), SOURCE_FP).block
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data, signal(10).data * 3.0)


def test_fit_lifecycle() -> None:
    stage = Stage(Demean())
    assert stage.status(SOURCE_FP) is StageStatus.UNCALIBRATED
    with pytest.raises(StaleModelError):
        stage.transform(signal(10), SOURCE_FP)
    stage.begin_calibration()
    assert stage.status(SOURCE_FP) is StageStatus.CALIBRATING
    stage.fit(signal(500), SOURCE_FP)
    assert stage.status(SOURCE_FP) is StageStatus.READY
    out = stage.transform(signal(500), SOURCE_FP).block
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data.mean(axis=1), 0.0, atol=1e-12)


def test_failed_fit_leaves_calibrating_state() -> None:
    class Broken(Demean):
        def fit(self, calibration: Block, params: Params) -> State:
            raise RuntimeError("boom")

    stage = Stage(Broken())
    stage.begin_calibration()
    with pytest.raises(RuntimeError):
        stage.fit(signal(10), SOURCE_FP)
    assert stage.status(SOURCE_FP) is StageStatus.UNCALIBRATED


def test_upstream_param_change_propagates_needs_refit() -> None:
    chain = [Stage(Gain()), Stage(Smooth()), Stage(Demean())]
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    chain[2].fit(signal(200), fps[2])
    assert [s.status(fp) for s, fp in zip(chain, fps, strict=True)] == [
        StageStatus.READY,
        StageStatus.READY,
        StageStatus.READY,
    ]

    chain[0].set_params(k=5.0)
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    assert chain[1].status(fps[1]) is StageStatus.READY  # no fit needed
    assert chain[2].status(fps[2]) is StageStatus.NEEDS_REFIT
    with pytest.raises(StaleModelError):
        chain[2].transform(signal(10), fps[2])

    chain[0].set_params(k=2.0)  # back to the fitted configuration
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    assert chain[2].status(fps[2]) is StageStatus.READY


def test_upstream_refit_propagates_needs_refit() -> None:
    chain = [Stage(Demean()), Stage(Demean())]
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    chain[0].fit(signal(200, seed=1), fps[0])
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    chain[1].fit(signal(200, seed=2), fps[1])
    assert chain[1].status(fps[1]) is StageStatus.READY

    chain[0].fit(signal(200, seed=3), fps[0])  # new calibration data, new model
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    assert chain[0].status(fps[0]) is StageStatus.READY
    assert chain[1].status(fps[1]) is StageStatus.NEEDS_REFIT


def test_own_param_change_and_source_change_need_refit() -> None:
    class ParamDemean(Demean):
        manifest = manifest(
            "pdemean",
            requires_fit=True,
            params=[{"name": "w", "type": "int", "default": 1}],
        )

    stage = Stage(ParamDemean())
    stage.fit(signal(100), SOURCE_FP)
    stage.set_params(w=2)
    assert stage.status(SOURCE_FP) is StageStatus.NEEDS_REFIT
    stage.set_params(w=1)
    assert stage.status(SOURCE_FP) is StageStatus.READY
    other_source = signal_fingerprint(SignalInfo(("Fz", "Cz"), 250.0, "amp"))
    assert stage.status(other_source) is StageStatus.NEEDS_REFIT


def test_channel_change_is_rejected_even_with_matching_fingerprint() -> None:
    stage = Stage(Demean())
    stage.fit(signal(100), SOURCE_FP)
    swapped = signal(10).pick(["Cz", "Fz"])
    with pytest.raises(StaleModelError, match="channels"):
        stage.transform(swapped, SOURCE_FP)


def test_blockwise_state_matches_one_pass_and_serializes() -> None:
    whole = signal(400)
    one_pass = Stage(Smooth()).transform(whole, SOURCE_FP).block
    stage = Stage(Smooth())
    outputs = []
    edges = [0, 1, 37, 38, 200, 399, 400]
    for a, b in itertools.pairwise(edges):
        outputs.append(stage.transform(whole.slice(a, b), SOURCE_FP).block)
        assert stage.state is not None
        stage.state = state_from_bytes(state_to_bytes(stage.state))  # save/resume
    joined = concat_blocks([b for b in outputs if isinstance(b, Block)])
    assert isinstance(one_pass, Block)
    np.testing.assert_allclose(joined.data, one_pass.data, rtol=0, atol=1e-12)


def test_param_change_resets_state() -> None:
    stage = Stage(Smooth())
    stage.transform(signal(10), SOURCE_FP)
    assert stage.state is not None
    stage.set_params(alpha=0.5)
    assert stage.state is None


def test_events_returned_with_sample_index() -> None:
    data = np.zeros((2, 20))
    data[0, 5:] = 10.0
    block = Block(data, INFO, 1000, 0.0)
    result = Stage(Smooth(), {"alpha": 0.9}).transform(block, SOURCE_FP)
    assert result.events == (Event(1005, "high"),)


def test_output_port_type_and_state_are_checked() -> None:
    with pytest.raises(PortTypeError):
        Stage(WrongOutput()).transform(signal(5), SOURCE_FP)
    with pytest.raises(TypeError):
        Stage(LeakyState()).transform(signal(5), SOURCE_FP)


def test_fit_on_processor_without_fit_is_refused() -> None:
    stage = Stage(Gain())
    with pytest.raises(RuntimeError):
        stage.fit(signal(5), SOURCE_FP)
    with pytest.raises(RuntimeError):
        stage.begin_calibration()


class LiveSmooth(Smooth):
    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "test.live_smooth",
            "version": "0.1.0",
            "type": "stream",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": [
                {"name": "alpha", "type": "float", "default": 0.1},
                {"name": "mode", "type": "string", "default": "a"},
            ],
            "live_params": ["alpha"],
        }
    )


def test_live_params_keep_state() -> None:
    stage = Stage(LiveSmooth())
    stage.transform(signal(10), SOURCE_FP)
    kept = stage.state
    stage.set_params(alpha=0.5)
    assert stage.state is kept  # live parameter: state carries over
    stage.set_params(alpha=0.5)
    assert stage.state is kept  # no change at all
    stage.set_params(alpha=0.2, mode="b")
    assert stage.state is None  # a non-live parameter changed


def test_live_params_of_a_fitted_node_need_no_refit() -> None:
    class ScaledDemean(Demean):
        manifest = Manifest.from_dict(
            {
                "spec_version": "0.2.0",
                "name": "test.sdemean",
                "version": "0.1.0",
                "type": "stream",
                "inputs": [{"name": "in", "type": "signal"}],
                "outputs": [{"name": "out", "type": "signal"}],
                "params": [
                    {"name": "k", "type": "float", "default": 1.0},
                    {"name": "w", "type": "int", "default": 1},
                ],
                "live_params": ["k"],
                "requires_fit": True,
            }
        )

    chain = [Stage(ScaledDemean()), Stage(Demean())]
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    chain[0].fit(signal(100, seed=1), fps[0])
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    chain[1].fit(signal(100, seed=2), fps[1])

    chain[0].set_params(k=3.0)  # acts in transform only: the model stays valid
    fps = chain_input_fingerprints(chain, SOURCE_FP)
    assert chain[0].status(fps[0]) is StageStatus.READY
    assert chain[1].status(fps[1]) is StageStatus.NEEDS_REFIT  # its input changed

    chain[0].set_params(w=2)  # not live: the model is stale
    assert chain[0].status(fps[0]) is StageStatus.NEEDS_REFIT
