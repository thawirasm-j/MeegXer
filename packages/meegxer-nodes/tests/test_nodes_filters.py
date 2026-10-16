import numpy as np
import pytest
from scipy import signal

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, SignalInfo, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.node import Processor
from meegxer.core.probes import MemoryProbe
from meegxer.core.registry import Registry
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.nodes.filters import (
    IIRFilter,
    IIRFilterZeroPhase,
    NotchFilter,
    NotchFilterZeroPhase,
)


def recording(seconds: float = 8.0, sfreq: float = 250.0) -> Block:
    src = SyntheticSource({"duration": seconds, "n_channels": 8, "sfreq": sfreq})
    block = src.read(10**7)
    assert block is not None
    return block


def run(processor: Processor, block: Block, **params: object) -> Block:
    out = Stage(processor, params).transform(block, signal_fingerprint(block.info))
    assert isinstance(out.block, Block)
    return out.block


def band_power(block: Block, f: float, width: float = 1.0) -> float:
    freqs, psd = signal.welch(block.data, fs=block.info.sfreq, nperseg=1024)
    return float(psd[:, np.abs(freqs - f) <= width].mean())


IIR_CASES = [
    {"kind": "highpass", "low": 1.0, "order": 4},
    {"kind": "lowpass", "high": 30.0, "order": 6},
    {"kind": "bandpass", "low": 1.0, "high": 40.0, "order": 4},
    {"kind": "bandstop", "low": 45.0, "high": 55.0, "order": 2},
]


@pytest.mark.parametrize("params", IIR_CASES)
def test_iir_blockwise_equals_one_pass(params: dict[str, object]) -> None:
    """Acceptance test 1 for the IIR filter."""
    check_blockwise_equivalence(IIRFilter(), recording(), params, trials=5)


@pytest.mark.parametrize(
    "params", [{"freq": 50.0}, {"freq": 50.0, "harmonics": 3, "quality": 20.0}]
)
def test_notch_blockwise_equals_one_pass(params: dict[str, object]) -> None:
    """Acceptance test 1 for the notch filter."""
    check_blockwise_equivalence(NotchFilter(), recording(), params, trials=5)


@pytest.mark.parametrize("params", IIR_CASES)
def test_iir_matches_scipy_reference(params: dict[str, object]) -> None:
    block = recording()
    resolved = IIRFilter.manifest.resolve_params(params)
    sos = IIRFilter.design(resolved, block.info.sfreq)
    zi = signal.sosfilt_zi(sos)[:, None, :] * block.data[None, :, :1]
    expected, _ = signal.sosfilt(sos, block.data, axis=-1, zi=zi)
    np.testing.assert_allclose(run(IIRFilter(), block, **params).data, expected,
                               rtol=1e-12, atol=1e-18)  # fmt: skip


def test_filters_do_what_they_say() -> None:
    block = recording(30.0)
    hp = run(IIRFilter(), block, kind="highpass", low=5.0)
    assert band_power(hp, 1.0, 0.5) < 0.01 * band_power(block, 1.0, 0.5)
    lp = run(IIRFilter(), block, kind="lowpass", high=20.0)
    assert band_power(lp, 50.0) < 1e-3 * band_power(block, 50.0)
    assert band_power(lp, 10.0) == pytest.approx(band_power(block, 10.0), rel=0.1)
    notched = run(NotchFilter(), block, freq=50.0)
    assert band_power(notched, 50.0, 0.5) < 0.01 * band_power(block, 50.0, 0.5)
    assert band_power(notched, 12.0) == pytest.approx(band_power(block, 12.0), rel=0.05)


def test_steady_state_start_has_no_step() -> None:
    info = SignalInfo(("Cz",), 250.0)
    dc = Block(np.full((1, 500), 1e-4), info, 0, 0.0)
    out = run(IIRFilter(), dc, kind="lowpass", high=30.0)
    np.testing.assert_allclose(out.data, 1e-4, rtol=1e-9)


def test_notch_harmonics_stop_below_nyquist() -> None:
    params = NotchFilter.manifest.resolve_params({"freq": 50.0, "harmonics": 10})
    sos = NotchFilter.design(params, 250.0)
    assert sos.shape[0] == 2  # 50 and 100 Hz; 150+ Hz is above Nyquist (125 Hz)


@pytest.mark.parametrize(
    ("processor", "params"),
    [
        (IIRFilter(), {"kind": "lowpass", "high": 200.0}),
        (IIRFilter(), {"kind": "bandpass", "low": 30.0, "high": 10.0}),
        (IIRFilter(), {"kind": "highpass", "low": 0.0}),
        (NotchFilter(), {"freq": 130.0}),
    ],
)
def test_invalid_frequencies_raise(
    processor: IIRFilter | NotchFilter, params: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        run(processor, recording(1.0), **params)


def test_zero_phase_variants_are_offline_only_with_same_params() -> None:
    for causal, zero in (
        (IIRFilter, IIRFilterZeroPhase),
        (NotchFilter, NotchFilterZeroPhase),
    ):
        assert zero.manifest.availability == "offline"
        assert causal.manifest.availability == "both"
        # the band-pass's ``phase`` picks the variant, so the variant lacks it
        shared = tuple(p for p in causal.manifest.params if p.name != "phase")
        assert zero.manifest.params == shared


@pytest.mark.parametrize("params", IIR_CASES)
def test_zero_phase_matches_scipy_sosfiltfilt(params: dict[str, object]) -> None:
    block = recording()
    sos = IIRFilter.design(IIRFilter.manifest.resolve_params(params), block.info.sfreq)
    expected = signal.sosfiltfilt(sos, block.data, axis=-1)
    np.testing.assert_allclose(run(IIRFilterZeroPhase(), block, **params).data,
                               expected, rtol=1e-12, atol=1e-18)  # fmt: skip


def lag_of_peak(a: np.ndarray, b: np.ndarray) -> int:
    corr = signal.correlate(a - a.mean(), b - b.mean(), mode="full")
    return int(np.argmax(corr)) - (len(b) - 1)


def test_zero_phase_has_no_delay_unlike_causal() -> None:
    sfreq = 250.0
    t = np.arange(5000) / sfreq
    x = np.sin(2 * np.pi * 8.0 * t)[None, :] * 1e-5
    block = Block(x, SignalInfo(("Cz",), sfreq), 0, 0.0)
    params = {"kind": "bandpass", "low": 4.0, "high": 12.0}
    zero = run(IIRFilterZeroPhase(), block, **params).data[0, 500:-500]
    causal = run(IIRFilter(), block, **params).data[0, 500:-500]
    ref = x[0, 500:-500]
    assert lag_of_peak(zero, ref) == 0
    assert lag_of_peak(causal, ref) > 0  # causal output lags the input


def test_zero_phase_runs_in_batch_regardless_of_block_size() -> None:
    probe = MemoryProbe()
    src = SyntheticSource({"duration": 8.0, "n_channels": 8, "sfreq": 250.0})
    graph = Graph.chain([src, Stage(NotchFilterZeroPhase()), probe])
    BatchScheduler(block_size=32).run(graph)
    out = probe.signal()
    expected = run(NotchFilterZeroPhase(), recording())
    np.testing.assert_allclose(out.data, expected.data, rtol=1e-12, atol=1e-18)
    assert band_power(out, 50.0, 0.5) < 0.01 * band_power(recording(), 50.0, 0.5)


def test_zero_phase_needs_enough_samples() -> None:
    short = Block(np.zeros((1, 10)), SignalInfo(("Cz",), 250.0), 0, 0.0)
    with pytest.raises(ValueError, match="longer recording"):
        run(IIRFilterZeroPhase(), short)


def _stream(stage: Stage, block: Block, sizes: list[int]) -> Block:
    fp = signal_fingerprint(block.info)
    outs, start = [], block.sample_index
    for size in sizes:
        result = stage.transform(block.slice(start, start + size), fp)
        assert isinstance(result.block, Block)
        outs.append(result.block)
        start += size
    tail = stage.flush(fp)
    if tail is not None:
        assert isinstance(tail.block, Block)
        outs.append(tail.block)
    return concat_blocks([b for b in outs if b.n_samples])


def _online_signal_processors() -> list[str]:
    registry = Registry.from_entry_points()
    return [
        name
        for name in registry.names()
        if name.startswith("meegxer.nodes.")
        and (m := registry.get(name).manifest).role == "processor"
        and m.kind != "analysis"
        and m.runs_online
        and not m.requires_fit
    ]


@pytest.mark.parametrize("name", _online_signal_processors())
def test_empty_blocks_from_a_delaying_node_upstream_change_nothing(name: str) -> None:
    """A delaying node (envelope, ASR) may emit empty blocks, the first one
    included (spec 0.2.5); the band-pass and notch crashed on them."""
    factory = Registry.from_entry_points().get(name)
    block = recording(4.0)

    def stage() -> Stage:
        processor = factory()
        assert isinstance(processor, Processor)
        return Stage(processor)

    plain = _stream(stage(), block, [100] * 10)
    gappy = _stream(stage(), block, [0, 100, 0, 0] + [100] * 9 + [0])
    np.testing.assert_array_equal(gappy.data, plain.data)
    assert gappy.sample_index == plain.sample_index
