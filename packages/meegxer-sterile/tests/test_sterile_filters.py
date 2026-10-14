import numpy as np
import pytest
from scipy import signal

from meegxer.core.block import Block, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.sterile.filters import IIRFilter, NotchFilter


def recording(seconds: float = 8.0, sfreq: float = 250.0) -> Block:
    src = SyntheticSource({"duration": seconds, "n_channels": 8, "sfreq": sfreq})
    block = src.read(10**7)
    assert block is not None
    return block


def run(processor: IIRFilter | NotchFilter, block: Block, **params: object) -> Block:
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
