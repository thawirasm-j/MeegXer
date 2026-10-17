"""Multitaper spectra, spectral parameterization, Morlet power and SSVEP CCA,
against the libraries they wrap and against signals with known answers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.node import Params
from meegxer.core.stage import Stage
from meegxer.nodes.features import PSD, BandPower, welch
from meegxer.nodes.specparam_node import SpecParam
from meegxer.nodes.ssvep import SSVEPCCA, canonical_correlation, references, subband
from meegxer.nodes.tfr import MorletPower

FS = 250.0


def windows(proc: Any, block: Block, **params: Any) -> list[FeatureBlock]:
    out = Stage(proc, params).transform(block, "fp").block
    assert isinstance(out, tuple)
    return list(out)


def block_of(x: np.ndarray, labels: tuple[str, ...] | None = None,
             bads: tuple[str, ...] = ()) -> Block:  # fmt: skip
    labels = labels or tuple(f"E{i}" for i in range(x.shape[0]))
    return Block(x, SignalInfo(labels, FS, "t", bads=bads), 0, 0.0)


def aperiodic_noise(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Noise whose power falls as 1/f^exponent."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    return np.fft.irfft(spec * f ** (-exponent / 2), n)


# --- multitaper ------------------------------------------------------------


def test_multitaper_psd_matches_mne_and_a_sine_power() -> None:
    from mne.time_frequency import psd_array_multitaper

    rng = np.random.default_rng(0)
    x = 1e-5 * rng.standard_normal((3, int(4 * FS)))
    got = windows(PSD(), block_of(x), method="multitaper", window=4.0, step=4.0,
                  fmin=0.0, fmax=125.0)[0]  # fmt: skip
    ref, freqs = psd_array_multitaper(x, FS, normalization="full", verbose=False)
    np.testing.assert_allclose(got.data, ref, rtol=1e-12)
    assert got.coords["frequency"] == tuple(round(float(f), 6) for f in freqs)
    narrow = windows(PSD(), block_of(x), method="multitaper", bandwidth=1.0,
                     window=4.0, step=4.0)[0]  # fmt: skip
    ref1, _ = psd_array_multitaper(x, FS, bandwidth=1.0, normalization="full",
                                   verbose=False)  # fmt: skip
    in_range = (freqs >= 1.0) & (freqs <= 45.0)  # the default fmin, fmax
    np.testing.assert_allclose(narrow.data, ref1[:, in_range], rtol=1e-12)
    assert not np.allclose(narrow.data, ref[:, in_range], rtol=0.01, atol=0)

    t = np.arange(int(4 * FS)) / FS
    sine = 20e-6 * np.sin(2 * np.pi * 10 * t)
    power = windows(BandPower(), block_of(sine[None]), method="multitaper",
                    window=4.0, step=4.0)[0]  # fmt: skip
    assert power.data[0, 2] == pytest.approx((20e-6) ** 2 / 2, rel=0.02)


def test_welch_stays_the_default() -> None:
    x = np.random.default_rng(1).standard_normal((2, int(2 * FS)))
    got = windows(PSD(), block_of(x), window=2.0, step=2.0, fmin=0.0, fmax=125.0)[0]
    params: Params = {"segment": 1.0, "overlap": 0.5, "taper": "hamming"}
    np.testing.assert_allclose(got.data, welch(x, FS, params)[1], rtol=1e-12)


# --- spectral parameterization ---------------------------------------------


def alpha_recording(seconds: float = 8.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = np.stack([aperiodic_noise(n, 1.5, rng) for _ in range(2)])
    x /= x.std()
    x[0] += 0.6 * np.sin(2 * np.pi * 10 * t)
    return 1e-5 * x


def test_specparam_matches_the_library_on_the_same_spectrum() -> None:
    from specparam import SpectralModel

    x = alpha_recording()
    got = windows(SpecParam(), block_of(x), window=4.0, step=4.0)[0]
    names = got.coords["measure"]
    assert names[:2] == ("offset", "exponent")
    assert names[-2:] == ("r_squared", "error")
    assert "alpha_frequency" in names and "beta_bandwidth" in names
    params: Params = {"segment": 1.0, "overlap": 0.5, "taper": "hamming"}
    freqs, psd = welch(x[:, : int(4 * FS)], FS, params)
    keep = (freqs >= 1) & (freqs <= 40) & (freqs > 0)
    model = SpectralModel(
        peak_width_limits=(1.0, 12.0),
        max_n_peaks=6,
        min_peak_height=0.0,
        peak_threshold=2.0,
        verbose=False,
    )
    model.fit(freqs[keep], psd[0, keep])
    np.testing.assert_allclose(got.data[0, :2], model.get_params("aperiodic"))
    assert got.data[0, names.index("r_squared")] == pytest.approx(
        model.get_metrics("gof")
    )


def test_specparam_recovers_the_exponent_and_the_alpha_peak() -> None:
    got = windows(SpecParam(), block_of(alpha_recording(16.0)), window=8.0,
                  step=8.0, segment=2.0)  # fmt: skip
    names = got[0].coords["measure"]
    for w in got:  # channel 1 is noise only; a noisy spectrum grows small peaks
        assert w.data[0, names.index("exponent")] == pytest.approx(1.5, abs=0.2)
        assert w.data[1, names.index("exponent")] == pytest.approx(1.5, abs=0.4)
        assert w.data[0, names.index("alpha_frequency")] == pytest.approx(10, abs=0.2)
        assert w.data[0, names.index("alpha_power")] > 0.5
        assert w.data[:, names.index("r_squared")].min() > 0.8


def test_specparam_knee_mode_adds_the_knee() -> None:
    got = windows(SpecParam(), block_of(alpha_recording()), aperiodic="knee",
                  window=4.0, step=4.0, bands="alpha:8-13")[0]  # fmt: skip
    assert got.coords["measure"] == (
        "offset", "knee", "exponent", "alpha_frequency", "alpha_power",
        "alpha_bandwidth", "r_squared", "error",
    )  # fmt: skip
    assert np.isfinite(got.data[:, :3]).all()


def test_specparam_leaves_a_failed_fit_as_nan() -> None:
    x = alpha_recording()
    x[1] = 0.0  # a flat channel
    got = windows(SpecParam(), block_of(x), window=4.0, step=4.0)[0]
    assert np.isfinite(got.data[0, :2]).all()
    assert got.data.shape == (2, len(got.coords["measure"]))


def test_specparam_refuses_too_few_frequencies() -> None:
    with pytest.raises(ValueError, match="at least 4 frequencies"):
        windows(SpecParam(), block_of(alpha_recording()), fmin=10.0, fmax=11.0,
                window=4.0, step=4.0)  # fmt: skip


# --- Morlet ---------------------------------------------------------------


def test_morlet_power_matches_mne_over_the_full_overlap() -> None:
    from mne.time_frequency import morlet, tfr_array_morlet

    rng = np.random.default_rng(2)
    x = 1e-5 * rng.standard_normal((2, int(4 * FS)))
    got = windows(MorletPower(), block_of(x), window=4.0, step=4.0, fmin=6.0,
                  fmax=30.0, fstep=2.0, n_cycles=5.0)[0]  # fmt: skip
    freqs = np.arange(6.0, 31.0, 2.0)
    tfr = tfr_array_morlet(x[None], FS, freqs, n_cycles=5.0, output="power",
                           verbose=False)[0]  # fmt: skip
    for j, w in enumerate(morlet(FS, freqs, n_cycles=5.0, zero_mean=True)):
        half = len(w) // 2
        inner = tfr[:, j, half : x.shape[1] - (len(w) - 1 - half)]
        np.testing.assert_allclose(got.data[:, j], inner.mean(axis=1), rtol=1e-9)
    assert got.coords["frequency"] == tuple(float(f) for f in freqs)


def test_morlet_power_peaks_at_a_sine_and_refuses_long_wavelets() -> None:
    t = np.arange(int(4 * FS)) / FS
    sine = 1e-5 * np.sin(2 * np.pi * 12 * t)
    got = windows(MorletPower(), block_of(sine[None]), window=4.0, step=4.0)[0]
    assert got.coords["frequency"][int(np.argmax(got.data[0]))] == 12.0
    db = windows(MorletPower(), block_of(sine[None]), window=4.0, step=4.0, db=True)
    np.testing.assert_allclose(db[0].data, 10 * np.log10(got.data), rtol=1e-12)
    with pytest.raises(ValueError, match=r"longer than the 1\.00 s window"):
        windows(MorletPower(), block_of(sine[None]), window=1.0, step=1.0)


# --- SSVEP ----------------------------------------------------------------


def ssvep(freq: float, seconds: float = 4.0, seed: int = 0) -> np.ndarray:
    """Eight channels: an SSVEP at ``freq`` (fundamental and 2nd harmonic, with
    channel-specific phase and gain) in independent noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    rows = []
    for ch in range(8):
        phase, gain = rng.uniform(0, 2 * np.pi), 0.3 + 0.1 * ch
        s = np.sin(2 * np.pi * freq * t + phase) + 0.5 * np.sin(4 * np.pi * freq * t)
        rows.append(gain * s + rng.standard_normal(t.size))
    return 1e-5 * np.array(rows)


def test_canonical_correlation_matches_the_textbook_formula() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 4))
    y = references(10.0, 2, 500, FS) + 0.5 * rng.standard_normal((500, 4))
    xc, yc = x - x.mean(0), y - y.mean(0)
    cxx, cyy, cxy = xc.T @ xc, yc.T @ yc, xc.T @ yc
    m = np.linalg.solve(cxx, cxy) @ np.linalg.solve(cyy, cxy.T)
    assert canonical_correlation(x, y) == pytest.approx(
        np.sqrt(np.max(np.linalg.eigvals(m).real)), rel=1e-9
    )
    # a rank-deficient montage (average reference) adds no spurious direction
    avg = x - x.mean(axis=1, keepdims=True)
    full = np.column_stack([avg, avg[:, :1] * 0])
    assert canonical_correlation(full, y) == pytest.approx(
        canonical_correlation(avg, y), rel=1e-9
    )


@pytest.mark.parametrize("method", ["cca", "fbcca"])
def test_ssvep_finds_the_stimulus_frequency(method: str) -> None:
    for target in (8.0, 10.0, 12.0, 15.0):
        got = windows(SSVEPCCA(), block_of(ssvep(target)), method=method,
                      window=2.0, step=1.0)  # fmt: skip
        assert len(got) == 3
        for w in got:
            assert w.dims == ("frequency",)
            assert w.coords["frequency"] == (8.0, 10.0, 12.0, 15.0)
            assert w.coords["frequency"][int(np.argmax(w.data))] == target


def test_ssvep_ignores_bad_channels() -> None:
    x = ssvep(12.0)
    t = np.arange(x.shape[1]) / FS
    x[0] = 1e-3 * np.sin(2 * np.pi * 8 * t)  # a broken channel picking up 8 Hz
    labels = tuple(f"E{i}" for i in range(8))
    fooled = windows(SSVEPCCA(), block_of(x, labels), method="cca", window=2.0,
                     step=2.0)[0]  # fmt: skip
    assert fooled.coords["frequency"][int(np.argmax(fooled.data))] == 8.0
    got = windows(SSVEPCCA(), block_of(x, labels, bads=("E0",)), method="cca",
                  window=2.0, step=2.0)[0]  # fmt: skip
    assert got.coords["frequency"][int(np.argmax(got.data))] == 12.0


def test_fbcca_subbands_follow_the_sample_rate() -> None:
    from scipy import signal as sps

    sos = subband(250.0, 16.0, 88.0)  # band-pass
    f, h = sps.sosfreqz(sos, worN=4096, fs=250.0)
    assert np.abs(h[(f > 20) & (f < 85)]).min() > 0.9  # within the 0.5 dB ripple
    assert np.abs(h[f < 12]).max() < 0.02
    assert np.abs(h[f > 100]).max() < 0.02
    sos = subband(128.0, 16.0, 88.0)  # no room above 88 Hz: high-pass
    f, h = sps.sosfreqz(sos, worN=4096, fs=128.0)
    assert np.abs(h[(f > 20)]).min() > 0.9
    with pytest.raises(ValueError, match="Nyquist"):
        subband(100.0, 48.0, 88.0)


def test_ssvep_frequencies_are_parsed_strictly() -> None:
    for bad in ("", "ten", "0, 10"):
        with pytest.raises(ValueError):
            windows(SSVEPCCA(), block_of(ssvep(10.0)), frequencies=bad)


# --- acceptance test 1 -------------------------------------------------------


@pytest.mark.parametrize(
    ("proc", "params"),
    [
        (PSD(), {"method": "multitaper"}),
        (BandPower(), {"method": "multitaper"}),
        (SpecParam(), {"window": 2.0, "step": 1.0}),
        (MorletPower(), {"window": 2.0, "step": 0.5, "fmin": 8.0}),
        (SSVEPCCA(), {}),
        (SSVEPCCA(), {"method": "cca"}),
    ],
)
def test_blockwise_equals_one_pass(proc: Any, params: dict[str, Any]) -> None:
    block = block_of(ssvep(10.0, seconds=5.0))
    assert check_blockwise_equivalence(proc, block, params, trials=2) == 0.0
