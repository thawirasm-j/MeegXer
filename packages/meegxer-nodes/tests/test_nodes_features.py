"""Feature nodes against references (acceptance test 6) and analytic truths.

PSD and the Hilbert envelope are compared with GNU Octave (``pwelch`` as MATLAB
runs it, ``hilbert``) on ``tests/data/m6_eeg.npz``; the stored results come from
``tools/make_m6_feature_references.py``. Connectivity and complexity have no
EEGLAB counterpart: they are checked against the libraries they wrap
(mne-connectivity, antropy) and against signals with known answers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import signal as sps

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, FeatureBlock, SignalInfo, concat_blocks
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage
from meegxer.nodes.features import (
    PSD,
    BandPower,
    Complexity,
    Connectivity,
    HilbertEnvelope,
    parse_bands,
)

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"
REC = np.load(DATA / "m6_eeg.npz")
REF = np.load(DATA / "m6_ref_features.npz")
SFREQ = float(REC["sfreq"])
INFO = SignalInfo(tuple(str(s) for s in REC["labels"]), SFREQ, "m6")


def recording(data: np.ndarray | None = None, info: SignalInfo = INFO) -> Block:
    x = REC["data"].astype(np.float64) if data is None else data
    return Block(x, info, 0, 100.0)


def windows(proc: Any, block: Block, **params: Any) -> list[FeatureBlock]:
    out = Stage(proc, params).transform(block, "fp").block
    assert isinstance(out, tuple)
    return list(out)


def test_psd_matches_matlab_pwelch() -> None:
    got = windows(PSD(), recording(), window=2.0, step=1.0, fmin=0.0, fmax=64.0)
    assert [w.window[0] for w in got] == list(REF["window_starts"])
    psd = np.stack([w.data for w in got])
    np.testing.assert_allclose(psd, REF["psd"], rtol=1e-9, atol=0)
    assert got[0].coords["frequency"] == tuple(float(f) for f in REF["freqs"])
    assert got[3].timestamp == pytest.approx(100.0 + 3.0)  # window start time
    db = windows(PSD(), recording(), window=2.0, step=1.0, fmin=0.0, fmax=64.0,
                 db=True)  # fmt: skip
    np.testing.assert_allclose(db[0].data, 10 * np.log10(REF["psd"][0]), rtol=1e-9)


def test_band_power_sums_the_pwelch_bins_and_finds_a_sine() -> None:
    got = windows(BandPower(), recording(), window=2.0, step=1.0)
    freqs, df = REF["freqs"], REF["freqs"][1] - REF["freqs"][0]
    alpha = REF["psd"][:, :, (freqs >= 8) & (freqs < 13)].sum(axis=2) * df
    np.testing.assert_allclose(np.stack([w.data[:, 2] for w in got]), alpha,
                               rtol=1e-9)  # fmt: skip
    t = np.arange(int(4 * SFREQ)) / SFREQ
    sine = 20e-6 * np.sin(2 * np.pi * 10 * t)
    info = SignalInfo(("Oz",), SFREQ, "s")
    power = windows(BandPower(), recording(sine[None], info), window=4.0, step=4.0)
    assert power[0].coords["band"] == ("delta", "theta", "alpha", "beta", "gamma")
    assert power[0].data[0, 2] == pytest.approx((20e-6) ** 2 / 2, rel=0.01)
    rel = windows(BandPower(), recording(sine[None], info), window=4.0, step=4.0,
                  relative=True)  # fmt: skip
    assert rel[0].data[0].sum() == pytest.approx(1.0)
    assert rel[0].data[0, 2] > 0.99


def test_bands_are_parsed_strictly() -> None:
    assert parse_bands("mu:8-12, beta : 13-30") == [("mu", 8.0, 12.0),
                                                    ("beta", 13.0, 30.0)]  # fmt: skip
    for bad in ("alpha", "alpha:13-8", ""):
        with pytest.raises(ValueError):
            parse_bands(bad)


def coupled(seed: int = 0) -> Block:
    """ch0 alpha-band noise; ch1 = ch0 (zero lag); ch2 = ch0 a quarter cycle
    later; ch3 independent. Each with a little sensor noise."""
    rng = np.random.default_rng(seed)
    fs, n = 250.0, 250 * 8
    sos = sps.butter(4, [8, 12], btype="bandpass", fs=fs, output="sos")
    src = sps.sosfiltfilt(sos, rng.standard_normal(n + 100))
    lag = round(fs / 10 / 4)  # 25 ms at 10 Hz
    x = np.stack([src[100:], src[100:], src[100 - lag : n + 100 - lag],
                  sps.sosfiltfilt(sos, rng.standard_normal(n))])  # fmt: skip
    x += 0.2 * x.std() * rng.standard_normal(x.shape)
    return Block(x, SignalInfo(("A", "B", "C", "D"), fs, "c"), 0, 0.0)


@pytest.mark.parametrize("method", ["coh", "imcoh", "plv", "wpli"])
def test_connectivity_matches_mne_connectivity(method: str) -> None:
    import mne
    from mne_connectivity import spectral_connectivity_epochs

    block = coupled()
    got = windows(Connectivity(), block, method=method, window=8.0, step=8.0)[0]
    epochs = block.data.reshape(4, 8, 250).transpose(1, 0, 2)  # 1 s pieces
    with mne.utils.use_log_level("error"):
        con = spectral_connectivity_epochs(epochs, method=method, sfreq=250.0,
                                           fmin=8.0, fmax=13.0, faverage=True,
                                           verbose=False)  # fmt: skip
    lower = con.get_data(output="dense")[:, :, 0]
    rows, cols = np.tril_indices(4, -1)
    np.testing.assert_allclose(got.data[rows, cols], lower[rows, cols], rtol=1e-12)
    sign = -1 if method == "imcoh" else 1
    np.testing.assert_allclose(got.data.T, sign * got.data)
    assert got.coords["channel"] == got.coords["target"] == ("A", "B", "C", "D")


def test_connectivity_tells_zero_lag_from_lagged_coupling() -> None:
    def con(method: str) -> np.ndarray:
        out = windows(Connectivity(), coupled(), method=method, window=8.0, step=8.0)
        return np.abs(out[0].data)

    plv, coh, imcoh, wpli = con("plv"), con("coh"), con("imcoh"), con("wpli")
    assert plv[0, 1] > 0.9 and plv[0, 2] > 0.9 and plv[0, 3] < 0.6
    assert coh[0, 1] > 0.9 and coh[0, 3] < 0.6
    assert imcoh[0, 1] < 0.2 < 0.6 < imcoh[0, 2]  # volume conduction is ignored
    assert wpli[0, 2] > 0.8


def test_complexity_matches_antropy_and_orders_signals() -> None:
    import antropy as ant

    rng = np.random.default_rng(1)
    t = np.arange(512) / SFREQ
    x = np.stack([np.sin(2 * np.pi * 6 * t), rng.standard_normal(512)])
    info = SignalInfo(("sine", "noise"), SFREQ, "c")
    got = windows(Complexity(), recording(x, info), window=4.0, step=4.0)[0]
    assert got.coords["measure"] == ("perm_entropy", "sample_entropy",
                                     "spectral_entropy", "hjorth_mobility",
                                     "hjorth_complexity", "lziv")  # fmt: skip
    noise = x[1]
    expected = [ant.perm_entropy(noise, order=3, normalize=True),
                ant.sample_entropy(noise, order=2),
                ant.spectral_entropy(noise, sf=SFREQ, method="welch", normalize=True),
                *ant.hjorth_params(noise),
                ant.lziv_complexity((noise > np.median(noise)).astype(np.uint8),
                                    normalize=True)]  # fmt: skip
    np.testing.assert_allclose(got.data[1], expected, rtol=1e-12)
    sine, white = got.data
    for k in (0, 1, 2, 5):  # entropies and LZ: a sine is far more regular
        assert sine[k] < white[k]
    assert sine[3] == pytest.approx(2 * np.sin(np.pi * 6 / SFREQ), rel=0.01)  # Hjorth
    only = windows(Complexity(), recording(x, info), window=4.0, step=4.0,
                   sample_entropy=False, spectral_entropy=False,
                   hjorth_mobility=False, hjorth_complexity=False)[0]  # fmt: skip
    assert only.coords["measure"] == ("perm_entropy", "lziv")


def envelope(block: Block, **params: Any) -> Block:
    stage = Stage(HilbertEnvelope(), params)
    parts = [stage.transform(block, "fp").block]
    tail = stage.flush("fp")
    assert tail is not None
    parts.append(tail.block)
    return concat_blocks([p for p in parts if isinstance(p, Block)])


def test_envelope_matches_matlab_hilbert_away_from_the_ends() -> None:
    alpha = REF["alpha"].astype(np.float64)
    out = envelope(recording(alpha))
    assert (out.sample_index, out.n_samples) == (0, alpha.shape[1])
    ref = REF["envelope"].astype(np.float64)
    err = np.abs(out.data - ref) / np.abs(ref).max(axis=1, keepdims=True)
    edge = int(SFREQ)  # the whole-recording transform has its own end effects
    assert err[:, edge:-edge].max() < 0.01
    assert np.median(err) < 5e-4


def test_envelope_of_an_am_tone_and_its_delay() -> None:
    fs = 250.0
    t = np.arange(int(6 * fs)) / fs
    truth = 1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    x = (truth * np.sin(2 * np.pi * 12 * t))[None] * 1e-5
    info = SignalInfo(("Cz",), fs, "am")
    out = envelope(Block(x, info, 0, 0.0))
    err = np.abs(out.data[0] / 1e-5 - truth)
    assert err[250:-250].max() < 0.01
    stage = Stage(HilbertEnvelope(), {"lookahead": 0.5})
    first = stage.transform(Block(x[:, :500], info, 0, 0.0), "fp").block
    assert isinstance(first, Block)
    assert first.sample_index == 0 and first.stop_index <= 500 - 125  # lags 0.5 s


def test_envelope_keeps_untouched_channels_aligned() -> None:
    rng = np.random.default_rng(2)
    info = SignalInfo(("Cz", "EOG"), 250.0, "s", ch_types=("eeg", "eog"))
    x = rng.standard_normal((2, 1000))
    stage = Stage(HilbertEnvelope(), {"lookahead": 0.2})
    parts = []
    for start in range(0, 1000, 137):
        res = stage.transform(Block(x[:, start : start + 137], info, start, 0.0), "f")
        assert isinstance(res.block, Block)
        parts.append(res.block)
    tail = stage.flush("f")
    assert tail is not None and isinstance(tail.block, Block)
    out = concat_blocks([*parts, tail.block])
    assert (out.sample_index, out.n_samples) == (0, 1000)
    np.testing.assert_array_equal(out.data[1], x[1])  # EOG untouched, in step
    whole = envelope(Block(x[:1], info.subset(["Cz"]), 0, 0.0), lookahead=0.2)
    np.testing.assert_allclose(out.data[0], whole.data[0], rtol=1e-12)


@pytest.mark.parametrize(
    "proc", [PSD(), BandPower(), Connectivity(), Complexity(), HilbertEnvelope()]
)
def test_blockwise_equals_one_pass(proc: Any) -> None:
    block = recording().slice(0, 1200)
    assert check_blockwise_equivalence(proc, block, trials=2) == 0.0


def test_batch_pipeline_writes_band_power_rows(tmp_path: Path) -> None:
    out = tmp_path / "bands.tsv"
    graph = Graph.from_dict({"spec_version": "0.2.7", "nodes": [
        {"node": "meegxer.core.synthetic",
         "params": {"duration": 10, "n_channels": 2, "sfreq": 250}},
        {"node": "meegxer.nodes.iir_filter", "params": {"kind": "highpass",
                                                          "low": 1.0}},
        {"node": "meegxer.nodes.band_power", "params": {"window": 2, "step": 1}},
        {"node": "meegxer.core.feature_tsv", "params": {"path": str(out)}},
        {"node": "meegxer.core.probe"},
    ]}, Registry.from_entry_points())  # fmt: skip
    BatchScheduler(block_size=100).run(graph)
    with out.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    assert rows[0][:4] == ["window_start", "window_stop", "timestamp", "Fp1|delta"]
    assert len(rows) == 1 + 9  # 2 s windows every 1 s over 10 s
