"""Display montages and filters for review pages (docs/REVIEW.md): computed in
the core from the raw samples of an offline probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.montage import display_filter, montage_matrix
from meegxer.core.offline import ProbeStore

LABELS = ("Fp1", "F7", "T3", "Cz", "Pz", "EOG")
TYPES = ("eeg", "eeg", "eeg", "eeg", "eeg", "eog")


def info(**kw: object) -> SignalInfo:
    return SignalInfo(LABELS, 100.0, ch_types=TYPES, **kw)  # type: ignore[arg-type]


def test_average_reference_leaves_out_bads_and_other_channels() -> None:
    m, labels = montage_matrix(info(bads=("Pz",)), "average")
    assert labels == LABELS
    x = np.array([1.0, 2.0, 3.0, 6.0, 100.0, 7.0])
    np.testing.assert_allclose(m @ x, [-2.0, -1.0, 0.0, 3.0, 97.0, 7.0])


def test_bipolar_pairs_the_electrodes_the_recording_has() -> None:
    m, labels = montage_matrix(info(), "longitudinal bipolar")
    # T3 is the old name of T7; Pz has no partner but Cz; EOG follows as it is
    assert labels == ("Fp1-F7", "F7-T3", "Cz-Pz", "EOG")
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 7.0])
    np.testing.assert_allclose(m @ x, [-1.0, -2.0, -8.0, 7.0])
    with pytest.raises(ValueError, match="none of the electrode pairs"):
        montage_matrix(SignalInfo(("A1", "A2"), 100.0), "transverse bipolar")


def test_laplacian_subtracts_the_nearest_neighbours() -> None:
    labels = ("C", "N", "S", "E", "W", "Far")
    pos = ((0.0, 0.0, 1.0), (0.0, 0.1, 1.0), (0.0, -0.1, 1.0), (0.1, 0.0, 1.0),
           (-0.1, 0.0, 1.0), (0.0, 0.9, 0.0))  # fmt: skip
    m, _ = montage_matrix(SignalInfo(labels, 100.0, positions=pos), "laplacian")
    x = np.array([10.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    assert (m @ x)[0] == pytest.approx(10.0 - 2.5)  # Far is not a neighbour of C
    with pytest.raises(ValueError, match="electrode positions"):
        montage_matrix(info(), "laplacian")


def test_display_filter_removes_drift_and_line_noise() -> None:
    t = np.arange(2000) / 200.0
    alpha = np.sin(2 * np.pi * 10 * t)
    data = (alpha + 5.0 + 2.0 * np.sin(2 * np.pi * 50 * t))[np.newaxis]
    out = display_filter(data, 200.0, highpass=1.0, lowpass=0.0, notch=50.0)
    middle = slice(400, 1600)
    np.testing.assert_allclose(out[0, middle], alpha[middle], atol=0.15)


def store_with(tmp_path: Path, data: np.ndarray, gap: tuple[int, int] | None = None,
               ) -> ProbeStore:  # fmt: skip
    signal = info()
    store = ProbeStore(tmp_path, signal, data.shape[1])
    if gap is None:
        store.write(Block(data, signal, 0, 0.0))
    else:
        a, b = gap
        store.write(Block(data[:, :a], signal, 0, 0.0))
        store.write(Block(data[:, b:], signal, b, b / 100.0))
    return store


def test_a_page_through_a_montage_is_computed_from_the_raw_samples(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 1000))
    store = store_with(tmp_path, data)
    page = store.derived_window(200, 100, 2000, montage="longitudinal bipolar")
    assert page["derived"] and page["bin_size"] == 1 and page["start_bin"] == 200
    assert page["ch_labels"] == ["Fp1-F7", "F7-T3", "Cz-Pz", "EOG"]
    expected = (data[0] - data[1])[200:300].astype(np.float32)
    np.testing.assert_allclose(page["mins"][0], expected, rtol=1e-5)
    # bads given by the reviewer leave the average reference at once
    avg = store.derived_window(0, 10, 2000, montage="average", bads=("Fp1",))
    ref = data[1:5, :10].astype(np.float32).mean(axis=0)
    np.testing.assert_allclose(avg["maxs"][3], data[3, :10] - ref, rtol=1e-4)
    # binned: min/max of the derived trace, not of the channels
    binned = store.derived_window(0, 1000, 100, montage="longitudinal bipolar")
    diff = (data[0] - data[1]).astype(np.float32)
    assert binned["bin_size"] == 10
    assert binned["mins"][0][3] == pytest.approx(diff[30:40].min(), rel=1e-5)
    # too long a page: the plain envelope, said so
    long = store.derived_window(0, 100_000_000, 100, montage="average")
    assert long["derived"] is False


def test_a_cut_is_filtered_on_each_side_and_stays_blank(tmp_path: Path) -> None:
    data = np.ones((6, 1000))
    store = store_with(tmp_path, data, gap=(400, 600))
    page = store.derived_window(0, 1000, 2000, montage="as recorded", highpass=1.0)
    trace = np.array(page["mins"][0], dtype=float)
    assert np.isnan(trace[400:600]).all()  # the cut
    # a constant is removed by the high-pass on each piece, nothing leaks
    assert np.nanmax(np.abs(trace)) < 0.2
    with pytest.raises(ValueError, match="below 50 Hz"):
        store.derived_window(0, 100, 100, lowpass=60.0)
