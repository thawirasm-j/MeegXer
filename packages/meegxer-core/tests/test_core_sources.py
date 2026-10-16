from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from meegxer.core.block import Block, concat_blocks
from meegxer.core.sinks import EdfSink, SetSink
from meegxer.core.sources import BidsSource, FileSource, SyntheticSource


def read_all(source: SyntheticSource | FileSource | BidsSource, chunk: int) -> Block:
    blocks = []
    while (block := source.read(chunk)) is not None:
        blocks.append(block)
    return concat_blocks(blocks)


def test_synthetic_is_block_size_invariant_and_deterministic() -> None:
    params = {"duration": 4.0, "n_channels": 6, "seed": 3}
    whole = read_all(SyntheticSource(params), 10_000)
    pieces = read_all(SyntheticSource(params), 37)
    assert whole.n_samples == 2000
    np.testing.assert_array_equal(whole.data, pieces.data)
    other_seed = read_all(SyntheticSource({**params, "seed": 4}), 10_000)
    assert not np.allclose(whole.data, other_seed.data)


def test_synthetic_spectrum_has_expected_peaks() -> None:
    src = SyntheticSource({"duration": 60.0, "n_channels": 20, "sfreq": 250.0})
    block = read_all(src, 100_000)
    labels = block.info.ch_labels
    freqs, psd = signal.welch(block.data, fs=250.0, nperseg=1000)

    def power(label: str, f: float) -> float:
        return float(psd[labels.index(label), np.argmin(np.abs(freqs - f))])

    def floor(label: str, f: float) -> float:
        band = (np.abs(freqs - f) > 1.5) & (np.abs(freqs - f) < 4)
        return float(np.median(psd[labels.index(label), band]))

    assert power("Oz", 12.0) > 10 * floor("Oz", 12.0)  # SSVEP, occipital only
    assert power("Fz", 12.0) < 3 * floor("Fz", 12.0)
    assert power("Fz", 50.0) > 10 * floor("Fz", 50.0)  # line noise everywhere
    assert power("O1", 10.0) > power("Fp1", 10.0)  # posterior alpha
    assert power("Cz", 2.0) > power("Cz", 30.0)  # pink background
    assert 1e-6 < np.std(block.data) < 1e-4  # volts, tens of microvolts


def test_synthetic_metadata() -> None:
    src = SyntheticSource({"n_channels": 3, "sfreq": 100.0, "duration": 0.0,
                           "start_time": 5.0})  # fmt: skip
    assert src.n_samples is None and src.info.ch_labels == ("Fp1", "Fp2", "F3")
    first = src.read(10)
    second = src.read(10)
    assert first is not None and second is not None
    assert second.sample_index == 10 and second.timestamp == pytest.approx(5.1)


@pytest.mark.parametrize(
    ("sink", "suffix", "atol"), [(SetSink, ".set", 1e-11), (EdfSink, ".edf", 2e-9)]
)
def test_file_round_trip(
    tmp_path: Path, sink: type[SetSink] | type[EdfSink], suffix: str, atol: float
) -> None:
    original = read_all(SyntheticSource({"duration": 2.0, "n_channels": 4}), 10_000)
    path = tmp_path / f"rec{suffix}"
    writer = sink({"path": str(path)})
    for start in range(0, original.n_samples, 300):
        writer.write(original.slice(start, min(start + 300, original.n_samples)))
    writer.close()

    source = FileSource({"path": str(path)})
    assert source.n_samples == original.n_samples
    loaded = read_all(source, 333)
    assert loaded.info.ch_labels == original.info.ch_labels
    assert loaded.info.sfreq == original.info.sfreq
    np.testing.assert_allclose(loaded.data, original.data, atol=atol)
    assert len(source.content_id()) == 64


def test_file_content_id_changes_with_content(tmp_path: Path) -> None:
    ids = []
    for seed in (0, 1):
        path = tmp_path / f"s{seed}.set"
        writer = SetSink({"path": str(path)})
        writer.write(read_all(SyntheticSource({"duration": 1.0, "seed": seed}), 10**6))
        writer.close()
        ids.append(FileSource({"path": str(path)}).content_id())
    assert ids[0] != ids[1]


def test_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        _ = FileSource({"path": "/nonexistent/x.set"}).info


def test_bids_source(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    mne_bids = pytest.importorskip("mne_bids")
    original = read_all(SyntheticSource({"duration": 2.0, "n_channels": 3}), 10_000)
    info = mne.create_info(list(original.info.ch_labels), original.info.sfreq, "eeg")
    raw = mne.io.RawArray(np.array(original.data), info, verbose="ERROR")
    bids_path = mne_bids.BIDSPath(subject="01", task="rest", root=tmp_path,
                                  datatype="eeg")  # fmt: skip
    mne_bids.write_raw_bids(raw, bids_path, format="EDF", allow_preload=True,
                            verbose=False)  # fmt: skip
    source = BidsSource({"root": str(tmp_path), "subject": "01", "task": "rest"})
    loaded = read_all(source, 500)
    assert loaded.info.ch_labels == original.info.ch_labels
    np.testing.assert_allclose(loaded.data, original.data, atol=2e-9)


ARTEFACTS = ("sine", "dc", "blinks", "saccades", "slow_eye", "ecg", "emg", "pops",
             "drift")  # fmt: skip


def test_synthetic_parts_add_up_and_switch_off() -> None:
    base = {"duration": 3.0, "n_channels": 8, "seed": 2}
    quiet = {"neural": False, "line": False, "ssvep": False}
    silent = read_all(SyntheticSource({**base, **quiet}), 1000)
    assert not silent.data.any()  # every part off: nothing
    default = read_all(SyntheticSource(base), 1000).data
    parts = [read_all(SyntheticSource({**base, **quiet, name: True}), 1000).data
             for name in ("neural", "line", "ssvep")]  # fmt: skip
    np.testing.assert_allclose(default, sum(parts), rtol=0, atol=1e-18)


def test_synthetic_artefacts_are_block_size_invariant() -> None:
    params = {"duration": 30.0, "n_channels": 20, "seed": 3, "pop_rate": 6.0,
              "emg_rate": 10.0, **dict.fromkeys(ARTEFACTS, True)}  # fmt: skip
    whole = read_all(SyntheticSource(params), 100_000)
    pieces = read_all(SyntheticSource(params), 37)
    np.testing.assert_array_equal(whole.data, pieces.data)


def only(**params: object) -> dict[str, object]:
    return {"duration": 60.0, "n_channels": 20, "sfreq": 250.0, "seed": 7,
            "neural": False, "line": False, "ssvep": False, **params}  # fmt: skip


def test_blinks_are_frontal_and_marked() -> None:
    src = SyntheticSource(only(blinks=True))
    data = read_all(src, 100_000)
    labels = data.info.ch_labels
    blinks = [e for e in src.annotations() if e["label"] == "blink"]
    assert 5 < len(blinks) < 40  # 17 per minute by default
    for e in blinks[:5]:
        peak = round((float(e["onset"]) + 0.15) * 250)  # middle of 300 ms
        fp1 = data.data[labels.index("Fp1"), peak] * 1e6
        assert fp1 > 100 or any(  # unless another blink overlaps
            abs(float(o["onset"]) - float(e["onset"])) < 0.3
            for o in blinks
            if o is not e
        )
        assert abs(data.data[labels.index("O1"), peak]) < 0.1 * abs(
            data.data[labels.index("Fp1"), peak]
        )


def test_saccades_are_opposite_at_f7_and_f8() -> None:
    data = read_all(SyntheticSource(only(saccades=True)), 100_000)
    labels = data.info.ch_labels
    f7, f8 = data.data[labels.index("F7")], data.data[labels.index("F8")]
    assert np.std(f7) > 20e-6
    np.testing.assert_allclose(f7, -f8)


def test_ecg_beats_match_the_heart_rate() -> None:
    src = SyntheticSource(only(ecg=True, ecg_bpm=60.0))
    beats = [e for e in src.annotations() if e["label"] == "ecg_r"]
    assert len(beats) == 60  # one per second for 60 s
    data = read_all(src, 100_000)
    t7 = data.data[data.info.ch_labels.index("T7")]
    r = round(float(beats[10]["onset"]) * 250)
    assert t7[r] == pytest.approx(5e-6, rel=0.05)  # unit R peak x 5 uV


def test_emg_is_muscle_band_and_temporal() -> None:
    src = SyntheticSource(only(emg=True, emg_rate=20.0))
    data = read_all(src, 100_000)
    labels = data.info.ch_labels
    freqs, psd = signal.welch(data.data, fs=250.0, nperseg=500)
    band = (freqs >= 20) & (freqs <= 80)
    low = freqs < 10
    t7 = psd[labels.index("T7")]
    assert t7[band].mean() > 20 * t7[low].mean()  # 20-80 Hz bursts
    assert t7[band].mean() > 10 * psd[labels.index("Pz")][band].mean()


def test_dc_offsets_differ_per_channel() -> None:
    data = read_all(SyntheticSource(only(dc=True, dc_mv=10.0)), 100_000)
    means = data.data.mean(axis=1)
    assert np.all(np.abs(means) <= 10e-3) and np.std(means) > 1e-3


def test_markers_reach_the_recording_events() -> None:
    from meegxer.core.recording import describe

    src = SyntheticSource(only(blinks=True, pops=True, pop_rate=6.0))
    labels = {e["label"] for e in describe(src)["events"]}
    assert labels == {"blink", "pop"}
    assert SyntheticSource(only(blinks=True, markers=False)).annotations() == []
    assert SyntheticSource(only(blinks=True, duration=0.0)).annotations() == []
