import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.ports import PortTypeError
from meegxer.core.sinks import (
    FeatureTsvSink,
    SetSink,
    discard,
    recover,
    spool_paths,
    unfinished_recordings,
    write_raw,
)


def features(start: int, value: float) -> FeatureBlock:
    return FeatureBlock(
        np.full((2, 3), value),
        dims=("channel", "band"),
        window=(start, start + 100),
        timestamp=start / 100.0,
        coords={"channel": ("Fz", "Cz"), "band": ("theta", "alpha", "beta")},
    )


def test_feature_tsv(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    sink = FeatureTsvSink({"path": str(path)})
    sink.write(features(0, 1.5))
    sink.write(features(100, 2.5))
    sink.close()
    with path.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    assert rows[0][:5] == ["window_start", "window_stop", "timestamp", "Fz|theta",
                           "Fz|alpha"]  # fmt: skip
    assert rows[0][-1] == "Cz|beta" and len(rows) == 3
    assert rows[2][:2] == ["100", "200"] and float(rows[2][3]) == 2.5

    before = path.read_bytes()  # an existing file is kept, its save skipped
    again = FeatureTsvSink({"path": str(path)})
    again.write(features(0, 1.0))
    again.close()
    assert path.read_bytes() == before
    assert again.skipped and "not saved" in again.skipped[0]


def test_feature_tsv_writes_missing_values_as_n_a(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    sink = FeatureTsvSink({"path": str(path)})
    sink.write(features(0, float("nan")))
    sink.close()
    with path.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    assert rows[1][3:] == ["n/a"] * 6


def test_feature_tsv_keeps_its_rows_when_the_layout_changes(tmp_path: Path) -> None:
    path = tmp_path / "f.tsv"
    sink = FeatureTsvSink({"path": str(path)})
    sink.write(features(0, 1.0))
    other = FeatureBlock(np.zeros(2), ("channel",), (100, 200), 1.0,
                         {"channel": ("Fz", "Cz")})  # fmt: skip
    sink.write(other)  # used to raise, ending a live run
    sink.write(features(200, 3.0))
    sink.close()
    assert sink.notes == [
        "feature layout changed at sample 100: f.tsv keeps the rows before"
    ]
    assert len(path.read_text().splitlines()) == 2  # header and the first window


def test_feature_tsv_rows_are_on_disk_as_they_come(tmp_path: Path) -> None:
    path = tmp_path / "f.tsv"
    sink = FeatureTsvSink({"path": str(path)})
    sink.write(features(0, 1.0))
    sink.write(features(100, 2.0))
    rows = list(csv.reader(path.open(), delimiter="\t"))  # before close
    assert len(rows) == 3 and float(rows[2][3]) == 2.0
    sink.close()


def test_sinks_check_block_type(tmp_path: Path) -> None:
    signal = Block(np.zeros((1, 3)), SignalInfo(("Cz",), 100.0), 0, 0.0)
    with pytest.raises(PortTypeError):
        FeatureTsvSink({"path": str(tmp_path / "f.tsv")}).write(signal)
    with pytest.raises(PortTypeError):
        SetSink({"path": str(tmp_path / "x.set")}).write(features(0, 1.0))
    with pytest.raises(ValueError, match="path"):
        SetSink({})


def test_feature_tsv_writes_units_into_a_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "features.tsv"
    sink = FeatureTsvSink({"path": str(path)})
    block = features(0, 1.0)
    sink.write(FeatureBlock(block.data, block.dims, block.window, block.timestamp,
                            block.coords, unit_dim="band",
                            units=("V^2", "V^2", "")))  # fmt: skip
    sink.close()
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["Cz|alpha"] == {"Units": "V^2"} and "Cz|beta" not in sidecar
    assert sidecar["timestamp"] == {"Units": "s"}

    plain = tmp_path / "plain.tsv"
    sink = FeatureTsvSink({"path": str(plain)})
    sink.write(features(0, 1.0))
    sink.close()
    assert not plain.with_suffix(".json").exists()  # no units stated, no sidecar


def signal(start: int, n: int, labels: tuple[str, ...] = ("Fz", "Cz")) -> Block:
    rng = np.random.default_rng(start)
    data = rng.standard_normal((len(labels), n)) * 1e-5
    return Block(data, SignalInfo(labels, 100.0), start, start / 100.0)


def read_set(path: Path) -> np.ndarray:
    import mne

    return np.asarray(mne.io.read_raw_eeglab(path, verbose="ERROR").get_data())


def test_set_writer_spools_until_close_and_writes_the_same_file(tmp_path: Path) -> None:
    blocks = [signal(0, 50), signal(50, 70), signal(120, 30)]
    path = tmp_path / "rec.set"
    sink = SetSink({"path": str(path)})
    for block in blocks:
        sink.write(block)
    data_path, header_path = spool_paths(path)
    assert data_path.stat().st_size == 2 * 150 * 8 and header_path.is_file()
    assert not path.exists()
    sink.close()
    assert not data_path.exists() and not header_path.exists()

    whole = np.hstack([b.data for b in blocks])
    header = {"ch_labels": ["Fz", "Cz"], "sfreq": 100.0, "bads": []}
    direct = tmp_path / "direct" / "rec.set"  # .set files name themselves
    write_raw(direct, "eeglab", whole, header, overwrite=False)
    assert path.read_bytes() == direct.read_bytes()  # sessions: same data, same file


def test_a_crashed_recording_is_recovered(tmp_path: Path) -> None:
    path = tmp_path / "rec.edf"
    script = f"""
import os, numpy as np
from meegxer.core.block import Block, SignalInfo
from meegxer.core.sinks import EdfSink
sink = EdfSink({{"path": {str(path)!r}}})
info = SignalInfo(("Fz", "Cz"), 100.0)
for k in range(3):
    sink.write(Block(np.full((2, 100), (k + 1) * 1e-5), info, 100 * k, float(k)))
os._exit(1)  # a crash: no close, nothing more flushed
"""
    done = subprocess.run([sys.executable, "-c", script], check=False)
    assert done.returncode == 1 and not path.exists()
    data_path, _ = spool_paths(path)
    with data_path.open("ab") as fh:
        fh.write(b"\0" * 12)  # a sample cut short mid-write
    assert recover(data_path) == path
    import mne

    raw = mne.io.read_raw_edf(path, verbose="ERROR")
    assert raw.n_times == 300 and raw.ch_names == ["Fz", "Cz"]
    np.testing.assert_allclose(raw.get_data()[0, ::100], [1e-5, 2e-5, 3e-5], rtol=1e-3)
    assert not any(p.exists() for p in spool_paths(path))
    with pytest.raises(FileNotFoundError, match="no spool"):
        recover(path)


def test_the_next_run_recovers_a_left_spool_first(tmp_path: Path) -> None:
    path = tmp_path / "rec.set"
    crashed = SetSink({"path": str(path)})
    crashed.write(signal(0, 40))
    crashed._spool.close()  # type: ignore[union-attr]  # as if the process died
    sink = SetSink({"path": str(path)})
    sink.write(signal(0, 60))
    sink.close()
    assert sink.notes == ["recovered a crashed run's recording to rec_recovered.set"]
    assert read_set(tmp_path / "rec_recovered.set").shape == (2, 40)
    assert read_set(path).shape == (2, 60)


def test_an_empty_left_spool_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "rec.set"
    _, header_path = spool_paths(path)
    header_path.write_text(json.dumps({"format": "eeglab", "ch_labels": ["Fz"],
                                       "sfreq": 100.0, "bads": []}))  # fmt: skip
    sink = SetSink({"path": str(path)})  # crashed before its first block
    sink.write(signal(0, 20))
    sink.close()
    assert not sink.notes and read_set(path).shape == (2, 20)
    assert not list(tmp_path.glob("*recovered*"))


def test_set_writer_keeps_what_came_before_a_channel_change(tmp_path: Path) -> None:
    path = tmp_path / "rec.set"
    sink = SetSink({"path": str(path)})
    sink.write(signal(0, 50))
    sink.write(signal(50, 50, ("Fz",)))
    sink.write(signal(100, 50))
    sink.close()
    assert sink.notes == [
        "channels changed at sample 50: rec.set keeps what came before"
    ]
    assert read_set(path).shape == (2, 50)


def test_set_writer_skips_an_existing_file_without_spooling(tmp_path: Path) -> None:
    path = tmp_path / "rec.set"
    path.write_bytes(b"old")
    sink = SetSink({"path": str(path)})
    sink.write(signal(0, 50))
    sink.close()
    assert path.read_bytes() == b"old" and sink.skipped
    assert not any(p.exists() for p in spool_paths(path))


def crash_while_recording(path: Path) -> None:
    """A process that records two blocks to ``path`` and dies."""
    script = f"""
import os, numpy as np
from meegxer.core.block import Block, SignalInfo
from meegxer.core.sinks import SetSink
sink = SetSink({{"path": {str(path)!r}}})
info = SignalInfo(("Fz", "Cz"), 100.0)
for k in range(2):
    sink.write(Block(np.zeros((2, 150)), info, 150 * k, float(k)))
os._exit(1)
"""
    assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 1


def test_unfinished_recordings_lists_crashed_runs_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEEGXER_CONFIG_DIR", str(tmp_path / "config"))  # its own
    crashed = tmp_path / "crashed.set"
    crash_while_recording(crashed)
    running = SetSink({"path": str(tmp_path / "running.set")})
    running.write(signal(0, 10))  # this process still writes it
    (found,) = unfinished_recordings()
    assert found["path"] == str(crashed.resolve())
    assert (found["format"], found["channels"], found["seconds"]) == ("eeglab", 2, 3.0)
    running.close()  # written: its marker goes
    discard(crashed)
    assert unfinished_recordings() == []
    assert not any(p.exists() for p in spool_paths(crashed))
