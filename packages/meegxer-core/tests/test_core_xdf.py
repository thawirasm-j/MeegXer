"""XDF input (M10.5): a LabRecorder file's signal stream through the File input,
in volts, with its marker streams as events on the nearest sample and a
BAD_gap mark where samples were lost. The files are written here, chunk by
chunk, as the XDF specification lays them out (pyxdf only reads)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pytest

from meegxer.core.recording import describe, recording_events
from meegxer.core.sources import FileSource
from meegxer.core.xdf import GAP, gaps, nearest_samples, read_xdf

SFREQ = 100.0
T0 = 1000.0  # LSL clock at the first sample


def _varlen(n: int) -> bytes:
    return b"\x01" + struct.pack("<B", n) if n < 256 else b"\x04" + struct.pack("<I", n)


def _chunk(tag: int, content: bytes) -> bytes:
    body = struct.pack("<H", tag) + content
    return b"\x04" + struct.pack("<I", len(body)) + body


def _header(name: str, kind: str, n: int, rate: float, fmt: str,
            channels: list[dict[str, str]] | None = None) -> str:  # fmt: skip
    desc = ""
    if channels:
        desc = "<desc><channels>" + "".join(
            "<channel>" + "".join(f"<{k}>{escape(v)}</{k}>" for k, v in c.items())
            + "</channel>" for c in channels) + "</channels></desc>"  # fmt: skip
    return (f'<?xml version="1.0"?><info><name>{name}</name><type>{kind}</type>'
            f"<channel_count>{n}</channel_count><nominal_srate>{rate}"
            f"</nominal_srate><channel_format>{fmt}</channel_format>"
            f"<created_at>0</created_at>{desc}</info>")  # fmt: skip


def write_xdf(path: Path, streams: list[dict[str, Any]]) -> None:
    """Each stream: name, type, rate, format ("float32" or "string"),
    channels (descriptions or None), stamps, values (samples x channels);
    a clock offset (0) every 5 s, as LabRecorder measures them."""
    out = bytearray(b"XDF:")
    out += _chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')
    for sid, s in enumerate(streams, start=1):
        values = s["values"]
        n = len(values[0]) if len(values) else 1
        xml = _header(str(s["name"]), str(s["type"]), n, float(s["rate"]),
                      str(s["format"]), s.get("channels"))  # fmt: skip
        out += _chunk(2, struct.pack("<I", sid) + xml.encode())
        for t in np.arange(T0 - 5.0, T0 + 60.0, 5.0):
            out += _chunk(4, struct.pack("<Idd", sid, t, 0.0))
        body = bytearray(struct.pack("<I", sid) + _varlen(len(values)))
        for stamp, row in zip(s["stamps"], values, strict=True):
            body += b"\x08" + struct.pack("<d", stamp)
            for v in row:
                if s["format"] == "string":
                    raw = str(v).encode()
                    body += _varlen(len(raw)) + raw
                else:
                    body += struct.pack("<f", v)
        out += _chunk(3, bytes(body))
    path.write_bytes(bytes(out))


EEG = [{"label": "Fz", "type": "EEG", "unit": "microvolts"},
       {"label": "Cz", "type": "EEG", "unit": "microvolts"},
       {"label": "VEOG", "type": "EOG", "unit": "microvolts"}]  # fmt: skip


def signal(n: int = 500, channels: list[dict[str, str]] | None = EEG,
           lose: tuple[int, int] | None = None) -> dict[str, Any]:  # fmt: skip
    """An EEG stream of ``n`` samples (channel k reads k + 1 µV plus the
    sample number / 1000), with samples ``lose`` cut out (a dropout)."""
    k = np.arange(n)
    values = (np.arange(1, 4)[np.newaxis] + k[:, np.newaxis] / 1000).astype(np.float32)
    stamps = T0 + k / SFREQ
    if lose is not None:
        keep = (k < lose[0]) | (k >= lose[1])
        values, stamps = values[keep], stamps[keep]
    return {"name": "amp", "type": "EEG", "rate": SFREQ, "format": "float32",
            "channels": channels, "stamps": stamps, "values": values}  # fmt: skip


def markers(name: str, at: list[tuple[float, str]]) -> dict[str, Any]:
    return {"name": name, "type": "Markers", "rate": 0.0, "format": "string",
            "channels": None, "stamps": [T0 + t for t, _ in at],
            "values": [[label] for _, label in at]}  # fmt: skip


def test_a_labrecorder_file_through_the_file_input(tmp_path: Path) -> None:
    path = tmp_path / "sub-01_task-oddball.xdf"
    write_xdf(path, [
        markers("cues", [(1.004, "standard"), (2.996, "target"), (-0.5, "early")]),
        signal(),
        markers("responses", [(3.5, "press")]),
    ])  # fmt: skip
    source = FileSource({"path": str(path)})
    info = source.info
    assert info.ch_labels == ("Fz", "Cz", "VEOG")
    assert info.ch_types == ("eeg", "eeg", "eog") and info.sfreq == SFREQ
    block = source.read(10)
    assert block is not None and block.sample_index == 0
    np.testing.assert_allclose(block.data[:, 0], [1e-6, 2e-6, 3e-6], rtol=1e-6)
    events = recording_events(source)  # the nearest sample to each marker
    assert [(e.sample_index, e.label) for e in events] == [
        (100, "standard"), (300, "target"), (350, "press")]  # fmt: skip
    notes = describe(source)["notes"]
    assert notes[0] == "XDF: signal stream 'amp', markers from 'cues', 'responses'"
    assert notes[-1].startswith("1 markers left out: no recorded sample")


def test_a_marker_stream_stating_a_rate(tmp_path: Path) -> None:
    """Strings are markers even when the stream states a rate (as the XDF
    example files' marker streams do)."""
    path = tmp_path / "rated.xdf"
    write_xdf(path, [signal(), markers("cues", [(1.0, "go")]) | {"rate": 10.0}])
    read = read_xdf(path)
    assert read.marker_streams == ("cues",)
    assert list(read.raw.annotations.description) == ["go"]


def test_streams_by_name_and_an_assumed_unit(tmp_path: Path) -> None:
    path = tmp_path / "two.xdf"
    bare = [{"label": "Fz"}, {"label": "Cz"}, {"label": "Pz"}]  # no unit stated
    write_xdf(path, [signal(channels=bare) | {"name": "other", "type": "EMG"},
                     signal(), markers("cues", [(1.0, "a")]),
                     markers("noise", [(2.0, "b")])])  # fmt: skip
    read = read_xdf(path)  # the first EEG stream, every marker stream
    assert read.stream == "amp" and read.marker_streams == ("cues", "noise")
    picked = read_xdf(path, stream="other", markers="noise", unit="mV")
    assert picked.raw.ch_names == ["Fz", "Cz", "Pz"]
    assert picked.raw.get_channel_types() == ["emg"] * 3  # the stream's type
    np.testing.assert_allclose(picked.raw.get_data()[:, 0], [1e-3, 2e-3, 3e-3],
                               rtol=1e-6)  # mV, as the user said  # fmt: skip
    assert "3 of 3 channels state no unit: taken to be in mV" in picked.notes
    assert list(picked.raw.annotations.description) == ["b"]
    with pytest.raises(ValueError, match="no signal stream 'amp2'"):
        read_xdf(path, stream="amp2")
    with pytest.raises(ValueError, match="no marker stream"):
        read_xdf(path, markers="clicks")


def test_lost_samples_are_marked_bad(tmp_path: Path) -> None:
    """pyxdf breaks a stream where its stamps jump by over 1 s (500 samples
    at 100 Hz is 5 s, the larger): the first sample after it gets BAD_gap."""
    path = tmp_path / "gap.xdf"
    write_xdf(path, [signal(n=2000, lose=(800, 1400)),  # 6 s lost
                     markers("cues", [(2.0, "before"), (11.0, "lost"),
                                      (15.0, "after")])])  # fmt: skip
    read = read_xdf(path)
    ann = read.raw.annotations
    gap = [(o, d) for o, d, lb in zip(ann.onset, ann.duration, ann.description,
                                      strict=True) if lb == GAP]  # fmt: skip
    assert gap == [(pytest.approx(8.0), pytest.approx(0.01))]  # sample 800 on
    assert any("6.000 s of samples lost at 8.000 s" in n for n in read.notes)
    after = next(o for o, lb in zip(ann.onset, ann.description, strict=True)
                 if lb == "after")  # fmt: skip
    assert after == pytest.approx(9.0)  # 15 s on the clock: sample 900, not 1500
    assert "lost" not in ann.description  # sent in the gap: no sample for it
    assert any(n.startswith("1 markers left out") for n in read.notes)


def test_nearest_sample_and_gaps() -> None:
    stamps = np.array([0.0, 0.01, 0.02, 0.03, 0.10, 0.11])
    at = np.array([0.004, 0.006, 0.105, -0.004, -0.006, 0.114, 0.116, 0.06])
    placed = nearest_samples(stamps, at, 100.0)
    assert list(placed) == [0, 1, 4, 0, -1, 5, -1, -1]  # 0.06: in the gap
    (after, missing), *more = gaps(stamps, 100.0)
    assert after == 4 and not more
    assert missing == pytest.approx(0.06)


def test_the_reference_example_file() -> None:
    """minimal.xdf from the XDF example files (xdf-modules/example-files,
    MIT): 3 int16 channels at 10 Hz from 5.1 s, their clock 0.1 s behind
    (offset -0.1), and a string marker stream stating a rate."""
    path = Path(__file__).parent / "data" / "xdf" / "minimal.xdf"
    source = FileSource({"path": str(path)})
    assert source.info.sfreq == 10.0 and source.info.ch_labels == ("Ch1", "Ch2", "Ch3")
    block = source.read(9)
    assert block is not None
    np.testing.assert_allclose(block.data[:, 0], [192e-6, 255e-6, 238e-6])
    np.testing.assert_allclose(block.data[:, 1], [12e-6, 22e-6, 32e-6])
    events = recording_events(source)
    # markers at 5.2 ... 5.8 s on samples 2 ... 8 (now 5.0 ... 5.8 s); 5.9 s is
    # after the last sample and left out
    words = [(e.sample_index, e.label) for e in events if not e.label.startswith("<")]
    assert words == [(2, "Hello"), (3, "World"), (4, "from"), (5, "LSL"),
                     (6, "Hello"), (7, "World"), (8, "from")]  # fmt: skip
    assert any(n.startswith("1 markers left out") for n in describe(source)["notes"])
