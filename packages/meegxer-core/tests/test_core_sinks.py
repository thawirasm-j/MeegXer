import csv
import json
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.ports import PortTypeError
from meegxer.core.sinks import FeatureTsvSink, SetSink


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


def test_feature_tsv_rejects_layout_change(tmp_path: Path) -> None:
    sink = FeatureTsvSink({"path": str(tmp_path / "f.tsv")})
    sink.write(features(0, 1.0))
    other = FeatureBlock(np.zeros(2), ("channel",), (100, 200), 1.0,
                         {"channel": ("Fz", "Cz")})  # fmt: skip
    with pytest.raises(ValueError, match="layout"):
        sink.write(other)


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
