"""Non-EEG channels are kept but left alone by every strip unless it lists their
type in ``channel_types`` (default: EEG only). Channels without a type count as
EEG."""

from pathlib import Path
from typing import ClassVar

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, SignalInfo, concat_blocks
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION, Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.registry import Registry
from meegxer.core.serialization import State
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.sterile.filters import IIRFilter
from meegxer.sterile.reference import Rereference

LABELS = ("Fz", "Cz", "VEOG", "Pz")
TYPES = ("eeg", "eeg", "eog", "eeg")


def block(types: tuple[str, ...] = TYPES, n: int = 2000) -> Block:
    rng = np.random.default_rng(3)
    data = rng.standard_normal((4, n)) * 1e-5 + 5e-5  # with an offset to filter out
    return Block(data, SignalInfo(LABELS, 250.0, "t", (), types), 0, 0.0)


def run(stage: Stage, b: Block) -> Block:
    out = stage.transform(b, signal_fingerprint(b.info)).block
    assert isinstance(out, Block)
    return out


def test_filters_leave_non_eeg_channels_alone() -> None:
    b = block()
    out = run(Stage(IIRFilter(), {"kind": "highpass", "low": 1.0}), b)
    assert out.info == b.info  # same channels, same order, same description
    np.testing.assert_array_equal(out.data[2], b.data[2])  # VEOG untouched
    assert abs(out.data[[0, 1, 3], 1000:].mean()) < 5e-6  # EEG offset removed


def test_listing_a_type_includes_it() -> None:
    b = block()
    stage = Stage(IIRFilter(), {"kind": "highpass", "low": 1.0},
                  channel_types=["eeg", "eog"])  # fmt: skip
    out = run(stage, b)
    assert abs(out.data[2, 1000:].mean()) < 5e-6


def test_average_reference_ignores_non_eeg() -> None:
    b = block()
    out = run(Stage(Rereference()), b)
    eeg = b.data[[0, 1, 3]]
    np.testing.assert_allclose(out.data[[0, 1, 3]], eeg - eeg.mean(axis=0), atol=1e-18)
    np.testing.assert_array_equal(out.data[2], b.data[2])


def test_untyped_streams_are_all_eeg_and_no_match_passes_through() -> None:
    untyped = block(types=())
    out = run(Stage(IIRFilter(), {"kind": "highpass", "low": 1.0}), untyped)
    assert abs(out.data[2, 1000:].mean()) < 5e-6  # processed like the rest
    only_eog = Stage(IIRFilter(), {"kind": "highpass", "low": 1.0},
                     channel_types=["emg"])  # fmt: skip
    same = run(only_eog, block())
    np.testing.assert_array_equal(same.data, block().data)


def test_selection_is_blockwise_equivalent() -> None:
    b = block()
    whole = run(Stage(IIRFilter(), {"kind": "highpass", "low": 1.0}), b)
    stage = Stage(IIRFilter(), {"kind": "highpass", "low": 1.0})
    fp = signal_fingerprint(b.info)
    parts = []
    for start in range(0, b.n_samples, 137):
        out = stage.transform(b.slice(start, min(start + 137, b.n_samples)), fp).block
        assert isinstance(out, Block)
        parts.append(out)
    np.testing.assert_allclose(concat_blocks(parts).data, whole.data, atol=1e-18)


class MeanFit(Processor):
    """Fits the mean of each channel it sees (to check what fit receives)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": SPEC_VERSION, "name": "test.meanfit", "version": "0.1.0",
        "type": "stream", "requires_fit": True,
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
    })  # fmt: skip

    def fit(self, calibration: Block, params: Params) -> State:
        return {"labels": list(calibration.info.ch_labels),
                "mean": calibration.data.mean(axis=1)}  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        out = block.data - np.asarray(model["mean"])[:, None]
        return TransformResult(Block(out, block.info, block.sample_index,
                                     block.timestamp), {})  # fmt: skip


def test_fit_sees_only_the_selected_channels() -> None:
    b = block()
    stage = Stage(MeanFit())
    model = stage.fit(b, signal_fingerprint(b.info))
    assert model.data["labels"] == ["Fz", "Cz", "Pz"]
    out = run(stage, b)
    np.testing.assert_array_equal(out.data[2], b.data[2])
    with_eog = Stage(MeanFit(), channel_types=["eeg", "eog"])
    assert with_eog.fit(b, signal_fingerprint(b.info)).data["labels"] == list(LABELS)
    assert stage.fingerprint("x") != with_eog.fingerprint("x")


def test_pipeline_field_round_trips_and_is_for_processors(tmp_path: Path) -> None:
    registry = Registry.from_entry_points()
    data = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic"},
        {"id": "hp", "node": "meegxer.sterile.iir_filter",
         "channel_types": ["eeg", "eog"]},
        {"id": "ref", "node": "meegxer.sterile.rereference"},
    ]}  # fmt: skip
    graph = Graph.from_dict(data, registry)
    nodes = graph.to_dict()["nodes"]
    assert nodes[1]["channel_types"] == ["eeg", "eog"]
    assert "channel_types" not in nodes[2]  # the default is not written
    bad = {
        "spec_version": SPEC_VERSION,
        "nodes": [
            {"id": "input", "node": "meegxer.core.synthetic", "channel_types": ["eeg"]}
        ],
    }
    with pytest.raises(GraphError, match="processors"):
        Graph.from_dict(bad, registry)


def test_file_source_keeps_eog_and_the_reference_skips_it(tmp_path: Path) -> None:
    info = mne.create_info(["Fz", "Cz", "VEOG", "STI"], 250.0,
                           ["eeg", "eeg", "eog", "stim"])  # fmt: skip
    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.standard_normal((4, 2500)) * 1e-5, info, verbose="error")
    path = tmp_path / "rec_raw.fif"
    raw.save(path, verbose="error")
    probe = MemoryProbe({"keep_seconds": 0.0})
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "ref", "node": "meegxer.sterile.rereference"},
    ]}, Registry.from_entry_points())  # fmt: skip
    graph.add(probe, id="tap")
    BatchScheduler().run(graph)
    out = probe.signal()
    assert out.info.ch_labels == ("Fz", "Cz", "VEOG")  # trigger channel left out
    assert out.info.ch_types == ("eeg", "eeg", "eog")
    np.testing.assert_allclose(out.data[2], raw.get_data()[2], atol=1e-18)
    np.testing.assert_allclose(out.data[0] + out.data[1], 0.0, atol=1e-18)
