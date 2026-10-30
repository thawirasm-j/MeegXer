"""Epochs and ERPs into files (M10.2/M10.3): the Epochs writer and the BIDS
epochs writer read the epochs rail and write it with MNE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import FeatureBlock
from meegxer.core.epoch_review import EpochVerdict
from meegxer.core.epochs_out import EpochsWriter, erp_path_for
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import FeatureProbe
from meegxer.core.registry import Registry
from meegxer.core.review import ReviewRequest

SFREQ = 100.0


def nodes(tmp_path: Path, writer: dict[str, Any], review: bool = False
          ) -> list[dict[str, Any]]:  # fmt: skip
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0\tgo\n"
                      "2.0\t0\tstop\n3.0\t0\tgo\n")  # fmt: skip
    rail = "rev:epochs" if review else "ep:epochs"
    return [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 5.0, "n_channels": 3, "sfreq": SFREQ},
         "recording": {"events": {"file": str(events)},
                       "channels": {"F3": {"type": "eog"}}}},
        {"id": "ep", "node": "meegxer.nodes.epochs",
         "params": {"tmin": -0.2, "tmax": 0.5}},
        *([{"id": "rev", "node": "meegxer.core.epoch_review",
            "params": {"file": str(tmp_path / "verdicts.tsv")}}] if review else []),
        {"id": "tap", "node": "meegxer.core.feature_probe", "input": rail,
         "params": {"keep_windows": 0}},
        {**writer, "input": rail},
    ]  # fmt: skip


def run(tmp_path: Path, writer: dict[str, Any], review: Any = None
        ) -> tuple[Graph, list[FeatureBlock]]:  # fmt: skip
    graph = Graph.from_dict(
        {"spec_version": SPEC_VERSION,
         "nodes": nodes(tmp_path, writer, review is not None)},
        Registry.from_entry_points(),
    )  # fmt: skip
    BatchScheduler(review=review).run(graph)
    tap = graph["tap"].impl
    assert isinstance(tap, FeatureProbe)
    return graph, [w for w in tap.windows if isinstance(w, FeatureBlock)]


def test_epochs_and_erps_written_as_fif(tmp_path: Path) -> None:
    out = tmp_path / "sub-01-epo.fif"
    _, windows = run(tmp_path, {"id": "w", "node": "meegxer.core.epochs_writer",
                                "params": {"path": str(out)}})  # fmt: skip
    epochs = mne.read_epochs(out, verbose="error")
    assert len(epochs) == 3 and epochs.event_id == {"go": 1, "stop": 2}
    assert epochs.info["sfreq"] == SFREQ and epochs.tmin == pytest.approx(-0.2)
    assert epochs.get_channel_types() == ["eeg", "eeg", "eog"]
    np.testing.assert_allclose(epochs.get_data(), np.concatenate(
        [w.data for w in windows]), rtol=1e-6, atol=1e-12)  # fmt: skip
    # the event at its sample: the window's start plus 0.2 s
    assert [int(s) for s in epochs.events[:, 0]] == [100, 200, 300]
    evokeds = mne.read_evokeds(erp_path_for(out), verbose="error")
    assert [e.comment for e in evokeds] == ["go", "stop"]
    np.testing.assert_allclose(evokeds[0].data, epochs["go"].get_data(
        picks="data").mean(axis=0), rtol=1e-6, atol=1e-12)  # fmt: skip
    assert evokeds[0].nave == 2


def test_epochs_as_an_eeglab_set_and_without_erps(tmp_path: Path) -> None:
    out = tmp_path / "epochs.set"
    run(tmp_path, {"id": "w", "node": "meegxer.core.epochs_writer",
                   "params": {"path": str(out), "erp": False}})  # fmt: skip
    epochs = mne.read_epochs_eeglab(out, verbose="error")
    assert len(epochs) == 3 and set(epochs.event_id) == {"go", "stop"}
    assert not erp_path_for(out).exists()


def test_rejected_epochs_are_left_out(tmp_path: Path) -> None:
    def review(req: ReviewRequest) -> list[EpochVerdict]:
        second = req.epochs[1]
        return [EpochVerdict(second.onset, second.duration, second.label, keep=False)]

    out = tmp_path / "kept-epo.fif"
    run(tmp_path, {"id": "w", "node": "meegxer.core.epochs_writer",
                   "params": {"path": str(out)}}, review)  # fmt: skip
    epochs = mne.read_epochs(out, verbose="error")
    assert len(epochs) == 2 and list(epochs.event_id) == ["go"]


def test_bids_epochs_writer(tmp_path: Path) -> None:
    root = tmp_path / "derivatives"
    params = {"root": str(root), "subject": "01", "task": "flanker", "desc": "clean"}
    run(tmp_path, {"id": "w", "node": "meegxer.core.bids_epochs", "params": params})
    folder = root / "sub-01" / "eeg"
    stem = "sub-01_task-flanker_desc-clean"
    epochs = mne.read_epochs(folder / f"{stem}_epo.fif", verbose="error")
    assert len(epochs) == 3
    assert len(mne.read_evokeds(folder / f"{stem}_ave.fif", verbose="error")) == 2
    sidecar = json.loads((folder / f"{stem}_epo.json").read_text())
    assert sidecar["Conditions"] == {"go": 2, "stop": 1}
    assert "meegxer.nodes.epochs" in sidecar["Description"]  # the chain
    assert (root / "dataset_description.json").is_file()
    ignored = set((root / ".bidsignore").read_text().split())
    assert {"*_epo.fif", "*_epo.json", "*_ave.fif"} <= ignored


def test_the_writer_wants_an_epochs_file_name() -> None:
    with pytest.raises(ValueError, match=r"-epo\.fif"):
        EpochsWriter({"path": "out.fif"})
    with pytest.raises(ValueError, match=r"\.fif or \.set"):
        EpochsWriter({"path": "out.edf"})
    assert erp_path_for(Path("a/s_epo.fif")) == Path("a/s_ave.fif")
    assert erp_path_for(Path("a/s.set")) == Path("a/s-ave.fif")
