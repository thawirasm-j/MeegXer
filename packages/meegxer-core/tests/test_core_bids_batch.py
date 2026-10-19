"""One pipeline over every recording of a BIDS dataset (bids_batch, OfflineJob)."""

from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pytest

from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.offline import OfflineJob
from meegxer.core.registry import Registry


def make_dataset(root: Path, subjects: tuple[str, ...]) -> Path:
    for i, subject in enumerate(subjects):
        info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
        data = np.random.default_rng(i).standard_normal((3, 250 * 8)) * 1e-5
        path = mne_bids.BIDSPath(root=root, subject=subject, task="rest",
                                 datatype="eeg")  # fmt: skip
        mne_bids.write_raw_bids(mne.io.RawArray(data, info, verbose="error"), path,
                                format="EDF", allow_preload=True,
                                verbose="error")  # fmt: skip
    return root


def every_subject(root: Path) -> dict[str, Any]:
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.bids",
         "params": {"root": str(root), "subject": "", "task": "rest"}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "main", "node": "meegxer.core.probe"},
    ]}  # fmt: skip


def run(pipeline: dict[str, Any], folder: Path) -> OfflineJob:
    registry = Registry.from_entry_points()
    job = OfflineJob(Graph.from_dict(pipeline, registry), folder=folder,
                     rebuild=lambda: Graph.from_dict(pipeline, registry))  # fmt: skip
    job.start()
    job.join(120)
    assert job.state == "done", job.error
    return job


def test_every_subject_is_run_skipped_once_done_and_failures_go_on(
    tmp_path: Path,
) -> None:
    root = make_dataset(tmp_path / "ds", ("01", "02", "03"))
    broken = root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf"
    broken.write_bytes(b"not an EDF file")
    deriv = root / "derivatives" / "meegxer"

    first = run(every_subject(root), tmp_path / "run1")
    rec = first.status()["recordings"]
    assert (rec["index"], rec["total"], rec["current"]) == (3, 3, "sub-03_task-rest")
    assert rec["done"] == ["sub-01_task-rest", "sub-03_task-rest"]
    assert len(rec["failed"]) == 1 and rec["failed"][0].startswith("sub-02_task-rest")
    assert sorted(p.name for p in deriv.glob("sub-*")) == ["sub-01", "sub-03"]
    assert set(first.status()["probes"]) == {"main"}  # the last recording's views

    again = run(every_subject(root), tmp_path / "run2")
    rec = again.status()["recordings"]
    # finished ones are skipped; the last runs again for the views, its files kept
    assert rec["skipped"] == ["sub-01_task-rest", "sub-03_task-rest"]
    assert rec["done"] == [] and len(rec["failed"]) == 1
    assert again.status()["probes"]["main"]["n_filled"] > 0


def test_a_dataset_without_matching_recordings_is_refused(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ds", ("01",))
    pipeline = every_subject(root)
    pipeline["nodes"][0]["params"]["task"] = "nothing"
    registry = Registry.from_entry_points()
    with pytest.raises(GraphError, match="no matching EEG recordings"):
        OfflineJob(Graph.from_dict(pipeline, registry),
                   rebuild=lambda: Graph.from_dict(pipeline, registry))  # fmt: skip
