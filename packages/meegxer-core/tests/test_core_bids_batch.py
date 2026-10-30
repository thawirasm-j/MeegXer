"""One pipeline over every recording of a BIDS dataset (bids_batch, OfflineJob)."""

import csv
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
from meegxer.core.session import SESSION_FORMAT


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


def test_workers_run_the_others_and_every_recording_gets_a_qc_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two workers, all but the last recording run in worker processes;
    the last runs in the job for the views. Each gets a QC report, and the
    dataset's qc.tsv lists every recording, the failed one too."""
    monkeypatch.setenv("MEEGXER_BATCH_JOBS", "2")
    root = make_dataset(tmp_path / "ds", ("01", "02", "03", "04"))
    (root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf").write_bytes(b"broken")
    pipeline = every_subject(root)
    registry = Registry.from_entry_points()
    document = {"meegxer_session": SESSION_FORMAT, "pipeline": pipeline}
    job = OfflineJob(Graph.from_dict(pipeline, registry), folder=tmp_path / "run",
                     rebuild=lambda: Graph.from_dict(pipeline, registry),
                     document=document)  # fmt: skip
    job.start()
    job.join(300)
    assert job.state == "done", job.error
    rec = job.status()["recordings"]
    assert sorted(rec["done"]) == ["sub-01_task-rest", "sub-03_task-rest",
                                   "sub-04_task-rest"]  # fmt: skip
    assert [f.split(":")[0] for f in rec["failed"]] == ["sub-02_task-rest"]
    assert set(job.status()["probes"]) == {"main"}  # the last one, run here
    deriv = root / "derivatives" / "meegxer"
    for sub in ("01", "03", "04"):
        assert (deriv / f"sub-{sub}" / "eeg" /
                f"sub-{sub}_task-rest_desc-clean_report.html").is_file()  # fmt: skip
    with (deriv / "qc.tsv").open(newline="") as fh:
        rows = {r["recording"]: r for r in csv.DictReader(fh, delimiter="\t")}
    assert sorted(rows) == [f"sub-0{k}_task-rest" for k in (1, 2, 3, 4)]
    assert rows["sub-02_task-rest"]["status"] == "failed"
    assert float(rows["sub-01_task-rest"]["kept_percent"]) == pytest.approx(100.0)
    assert rows["sub-01_task-rest"]["report"] == (
        "sub-01/eeg/sub-01_task-rest_desc-clean_report.html"
    )
    assert (deriv / "qc.html").is_file()
