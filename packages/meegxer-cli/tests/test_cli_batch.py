"""Batch at scale (M10.8): a --bids run in worker processes, with a QC report
per recording and the dataset's summary."""

import csv
import json
import os
from pathlib import Path

import mne
import mne_bids
import numpy as np
import pytest
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core.manifest import SPEC_VERSION

runner = CliRunner()


def dataset(root: Path, subjects: tuple[str, ...]) -> Path:
    for i, subject in enumerate(subjects):
        info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
        data = np.random.default_rng(i).standard_normal((3, 250 * 10)) * 1e-5
        raw = mne.io.RawArray(data, info, verbose="error")
        path = mne_bids.BIDSPath(root=root, subject=subject, task="rest",
                                 datatype="eeg")  # fmt: skip
        mne_bids.write_raw_bids(raw, path, format="EDF", allow_preload=True,
                                verbose="error")  # fmt: skip
    return root


def pipeline(path: Path) -> Path:
    path.write_text(json.dumps({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic"},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
    ]}))  # fmt: skip
    return path


def rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["recording"]: r for r in csv.DictReader(fh, delimiter="\t")}


def test_jobs_run_recordings_in_workers_with_qc_reports(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01", "02", "03"))
    (root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf").write_bytes(b"broken")
    pipe = pipeline(tmp_path / "clean.json")
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                 "--jobs", "2"])  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "done 2, skipped 0, failed 1: sub-02_task-rest" in result.output
    deriv = root / "derivatives" / "meegxer"
    assert f"QC summary: {deriv / 'qc.html'}" in result.output
    summary = rows(deriv / "qc.tsv")
    assert sorted(summary) == ["sub-01_task-rest", "sub-02_task-rest",
                               "sub-03_task-rest"]  # fmt: skip
    assert summary["sub-02_task-rest"]["status"] == "failed"
    assert summary["sub-01_task-rest"]["duration_s"] == "10.0"
    report = deriv / summary["sub-03_task-rest"]["report"]
    assert report.is_file() and "Processed" in report.read_text(encoding="utf-8")
    ignored = (deriv / ".bidsignore").read_text().split()
    assert {"*_report.html", "*_qc.json", "qc.tsv", "qc.html"} <= set(ignored)

    # a rerun skips the finished ones; their rows stay in the summary
    again = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                "--jobs", "0", "--subjects", "01,03"])  # fmt: skip
    assert "done 0, skipped 2, failed 0" in again.output, again.output
    assert sorted(rows(deriv / "qc.tsv")) == sorted(summary)


def test_no_report_and_a_bad_jobs_value(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01",))
    pipe = pipeline(tmp_path / "clean.json")
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                 "--no-report"])  # fmt: skip
    assert result.exit_code == 0, result.output
    deriv = root / "derivatives" / "meegxer"
    assert not (deriv / "qc.tsv").exists()
    assert not list(deriv.rglob("*_report.html"))
    bad = runner.invoke(app, ["run", str(pipe), "--bids", str(root), "--jobs", "-1"])
    assert bad.exit_code == 1 and "--jobs must be 0" in bad.output


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs two CPUs")
def test_jobs_zero_means_one_per_cpu(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01", "02"))
    pipe = pipeline(tmp_path / "clean.json")
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                 "--jobs", "0", "--no-report"])  # fmt: skip
    assert result.exit_code == 0 and "done 2" in result.output, result.output
