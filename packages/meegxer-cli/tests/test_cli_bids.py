"""Many recordings into BIDS derivatives (spec/bids-derivatives.md)."""

import json
from pathlib import Path

import mne
import mne_bids
import numpy as np
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
        {"id": "hp", "node": "meegxer.sterile.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
    ]}))  # fmt: skip
    return path


def test_all_subjects_continue_past_a_failure_and_resume(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01", "02", "03"))
    broken = root / "sub-02" / "eeg" / "sub-02_task-rest_eeg.edf"
    broken.write_bytes(b"not an EDF file")
    pipe = pipeline(tmp_path / "clean.json")
    deriv = root / "derivatives" / "meegxer"

    first = runner.invoke(
        app, ["run", str(pipe), "--bids", str(root), "--task", "rest"]
    )
    assert first.exit_code == 1, first.output
    assert "done 2, skipped 0, failed 1: sub-02_task-rest" in first.output
    assert sorted(p.name for p in deriv.glob("sub-*")) == ["sub-01", "sub-03"]

    again = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert "done 0, skipped 2, failed 1" in again.output, again.output

    redo = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                               "--subjects", "01,03", "--overwrite"])  # fmt: skip
    assert redo.exit_code == 0 and "done 2, skipped 0, failed 0" in redo.output


def test_single_recording_with_bids_out(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("05",))
    edf = root / "sub-05" / "eeg" / "sub-05_task-rest_eeg.edf"
    out = tmp_path / "deriv"
    result = runner.invoke(app, ["run", str(pipeline(tmp_path / "p.json")),
                                 "-i", str(edf), "--bids-out", str(out)])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "sub-05" / "eeg" / "sub-05_task-rest_desc-clean_eeg.set").is_file()
    assert (out / "sub-05" / "eeg"
            / "sub-05_task-rest_desc-clean_meegxer.json").is_file()  # fmt: skip


def test_bids_refuses_single_recording_options(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(pipeline(tmp_path / "p.json")),
                                 "--bids", str(tmp_path), "-o", "x.set"])  # fmt: skip
    assert result.exit_code == 1 and "no -i, -o or --session" in result.output
