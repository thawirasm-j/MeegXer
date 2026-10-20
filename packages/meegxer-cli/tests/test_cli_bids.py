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
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
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


def test_interrupted_recording_is_redone_not_skipped(tmp_path: Path) -> None:
    """Ctrl+C while a recording is written leaves a partial .set but no session
    file (written last): the next run must redo it, not skip or refuse it."""
    root = dataset(tmp_path / "ds", ("01",))
    pipe = pipeline(tmp_path / "clean.json")
    folder = root / "derivatives" / "meegxer" / "sub-01" / "eeg"
    folder.mkdir(parents=True)
    partial = folder / "sub-01_task-rest_desc-clean_eeg.set"
    partial.write_bytes(b"half a file")
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert result.exit_code == 0, result.output
    assert "done 1, skipped 0, failed 0" in result.output
    assert partial.read_bytes() != b"half a file"
    assert (folder / "sub-01_task-rest_desc-clean_meegxer.json").is_file()


def test_recording_settings_apply_to_every_recording(tmp_path: Path) -> None:
    """The pipeline input's Recording settings (renames, types, montage, bad
    channels) carry over to each recording the loop runs."""
    root = dataset(tmp_path / "ds", ("01", "02"))
    pipe = tmp_path / "clean.json"
    pipe.write_text(json.dumps({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic",
         "recording": {"channels": {"Fz": {"rename": "FZ"}, "Pz": {"bad": True}}}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
    ]}))  # fmt: skip
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert result.exit_code == 0, result.output
    for subject in ("01", "02"):
        folder = root / "derivatives" / "meegxer" / f"sub-{subject}" / "eeg"
        name = f"sub-{subject}_task-rest_desc-clean_channels.tsv"
        channels = (folder / name).read_text()
        assert "FZ\tEEG" in channels and "Pz\tEEG\tV\tbad" in channels


def test_a_plain_file_writer_is_left_out_of_a_dataset_run(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01", "02"))
    pipe = tmp_path / "clean.json"
    pipe.write_text(json.dumps({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic"},
        {"id": "out", "node": "meegxer.core.set_writer",
         "params": {"path": str(tmp_path / "one.set")}},
    ]}))  # fmt: skip
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert result.exit_code == 0, result.output
    assert "done 2, skipped 0, failed 0" in result.output
    assert result.output.count("note: out: not written") == 2
    assert not (tmp_path / "one.set").exists()  # each would overwrite the last


def test_unreviewed_recordings_stop_at_the_review_strip(tmp_path: Path) -> None:
    root = dataset(tmp_path / "ds", ("01", "02"))
    pipe = tmp_path / "review.json"
    pipe.write_text(json.dumps({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic"},
        {"id": "rev", "node": "meegxer.core.review"},
    ]}))  # fmt: skip
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert result.exit_code == 2, result.output
    assert "sub-01_task-rest: NOT REVIEWED" in result.output
    assert "warning: 2 not reviewed" in result.output
    deriv = root / "derivatives" / "meegxer"
    assert not list(deriv.rglob("*_eeg.set"))  # nothing processed past the strip
