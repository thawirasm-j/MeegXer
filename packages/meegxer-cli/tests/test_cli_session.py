"""Acceptance test 4 (docs/DESIGN.md §6): save a session, reload it elsewhere,
and the offline run writes an identical output file."""

import json
import shutil
from pathlib import Path

import mne
import numpy as np
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.session import read_session, sha256_file

runner = CliRunner()
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4",
            "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]  # fmt: skip


def recording(path: Path) -> Path:
    rng = np.random.default_rng(7)
    sfreq, n = 250.0, 250 * 60
    t = np.arange(n) / sfreq
    sources = np.vstack([np.sin(2 * np.pi * 10 * t), rng.laplace(size=n),
                         np.sign(np.sin(2 * np.pi * 0.3 * t))])  # fmt: skip
    mixing = rng.standard_normal((len(CHANNELS), 3))
    data = (mixing @ sources + 0.5 * rng.standard_normal((len(CHANNELS), n))) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(CHANNELS, sfreq, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_montage("colin27_1020")
    raw.save(path, verbose="error")
    return path


def test_session_reruns_byte_identical_on_another_machine(tmp_path: Path) -> None:
    study = tmp_path / "pc1" / "study"
    study.mkdir(parents=True)
    rec = recording(study / "sub-01_raw.fif")
    pipeline = study / "clean.json"
    pipeline.write_text(json.dumps({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(rec)}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "ica", "node": "meegxer.nodes.ica", "params": {"max_iter": 200}},
    ]}))  # fmt: skip

    args = ["run", str(pipeline), "-o", str(study / "clean.set"),
            "--session", str(study / "sub-01.meegxer.json")]  # fmt: skip
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "fitted: ica" in first.output
    saved = json.loads((study / "sub-01.meegxer.json").read_text())
    assert saved["outputs"] == [
        {"path": "clean.set", "sha256": sha256_file(study / "clean.set")}
    ]
    assert saved["models"]["ica"]["fitted_on"]["channels"] == CHANNELS

    # "another machine": the study folder elsewhere, without the output
    other = tmp_path / "pc2" / "copied"
    shutil.copytree(study, other)
    (other / "clean.set").unlink()
    session = other / "sub-01.meegxer.json"
    check = runner.invoke(app, ["session", "check", str(session)])
    assert check.exit_code == 0 and "model ica: reused" in check.output, check.output

    again = runner.invoke(app, ["run", str(session)])
    assert again.exit_code == 0, again.output
    assert "model ica: reused" in again.output and "fitted" not in again.output
    recorded = read_session(session)["outputs"][0]
    assert Path(recorded["path"]) == other / "clean.set"
    assert sha256_file(other / "clean.set") == recorded["sha256"]


def test_session_errors_are_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.meegxer.json"
    bad.write_text(json.dumps({"meegxer_session": "9.0", "pipeline": {}}))
    result = runner.invoke(app, ["run", str(bad)])
    assert result.exit_code == 1 and "not supported" in result.output
