"""Custom Python scripts in a standalone MNE script (M11.7 step 2): the
script's source inlined, called as MeegXer calls it, giving what MeegXer's
offline run gives (to single precision: MeegXer sends a plugin float32); a
script without standalone code yet is named."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.api import export_mne, run
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.mne_export import NotExportable
from meegxer.plugins import index

PROCESS = '''"""Band-pass, then mark the largest peak."""

import numpy as np


def process(raw, low: float = 1.0, high: float = 30.0, mark: str = "peak"):
    raw.filter(low, high, verbose="error")
    t = raw.times[int(np.argmax(np.abs(raw.get_data()).max(axis=0)))]
    raw.annotations.append(t, 0.0, mark)
    return raw
'''
TRANSFORM = '''def transform(data, state, gain: float = 2.0):
    """Scale the signal (no state)."""
    return data * gain, state
'''
EPOCHS = '''def process_epochs(epochs, keep: str = "go"):
    """Keep one condition, centre each epoch's channels."""
    kept = epochs[keep]
    kept.apply_function(lambda x: x - x.mean(), picks="all")
    return kept
'''
FEATURES = '''def features(data, info):
    """Variance."""
    return {"var": data.var(axis=1)}
'''


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """20 s: four EEG channels (Oz bad) and an EOG, "go"/"stop" events."""
    rng = np.random.default_rng(5)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz", "EOG1"], 250.0,
                           ["eeg"] * 4 + ["eog"])  # fmt: skip
    raw = mne.io.RawArray(rng.standard_normal((5, 5000)) * 1e-5, info,
                          verbose="error")  # fmt: skip
    raw.info["bads"] = ["Oz"]
    raw.set_annotations(mne.Annotations([2, 6, 10, 14], [0] * 4,
                                        ["go", "stop", "go", "stop"]))  # fmt: skip
    path = tmp_path / "rec_raw.fif"
    raw.save(path, verbose="error")
    return path


def add(tmp_path: Path, name: str, text: str) -> str:
    """Add the script (its notice accepted) and give its node name."""
    path = tmp_path / f"{name}.py"
    path.write_text(text)
    index.add_local(index.script_command(path), accept_risk=True, script=path)
    return f"script.{name}"


def pipeline(*nodes: dict[str, Any]) -> dict[str, Any]:
    source = {"id": "input", "node": "meegxer.core.file"}
    return {"spec_version": SPEC_VERSION, "nodes": [source, *nodes]}


def close(got: Any, want: Any) -> None:
    np.testing.assert_allclose(got.get_data(), want.get_data(), rtol=1e-5,
                               atol=1e-11)  # MeegXer sends plugins float32  # fmt: skip


def test_scripts_run_inlined_as_their_own_program(
    recording: Path, tmp_path: Path
) -> None:
    """process (with the events it adds), transform, Epochs, process_epochs:
    the script, run with nothing of MeegXer, gives MeegXer's signal and
    epochs."""
    pipe = pipeline(
        {"id": "bp", "node": add(tmp_path, "mnex_bandpass", PROCESS),
         "params": {"high": 20.0}},
        {"id": "gain", "node": add(tmp_path, "mnex_gain", TRANSFORM)},
        {"id": "ep1", "node": "meegxer.nodes.epochs",
         "params": {"events": "go,stop,peak", "tmin": -0.2, "tmax": 0.5}},
        {"id": "keep", "node": add(tmp_path, "mnex_keep", EPOCHS)},
    )  # fmt: skip
    script = tmp_path / "run_mne.py"
    text = export_mne(pipe, script)
    assert "custom script mnex_bandpass.py: process(low=1.0, high=20.0" in text
    assert "def process(raw, low: float = 1.0" in text  # the source, as written
    done = subprocess.run(
        [sys.executable, "-I", str(script), str(recording), "-o",
         str(tmp_path / "end_raw.fif"), "--epochs", str(tmp_path / "go-epo.fif")],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    want = run(pipe, recording)
    got = mne.io.read_raw_fif(tmp_path / "end_raw.fif", verbose="error")
    close(got, want)
    peaks = [a for a in got.annotations if a["description"] == "peak"]
    want_peaks = [a for a in want.annotations if a["description"] == "peak"]
    assert len(peaks) == len(want_peaks) == 1  # the event the script added
    assert peaks[0]["onset"] == pytest.approx(want_peaks[0]["onset"])  # its sample
    epochs = mne.read_epochs(tmp_path / "go-epo.fif", verbose="error")
    want_epochs = run(pipe, recording, output="keep:epochs")
    assert len(epochs) == len(want_epochs) == 2  # the go trials kept
    close(epochs, want_epochs)
    assert set(epochs.get_channel_types()) == {"eeg", "eog"}  # types kept


def test_a_script_without_standalone_code_is_named(
    recording: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "mnex_folder"
    folder.mkdir()
    (folder / "mnex_folder.py").write_text(TRANSFORM)
    index.add_local(index.script_command(folder), accept_risk=True, script=folder)
    for node in (add(tmp_path, "mnex_var", FEATURES), "script.mnex_folder"):
        with pytest.raises(NotExportable, match=rf"s1 \({node}\)"):
            export_mne(pipeline({"id": "s1", "node": node}))
