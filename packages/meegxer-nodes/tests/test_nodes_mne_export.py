"""A pipeline as a standalone MNE script (M11.7 step 2, decided 2026-11-27):
each strip writes its own MNE/SciPy code, and that code gives what the strip
gives in MeegXer's offline run (``meegxer.api.run``); a strip without code
yet is named; the script needs no MeegXer."""

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


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """20 s: six EEG channels (Oz bad) and an EOG, events and a BAD mark."""
    rng = np.random.default_rng(3)
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz", "C3", "C4", "EOG1"], 250.0,
                           ["eeg"] * 6 + ["eog"])  # fmt: skip
    raw = mne.io.RawArray(rng.standard_normal((7, 5000)) * 1e-5, info,
                          verbose="error")  # fmt: skip
    raw.set_montage("standard_1020", on_missing="ignore", verbose="error")
    raw.info["bads"] = ["Oz"]
    labels = ["go", "stop", "go", "stop", "BAD_x"]
    raw.set_annotations(mne.Annotations([2, 6, 10, 14, 17.5], [0, 0, 0, 0, 1.0],
                                        labels))  # fmt: skip
    path = tmp_path / "rec_raw.fif"
    raw.save(path, verbose="error")
    return path


def strip(node_id: str, node: str, **params: Any) -> dict[str, Any]:
    return {"id": node_id, "node": node, "params": params}


def pipeline(*strips: dict[str, Any]) -> dict[str, Any]:
    source = {"id": "input", "node": "meegxer.core.file"}
    return {"spec_version": SPEC_VERSION, "nodes": [source, *strips]}


def exported(pipe: dict[str, Any], path: Path) -> tuple[Any, Any]:
    """The exported script's pipeline(path): (raw, epochs)."""
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(export_mne(pipe), "run_mne.py", "exec"), scope)
    return scope["pipeline"](str(path))  # type: ignore[no-any-return]


CASES = {
    "iir zero-phase": [strip("f", "meegxer.nodes.iir_filter", kind="bandpass",
                             low=1.0, high=30.0)],
    "iir causal": [strip("f", "meegxer.nodes.iir_filter", kind="highpass",
                         low=0.5, phase="causal")],
    "notch zero-phase": [strip("n", "meegxer.nodes.notch_zerophase", freq=50.0,
                               harmonics=2)],
    "notch causal": [strip("n", "meegxer.nodes.notch", freq=50.0)],
    "fir zero-phase": [strip("f", "meegxer.nodes.fir_filter", kind="lowpass",
                             high=40.0)],
    "fir causal": [strip("f", "meegxer.nodes.fir_filter", kind="bandpass",
                         low=1.0, high=40.0, phase="causal")],
    "average reference": [strip("r", "meegxer.nodes.rereference")],
    "channel reference": [strip("r", "meegxer.nodes.rereference", mode="channels",
                                channels="Cz")],
    "pick channels": [strip("p", "meegxer.nodes.pick_channels", keep_eog=True,
                            drop_bads=True)],
    "resample": [strip("rs", "meegxer.nodes.resample", sfreq=100.0)],
    "interpolate bads": [strip("i", "meegxer.nodes.interpolate_bads")],
    "csd": [strip("c", "meegxer.nodes.csd", stiffness=3.0)],
    "csd after interpolation": [strip("i", "meegxer.nodes.interpolate_bads"),
                                strip("c", "meegxer.nodes.csd")],
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_strip_s_code_gives_what_the_strip_gives(
    recording: Path, name: str
) -> None:
    pipe = pipeline(*CASES[name])
    want = run(pipe, recording)
    got, _ = exported(pipe, recording)
    assert got.ch_names == want.ch_names
    assert got.info["sfreq"] == want.info["sfreq"]
    np.testing.assert_allclose(got.get_data(), want.get_data(), rtol=1e-9,
                               atol=1e-16)  # fmt: skip


def test_a_chain_with_epochs_runs_as_its_own_program(
    recording: Path, tmp_path: Path
) -> None:
    """Filter, re-reference, resample, epochs and the writers: the script,
    run with nothing of MeegXer, gives MeegXer's signal and epochs and
    writes the writers' files."""
    out_set, out_epo = tmp_path / "clean.set", tmp_path / "ep-epo.fif"
    pipe = pipeline(
        strip("f", "meegxer.nodes.iir_filter", low=1.0, high=30.0),
        strip("r", "meegxer.nodes.rereference"),
        strip("rs", "meegxer.nodes.resample", sfreq=125.0),
        strip("ep1", "meegxer.nodes.epochs", events="go,stop", tmin=-0.2, tmax=0.5),
        strip("set", "meegxer.core.set_writer", path=str(out_set)),
        {"id": "epw", "node": "meegxer.core.epochs_writer", "input": "ep1:epochs",
         "params": {"path": str(out_epo)}},
    )  # fmt: skip
    script = tmp_path / "erp_mne.py"
    text = export_mne(pipe, script)
    imports = [line for line in text.splitlines()
               if line.startswith(("import ", "from "))]  # fmt: skip
    assert imports and not any("meegxer" in line for line in imports)
    assert "  ep1          epochs -0.2 to 0.5 s around go, stop" in text
    done = subprocess.run(
        [sys.executable, "-I", str(script), str(recording), "-o",
         str(tmp_path / "end_raw.fif"), "--epochs", str(tmp_path / "all-epo.fif")],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    want = run({**pipe, "nodes": pipe["nodes"][:-2]}, recording)
    got = mne.io.read_raw_fif(tmp_path / "end_raw.fif", verbose="error")
    # through a file: .fif keeps single precision (MNE's default)
    np.testing.assert_allclose(got.get_data(), want.get_data(), rtol=1e-6,
                               atol=1e-12)  # fmt: skip
    epochs = mne.read_epochs(tmp_path / "all-epo.fif", verbose="error")
    want_epochs = run({**pipe, "nodes": pipe["nodes"][:-2]}, recording, output="ep1")
    np.testing.assert_allclose(epochs.get_data(), want_epochs.get_data(),
                               rtol=1e-6, atol=1e-12)  # fmt: skip
    assert out_set.exists() and out_epo.exists()
    assert (tmp_path / "ep-ave.fif").exists()  # the ERPs beside the epochs


def test_a_strip_without_code_is_named(recording: Path) -> None:
    with pytest.raises(NotExportable, match=r"ica1 \(meegxer\.nodes\.ica\).*"
                                            r"meegxer export"):  # fmt: skip
        export_mne(pipeline(strip("ica1", "meegxer.nodes.ica")))
    # an Epochs strip rejecting other words than BAD has no MNE equivalent yet
    with pytest.raises(NotExportable, match=r"ep1 \(meegxer\.nodes\.epochs\)"):
        export_mne(pipeline(strip("ep1", "meegxer.nodes.epochs", reject="BLINK")))


def test_windows_paths_and_quotes_stay_as_written() -> None:
    """A writer's Windows path (``C:\\Users`` would be an escape) and three
    quotes in the strips' list: the script compiles and its docstring shows
    them as written."""
    path = r'C:\Users\me\data\x"""y.set'
    text = export_mne(pipeline(strip("set", "meegxer.core.set_writer", path=path)))
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(text, "run_mne.py", "exec"), scope)
    assert path in scope["__doc__"]
