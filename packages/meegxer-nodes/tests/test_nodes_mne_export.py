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
    with pytest.raises(NotExportable, match=r"b1 \(meegxer\.nodes\.detect_blinks\)"
                                            r".*meegxer export"):  # fmt: skip
        export_mne(pipeline(strip("b1", "meegxer.nodes.detect_blinks")))
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


def test_ica_is_fitted_again_and_stored_in_the_set(tmp_path: Path) -> None:
    """ICA (fitted again in the script), ICLabel and the removal give
    MeegXer's signal; a .set writer after it stores the decomposition, less
    the removed components, as MeegXer's writer does."""
    names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4", "Cz", "Fz",
             "P3", "P4", "Pz", "O1", "O2", "T7"]  # fmt: skip
    rng = np.random.default_rng(11)
    n, sfreq = 7500, 250.0
    t = np.arange(n) / sfreq
    sources = rng.laplace(size=(len(names), n))
    sources[0] = (np.sin(2 * np.pi * 0.3 * t) > 0.95) * 8.0  # blinks
    mixing = rng.standard_normal((len(names), len(names)))
    mixing[:, 0] = [6, 6, 3, 3, 2, 2, 0.5, 0.5, 1, 3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.5]
    info = mne.create_info(names, sfreq, "eeg")
    raw = mne.io.RawArray(mixing @ sources * 5e-6, info, verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    rec = tmp_path / "ica_raw.fif"
    raw.save(rec, verbose="error")
    ours, theirs = tmp_path / "meegxer.set", tmp_path / "exported.set"

    def pipe(path: Path) -> dict[str, Any]:
        return pipeline(
            strip("ica1", "meegxer.nodes.ica", threshold=0.3, max_iter=500),
            strip("set", "meegxer.core.set_writer", path=str(path)),
        )

    want = run(pipe(ours), rec)
    text = export_mne(pipe(theirs))
    assert "pip install mne numpy scipy python-picard mne-icalabel onnxruntime" in text
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(text, "run_mne.py", "exec"), scope)
    got, _ = scope["pipeline"](str(rec))
    np.testing.assert_allclose(got.get_data(), want.get_data(), rtol=1e-9,
                               atol=1e-16)  # fmt: skip
    assert not np.allclose(got.get_data(), raw.get_data())  # components removed
    a, b = (
        mne.preprocessing.read_ica_eeglab(p, verbose="error") for p in (ours, theirs)
    )
    assert a.n_components_ == b.n_components_ < len(names)  # removed ones left out
    np.testing.assert_allclose(b.get_components(), a.get_components(), rtol=1e-6)
    # a re-reference between them: the decomposition unmixes the channels no
    # more, so neither writer stores it
    from scipy.io import loadmat

    def rereferenced(path: Path) -> dict[str, Any]:
        doc = pipe(path)
        doc["nodes"].insert(2, strip("r", "meegxer.nodes.rereference"))
        return doc

    ours, theirs = tmp_path / "meegxer_ref.set", tmp_path / "exported_ref.set"
    run(rereferenced(ours), rec)
    scope = {"__name__": "exported"}
    exec(compile(export_mne(rereferenced(theirs)), "run_mne.py", "exec"), scope)
    scope["pipeline"](str(rec))
    for p in (ours, theirs):
        assert p.exists()
        assert np.size(loadmat(str(p)).get("icaweights", [])) == 0
