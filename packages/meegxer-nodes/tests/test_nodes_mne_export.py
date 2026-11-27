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


def exported_all(pipe: dict[str, Any], path: Path) -> tuple[Any, Any, Any]:
    """The exported script's pipeline(path): (raw, epochs, features)."""
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(export_mne(pipe), "run_mne.py", "exec"), scope)
    return scope["pipeline"](str(path))  # type: ignore[no-any-return]


def exported(pipe: dict[str, Any], path: Path) -> tuple[Any, Any]:
    """The exported script's signal and epochs."""
    raw, epochs, _ = exported_all(pipe, path)
    return raw, epochs


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
    "hilbert envelope": [strip("h", "meegxer.nodes.hilbert_envelope")],
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
    with pytest.raises(NotExportable, match=r"e1 \(meegxer\.nodes\.eog_regression\)"
                                            r".*meegxer export"):  # fmt: skip
        export_mne(pipeline(strip("e1", "meegxer.nodes.eog_regression")))
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
    needs = next(line for line in text.splitlines() if "pip install" in line)
    for package in ("mne", "scipy", "python-picard", "mne-icalabel", "onnxruntime"):
        assert f" {package}" in needs
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(text, "run_mne.py", "exec"), scope)
    got, _, _ = scope["pipeline"](str(rec))
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


@pytest.fixture
def artefacts(tmp_path: Path) -> Path:
    """30 s at 500 Hz: jumps, a flat stretch, muscle bursts and blinks on an
    EOG channel, "go" events."""
    rng = np.random.default_rng(21)
    sfreq, n = 500.0, 15000
    names = ["Fz", "Cz", "Pz", "C3", "C4", "EOG1"]
    x = rng.standard_normal((6, n)) * 5e-6
    x[1, 4000] += 400e-6  # a jump
    x[2, 9000:9600] = 0.0  # flat
    for start in (6000, 21 * 500):  # muscle: 120 Hz bursts
        t = np.arange(400) / sfreq
        x[:5, start : start + 400] += 60e-6 * np.sin(2 * np.pi * 120 * t)
    for peak in (1500, 4500, 7500, 10500, 13500):  # blinks
        x[5, peak - 100 : peak + 100] += 300e-6 * np.hanning(200)
    info = mne.create_info(names, sfreq, ["eeg"] * 5 + ["eog"])
    raw = mne.io.RawArray(x, info, verbose="error")
    raw.set_annotations(mne.Annotations([3, 8, 13, 18, 23], [0] * 5, ["go"] * 5))
    path = tmp_path / "art_raw.fif"
    raw.save(path, verbose="error")
    return path


def events(raw: Any) -> list[tuple[float, float, str]]:
    return sorted((round(a["onset"], 6), round(a["duration"], 6), a["description"])
                  for a in raw.annotations)  # fmt: skip


@pytest.mark.parametrize(
    "detector",
    [strip("d", "meegxer.nodes.detect_amplitude", peak=150.0, bad_percent=1.0),
     strip("d", "meegxer.nodes.detect_amplitude", peak=0.0, flat=0.5),
     strip("d", "meegxer.nodes.detect_muscle", zscore=3.0),
     strip("d", "meegxer.nodes.detect_blinks", mark_label="BAD_blink")],
    ids=["jumps", "flat", "muscle", "blinks"],
)  # fmt: skip
def test_each_detector_s_events_are_meegxer_s(
    artefacts: Path, detector: dict[str, Any]
) -> None:
    """The events a detector adds, and the epochs a BAD-rejecting Epochs
    strip keeps after it, as MeegXer's run gives them."""
    epochs = strip("ep1", "meegxer.nodes.epochs", events="go", tmin=-0.2,
                   tmax=0.8, reject="BAD")  # fmt: skip
    pipe = pipeline(detector, epochs)
    want = run(pipe, artefacts)
    got, got_epochs = exported(pipe, artefacts)
    assert len(events(got)) > 5  # it found something
    assert events(got) == events(want)
    np.testing.assert_array_equal(got.get_data(), want.get_data())  # unchanged
    want_epochs = run(pipe, artefacts, output="ep1")
    assert len(got_epochs) == len(want_epochs)


FEATURES = {
    "psd welch": strip("f1", "meegxer.nodes.psd"),
    "psd multitaper in dB": strip("f1", "meegxer.nodes.psd", method="multitaper",
                                  db=True, fmax=30.0, window=4.0, step=2.0),
    "relative band power": strip("f1", "meegxer.nodes.band_power", relative=True,
                                 taper="hann", segment=0.5),
    "morlet": strip("f1", "meegxer.nodes.morlet", fmin=8.0, fmax=20.0, fstep=2.0,
                    db=True),
    "specparam": strip("f1", "meegxer.nodes.specparam", fmax=30.0),
    "connectivity": strip("f1", "meegxer.nodes.connectivity", method="coh"),
    "imaginary coherence": strip("f1", "meegxer.nodes.connectivity",
                                 method="imcoh", window=3.0, segment=0.75),
    "complexity": strip("f1", "meegxer.nodes.complexity", window=4.0, step=2.0),
    "ssvep fbcca": strip("f1", "meegxer.nodes.ssvep_cca", detections=True),
    "ssvep cca": strip("f1", "meegxer.nodes.ssvep_cca", method="cca",
                       detections=True, min_score=0.3, event_prefix="hit_"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_each_feature_strip_writes_meegxer_s_tsv(
    recording: Path, tmp_path: Path, name: str
) -> None:
    """A feature strip and a feature TSV writer: the exported script's file
    (and its units sidecar) is MeegXer's, byte for byte; pipeline(path) gives
    the windows as arrays."""

    def pipe(out: Path) -> dict[str, Any]:
        return pipeline(
            strip("hp", "meegxer.nodes.iir_filter", low=1.0, high=40.0),
            FEATURES[name],
            strip("tsv", "meegxer.core.feature_tsv", path=str(out)),
            strip("r", "meegxer.nodes.rereference"),  # the chain goes on
        )

    ours, theirs = tmp_path / "meegxer.tsv", tmp_path / "exported.tsv"
    want = run(pipe(ours), recording)
    got, _, features = exported_all(pipe(theirs), recording)
    assert theirs.read_text() == ours.read_text()
    assert (
        theirs.with_suffix(".json").read_text() == ours.with_suffix(".json").read_text()
    )
    feats = features["f1"]
    rows = ours.read_text().splitlines()
    assert len(feats["windows"]) == len(rows) - 1 > 3
    if feats["dims"][0] == "channel":
        assert feats["values"].shape[1] == 6  # the EEG channels, bad Oz kept
    assert events(got) == events(want)  # detection events too
    np.testing.assert_allclose(got.get_data(), want.get_data(), rtol=1e-9,
                               atol=1e-16)  # fmt: skip


def test_strips_with_settings_not_exported_yet_are_named() -> None:
    for node, params in (
        ("meegxer.nodes.specparam", {"fit_unfiltered": True}),
        ("meegxer.nodes.hilbert_envelope", {"method": "fir"}),
    ):
        with pytest.raises(NotExportable, match=r"s1 \(" + node.replace(".", r"\.")):
            export_mne(pipeline(strip("s1", node, **params)))


@pytest.fixture
def motor(tmp_path: Path) -> Path:
    """60 s at 250 Hz: "left" and "right" trials (every 4 s, alternating),
    each with 10 Hz power on the opposite side, and noise."""
    rng = np.random.default_rng(31)
    sfreq, n = 250.0, 15000
    names = ["C3", "C4", "Cz", "Fz", "Pz", "Oz"]
    x = rng.standard_normal((6, n)) * 2e-6
    onsets, labels = [], []
    for k, start in enumerate(range(500, n - 1000, 1000)):
        label = "left" if k % 2 == 0 else "right"
        t = np.arange(500) / sfreq
        x[1 if label == "left" else 0, start + 125 : start + 625] += 8e-6 * np.sin(
            2 * np.pi * 10 * t
        )
        onsets.append(start / sfreq)
        labels.append(label)
    raw = mne.io.RawArray(x, mne.create_info(names, sfreq, "eeg"), verbose="error")
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), labels))
    path = tmp_path / "motor_raw.fif"
    raw.save(path, verbose="error")
    return path


def test_csp_and_lda_are_fitted_again_as_meegxer_fits_them(
    motor: Path, tmp_path: Path
) -> None:
    """CSP (fitted on the trials) and LDA (on the CSP windows in the trials):
    the windows, the class probabilities and the pred_ events are MeegXer's."""

    def pipe(out: Path) -> dict[str, Any]:
        return pipeline(
            strip("bp", "meegxer.nodes.iir_filter", low=8.0, high=30.0),
            strip("csp1", "meegxer.nodes.csp", window=1.0, step=0.25,
                  n_components=2),
            {"id": "lda1", "node": "meegxer.nodes.lda", "input": "csp1",
             "params": {"tmin": 0.5, "tmax": 2.5}},
            strip("tsv", "meegxer.core.feature_tsv", path=str(out)),
        )  # fmt: skip

    ours, theirs = tmp_path / "meegxer.tsv", tmp_path / "exported.tsv"
    want = run(pipe(ours), motor)
    text = export_mne(pipe(theirs))
    assert "scikit-learn" in text and "fitted again on the recording" in text
    got, _, features = exported_all(pipe(theirs), motor)
    assert features["csp1"]["values"].shape[1:] == (2,)
    a, b = (np.loadtxt(p, delimiter="\t", skiprows=1) for p in (ours, theirs))
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)
    assert theirs.read_text().splitlines()[0] == ours.read_text().splitlines()[0]
    preds = [e for e in events(got) if e[2].startswith("pred_")]
    assert len(preds) > 100 and preds == [e for e in events(want)
                                          if e[2].startswith("pred_")]  # fmt: skip


def test_a_set_input_s_own_ica_is_stored_again(tmp_path: Path) -> None:
    """A .set recording carrying an ICA (no classes: ICLabel runs), a filter,
    a .set writer: the decomposition and its classes stored as MeegXer's."""
    from scipy.io import loadmat

    from meegxer.core.eeglab_ica import Decomposition, write_decomposition

    names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4", "Cz", "Fz",
             "P3", "P4", "Pz", "O1", "O2", "T7"]  # fmt: skip
    rng = np.random.default_rng(41)
    info = mne.create_info(names, 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((16, 5000)) * 1e-5, info, verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    rec = tmp_path / "rec.set"
    raw.export(rec, fmt="eeglab", verbose="error")
    w = np.linalg.qr(rng.standard_normal((16, 16)))[0] * 1e5
    write_decomposition(rec, Decomposition(tuple(names), w, np.linalg.inv(w)),
                        tuple(names))  # fmt: skip
    ours, theirs = tmp_path / "meegxer.set", tmp_path / "exported.set"

    def pipe(out: Path) -> dict[str, Any]:
        return pipeline(
            strip("f", "meegxer.nodes.iir_filter", low=1.0),
            strip("set", "meegxer.core.set_writer", path=str(out)),
        )

    run(pipe(ours), rec)
    scope: dict[str, Any] = {"__name__": "exported"}
    exec(compile(export_mne(pipe(theirs)), "run_mne.py", "exec"), scope)
    scope["pipeline"](str(rec))
    a, b = (
        mne.preprocessing.read_ica_eeglab(p, verbose="error") for p in (ours, theirs)
    )
    np.testing.assert_allclose(b.get_components(), a.get_components(), rtol=1e-6)

    def classes(p: Path) -> Any:
        eeg = loadmat(str(p), squeeze_me=True, struct_as_record=False)
        return eeg["etc"].ic_classification.ICLabel.classification

    np.testing.assert_allclose(classes(theirs), classes(ours), rtol=1e-6)
    # without a .set writer, nothing reads it
    assert "input_ica" not in export_mne(
        pipeline(strip("f", "meegxer.nodes.iir_filter"))
    )
