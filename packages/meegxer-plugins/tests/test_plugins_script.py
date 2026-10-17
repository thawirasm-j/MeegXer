"""Custom scripts: a plain process(raw, ...) function becomes an offline strip."""

import textwrap
from pathlib import Path

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block
from meegxer.core.graph import Graph
from meegxer.core.registry import Registry
from meegxer.plugins import index
from meegxer.plugins.cli import main as cli_main
from meegxer.plugins.script import ScriptError, load_script, script_manifest

SDK = Path(__file__).resolve().parents[1] / "sdk" / "python" / "meegxer_plugin.py"

BANDPASS = '''
    """Band-pass with MNE (a custom script)."""

    from typing import Literal


    def process(raw, low: float = 1.0, high: float = 40.0,
                method: Literal["fir", "iir"] = "iir", mark: bool = True):
        """Band-pass the recording.

        Parameters
        ----------
        raw : mne.io.Raw
            The recording.
        low : float
            Low cut-off (Hz).
        high : float
            High cut-off (Hz).
        method : str
            Filter design.
        mark : bool
            Add a marker at 1 s.

        Returns
        -------
        raw : mne.io.Raw
        """
        raw.filter(low, high, method=method, verbose="error")
        if mark:
            raw.annotations.append(1.0, 0.5, "checked")
        return raw
'''


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_packaged_sdk_is_the_published_sdk() -> None:
    from meegxer.plugins import _sdk

    assert Path(_sdk.__file__).read_bytes() == SDK.read_bytes()


def test_settings_come_from_the_signature_and_docstring(tmp_path: Path) -> None:
    path = write(tmp_path, "My Bandpass.py", BANDPASS)
    manifest = script_manifest(load_script(path), path)
    assert manifest["name"] == "script.my_bandpass"
    assert manifest["availability"] == "offline"
    assert manifest["description"] == "Band-pass with MNE (a custom script)."
    params = {p["name"]: p for p in manifest["params"]}
    assert params["low"] == {"name": "low", "type": "float", "default": 1.0,
                             "unit": "Hz", "description": "Low cut-off"}  # fmt: skip
    assert params["method"]["type"] == "enum"
    assert params["method"]["choices"] == ["fir", "iir"]
    assert params["mark"]["type"] == "bool"
    assert "raw" not in params


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("x = 1\n", "needs one function"),
        ("def process(raw, k):\n    return raw\n", "needs a default"),
        ("def process(raw, k=[1]):\n    return raw\n", "use bool, int, float"),
    ],
)
def test_scripts_that_break_the_conventions_are_explained(
    tmp_path: Path, body: str, message: str
) -> None:
    path = write(tmp_path, "bad.py", body)
    with pytest.raises(ScriptError, match=message):
        script_manifest(load_script(path), path)


def test_a_script_strip_runs_offline_after_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(index.INDEX_ENV, str(tmp_path / "plugins.json"))
    monkeypatch.delenv("MEEGXER_ALLOW_UNSANDBOXED", raising=False)
    script = write(tmp_path, "bandpass.py", BANDPASS)

    # without accepting the notice nothing is installed (stdin is not a terminal)
    assert cli_main(["add", "--script", str(script)]) == 1
    assert "HIPAA" in capsys.readouterr().err and index.load_index() == []
    assert cli_main(["add", "--script", str(script), "--accept-risk"]) == 0
    assert "local, unsandboxed" in capsys.readouterr().out

    pipeline = {"spec_version": "0.2.9", "nodes": [
        {"id": "input", "node": "meegxer.core.synthetic",
         "params": {"duration": 4.0, "n_channels": 3, "sfreq": 200.0}},
        {"id": "bp", "node": "script.bandpass", "params": {"high": 30.0}},
    ]}  # fmt: skip
    graph = Graph.from_dict(pipeline, Registry.from_entry_points())
    result = BatchScheduler().run(graph, keep=["input"])
    graph["bp"].impl.processor.close()  # type: ignore[union-attr]
    raw_in, out = result.outputs["input"], result.outputs["bp"]
    assert isinstance(raw_in, Block) and isinstance(out, Block)

    expected = mne.io.RawArray(
        raw_in.data.copy(), mne.create_info(list(raw_in.info.ch_labels), 200.0, "eeg"),
        verbose="error",
    ).filter(1.0, 30.0, method="iir", verbose="error").get_data()  # fmt: skip
    np.testing.assert_allclose(out.data, expected, atol=1e-9)  # float32 on the wire
    marks = [e for e in result.events.get("bp", []) if e.label == "checked"]
    assert [(m.sample_index, m.duration) for m in marks] == [(200, 100)]

    # what ran is recorded: the command and a hash of the script
    origin = type(graph["bp"].impl.processor).provenance()  # type: ignore[union-attr]
    assert origin["unsandboxed"] and origin["script"] == "bandpass.py"
    assert origin["sha256"] == index.code_sha256(origin["command"])


# -- live scripts: transform(data, state, ...) ------------------------------------

HIGHPASS = '''
    """Causal high-pass (a live custom script)."""

    from scipy import signal


    def transform(data, state, info, cutoff: float = 1.0):
        """High-pass, keeping the filter state between blocks.

        Parameters
        ----------
        cutoff : float
            Cut-off (Hz).
        """
        if state is None:
            sos = signal.butter(2, cutoff, "highpass", fs=info["sfreq"], output="sos")
            state = {"sos": sos, "zi": signal.sosfilt_zi(sos)[:, None, :]
                     * data[:, :1][None]}
        out, zf = signal.sosfilt(state["sos"], data, axis=-1, zi=state["zi"])
        return out, {"sos": state["sos"], "zi": zf}
'''

ZSCORE = '''
    """Standardize with calibration statistics."""


    def fit(raw):
        x = raw.get_data()
        return {"mean": x.mean(axis=1, keepdims=True),
                "std": x.std(axis=1, keepdims=True)}


    def transform(data, state, model, scale: float = 1.0):
        return (data - model["mean"]) / model["std"] * scale, state
'''

PER_BLOCK_DEMEAN = """
    def transform(data, state):  # forgets everything between blocks
        return data - data.mean(axis=1, keepdims=True), state
"""


def script_class(path: Path) -> type:
    from meegxer.plugins.adapter import plugin_class, probe_manifest
    from meegxer.plugins.worker import LocalLauncher

    launcher = LocalLauncher(tuple(index.script_command(path)), allow=True)
    return plugin_class(probe_manifest(launcher), launcher)


def test_live_script_settings_skip_state_info_and_model(tmp_path: Path) -> None:
    path = write(tmp_path, "hp.py", HIGHPASS)
    manifest = script_manifest(load_script(path), path)
    assert manifest["availability"] == "both" and "requires_fit" not in manifest
    assert [p["name"] for p in manifest["params"]] == ["cutoff"]
    path = write(tmp_path, "z.py", ZSCORE)
    manifest = script_manifest(load_script(path), path)
    assert manifest["requires_fit"] is True
    assert [p["name"] for p in manifest["params"]] == ["scale"]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("def process(raw):\n    return raw\n\ndef transform(d, state):\n"
         "    return d, state\n", "it has process and transform"),
        ("def transform(data, k=1.0):\n    return data\n", "needs a state"),
        ("def fit(raw):\n    return {}\n\ndef process(raw):\n    return raw\n",
         "fit goes with transform"),
    ],
)  # fmt: skip
def test_live_script_conventions_are_explained(
    tmp_path: Path, body: str, message: str
) -> None:
    path = write(tmp_path, "bad.py", body)
    with pytest.raises(ScriptError, match=message):
        script_manifest(load_script(path), path)


def test_a_live_script_runs_the_same_in_both_schedulers(tmp_path: Path) -> None:
    from scipy import signal

    from meegxer.core.probes import MemoryProbe
    from meegxer.core.realtime import RealtimeScheduler
    from meegxer.core.sources import SyntheticSource
    from meegxer.core.stage import Stage

    cls = script_class(write(tmp_path, "hp.py", HIGHPASS))

    def graph(probe: MemoryProbe) -> Graph:
        src = SyntheticSource({"duration": 2.0, "n_channels": 3, "sfreq": 200.0})
        return Graph.chain([src, Stage(cls(), {"cutoff": 2.0}), probe])

    batch_probe, live_probe, raw_probe = MemoryProbe(), MemoryProbe(), MemoryProbe()
    batch = graph(batch_probe)
    BatchScheduler(block_size=37).run(batch)
    live = graph(live_probe)
    scheduler = RealtimeScheduler(live)
    scheduler.start()
    assert scheduler.wait(20)
    assert scheduler.status()["errors"] == []
    BatchScheduler().run(Graph.chain([
        SyntheticSource({"duration": 2.0, "n_channels": 3, "sfreq": 200.0}), raw_probe
    ]))  # fmt: skip
    x = raw_probe.signal().data
    sos = signal.butter(2, 2.0, "highpass", fs=200.0, output="sos")
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[:, :1][None]
    expected, _ = signal.sosfilt(sos, x, axis=-1, zi=zi)
    for probe in (batch_probe, live_probe):
        np.testing.assert_allclose(probe.signal().data, expected, rtol=1e-5, atol=1e-9)
    for g in (batch, live):
        g.nodes[1].impl.processor.close()  # type: ignore[union-attr]


def test_a_calibrated_live_script_fits_on_the_recording(tmp_path: Path) -> None:
    from meegxer.core.sources import SyntheticSource
    from meegxer.core.stage import Stage

    cls = script_class(write(tmp_path, "z.py", ZSCORE))
    stage = Stage(cls(), {"scale": 1.0})
    graph = Graph.chain([
        SyntheticSource({"duration": 4.0, "n_channels": 3, "sfreq": 200.0}), stage
    ])  # fmt: skip
    result = BatchScheduler().run(graph)
    out = result.outputs[graph.nodes[1].id]
    assert isinstance(out, Block) and result.fitted == [graph.nodes[1].id]
    np.testing.assert_allclose(out.data.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(out.data.std(axis=1), 1.0, rtol=1e-3)
    stage.processor.close()


LABELLED = '''
    """Learns from labelled trials."""

    FIT_EVENTS = True


    def fit(raw):
        notes = raw.annotations
        return {"labels": sorted(set(notes.description)), "n": len(notes),
                "first": float(notes.onset[0])}


    def transform(data, state, model):
        return data, state
'''


def test_a_script_fits_on_labelled_trials(tmp_path: Path) -> None:
    """FIT_EVENTS: the manifest says fit_events, and fit(raw) finds the
    calibration's events in raw.annotations, timed from the data's start."""
    from meegxer.core.registry import Registry
    from meegxer.core.stage import Stage

    path = write(tmp_path, "labelled.py", LABELLED)
    assert script_manifest(load_script(path), path)["fit_events"] is True
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.5\t0\tleft\n"
                      "3.0\t0\tright\n4.5\t0\tleft\n")  # fmt: skip
    cls = script_class(path)
    registry = Registry([*Registry.from_entry_points().as_mapping().values(), cls])
    graph = Graph.from_dict({"spec_version": "0.2.16", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 6.0, "n_channels": 3, "sfreq": 200.0},
         "recording": {"events": {"file": str(events)}}},
        {"id": "s", "node": "script.labelled"},
    ]}, registry)  # fmt: skip
    result = BatchScheduler().run(graph)
    impl = graph["s"].impl
    assert isinstance(impl, Stage) and result.fitted == ["s"]
    assert impl.model is not None
    assert impl.model.data == {"labels": ["left", "right"], "n": 3, "first": 1.5}
    impl.processor.close()
    # the conformance suite gives such a script labelled trials to fit on
    assert cli_main(["check", "--script", str(path)]) == 0


def test_a_live_script_that_is_not_blockwise_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(index.INDEX_ENV, str(tmp_path / "plugins.json"))
    script = write(tmp_path, "demean.py", PER_BLOCK_DEMEAN)
    assert cli_main(["add", "--script", str(script), "--accept-risk"]) == 1
    err = capsys.readouterr().err
    assert "FAIL  acceptance test 1" in err and "was not added" in err
    assert index.load_index() == []


# -- credit -------------------------------------------------------------------------

CREDITED = '''
    """Notch with credit."""

    __author__ = ["Jane Doe", "TrimLAB"]
    __license__ = "BSD-3-Clause"
    REFERENCES = [
        "Doe J (2025). A better notch. J Neural Eng 1:1.",
        {"citation": "Gramfort A et al. (2013). MNE-Python. Front Neurosci 7:267.",
         "doi": "https://doi.org/10.3389/fnins.2013.00267"},
    ]


    def process(raw):
        return raw
'''


def test_scripts_credit_their_authors_and_references(tmp_path: Path) -> None:
    path = write(tmp_path, "notch.py", CREDITED)
    manifest = script_manifest(load_script(path), path)
    assert manifest["authors"] == ["Jane Doe", "TrimLAB"]
    assert manifest["license"] == "BSD-3-Clause"
    assert manifest["references"] == [
        {"citation": "Doe J (2025). A better notch. J Neural Eng 1:1."},
        {"citation": "Gramfort A et al. (2013). MNE-Python. Front Neurosci 7:267.",
         "doi": "10.3389/fnins.2013.00267"},  # the URL prefix is dropped
    ]  # fmt: skip
    from meegxer.core.manifest import Manifest

    assert Manifest.from_dict(manifest).references[1].doi.startswith("10.3389")
    bad = write(
        tmp_path, "bad.py", "REFERENCES = [42]\ndef process(raw):\n    return raw\n"
    )
    with pytest.raises(ScriptError, match="REFERENCES"):
        script_manifest(load_script(bad), bad)


def test_the_check_mentions_missing_credit_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = write(tmp_path, "plain.py", "def process(raw):\n    return raw\n")
    assert cli_main(["check", "--script", str(plain)]) == 0
    assert "PASS  credits" in capsys.readouterr().out
    credited = write(tmp_path, "credited.py", CREDITED)
    assert cli_main(["check", "--script", str(credited)]) == 0
    assert "2 author(s), 2 reference(s)" in capsys.readouterr().out


# -- feature extraction: features(data, ...) ---------------------------------------

BANDPOWER = '''
    """Band power per channel (a feature-extraction script)."""

    import numpy as np
    from scipy import signal


    def features(data, info, fmin: float = 8.0, fmax: float = 13.0):
        """Mean power in a band and the line length.

        Parameters
        ----------
        fmin : float
            Lower band edge (Hz).
        fmax : float
            Upper band edge (Hz).
        """
        f, p = signal.welch(data, info["sfreq"], nperseg=min(128, data.shape[1]))
        band = (f >= fmin) & (f <= fmax)
        return {"power": p[:, band].mean(axis=1),
                "line_length": np.abs(np.diff(data, axis=1)).sum(axis=1)}
'''


def test_feature_script_settings_include_the_windowing(tmp_path: Path) -> None:
    path = write(tmp_path, "bp.py", BANDPOWER)
    manifest = script_manifest(load_script(path), path)
    assert manifest["type"] == "analysis" and manifest["availability"] == "both"
    assert manifest["outputs"] == [{"name": "out", "type": "features"}]
    params = {p["name"]: p for p in manifest["params"]}
    assert list(params) == ["fmin", "fmax", "window", "step"]
    assert params["window"]["default"] == 2.0 and params["window"]["unit"] == "s"
    assert params["step"]["default"] == 0.5

    own = "def features(data, window: float = 4.0):\n    return data.mean(axis=1)\n"
    path = write(tmp_path, "own.py", own)
    params = {p["name"]: p for p in script_manifest(load_script(path), path)["params"]}
    assert params["window"]["default"] == 4.0 and params["window"]["unit"] == "s"
    assert params["window"]["description"] == "Length of each analysis window"
    assert params["step"]["default"] == 0.5


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("def features(data, state):\n    return {}\n", "keeps no state"),
        ("def fit(raw):\n    return {}\n\ndef features(data):\n    return {}\n",
         "fit goes with transform"),
        ("def features(data):\n    return {}\n\ndef transform(d, state):\n"
         "    return d, state\n", "it has transform and features"),
    ],
)  # fmt: skip
def test_feature_script_conventions_are_explained(
    tmp_path: Path, body: str, message: str
) -> None:
    path = write(tmp_path, "bad.py", body)
    with pytest.raises(ScriptError, match=message):
        script_manifest(load_script(path), path)


def test_a_feature_script_gives_the_same_windows_however_the_stream_is_cut(
    tmp_path: Path,
) -> None:
    from scipy import signal

    from meegxer.core.block import FeatureBlock
    from meegxer.core.conformance import check_blockwise_equivalence
    from meegxer.core.node import output_blocks
    from meegxer.core.sources import SyntheticSource
    from meegxer.core.stage import Stage, signal_fingerprint

    cls = script_class(write(tmp_path, "bp.py", BANDPOWER))
    x = SyntheticSource({"duration": 4.0, "n_channels": 3, "sfreq": 200.0}).read(10**6)
    assert x is not None
    params = {"fmin": 8.0, "fmax": 13.0, "window": 1.0, "step": 0.5}
    stage = Stage(cls(), params)
    out = stage.transform(x, signal_fingerprint(x.info)).block
    stage.processor.close()
    windows = [w for w in output_blocks(out) if isinstance(w, FeatureBlock)]
    assert [w.window for w in windows] == [(i, i + 200) for i in range(0, 601, 100)]
    first = windows[0]
    assert first.dims == ("channel", "feature")
    assert first.coords == {"channel": x.info.ch_labels,
                            "feature": ("power", "line_length")}  # fmt: skip
    f, p = signal.welch(x.data[:, :200], 200.0, nperseg=128)
    band = (f >= 8.0) & (f <= 13.0)
    np.testing.assert_allclose(first.data[:, 0], p[:, band].mean(axis=1), rtol=1e-6)

    processor = cls()
    try:
        worst = check_blockwise_equivalence(processor, x, params, trials=3)
    finally:
        processor.close()
    assert worst < 1e-9


def test_a_feature_script_that_returns_the_wrong_shape_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(index.INDEX_ENV, str(tmp_path / "plugins.json"))
    bad = "def features(data):\n    return {'mean': 1.0}\n"
    assert cli_main(["check", "--script", str(write(tmp_path, "bad.py", bad))]) == 1
    assert "one value per channel" in capsys.readouterr().out
    good = write(tmp_path, "bp.py", BANDPOWER)
    assert cli_main(["check", "--script", str(good)]) == 0
    report = capsys.readouterr().out
    assert "PASS  transform" in report and "feature windows" in report
    assert "PASS  restart with host-held state" in report
    assert cli_main(["add", "--script", str(good), "--accept-risk"]) == 0
    assert [e["manifest"]["name"] for e in index.load_index()] == ["script.bp"]


def test_a_script_chooses_its_menu_category(tmp_path: Path) -> None:
    body = 'CATEGORY = "Artifacts"\n\ndef process(raw):\n    return raw\n'
    path = write(tmp_path, "clean.py", body)
    manifest = script_manifest(load_script(path), path)
    assert manifest["category"] == "artifacts"
    from meegxer.core.manifest import Manifest

    assert Manifest.from_dict(manifest).category == "artifacts"  # valid per spec
    spaced = write(tmp_path, "tf.py", 'CATEGORY = "time frequency"\n'
                   "def features(data):\n    return data.mean(axis=1)\n")  # fmt: skip
    assert script_manifest(load_script(spaced), spaced)["category"] == "time-frequency"
    bad = write(
        tmp_path, "bad.py", 'CATEGORY = "!!"\ndef process(raw):\n    return raw\n'
    )
    with pytest.raises(ScriptError, match="CATEGORY"):
        script_manifest(load_script(bad), bad)
