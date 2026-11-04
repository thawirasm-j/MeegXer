"""Sterile mode (rule 10, M11.1): every built-in strip runs in the Sterile
worker, and the output is the local run's, exactly (the exit criterion of M11:
"Sterile mode gives the same output as a local run").

The worker runs as a local process (``local=True``, the development
launcher), or in the Sterile image when ``MEEGXER_TEST_STERILE_IMAGE`` names
it (CI's container job builds it with ``meegxer sterile build``): the same
program, in the container (``docs/STERILE.md``)."""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo, Windows
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.node import Sink, TransformResult
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage
from meegxer.core.sterile import sterilize
from meegxer.plugins import sterile
from meegxer.plugins.sterile import SterileError, SterileProcessor, from_wire, to_wire

EEG = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
       "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]  # fmt: skip
SFREQ = 250.0
IMAGE = os.environ.get("MEEGXER_TEST_STERILE_IMAGE", "")
# the worker: the image under test, else a local process
WORKER: dict[str, Any] = {"image": IMAGE} if IMAGE else {"local": True}


@pytest.fixture(autouse=True)
def unsandboxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")  # the local worker


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """40 s of 19 EEG channels (10-20 positions, T7 bad) and an EOG channel,
    with alpha in the left trials, beta in the right ones, and blinks."""
    rng = np.random.default_rng(7)
    n = int(40 * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.standard_normal((len(EEG) + 1, n)) * 5e-6
    onsets = np.arange(2.0, 38.0, 4.0)
    for k, onset in enumerate(onsets):
        span = (t >= onset + 0.5) & (t < onset + 2.5)
        freq = 10.0 if k % 2 == 0 else 22.0
        rows = [8] if k % 2 == 0 else [10]  # C3 / C4
        data[np.ix_(rows, span)] += 8e-6 * np.sin(2 * np.pi * freq * t[span])
        blink = (t >= onset + 3.0) & (t < onset + 3.3)
        data[-1, blink] += 150e-6 * np.hanning(blink.sum())
        data[:2, blink] += 60e-6 * np.hanning(blink.sum())
    info = mne.create_info([*EEG, "EOG"], SFREQ, [*["eeg"] * len(EEG), "eog"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_montage("standard_1020", on_missing="ignore", verbose="error")
    raw.info["bads"] = ["T7"]
    labels = ["left" if k % 2 == 0 else "right" for k in range(len(onsets))]
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), labels))
    path = tmp_path_factory.mktemp("rec") / "rec_raw.fif"
    raw.save(path, verbose="error")
    return path


def tap(node: str) -> dict[str, Any]:
    return {"id": f"tap_{node.replace(':', '_')}", "node": "meegxer.core.probe",
            "input": node}  # fmt: skip


def ftap(node: str) -> dict[str, Any]:
    return {"id": f"ftap_{node.replace(':', '_')}",
            "node": "meegxer.core.feature_probe", "input": node,
            "params": {"keep_windows": 0}}  # fmt: skip


def n(node_id: str, name: str, inp: str | None = None, **params: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": node_id, "node": f"meegxer.nodes.{name}",
                             "params": params}  # fmt: skip
    if inp:
        entry["input"] = inp
    return entry


PIPELINES: dict[str, list[dict[str, Any]]] = {
    "preprocessing": [
        n("iir", "iir_filter", kind="bandpass", low=1.0, high=40.0),
        tap("iir"),
        n("fir", "fir_filter", kind="highpass", low=0.5),
        tap("fir"),
        n("notch", "notch"),
        n("notch0", "notch_zerophase"),
        tap("notch0"),
        n("bads", "interpolate_bads"),
        n("ref", "rereference"),
        tap("ref"),
        n("csd", "csd"),
        tap("csd"),
        n("rs", "resample", sfreq=125.0),
        n("pick", "pick_channels"),
        tap("pick"),
    ],
    "ica": [
        n("hp", "fir_filter_zerophase", kind="highpass", low=1.0),
        n("eog", "eog_regression"),
        tap("eog"),
        n("ica", "ica", n_components=6, max_iter=300),
        tap("ica"),
        tap("ica:ics"),
    ],
    "detectors": [
        n("bp", "iir_filter_zerophase", kind="bandpass", low=1.0, high=40.0),
        n("amp", "detect_amplitude"),
        n("blinks", "detect_blinks"),
        n("muscle", "detect_muscle", low=60.0, high=100.0),
        {"id": "cut", "node": "meegxer.core.reject_segments"},
        tap("cut"),
        n("ep", "epochs", events="left, right", tmin=-0.2, tmax=1.0),
        {"id": "erp", "node": "meegxer.core.erp_probe", "input": "ep:epochs"},
        ftap("ep:epochs"),
        n("env", "hilbert_envelope", "ep", low=8.0),
        tap("env"),
        n("env2", "hilbert_envelope_analytic", "ep"),
        tap("env2"),
    ],
    "features": [
        n("bp", "iir_filter", kind="bandpass", low=1.0, high=45.0),
        n("psd", "psd", "bp"),
        ftap("psd"),
        n("bands", "band_power", "bp", relative=True),
        ftap("bands"),
        n("fooof", "specparam", "bp"),
        ftap("fooof"),
        n("cx", "complexity", "bp", window=4.0, step=4.0),
        ftap("cx"),
        n("conn", "connectivity", "bp"),
        ftap("conn"),
        n("tf", "morlet", "bp", window=2.0, step=2.0, fmin=8.0, fmax=12.0),
        ftap("tf"),
        n("ssvep", "ssvep_cca", "bp", detections=True, subband_high=60.0),
        ftap("ssvep"),
    ],
    "bci": [
        n("bp", "iir_filter_zerophase", kind="bandpass", low=8.0, high=30.0),
        n("csp", "csp", n_components=2),
        ftap("csp"),
        n("lda", "lda", "csp"),
        ftap("lda"),
    ],
}


def build(path: Path, nodes: list[dict[str, Any]]) -> Graph:
    entries = [{"id": "input", "node": "meegxer.core.file",
                "params": {"path": str(path)}}, *nodes]  # fmt: skip
    return Graph.from_dict(
        {"spec_version": SPEC_VERSION, "nodes": entries}, Registry.from_entry_points()
    )


def outputs(graph: Graph) -> dict[str, Any]:
    """What every probe kept (blocks, feature windows, ERP sums): its state."""
    taps: dict[str, Any] = {}
    for node in graph.nodes:
        if node.id.startswith(("tap_", "ftap_", "erp")):
            impl = node.impl
            names = getattr(impl, "__slots__", ()) or vars(impl)
            taps[node.id] = {k: getattr(impl, k) for k in names if k != "params"}
    return taps


def same(a: Any, b: Any, where: str = "") -> None:
    """``a`` and ``b`` exactly equal (arrays bit for bit, nan where nan)."""
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, where
        np.testing.assert_array_equal(a, b, err_msg=where, strict=True)
    elif is_dataclass(a) and not isinstance(a, type):
        assert type(a) is type(b), where
        for f in fields(a):
            same(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), where
        for k in a:
            same(a[k], b[k], f"{where}[{k!r}]")
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), where
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", list(PIPELINES))
def test_sterile_output_is_the_local_output(recording: Path, name: str) -> None:
    """Each pipeline, offline, in blocks of 2 s (feature windows, epochs and
    trials cross the blocks): every tap's content is the same, bit for bit."""
    local = build(recording, PIPELINES[name])
    BatchScheduler(block_size=500).run(local)
    isolated = build(recording, PIPELINES[name])
    with sterilize(isolated, **WORKER) as isolation:
        BatchScheduler(block_size=500).run(isolated)
        proxied = set(isolation.proxied)
    built_in = {node.id for node in isolated.nodes if isinstance(node.impl, Stage)
                and node.impl.manifest.name.startswith("meegxer.nodes.")}  # fmt: skip
    assert proxied == built_in  # every built-in strip ran in the worker
    assert "cut" not in proxied  # a reviewer's marks stay on the host
    got, want = outputs(isolated), outputs(local)
    assert got.keys() == want.keys() and want
    same(got, want, name)


def test_every_built_in_processor_is_covered() -> None:
    """The pipelines above hold every built-in processor (the keyboard is a sink)."""
    used = {e["node"] for nodes in PIPELINES.values() for e in nodes}
    registry = Registry.from_entry_points()
    processors = {
        name for name in registry.names() if name.startswith("meegxer.nodes.")
        and not issubclass(registry.get(name), Sink)
    }  # fmt: skip
    assert processors - used == set()


def test_values_travel_exactly() -> None:
    """The codec keeps dtypes (float64 stays float64; complex and int16 too),
    non-finite floats, tuples, mappings and MeegXer's block types."""
    info = SignalInfo(("Cz", "Pz"), 250.0, "amp", bads=("Pz",),
                      positions=((0.0, 0.1, 0.2), None))  # fmt: skip
    block = Block(np.arange(6.0).reshape(2, 3), info, 10, 1.5)
    coords: dict[str, Any] = {"channel": ("Cz", "Pz"), "f": (1.0, 2.0)}
    window = FeatureBlock(np.ones((2, 2), np.float32), ("channel", "f"), (0, 250),
                          0.0, coords, unit="dB")  # fmt: skip
    value = [
        block,
        window,
        Windows((window,), 250.0, 0, 250),
        (Event(12, "left", 3, {"score": 0.5}),),
        {
            "z": np.array([1 + 2j, 3 - 1j]),
            "i": np.arange(3, dtype=np.int16),
            "nan": float("nan"),
            "inf": float("-inf"),
            "none": None,
            "flag": True,
        },
        TransformResult(block, {"zi": np.zeros((2, 4))}, ()),
    ]
    arrays: dict[str, Any] = {}
    back = from_wire(to_wire(value, arrays), arrays)
    same(back[:4], value[:4])
    assert back[4]["z"].dtype == np.complex128 and back[4]["i"].dtype == np.int16
    np.testing.assert_array_equal(back[4]["z"], value[4]["z"])  # type: ignore[index]
    assert np.isnan(back[4]["nan"]) and back[4]["inf"] == float("-inf")
    assert isinstance(back[5], TransformResult) and back[5].events == ()
    with pytest.raises(TypeError, match="cannot travel"):
        to_wire(object(), {})


def test_a_nodes_error_is_raised_as_locally(recording: Path) -> None:
    """A node's ValueError in the worker is a ValueError on the host, with its
    words (the GUI shows them as they are)."""
    graph = build(recording, [n("lp", "iir_filter", kind="lowpass", high=40.0)])
    with sterilize(graph, **WORKER):
        stage = graph["lp"].impl
        assert isinstance(stage, Stage) and isinstance(
            stage.processor, SterileProcessor
        )
        bad = Block(np.zeros((1, 10)), SignalInfo(("Cz",), 50.0), 0, 0.0)  # 40 > 25 Hz
        with pytest.raises(ValueError) as raised:
            stage.processor.transform(bad, {}, None, stage.params)
    with pytest.raises(ValueError) as local:
        stage.processor.target.transform(bad, {}, None, stage.params)
    assert str(raised.value) == str(local.value)


def test_a_killed_worker_is_restarted(recording: Path) -> None:
    """The host holds model and state: a worker that dies mid-run is started
    again and the output is unchanged (models are sent to it again)."""
    nodes = PIPELINES["bci"]
    local = build(recording, nodes)
    BatchScheduler(block_size=500).run(local)
    isolated = build(recording, nodes)
    with sterilize(isolated, **WORKER) as isolation:
        assert isinstance(isolation, sterile.Isolation)
        worker = isolation.worker
        original = worker.call
        calls = {"n": 0}
        killed: list[int] = []

        def kill_then_call(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            proc = worker._proc
            if calls["n"] == 5 and proc is not None:  # the container dies
                proc.kill()
                proc.wait()
                killed.append(proc.pid)
            return original(*args, **kwargs)

        worker.call = kill_then_call  # type: ignore[method-assign]
        BatchScheduler(block_size=500).run(isolated)
        again = worker._proc
        assert killed and again is not None and again.pid != killed[0]  # a new one
    same(outputs(isolated), outputs(local))


def test_local_code_is_refused(recording: Path) -> None:
    """A custom script or local plugin would run without isolation: refused."""
    from meegxer.plugins.adapter import plugin_class
    from meegxer.plugins.worker import LocalLauncher

    manifest = {"spec_version": SPEC_VERSION, "name": "lab.script", "version": "0.1.0",
                "role": "processor", "type": "stream",
                "inputs": [{"name": "in", "type": "signal"}],
                "outputs": [{"name": "out", "type": "signal"}]}  # fmt: skip
    cls = plugin_class(manifest, LocalLauncher(("python", "script.py")))
    graph = build(recording, [n("iir", "iir_filter")])
    stage = graph["iir"].impl
    assert isinstance(stage, Stage)
    stage.processor = cls()
    with pytest.raises(SterileError, match="without isolation"):
        sterilize(graph, **WORKER)


def test_a_worker_of_another_version_is_refused(
    recording: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image is built from the installed MeegXer: an old one is refused
    with what to do."""
    monkeypatch.setattr(sterile, "__version__", "9.9.9")
    graph = build(recording, [n("iir", "iir_filter")])
    with sterilize(graph, **WORKER), pytest.raises(SterileError, match="build"):
        BatchScheduler().run(graph)


def test_a_sterile_run_saves_the_same_session(recording: Path) -> None:
    """Models stay on the host: the session (with the fitted CSP and LDA) is
    the local run's, and the strips are still recorded as built-ins."""
    from meegxer.core.registry import node_origin
    from meegxer.core.session import build_session

    local = build(recording, PIPELINES["bci"])
    BatchScheduler(block_size=500).run(local)
    isolated = build(recording, PIPELINES["bci"])
    with sterilize(isolated, **WORKER):
        BatchScheduler(block_size=500).run(isolated)
        session = build_session(isolated)
        stage = isolated["csp"].impl
        assert isinstance(stage, Stage)
        assert node_origin(type(stage.processor)) == "builtin"
    want = build_session(local)
    for key in ("environment", "created"):
        session.pop(key, None), want.pop(key, None)
    assert session["models"].keys() == want["models"].keys() == {"csp", "lda"}
    assert session == want
