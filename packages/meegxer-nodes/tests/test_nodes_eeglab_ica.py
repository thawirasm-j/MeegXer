"""An EEGLAB file's own decomposition (M10.6): a .set carrying EEGLAB's
icaweights starts the ICs rail at the File input, with the file's ICLabel
classes, else ICLabel's computed by the ICA strip (reached through the node
registry); an ICA strip after the input rewrites the rail. The .set writer stores the
decomposition that unmixes its data (less the components a strip removed, as
pop_subcomp). The decomposition and classes are EEGLAB's reference ones
(tools/make_m6_ica_references.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block
from meegxer.core.eeglab_ica import (
    CLASSES,
    Decomposition,
    read_decomposition,
    write_decomposition,
)
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.ports import PortTypeError
from meegxer.core.probes import MemoryProbe
from meegxer.core.recording import describe
from meegxer.core.registry import Registry
from meegxer.core.sinks import SetSink
from meegxer.core.sources import FileSource
from meegxer.core.stage import Stage
from meegxer.nodes.ica import rejected

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"
REC = np.load(DATA / "m6_ica.npz")
REF = np.load(DATA / "m6_ref_ica.npz")
LABELS = tuple(str(s) for s in REC["labels"])
W = REF["icaweights"]  # EEGLAB's (18 x 19), after eeg_checkset's RMS scaling


def export(path: Path) -> Path:
    """The recording as a .set, without a decomposition."""
    info = mne.create_info(list(LABELS), float(REC["sfreq"]), "eeg")
    raw = mne.io.RawArray(REC["data"].astype(np.float64), info, verbose="error")
    raw.set_montage(mne.channels.make_dig_montage(
        dict(zip(LABELS, REC["positions"], strict=True)), coord_frame="head"),
        verbose="error")  # fmt: skip
    raw.export(path, fmt="eeglab", verbose="error")
    return path


def write_set(path: Path, classes: bool) -> Path:
    export(path)
    proba = REF["probabilities"] if classes else None
    write_decomposition(path, Decomposition(LABELS, W, np.linalg.pinv(W), proba),
                        LABELS)  # fmt: skip
    return path


def test_the_file_reads_back_as_eeglab_wrote_it(tmp_path: Path) -> None:
    dec = read_decomposition(write_set(tmp_path / "rec.set", classes=True))
    assert dec is not None and dec.channels == LABELS
    np.testing.assert_allclose(dec.weights, W, rtol=1e-6, atol=1e-12)
    assert dec.probabilities is not None and dec.classified_by == "the file"
    np.testing.assert_allclose(dec.probabilities, REF["probabilities"])
    plain = tmp_path / "plain.set"
    two = mne.create_info(["Cz", "Pz"], 100.0, "eeg")
    mne.io.RawArray(np.zeros((2, 100)), two, verbose="error").export(
        plain, fmt="eeglab", verbose="error")  # fmt: skip
    assert read_decomposition(plain) is None  # no icaweights: no decomposition


def run_tap(path: Path) -> tuple[FileSource, Block]:
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "tap", "node": "meegxer.core.probe", "input": "input:ics",
         "params": {"keep_seconds": 0}},
    ]}, Registry.from_entry_points())  # fmt: skip
    BatchScheduler(block_size=1000).run(graph)
    probe = graph["tap"].impl
    source = graph["input"].impl
    assert isinstance(probe, MemoryProbe) and isinstance(source, FileSource)
    return source, probe.signal()


def test_the_input_starts_the_ics_rail_with_the_files_classes(tmp_path: Path) -> None:
    source, ics = run_tap(write_set(tmp_path / "rec.set", classes=True))
    assert ics.info.ch_labels[:2] == ("IC01", "IC02") and ics.info.n_channels == 18
    x = REC["data"].astype(np.float64)
    np.testing.assert_allclose(ics.data, W @ x, rtol=1e-4, atol=1e-9)  # single
    best = REF["probabilities"].argmax(axis=1)
    assert ics.info.ch_classes == tuple(CLASSES[b] for b in best)
    notes = describe(source)["notes"]
    assert notes[0] == ("EEGLAB decomposition: 18 components on 19 channels start "
                        "the ICs rail; ICLabel classes from the file")  # fmt: skip
    model: dict[str, Any] | None = source.ica_model()  # for the IC head plot
    assert model is not None and model["channels"] == list(LABELS)


def test_without_classes_iclabel_is_computed(tmp_path: Path) -> None:
    source, ics = run_tap(write_set(tmp_path / "rec.set", classes=False))
    dec = source.decomposition()
    assert dec is not None and dec.classified_by == "ICLabel (computed by MeegXer)"
    assert dec.probabilities is not None
    # EEGLAB's ICLabel on the same decomposition (the ICA strip matches it to
    # 1e-4 on the same data; here through a .set: single precision, its
    # chanlocs)
    assert np.abs(dec.probabilities - REF["probabilities"]).max() < 0.02
    assert ics.info.ch_classes  # the rail carries them


def test_an_ica_strip_after_the_input_rewrites_the_rail(tmp_path: Path) -> None:
    path = write_set(tmp_path / "rec.set", classes=True)
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "ica1", "node": "meegxer.nodes.ica"},
        {"id": "psd1", "node": "meegxer.nodes.psd", "input": "ica1:ics"},
    ]}, Registry.from_entry_points())  # fmt: skip
    assert graph["psd1"].input == "ica1:ics"
    early = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "psd1", "node": "meegxer.nodes.psd", "input": "input:ics"},
    ]}, Registry.from_entry_points())  # fmt: skip
    assert early["psd1"].input == "input:ics"


def write_through(tmp_path: Path, path: Path, *strips: dict[str, Any]) -> Any:
    out = tmp_path / "out.set"
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        *strips,
        {"id": "set", "node": "meegxer.core.set_writer",
         "params": {"path": str(out)}},
    ]}, Registry.from_entry_points())  # fmt: skip
    return graph, BatchScheduler(block_size=1000).run(graph), out


def test_the_set_writer_keeps_the_files_decomposition(tmp_path: Path) -> None:
    _, result, out = write_through(tmp_path, write_set(tmp_path / "rec.set", True))
    dec = read_decomposition(out)
    assert dec is not None and dec.channels == LABELS
    np.testing.assert_allclose(dec.weights, W, rtol=1e-5, atol=1e-12)
    assert dec.classified_by == "the file" and dec.probabilities is not None
    np.testing.assert_allclose(dec.probabilities, REF["probabilities"])
    assert ("set: stored the ICA decomposition (18 components, with its ICLabel "
            "classes)") in result.notes  # fmt: skip
    assert out.read_bytes()[:116] == SetSink.MAT_HEADER  # the same data, the same file


def test_after_the_ica_strip_the_removed_components_are_dropped(tmp_path: Path) -> None:
    plain = export(tmp_path / "rec.set")
    graph, result, out = write_through(tmp_path, plain, {"id": "ica1",
                                       "node": "meegxer.nodes.ica"})  # fmt: skip
    stage = graph["ica1"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    model = stage.model.data
    removed = rejected(model, stage.params)
    assert removed  # the eye and muscle components
    keep = [k for k in range(np.asarray(model["weights"]).shape[0]) if k not in removed]
    dec = read_decomposition(out)  # MNE's reader: icawinv, as after pop_subcomp
    assert dec is not None and dec.n_components == len(keep)
    np.testing.assert_allclose(dec.winv, np.asarray(model["winv"])[:, keep],
                               rtol=1e-4, atol=1e-9)  # fmt: skip
    assert dec.probabilities is not None
    np.testing.assert_allclose(dec.probabilities,
                               np.asarray(model["probabilities"])[keep])  # fmt: skip
    ica = mne.preprocessing.read_ica_eeglab(str(out), verbose="error")
    assert ica.n_components_ == len(keep)
    assert any("stored the ICA decomposition" in n for n in result.notes)


def test_a_spatial_strip_after_it_leaves_the_decomposition_out(tmp_path: Path) -> None:
    _, result, out = write_through(
        tmp_path, write_set(tmp_path / "rec.set", True),
        {"id": "reref1", "node": "meegxer.nodes.rereference"},
    )  # fmt: skip
    assert read_decomposition(out) is None
    assert ("set: ICA decomposition not stored: reref1 transforms the channels "
            "after it") in result.skipped  # fmt: skip


@pytest.mark.parametrize("node", ["meegxer.nodes.rereference", "meegxer.nodes.ica",
                                  "meegxer.nodes.csd", "meegxer.nodes.interpolate_bads",
                                  "meegxer.nodes.eog_regression",
                                  "meegxer.core.ica_writer"])  # fmt: skip
def test_electrode_only_strips_refuse_the_ics(tmp_path: Path, node: str) -> None:
    """Spec 0.2.22: strips that only make sense on electrodes (and the MNE ICA
    writer, which stores the decomposition of electrodes) take sensor space,
    so a pipeline wiring the ICs into them is refused; a spectrum takes them."""
    path = write_set(tmp_path / "rec.set", classes=True)
    params = {"path": str(tmp_path / "out-ica.fif")} if "writer" in node else {}
    nodes: list[dict[str, Any]] = [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "x", "node": node, "input": "input:ics", "params": params},
    ]
    with pytest.raises(PortTypeError, match="takes sensor space"):
        Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes},
                        Registry.from_entry_points())  # fmt: skip
    nodes[1] = {"id": "x", "node": "meegxer.nodes.psd", "input": "input:ics"}
    Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes},
                    Registry.from_entry_points())  # fmt: skip
