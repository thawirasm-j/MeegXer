"""MNE ICA files on the ICs rail (M10.7b, docs/RAILS.md): an ``-ica.fif``
given to the File input (``ica``) or to ``meegxer.api.run`` (``ica=``)
starts the ICs rail with MNE's own sources; the *MNE ICA (.fif)* writer
stores the rail's decomposition as MNE reads it back (``read_ica``), with the
components an ICA strip removed marked ``exclude``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.api import run
from meegxer.core.batch import BatchScheduler
from meegxer.core.eeglab_ica import Decomposition, write_decomposition
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.mne_ica import ICLABEL_NAMES, IcaWriter
from meegxer.core.recording import describe
from meegxer.core.registry import Registry
from meegxer.core.session import share_safe
from meegxer.core.sources import FileSource
from meegxer.core.stage import Stage
from meegxer.nodes.ica import rejected

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"
REC = np.load(DATA / "m6_ica.npz")
REF = np.load(DATA / "m6_ref_ica.npz")
LABELS = tuple(str(s) for s in REC["labels"])
W = REF["icaweights"]  # EEGLAB's, on the same recording


def recording() -> mne.io.RawArray:
    """The ICA test recording (19 channels, 60 s, with positions)."""
    info = mne.create_info(list(LABELS), float(REC["sfreq"]), "eeg")
    raw = mne.io.RawArray(REC["data"].astype(np.float64), info, verbose="error")
    raw.set_montage(mne.channels.make_dig_montage(
        dict(zip(LABELS, REC["positions"], strict=True)), coord_frame="head"),
        verbose="error")  # fmt: skip
    return raw


@pytest.fixture(scope="module")
def fitted() -> tuple[mne.io.RawArray, Any]:
    """MNE's own ICA of the recording, on average-referenced data (an SSP
    projector, which MNE applies before unmixing), one component excluded."""
    raw = recording()
    raw.set_eeg_reference(projection=True, verbose="error")
    ica = mne.preprocessing.ICA(n_components=8, method="picard", max_iter=500,
                                random_state=0)  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(raw, verbose="error")
    ica.exclude = [2]
    return raw, ica


def graph(*nodes: dict[str, Any]) -> Graph:
    return Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": list(nodes)},
                           Registry.from_entry_points())  # fmt: skip


def test_an_mne_ica_file_starts_the_ics_rail(
    tmp_path: Path, fitted: tuple[mne.io.RawArray, Any]
) -> None:
    raw, ica = fitted
    rec, ica_file = tmp_path / "rec.fif", tmp_path / "rec-ica.fif"
    raw.save(rec, verbose="error")
    ica.save(ica_file, verbose="error")
    sources = ica.get_sources(raw).get_data()  # MNE's
    g = graph({"id": "input", "node": "meegxer.core.file",
               "params": {"path": str(rec), "ica": str(ica_file)}},
              {"id": "tap", "node": "meegxer.core.probe", "input": "input:ics",
               "params": {"keep_seconds": 0}})  # fmt: skip
    BatchScheduler(block_size=1000).run(g)
    tap = g["tap"].impl
    ics = tap.signal()  # type: ignore[union-attr]
    assert ics.info.ch_labels[:2] == ("IC01", "IC02") and ics.info.n_channels == 8
    # MeegXer's weights take µV (as EEGLAB's): MNE's sources, read as volts
    np.testing.assert_allclose(ics.data * 1e6, sources, rtol=1e-6,
                               atol=1e-6 * np.abs(sources).max())  # fmt: skip
    assert ics.info.ch_classes  # ICLabel's, computed
    source = g["input"].impl
    assert isinstance(source, FileSource)
    notes = describe(source)["notes"]
    assert notes[0] == ("MNE ICA decomposition: 8 components on 19 channels "
                        "start the ICs rail; ICLabel classes computed when a run "
                        "first reads them")  # fmt: skip
    assert "the file marks component 3 for removal" in notes[1]


def test_the_api_takes_an_ica_object_or_file(
    tmp_path: Path, fitted: tuple[mne.io.RawArray, Any]
) -> None:
    raw, ica = fitted
    sources = ica.get_sources(raw).get_data()
    pipeline = {"spec_version": SPEC_VERSION,
                "nodes": [{"id": "input", "node": "meegxer.core.file"}]}  # fmt: skip
    ics = run(pipeline, raw, ica=ica, output="input:ics")
    np.testing.assert_allclose(ics.get_data() * 1e6, sources, rtol=1e-6,
                               atol=1e-6 * np.abs(sources).max())  # fmt: skip
    ica.save(tmp_path / "rec-ica.fif", verbose="error")
    again = run(pipeline, raw, ica=tmp_path / "rec-ica.fif", output="input:ics")
    np.testing.assert_allclose(again.get_data(), ics.get_data())


def test_the_writer_stores_a_files_decomposition_as_mne_reads_it(
    tmp_path: Path,
) -> None:
    """A .set carrying EEGLAB's decomposition and classes, written as an MNE
    ICA file: its sources are EEGLAB's activations (µV), its labels_ the
    classes, as mne-icalabel names them."""
    rec = tmp_path / "rec.set"
    recording().export(rec, fmt="eeglab", verbose="error")
    write_decomposition(rec, Decomposition(LABELS, W, np.linalg.pinv(W),
                                           REF["probabilities"]), LABELS)  # fmt: skip
    out = tmp_path / "out-ica.fif"
    g = graph({"id": "input", "node": "meegxer.core.file",
               "params": {"path": str(rec)}},
              {"id": "icaw", "node": "meegxer.core.ica_writer",
               "params": {"path": str(out)}})  # fmt: skip
    result = BatchScheduler(block_size=1000).run(g)
    assert "icaw: wrote out-ica.fif: 18 components" in result.notes
    ica = mne.preprocessing.read_ica(out, verbose="error")
    x = mne.io.read_raw_eeglab(rec, preload=True, verbose="error")
    acts = W @ (x.get_data() * 1e6)  # EEGLAB's activations, on its µV
    np.testing.assert_allclose(ica.get_sources(x).get_data(), acts, rtol=1e-5,
                               atol=1e-6 * np.abs(acts).max())  # fmt: skip
    best = REF["probabilities"].argmax(axis=1)
    assert ica.labels_[ICLABEL_NAMES[2]] == [int(k) for k in np.flatnonzero(best == 2)]
    assert ica.exclude == []


def test_after_the_ica_strip_its_removed_components_are_excluded(
    tmp_path: Path,
) -> None:
    rec = tmp_path / "rec.set"
    recording().export(rec, fmt="eeglab", verbose="error")
    out = tmp_path / "out-ica.fif"
    g = graph({"id": "input", "node": "meegxer.core.file",
               "params": {"path": str(rec)}},
              {"id": "ica1", "node": "meegxer.nodes.ica"},
              {"id": "icaw", "node": "meegxer.core.ica_writer",
               "params": {"path": str(out)}})  # fmt: skip
    BatchScheduler(block_size=1000).run(g)
    stage = g["ica1"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    model = stage.model.data
    removed = rejected(model, stage.params)
    assert removed
    ica = mne.preprocessing.read_ica(out, verbose="error")
    weights = np.asarray(model["weights"])
    assert ica.n_components_ == weights.shape[0]  # every component, MNE's way
    assert ica.exclude == removed
    x = mne.io.read_raw_eeglab(rec, preload=True, verbose="error").get_data()
    acts = weights @ x * 1e6
    np.testing.assert_allclose(ica.get_sources(recording()).get_data(), acts,
                               rtol=1e-5, atol=1e-6 * np.abs(acts).max())  # fmt: skip


def test_the_writer_says_why_it_writes_nothing(tmp_path: Path) -> None:
    rec = tmp_path / "rec.set"
    recording().export(rec, fmt="eeglab", verbose="error")
    g = graph({"id": "input", "node": "meegxer.core.file",
               "params": {"path": str(rec)}},
              {"id": "icaw", "node": "meegxer.core.ica_writer",
               "params": {"path": str(tmp_path / "out-ica.fif")}})  # fmt: skip
    result = BatchScheduler(block_size=1000).run(g)
    assert any("no ICA decomposition before it" in s for s in result.skipped)
    assert not (tmp_path / "out-ica.fif").exists()
    with pytest.raises(ValueError, match=r"-ica\.fif"):
        IcaWriter({"path": str(tmp_path / "out.fif")})


def test_a_shared_session_names_no_ica_file() -> None:
    session = {"meegxer_session": "1.0", "pipeline": {
        "spec_version": SPEC_VERSION, "nodes": [
            {"id": "input", "node": "meegxer.core.file",
             "params": {"path": "/data/JaneDoe.fif", "ica": "/data/JaneDoe-ica.fif"}},
        ]}}  # fmt: skip
    params = share_safe(session)["pipeline"]["nodes"][0]["params"]
    assert params == {"path": "<input>.fif", "ica": ""}


def test_an_ica_file_given_as_the_recording_says_where_it_goes(
    tmp_path: Path, fitted: tuple[mne.io.RawArray, Any]
) -> None:
    """Both of the File input's fields take .fif: an ICA file chosen as the
    recording says so, instead of MNE's "No raw data in …" (§H5)."""
    ica_file = tmp_path / "rec-ica.fif"
    fitted[1].save(ica_file, verbose="error")
    source = FileSource({"path": str(ica_file)})
    with pytest.raises(ValueError, match="is an MNE ICA file, not a recording"):
        _ = source.info
