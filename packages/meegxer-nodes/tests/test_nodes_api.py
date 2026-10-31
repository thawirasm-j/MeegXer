"""The Python API (M10.7, docs/PYTHON_API.md): ``meegxer.api.run(pipeline,
raw)`` runs the batch scheduler over an MNE Raw (or a file) and gives back
the signal at the chain's end as Raw, or an Epochs strip's epochs as
Epochs: the same as a file writer at the same place would write."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.api import run
from meegxer.core.eeglab_ica import Decomposition, write_decomposition
from meegxer.core.graph import GraphError
from meegxer.core.manifest import SPEC_VERSION

LABELS = ["Fz", "Cz", "Pz", "Oz"]
SFREQ = 250.0
ONSETS = [1.0, 3.0, 5.0, 7.0]


def recording() -> mne.io.RawArray:
    """10 s of noise on four 10-20 channels, a "go" event every 2 s from 1 s."""
    rng = np.random.default_rng(7)
    info = mne.create_info(LABELS, SFREQ, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, int(10 * SFREQ))) * 1e-5, info,
                          verbose="error")  # fmt: skip
    raw.set_montage("colin27_1020", verbose="error")
    raw.set_annotations(mne.Annotations(ONSETS, [0.0] * 4, ["go"] * 4))
    return raw


def pipeline(*writers: dict[str, Any]) -> dict[str, Any]:
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file"},
        {"id": "bp1", "node": "meegxer.nodes.iir_filter",
         "params": {"low": 1.0, "high": 30.0}},
        {"id": "ep1", "node": "meegxer.nodes.epochs",
         "params": {"events": "go", "tmin": -0.2, "tmax": 0.5}},
        *writers,
    ]}  # fmt: skip


def test_raw_in_raw_out_as_the_set_writer_writes_it(tmp_path: Path) -> None:
    raw = recording()
    before = raw.get_data().copy()
    out = tmp_path / "out.set"
    clean = run(pipeline({"id": "set", "node": "meegxer.core.set_writer",
                          "params": {"path": str(out)}}), raw)  # fmt: skip
    assert isinstance(clean, mne.io.BaseRaw)
    np.testing.assert_array_equal(raw.get_data(), before)  # the caller's is kept
    written = mne.io.read_raw_eeglab(out, preload=True, verbose="error")
    assert clean.ch_names == written.ch_names == LABELS
    np.testing.assert_allclose(clean.get_data(), written.get_data(), rtol=1e-6,
                               atol=1e-12)  # .set stores single precision  # fmt: skip
    assert not np.allclose(clean.get_data(), before)  # filtered
    assert list(clean.annotations.description) == ["go"] * 4
    np.testing.assert_allclose(clean.annotations.onset, ONSETS)
    assert set(clean.get_channel_types()) == {"eeg"}
    assert clean.get_montage() is not None  # positions kept
    assert clean.info["meas_date"] is None


def test_an_epochs_strips_epochs_as_the_epochs_writer_writes_them(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out-epo.fif"
    epochs = run(pipeline({"id": "epochs", "node": "meegxer.core.epochs_writer",
                           "input": "ep1:epochs",
                           "params": {"path": str(out), "erp": False}}),
                 recording(), output="ep1")  # fmt: skip
    assert isinstance(epochs, mne.BaseEpochs)
    written = mne.read_epochs(out, verbose="error")
    assert len(epochs) == len(written) == len(ONSETS)
    np.testing.assert_allclose(epochs.get_data(), written.get_data(), rtol=1e-6,
                               atol=1e-12)  # .fif stores single precision  # fmt: skip
    assert list(epochs.event_id) == ["go"] and epochs.tmin == pytest.approx(-0.2)
    same = run(pipeline(), recording(), output="ep1:epochs")  # the port named
    np.testing.assert_array_equal(same.get_data(), epochs.get_data())


def test_a_pipeline_file_and_a_recording_file(tmp_path: Path) -> None:
    """A pipeline on disk; a recording file read by the File input, here an
    EEGLAB .set carrying an ICA: ``input:ics`` gives its components."""
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(pipeline()))
    raw = recording()
    rec = tmp_path / "rec.set"
    raw.export(rec, fmt="eeglab", verbose="error")
    w = np.linalg.qr(np.random.default_rng(1).standard_normal((4, 4)))[0] * 1e5
    write_decomposition(rec, Decomposition(tuple(LABELS), w, np.linalg.inv(w)),
                        tuple(LABELS))  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ICLabel cannot classify 4 channels
        ics = run(path, rec, output="input:ics")
    assert ics.ch_names == ["IC01", "IC02", "IC03", "IC04"]
    assert set(ics.get_channel_types()) == {"misc"}
    x = mne.io.read_raw_eeglab(rec, preload=True, verbose="error").get_data()
    np.testing.assert_allclose(ics.get_data(), w @ x, rtol=1e-5, atol=1e-9)
    end = run(str(path), raw)  # str paths too; the chain's end
    assert end.ch_names == LABELS


def test_outputs_it_cannot_give_say_why() -> None:
    with pytest.raises(GraphError, match="no strip 'nope'"):
        run(pipeline(), recording(), output="nope")
    with pytest.raises(GraphError, match="has no output 'x'"):
        run(pipeline(), recording(), output="bp1:x")
    psd = pipeline()
    psd["nodes"].append({"id": "psd1", "node": "meegxer.nodes.psd"})
    with pytest.raises(GraphError, match="feature writer"):
        run(psd, recording(), output="psd1")
