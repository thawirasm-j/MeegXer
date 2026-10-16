"""BIDS-derivatives output (spec/bids-derivatives.md)."""

import csv
import json
from pathlib import Path
from typing import Any

import mne
import mne_bids
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.session import load_session, read_session

CHANNELS = ["Fp1", "Fz", "Cz", "Pz", "O1", "VEOG"]


def make_dataset(root: Path, subjects: tuple[str, ...] = ("01",)) -> Path:
    for i, subject in enumerate(subjects):
        info = mne.create_info(CHANNELS, 250.0, ["eeg"] * 5 + ["eog"])
        data = np.random.default_rng(i).standard_normal((6, 250 * 20)) * 1e-5
        raw = mne.io.RawArray(data, info, verbose="error")
        raw.set_montage("colin27_1020", on_missing="ignore")
        raw.set_annotations(mne.Annotations([2.0, 5.0], [0.0, 1.0], ["go", "stop"]))
        raw.info["line_freq"] = 50
        path = mne_bids.BIDSPath(root=root, subject=subject, task="rest",
                                 datatype="eeg")  # fmt: skip
        mne_bids.write_raw_bids(raw, path, format="EDF", allow_preload=True,
                                verbose="error")  # fmt: skip
    return root


def bids_pipeline(root: Path, **writer: Any) -> dict[str, Any]:
    return {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.bids",
         "params": {"root": str(root), "subject": "01", "task": "rest"},
         "recording": {"channels": {"Pz": {"bad": True}}}},
        {"id": "hp", "node": "meegxer.nodes.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
        {"id": "out", "node": "meegxer.core.bids_writer", "params": writer},
        {"id": "bp", "node": "meegxer.nodes.band_power"},
        {"id": "feat", "node": "meegxer.core.bids_features", "params": writer},
    ]}  # fmt: skip


def run(data: dict[str, Any]) -> None:
    BatchScheduler().run(Graph.from_dict(data, Registry.from_entry_points()))


def tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_derivative_is_readable_by_bids_tools(tmp_path: Path) -> None:
    raw_root = make_dataset(tmp_path / "ds")
    run(bids_pipeline(raw_root))
    deriv = raw_root / "derivatives" / "meegxer"
    folder = deriv / "sub-01" / "eeg"
    stem = "sub-01_task-rest_desc-clean"
    for suffix in ("eeg.set", "eeg.json", "channels.tsv", "events.tsv",
                   "electrodes.tsv", "coordsystem.json", "bp_features.tsv",
                   "bp_features.json", "meegxer.json"):  # fmt: skip
        assert (folder / f"{stem}_{suffix}").is_file(), suffix

    back = mne_bids.read_raw_bids(
        mne_bids.BIDSPath(root=deriv, subject="01", task="rest", description="clean",
                          datatype="eeg", suffix="eeg", extension=".set"),
        verbose="error",
    )  # fmt: skip
    assert back.ch_names == CHANNELS
    assert back.get_channel_types()[-1] == "eog"
    assert [a["description"] for a in back.annotations] == ["go", "stop"]
    assert back.info["bads"] == ["Pz"]  # the recording edit, in channels.tsv
    assert back.info["line_freq"] == 50.0  # from the raw sidecar

    description = json.loads((deriv / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "MeegXer"
    assert description["DatasetLinks"] == {"raw": "../.."}
    links = description["ReferencesAndLinks"]  # what the pipeline's nodes cite
    assert any("10.3389/fnins.2013.00267" in r for r in links)  # MNE
    assert any("10.1038/s41597-019-0104-8" in r for r in links)  # EEG-BIDS
    assert len(links) == len(set(links))
    assert "*_meegxer.json" in (deriv / ".bidsignore").read_text()
    sidecar = json.loads((folder / f"{stem}_eeg.json").read_text())
    assert sidecar["Sources"] == ["bids:raw:sub-01/eeg/sub-01_task-rest_eeg.edf"]
    assert sidecar["EEGChannelCount"] == 5 and sidecar["EOGChannelCount"] == 1
    assert "meegxer.nodes.iir_filter" in sidecar["Description"]

    rows = tsv(folder / f"{stem}_bp_features.tsv")
    assert rows and {"window_start", "window_stop", "onset"} <= set(rows[0])
    session = read_session(folder / f"{stem}_meegxer.json")
    outputs = {Path(o["path"]).name for o in session["outputs"]}
    assert f"{stem}_eeg.set" in outputs and f"{stem}_bp_features.tsv" in outputs
    load_session(session, Registry.from_entry_points())  # a valid session


def test_rerun_needs_overwrite_and_is_byte_identical(tmp_path: Path) -> None:
    raw_root = make_dataset(tmp_path / "ds")
    run(bids_pipeline(raw_root))
    signal = next((raw_root / "derivatives").rglob("*_eeg.set"))
    first = signal.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(bids_pipeline(raw_root))
    run(bids_pipeline(raw_root, overwrite=True))
    assert signal.read_bytes() == first


def test_dataset_description_is_merged_not_replaced(tmp_path: Path) -> None:
    raw_root = make_dataset(tmp_path / "ds", ("01", "02"))
    deriv = raw_root / "derivatives" / "meegxer"
    run(bids_pipeline(raw_root))
    path = deriv / "dataset_description.json"
    data = json.loads(path.read_text())
    data["Authors"] = ["TrimLAB"]
    path.write_text(json.dumps(data))
    second = bids_pipeline(raw_root)
    second["nodes"][0]["params"]["subject"] = "02"
    run(second)
    merged = json.loads(path.read_text())
    assert merged["Authors"] == ["TrimLAB"] and len(merged["GeneratedBy"]) == 1
    assert len(merged["ReferencesAndLinks"]) == len(set(merged["ReferencesAndLinks"]))
    assert (deriv / "sub-01").is_dir() and (deriv / "sub-02").is_dir()


def test_file_input_takes_entities_from_its_name(tmp_path: Path) -> None:
    raw_root = make_dataset(tmp_path / "ds")
    edf = raw_root / "sub-01" / "eeg" / "sub-01_task-rest_eeg.edf"
    data = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(edf)}},
        {"id": "out", "node": "meegxer.core.bids_writer", "params": {"desc": "raw2"}},
    ]}  # fmt: skip
    run(data)
    assert (raw_root / "derivatives" / "meegxer" / "sub-01" / "eeg"
            / "sub-01_task-rest_desc-raw2_eeg.set").is_file()  # fmt: skip


def test_inputs_without_entities_or_root_are_explained(tmp_path: Path) -> None:
    raw = mne.io.RawArray(np.zeros((2, 500)), mne.create_info(["a", "b"], 250.0, "eeg"),
                          verbose="error")  # fmt: skip
    loose = tmp_path / "recording_raw.fif"
    raw.save(loose, verbose="error")
    writer: dict[str, Any] = {"id": "out", "node": "meegxer.core.bids_writer"}
    data = {"spec_version": SPEC_VERSION, "nodes": [
        {"id": "input", "node": "meegxer.core.file", "params": {"path": str(loose)}},
        writer,
    ]}  # fmt: skip
    with pytest.raises(ValueError, match="needs subject and task"):
        run(data)
    writer["params"] = {"subject": "07", "task": "rest"}
    with pytest.raises(ValueError, match="set 'root'"):
        run(data)
    writer["params"]["root"] = str(tmp_path / "out")
    run(data)
    assert (tmp_path / "out" / "sub-07" / "eeg"
            / "sub-07_task-rest_desc-clean_eeg.set").is_file()  # fmt: skip
