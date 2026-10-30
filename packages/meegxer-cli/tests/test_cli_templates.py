"""Templates and sample data (M10.9): a pipeline from a template, run over its
sample dataset, each subject fetched (here from a fake fetcher, no network)
the first time a run reads it."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from typer.testing import CliRunner

from meegxer.cli import app
from meegxer.core import samples
from meegxer.core.bids_out import find_recordings
from meegxer.core.graph import Graph
from meegxer.core.registry import Registry
from meegxer.core.templates import FLANKER_STIMULI, TEMPLATES, template_pipeline

runner = CliRunner()
CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "P3", "P4"]


def fake_raw(seconds: float, seed: int, eog: bool = False) -> Any:
    rng = np.random.default_rng(seed)
    sfreq = 100.0
    names = CHANNELS + (["HEOG"] if eog else [])
    types = ["eeg"] * len(CHANNELS) + (["eog"] if eog else [])
    info = mne.create_info(names, sfreq, types)
    raw = mne.io.RawArray(rng.normal(0, 10e-6, (len(names), int(seconds * sfreq))),
                          info, verbose="error")  # fmt: skip
    raw.set_montage("standard_1020", verbose="error")
    raw.info["line_freq"] = 60.0
    return raw


@pytest.fixture
def fake_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Three EEGBCI subjects and the ERP CORE subject, from fake fetchers;
    the subjects fetched are recorded."""
    monkeypatch.setenv(samples.SAMPLE_ENV, str(tmp_path / "samples"))
    asked: list[str] = []

    def eegbci(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
        asked.append(f"eegbci-{subject}")
        return [(t, fake_raw(30.0, int(subject) * 2 + k))
                for k, t in enumerate(samples.EEGBCI_RUNS)]  # fmt: skip

    def erp_core(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
        asked.append(f"erp_core-{subject}")
        raw = fake_raw(30.0, 9, eog=True)
        labels = [*FLANKER_STIMULI.split(", "), "response/left"] * 4
        raw.set_annotations(mne.Annotations(np.arange(2.0, 2.0 + len(labels)),
                                            0.0, labels))  # fmt: skip
        return [("flanker", raw)]

    three = ("001", "002", "003")
    eeg = replace(samples.SAMPLES["eegbci"], subjects=three, fetch=eegbci)
    monkeypatch.setitem(samples.SAMPLES, "eegbci", eeg)
    erp = replace(samples.SAMPLES["erp_core"], fetch=erp_core)
    monkeypatch.setitem(samples.SAMPLES, "erp_core", erp)
    return asked


def test_every_template_builds() -> None:
    registry = Registry.from_entry_points()
    for name in TEMPLATES:
        graph = Graph.from_dict(template_pipeline(name), registry)
        assert graph.source.id == "input"


def test_a_sample_lists_every_subject_before_any_is_fetched(
    fake_samples: list[str],
) -> None:
    root = samples.create("eegbci")
    listed = find_recordings(root)
    tasks = ("resteyesopen", "resteyesclosed")
    assert [(r["subject"], r["task"]) for r in listed] == [
        (s, t) for s in ("001", "002", "003") for t in tasks
    ]
    closed = find_recordings(root, subjects=["002"], tasks=["resteyesclosed"])
    assert closed == [{"root": str(root), "subject": "002", "session": "",
                       "task": "resteyesclosed", "run": "",
                       "datatype": "eeg"}]  # fmt: skip
    assert fake_samples == []  # nothing downloaded
    description = json.loads((root / "dataset_description.json").read_text())
    assert description["Name"] == "EEGBCI resting state"
    assert description["License"] == "ODC-By-1.0"  # from the dataset's page
    assert description["SourceDatasets"][0]["DOI"] == "doi:10.13026/C28G6P"
    with pytest.raises(ValueError, match="no subject '200'"):
        samples.ensure_recording({"root": str(root), "subject": "200"})


def test_rest_template_over_the_sample_fetches_only_what_it_runs(
    tmp_path: Path, fake_samples: list[str]
) -> None:
    pipe = tmp_path / "rest.json"
    made = runner.invoke(app, ["new", "rest", str(pipe), "--sample"])
    assert made.exit_code == 0, made.output
    root = samples.sample_dir() / "eegbci"
    assert f"--bids {root}" in made.output
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                 "--subjects", "002"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "downloading this sample subject" in result.output
    assert "done 2, skipped 0, failed 0" in result.output
    assert fake_samples == ["eegbci-002"]  # once, for both of its tasks
    folder = root / "derivatives" / "meegxer" / "sub-002" / "eeg"
    stem = "sub-002_task-resteyesclosed_desc-clean"
    assert (folder / f"{stem}_psd_features.tsv").is_file()
    raw_sidecar = root / "sub-002" / "eeg" / "sub-002_task-resteyesclosed_eeg.json"
    assert json.loads(raw_sidecar.read_text())["TaskDescription"].startswith(
        "One minute at rest with the eyes closed"
    )
    assert (folder / f"{stem}_bands_features.tsv").is_file()

    again = runner.invoke(app, ["run", str(pipe), "--bids", str(root),
                                "--subjects", "002,003"])  # fmt: skip
    assert "done 2, skipped 2, failed 0" in again.output, again.output
    assert fake_samples == ["eegbci-002", "eegbci-003"]  # 002 is reused
    listing = runner.invoke(app, ["templates"])
    assert "2 of 3 subjects fetched" in listing.output


def test_erp_template_over_the_sample_writes_epochs_and_erps(
    tmp_path: Path, fake_samples: list[str]
) -> None:
    pipe = tmp_path / "erp.json"
    assert runner.invoke(app, ["new", "erp", str(pipe), "--sample"]).exit_code == 0
    root = samples.sample_dir() / "erp_core"
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root)])
    assert result.exit_code == 0, result.output
    folder = root / "derivatives" / "meegxer" / "sub-001" / "eeg"
    epochs = mne.read_epochs(folder / "sub-001_task-flanker_desc-clean_epo.fif",
                             verbose="error")  # fmt: skip
    assert set(epochs.event_id) == set(FLANKER_STIMULI.split(", "))  # no responses
    assert epochs.get_montage() is not None  # the channels' positions are kept
    assert len(mne.read_evokeds(folder / "sub-001_task-flanker_desc-clean_ave.fif",
                                verbose="error")) == 4  # fmt: skip


def test_a_failed_download_fails_that_subject_only(
    tmp_path: Path, fake_samples: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    good = samples.SAMPLES["eegbci"].fetch

    def flaky(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
        if subject == "001":
            raise OSError("no network")
        return good(subject, downloads)

    monkeypatch.setitem(samples.SAMPLES, "eegbci",
                        replace(samples.SAMPLES["eegbci"], fetch=flaky))  # fmt: skip
    pipe = tmp_path / "rest.json"
    runner.invoke(app, ["new", "rest", str(pipe), "--sample"])
    root = samples.sample_dir() / "eegbci"
    result = runner.invoke(app, ["run", str(pipe), "--bids", str(root), "--subjects",
                                 "001,002", "--task", "resteyesopen"])  # fmt: skip
    assert result.exit_code == 1
    assert "sub-001_task-resteyesopen: FAILED: no network" in result.output
    assert "done 1, skipped 0, failed 1" in result.output


def test_new_refuses_an_unknown_template_or_an_existing_file(tmp_path: Path) -> None:
    assert "no template 'nope'" in runner.invoke(
        app, ["new", "nope", str(tmp_path / "x.json")]).output  # fmt: skip
    (tmp_path / "x.json").write_text("{}")
    again = runner.invoke(app, ["new", "erp", str(tmp_path / "x.json")])
    assert "exists" in again.output
