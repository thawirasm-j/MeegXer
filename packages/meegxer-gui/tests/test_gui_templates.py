"""New from template (M10.9): templates load into the chain as they are, and
with the sample data the first subject is fetched in the background."""

import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from meegxer.core import samples
from meegxer.core.control import manifest_to_dict
from meegxer.core.graph import Graph
from meegxer.core.registry import Registry
from meegxer.core.templates import TEMPLATES, template_pipeline
from meegxer.gui.pipeline_doc import PipelineDoc
from meegxer.gui.template_dialog import TemplateDialog


def inputs(pipeline: dict[str, Any]) -> dict[str, str]:
    return {n["id"]: n["input"] for n in pipeline["nodes"] if "input" in n}


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_a_template_loads_and_saves_unchanged(name: str) -> None:
    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    template = template_pipeline(name)
    doc.load_pipeline(template)
    saved = doc.to_pipeline()
    assert [n["id"] for n in saved["nodes"] if n["id"] in
            {m["id"] for m in template["nodes"]}] == [
        m["id"] for m in template["nodes"]]  # fmt: skip
    assert inputs(template).items() <= inputs(saved).items()
    Graph.from_dict(saved, registry)


def test_the_dialog_describes_each_template_and_its_sample(qapp: QApplication) -> None:
    dialog = TemplateDialog()
    dialog.templates.setCurrentRow(list(TEMPLATES).index("rest"))
    assert "spectrum" in dialog.description.text()
    assert "EEGBCI" in dialog.sample.text()
    note = dialog.sample_note.text()
    assert note.count("109") == 1 and "first time a run reads it" in note
    dialog.sample.setChecked(True)
    assert dialog.choice() == ("rest", True)


def test_new_from_template_on_the_sample_fetches_the_first_subject(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    monkeypatch.setenv(samples.SAMPLE_ENV, str(tmp_path / "samples"))
    asked: list[str] = []

    def fetch(subject: str, downloads: Path | None) -> list[tuple[str, Any]]:
        asked.append(subject)
        info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], 100.0, "eeg")
        data = np.random.default_rng(1).normal(0, 1e-5, (4, 2000))
        raw = mne.io.RawArray(data, info, verbose="error")
        raw.set_montage("standard_1020", verbose="error")
        raw.set_annotations(mne.Annotations([1.0], [0.0], ["T0"]))
        return [(t, raw.copy()) for t in samples.EEGBCI_RUNS]

    sample = replace(samples.SAMPLES["eegbci"], subjects=("001", "002"), fetch=fetch)
    monkeypatch.setitem(samples.SAMPLES, "eegbci", sample)
    monkeypatch.setattr(TemplateDialog, "exec", lambda self: 1)
    monkeypatch.setattr(TemplateDialog, "choice", lambda self: ("rest", True))
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.new_from_template()
        deadline = time.monotonic() + 30
        while "New from template" not in win.logbox.toPlainText():
            assert "Could not" not in win.logbox.toPlainText(), win.logbox.toPlainText()
            assert time.monotonic() < deadline, win.logbox.toPlainText()
            qapp.processEvents()
            time.sleep(0.02)
        assert asked == ["001"]  # the first subject only
        source = win.doc.source
        assert source is not None and source.node == "meegxer.core.bids"
        assert source.params["root"] == str(samples.sample_dir() / "eegbci")
        assert [s.id for s in win.doc.strips][-1] == "bands_out"
    finally:
        win.close()


def test_a_live_template_offers_no_sample(qapp: QApplication) -> None:
    dialog = TemplateDialog()
    dialog.templates.setCurrentRow(list(TEMPLATES).index("ssvep_bci"))
    assert "keyboard" in dialog.description.text()
    assert dialog.sample.isHidden() and dialog.sample_note.isHidden()
    assert dialog.choice() == ("ssvep_bci", False)
    dialog.templates.setCurrentRow(list(TEMPLATES).index("rest"))
    assert not dialog.sample.isHidden()
