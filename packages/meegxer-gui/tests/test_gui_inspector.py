from typing import Any

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QToolButton, QWidget

from meegxer.core.control import manifest_to_dict
from meegxer.core.registry import Registry
from meegxer.gui.inspector import Inspector
from meegxer.gui.labels import choice_label, param_label, unit_label
from meegxer.gui.pipeline_doc import PipelineDoc


@pytest.fixture
def doc() -> PipelineDoc:
    registry = Registry.from_entry_points()
    d = PipelineDoc(
        [manifest_to_dict(registry.get(n).manifest) for n in registry.names()]
    )
    d.set_source("meegxer.core.synthetic")
    d.insert(1, "meegxer.nodes.iir_filter")
    d.insert(2, "meegxer.nodes.ica")
    return d


def test_every_installed_parameter_has_a_friendly_label(doc: PipelineDoc) -> None:
    raw = {
        p["name"]
        for m in doc.manifests.values()
        for p in m["params"]
        if param_label(p["name"]) == p["name"]
    }
    assert not raw, raw  # each reads as words, not the machine name
    assert param_label("sfreq") == "Sampling rate"
    assert param_label("blink_uv") == "Blink size"
    assert param_label("my_new_hz") == "My new Hz"  # plugins: tidied mechanically
    assert param_label("x", {"label": "From the manifest"}) == "From the manifest"
    assert choice_label("method", "imcoh") == "Imaginary coherence"
    assert unit_label("uV") == "µV"


def test_strip_values_use_friendly_choices_and_units(doc: PipelineDoc) -> None:
    hp = doc.get("iirfilter1")
    assert doc.format_param(hp, "kind") == "Band-pass"
    assert doc.format_param(doc.get("input"), "noise_uv") == "10 µV"


def dispose(qapp: QApplication, widget: QWidget) -> None:
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def show(
    qapp: QApplication, doc: PipelineDoc, strip: str, *, advanced: bool
) -> tuple[Inspector, list[Any]]:
    ins = Inspector()
    seen: list[Any] = []
    ins.param_changed.connect(lambda *a: seen.append(a))
    ins.show_strip(doc, doc.get(strip), advanced=advanced, running=True, node_status={})
    ins.show()
    qapp.processEvents()
    return ins, seen


def test_inspector_labels_choices_and_folded_advanced(
    qapp: QApplication, doc: PipelineDoc
) -> None:
    ins, seen = show(qapp, doc, "iirfilter1", advanced=False)
    texts = {lb.text() for lb in ins.findChildren(QLabel)}
    assert {"Type", "Low cut-off"} <= texts
    combo = next(c for c in ins.findChildren(QComboBox) if c.findData("highpass") >= 0)
    combo.setCurrentIndex(combo.findData("highpass"))
    assert combo.currentText() == "High-pass"
    assert seen == [("iirfilter1", "kind", "highpass")]  # the manifest value
    dispose(qapp, ins)

    ins, _ = show(qapp, doc, "ica1", advanced=False)
    fold = next(b for b in ins.findChildren(QToolButton) if "Advanced" in b.text())
    assert not fold.isChecked()  # friendly mode: folded away, still reachable
    fold.click()
    assert ins.advanced_open
    ins.show_strip(doc, doc.get("ica1"), advanced=False, running=True, node_status={})
    fold = next(b for b in ins.findChildren(QToolButton) if "Advanced" in b.text())
    assert fold.isChecked()  # stays open while editing
    dispose(qapp, ins)

    ins, _ = show(qapp, doc, "ica1", advanced=True)
    fold = next(b for b in ins.findChildren(QToolButton) if "Advanced" in b.text())
    assert fold.isChecked()  # advanced mode opens it
    texts = {lb.text() for lb in ins.findChildren(QLabel)}
    assert {"Calibration", "Channels processed", "Shown on the strip"} <= texts
    assert "Credits" in texts  # ICA cites MNE, Picard, ICLabel, ...
    assert any("doi.org/10.1016/j.neuroimage.2019.05.026" in t for t in texts)
    dispose(qapp, ins)


def test_one_step_changes_a_value_by_a_sensible_amount(doc: PipelineDoc) -> None:
    from meegxer.gui.inspector import param_step

    def spec(name: str, node: str) -> dict[str, Any]:
        return next(p for p in doc.manifest(node)["params"] if p["name"] == name)

    ica, hp = "meegxer.nodes.ica", "meegxer.nodes.iir_filter"
    assert param_step(spec("threshold", ica), 0.9) == pytest.approx(0.1)
    assert param_step(spec("low", hp), 1.0) == pytest.approx(0.1)
    assert param_step(spec("high", hp), 40.0) == pytest.approx(5.0)
    assert param_step(spec("blink_uv", "meegxer.core.synthetic"), 150.0) == 20.0
    assert param_step(spec("n_channels", "meegxer.core.synthetic"), 16) == 1
    assert param_step(spec("max_iter", ica), 1000) == 100
    assert param_step({"type": "float", "step": 0.25, "min": None, "max": None},
                      3.0) == 0.25  # the manifest's own step wins  # fmt: skip


def test_editing_keeps_the_inspector_where_it_was(
    qapp: QApplication, doc: PipelineDoc
) -> None:
    ins = Inspector()
    ins.resize(500, 250)
    ins.show()

    def show(strip: str) -> None:
        ins.show_strip(doc, doc.get(strip), advanced=True, running=True,
                       node_status={})  # fmt: skip
        qapp.processEvents()

    show("input")  # many settings: the page scrolls
    bar = ins.verticalScrollBar()
    assert bar.maximum() > 200
    bar.setValue(200)
    show("input")  # a value changed: same strip, same place
    assert bar.value() == 200
    show("ica1")  # another strip starts at the top
    assert bar.value() == 0
    dispose(qapp, ins)


def test_an_output_chooses_the_feature_rail_it_reads(
    qapp: QApplication, doc: PipelineDoc
) -> None:
    psd = doc.insert(len(doc.strips), "meegxer.nodes.psd")
    doc.insert(len(doc.strips), "meegxer.nodes.band_power")
    tsv = doc.insert(len(doc.strips), "meegxer.core.feature_tsv")
    ins, _ = show(qapp, doc, tsv.id, advanced=False)
    chosen: list[tuple[str, str]] = []
    ins.input_changed.connect(lambda *a: chosen.append(a))
    box = ins.findChild(QComboBox, "reads")
    assert box is not None
    assert [box.itemText(i) for i in range(box.count())] == [
        "Nearest (Band power)", "Band power", "PSD"]  # fmt: skip
    box.setCurrentIndex(box.findData(psd.id))
    assert chosen == [(tsv.id, psd.id)]
    dispose(qapp, ins)
    ins, _ = show(qapp, doc, psd.id, advanced=False)  # the signal, or ICA's ICs
    box = ins.findChild(QComboBox, "reads")
    assert box is not None
    assert [box.itemText(i) for i in range(box.count())] == [
        "The signal", "ICs (ICA + ICLabel)"]  # fmt: skip
    dispose(qapp, ins)
    doc.remove("ica1")
    ins, _ = show(qapp, doc, psd.id, advanced=False)  # no rail output: no choice
    assert ins.findChild(QComboBox, "reads") is None
    dispose(qapp, ins)


def test_path_parameters_have_a_browse_button(
    qapp: QApplication, doc: PipelineDoc, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QFileDialog

    from meegxer.gui.inspector import PathEdit

    asked: list[tuple[str, str]] = []

    def save(parent: Any, caption: str, start: str, filters: str) -> tuple[str, str]:
        asked.append((start, filters))
        return str(tmp_path / "clean"), ""  # typed without its extension

    monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
    doc.insert(len(doc.strips), "meegxer.core.set_writer")
    writer = doc.strips[-1]
    writer.params["path"] = str(tmp_path / "old.set")
    ins, seen = show(qapp, doc, writer.id, advanced=True)
    (edit,) = ins.findChildren(PathEdit)
    assert edit.button.text() == "Browse…"
    edit.button.click()
    assert asked == [(str(tmp_path), "Supported files (*.set);;All files (*)")]
    assert seen[-1] == (writer.id, "path", str(tmp_path / "clean.set"))
    assert edit.text() == str(tmp_path / "clean.set")
    dispose(qapp, ins)

    picked = str(tmp_path / "rec.edf")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *a: (picked, ""))  # fmt: skip
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a: str(tmp_path))  # fmt: skip
    doc.set_source("meegxer.core.file")
    ins, seen = show(qapp, doc, "input", advanced=True)
    ins.findChildren(PathEdit)[0].button.click()
    assert seen[-1] == ("input", "path", picked)
    dispose(qapp, ins)
    doc.set_source("meegxer.core.bids")
    ins, seen = show(qapp, doc, "input", advanced=True)
    (root,) = [e for e in ins.findChildren(PathEdit) if e.kind == "folder"]
    root.button.click()
    assert seen[-1] == ("input", "root", str(tmp_path))
    dispose(qapp, ins)


def test_epoch_times_are_shown_in_ms_and_kept_in_seconds(
    qapp: QApplication, doc: PipelineDoc
) -> None:
    """The Epochs strip's trial and baseline times: edited in ms, 50 ms a
    step; the pipeline keeps seconds, as MNE takes them."""
    from PySide6.QtWidgets import QDoubleSpinBox

    strip = doc.insert(len(doc.strips), "meegxer.nodes.epochs")
    ins, seen = show(qapp, doc, strip.id, advanced=False)
    fields = [f for f in ins.findChildren(QDoubleSpinBox) if f.suffix() == " ms"]
    assert sorted(f.value() for f in fields) == [-200.0, -200.0, 0.0, 800.0]
    assert all(f.singleStep() == 50.0 for f in fields)
    tmax = next(f for f in fields if f.value() == 800.0)
    tmax.setValue(650.0)  # typed in ms ...
    assert seen[-1][1] == "tmax"
    assert seen[-1][2] == pytest.approx(0.65)  # ... sent in seconds
    dispose(qapp, ins)
