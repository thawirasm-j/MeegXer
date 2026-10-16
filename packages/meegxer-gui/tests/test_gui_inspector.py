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
    d.insert(1, "meegxer.sterile.iir_filter")
    d.insert(2, "meegxer.sterile.ica")
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
    dispose(qapp, ins)
