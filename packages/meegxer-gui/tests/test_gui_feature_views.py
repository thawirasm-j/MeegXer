"""Feature views: taps on feature strips and the ICs rail, the feature view
(spectrum or over time), components coloured by their ICLabel class."""

import subprocess
import time
from pathlib import Path

import mne
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from meegxer.core.control import manifest_to_dict
from meegxer.core.registry import Registry
from meegxer.gui.colors import IC_CLASS_COLORS, class_style
from meegxer.gui.feature_view import FeatureSeries, FeatureView, pick
from meegxer.gui.pipeline_doc import FEATURE_PROBE, PROBE, PipelineDoc, tap_id
from meegxer.gui.signal_view import SignalView, Window
from meegxer.gui.theme import load_theme


def dispose(qapp: QApplication, *widgets: QWidget) -> None:
    for w in widgets:
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def make_doc() -> PipelineDoc:
    registry = Registry.from_entry_points()
    return PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                        for n in registry.names()])  # fmt: skip


def wait_for(qapp: QApplication, done, seconds: float = 60.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + seconds
    while not done():
        assert time.monotonic() < deadline
        qapp.processEvents()
        time.sleep(0.05)


def test_feature_strips_and_rail_outputs_get_taps() -> None:
    doc = make_doc()
    doc.set_source("meegxer.core.synthetic")
    for node in ["meegxer.nodes.ica", "meegxer.nodes.psd"]:
        doc.insert(len(doc.strips), node)
    nodes = {n["id"]: n for n in doc.to_pipeline()["nodes"]}
    assert nodes[tap_id("psd1")]["node"] == FEATURE_PROBE
    assert nodes[tap_id("psd1")]["input"] == "psd1"
    assert tap_id("ica1:ics") == "tap_ica1__ics"
    assert (nodes["tap_ica1__ics"]["node"], nodes["tap_ica1__ics"]["input"]) == (
        PROBE, "ica1:ics")  # fmt: skip
    assert [doc.tap_kind(k) for k in ("ica1", "psd1", "ica1:ics")] == [
        "signal", "features", "rail"]  # fmt: skip
    saved = doc.to_pipeline(taps=False)
    assert not any(n["id"].startswith("tap_") for n in saved["nodes"])
    doc.load_pipeline(doc.to_pipeline())  # taps are not strips
    assert [s.id for s in doc.strips] == ["input", "ica1", "psd1"]


def test_pick_orders_series_and_frequency() -> None:
    values = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)  # windows, band, channel
    out = pick(values, ("band", "channel"), {"band": 1})
    assert out.shape == (2, 4, 1)
    np.testing.assert_array_equal(out[:, :, 0], values[:, 1, :])
    spec = pick(values, ("frequency", "channel"), {})
    assert spec.shape == (2, 4, 3)  # channels as series, then frequencies


def test_feature_view_draws_a_spectrum_or_values_over_time(
    qapp: QApplication,
) -> None:
    view = FeatureView(load_theme("dark"))
    freqs = (1.0, 2.0, 4.0, 8.0)
    psd = np.full((1, 2, 4), 1e-12)
    view.refresh(FeatureSeries("tap_psd1", ("channel", "frequency"),
                               {"channel": ("Fz", "Cz"), "frequency": freqs},
                               np.array([2.0]), psd), "PSD", 0.0, 60.0)  # fmt: skip
    curves = [c for c in view.plot._curves if c.isVisible()]
    assert len(curves) == 2
    x, y = curves[0].getData()
    assert list(x) == list(freqs) and y[0] == pytest.approx(-120.0)  # in dB
    assert "spectrum" in view.legend.text()
    assert not view.slice.isVisible()

    bands = ("theta", "alpha")
    values = np.arange(5 * 2 * 2.0).reshape(5, 2, 2)  # windows, channel, band
    band_power = FeatureSeries("tap_bp1", ("channel", "band"),
                               {"channel": ("Fz", "Cz"), "band": bands},
                               np.arange(5.0), values)  # fmt: skip
    view.refresh(band_power, "Band power", 0.0, 10.0)
    assert [view.slice.itemText(i) for i in range(view.slice.count())] == [
        "theta", "alpha"]  # fmt: skip
    view.slice.setCurrentIndex(1)
    view.refresh(band_power, "Band power", 0.0, 10.0)
    x, y = next(c for c in view.plot._curves if c.isVisible()).getData()
    assert list(x) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(y, values[:, 0, 1])  # Fz, alpha
    assert "over time" in view.legend.text()
    view.refresh(None, "Band power", 0.0, 10.0)
    assert "no feature windows yet" in view.legend.text()
    dispose(qapp, view)


def test_components_are_coloured_and_named_by_class(qapp: QApplication) -> None:
    view = SignalView(load_theme("dark"))
    t = np.arange(500) / 50
    mid = np.random.default_rng(0).standard_normal((2, 500)) * 1e-6
    win = Window("tap_ica1__ics", ("IC01", "IC02"), t, mid, mid,
                 np.zeros(500, dtype=bool), 50.0, ("eye", "brain"),
                 (0.97, 0.81))  # fmt: skip
    view.refresh(lambda p, s: win, "tap_ica1__ics", [])
    texts = [a.text() for a in view.channels.menu().actions() if a.isCheckable()]
    assert texts == ["IC01 · eye 0.97", "IC02 · brain 0.81"]
    pens = [c.opts["pen"].color().name() for c in view.plot._curves if c.isVisible()]
    assert pens == [IC_CLASS_COLORS["eye"].lower(), IC_CLASS_COLORS["brain"].lower()]
    assert "ICLabel class" in view.legend.text()
    view.channels.toggle_channel("IC02")  # menu keys stay the labels
    assert view.channels.selected(win.labels) == [0]
    assert class_style("other", load_theme("dark")).dashed
    dispose(qapp, view)


def test_clicking_a_rail_name_taps_it(qapp: QApplication) -> None:
    from meegxer.gui.chain import ChainView, RailBoard

    doc = make_doc()
    doc.set_source("meegxer.core.synthetic")
    for node in ["meegxer.nodes.ica", "meegxer.nodes.band_power"]:
        doc.insert(len(doc.strips), node)
    chain = ChainView()
    chain.rebuild(doc, load_theme("dark"), statuses={}, taps={"ica1:ics": "main"},
                  advanced=False, selected=None)  # fmt: skip
    chain.resize(1600, 400)
    chain.show()
    qapp.processEvents()
    board = chain.widget()
    assert isinstance(board, RailBoard)
    clicked: list[tuple[str, bool]] = []
    chain.tap_clicked.connect(lambda key, main: clicked.append((key, main)))
    from PySide6.QtGui import QFontMetrics

    fm = QFontMetrics(board.font_)
    for rail, x0, _ in board.spans():
        box = board.label_box(rail, x0, fm)
        point = QPointF(box.right() - 4, box.center().y())
        board.mousePressEvent(QMouseEvent(
            QEvent.Type.MouseButtonPress, point, point, Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))  # fmt: skip
    assert sorted(clicked) == [("bandpower1", False), ("ica1:ics", False)]
    chain.close()
    dispose(qapp, chain)


def test_live_feature_tap_shows_band_power_and_back(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        strip = win.doc.insert(len(win.doc.strips), "meegxer.nodes.band_power")
        strip.params.update({"window": 0.5, "step": 0.25})
        win._structure_changed("added")
        win.click_tap("bandpower1", False)
        assert win.view_tap == "bandpower1"
        assert win.chain_taps()["bandpower1"] == "main"
        assert "main" not in [v for k, v in win.chain_taps().items()
                              if k != "bandpower1"]  # fmt: skip
        win.start()
        assert win.running
        wait_for(qapp, lambda: n_windows(win, "tap_bandpower1") > 4)
        win.redraw()
        assert win.views.currentWidget() is win.feature_view
        assert "over time" in win.feature_view.legend.text()
        assert any(c.isVisible() for c in win.feature_view.plot._curves)
        win.click_tap("notch1", False)  # a signal tap: back to the signal
        assert win.view_tap is None and win.taps["notch1"] == "main"
        win.redraw()
        assert win.views.currentWidget() is win.view
        win.stop()
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)


def n_windows(win, probe: str) -> int:  # type: ignore[no-untyped-def]
    buf = win.feature_buffers.get(probe)
    return 0 if buf is None else len(buf.windows)


def test_offline_feature_and_ics_taps(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
    data = np.random.default_rng(1).standard_normal((8, 250 * 40)) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(names, 250.0, "eeg"), verbose="error")
    rec = tmp_path / "rec_raw.fif"
    raw.save(rec, verbose="error")
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        win.doc.source.recording = {"montage": "colin27_1020"}
        for node in ["meegxer.nodes.ica", "meegxer.nodes.psd"]:
            win.insert_strip(len(win.doc.strips), node)
        win.start()
        wait_for(qapp, lambda: win.offline_state in ("done", "error"), 180)
        assert win.offline_state == "done", win.logbox.toPlainText()
        qapp.processEvents()
        win.click_tap("psd1", False)
        win.redraw()
        assert win.views.currentWidget() is win.feature_view
        assert "mean over the span" in win.feature_view.legend.text()
        curves = [c for c in win.feature_view.plot._curves if c.isVisible()]
        assert len(curves) == len(names)
        win.click_tap("ica1:ics", False)
        assert win._shown() == ("tap_ica1__ics", [])
        win.redraw()
        assert win.views.currentWidget() is win.view
        actions = [a.text() for a in win.view.channels.menu().actions()
                   if a.isCheckable()]  # fmt: skip
        assert actions and actions[0].startswith("IC01 · ")
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)
