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
from meegxer.gui.feature_view import FeatureSeries, FeatureView, nearest_point, pick
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
    # no place on the scalp: listed beside the head, named by class
    texts = [b.text() for b in view.channels.picker.boxes.values()]
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
        for node in ["meegxer.nodes.iir_filter", "meegxer.nodes.ica",
                     "meegxer.nodes.psd"]:  # fmt: skip
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
        boxes = [b.text() for b in win.view.channels.picker.boxes.values()]
        assert boxes and boxes[0].startswith("IC01 · ")
        # spectrum: the head plot maps the components back on the electrodes,
        # from ICA's mixing matrix read from the core (get_model)
        from meegxer.gui.headplot import HeadMap

        win.show()  # the head plot draws only when shown
        win.view._set_mode("spectrum")
        qapp.processEvents()
        win.redraw()
        assert win.view.mixing is not None and win.view.mixing[0] == tuple(names)
        head = win.view.head.map
        assert isinstance(head, HeadMap), win.view.head.message
        assert {e.label for e in head.electrodes} == set(names)
        assert head.caption.endswith("8 of 8 components, back-projected")
        # every filter strip's mark reaches the view, in chain order (spec
        # 0.2.19): the 1/f fit skips those bands. The chain before the band-pass
        # is the daemon's or the default one.
        from meegxer.nodes.filters import filters_of, notch_filters

        expected: list[tuple[str, float, float]] = []
        for strip in win.doc.strips:
            p = strip.params
            if strip.node == "meegxer.nodes.iir_filter":
                expected += filters_of(str(p["kind"]), float(p["low"]),
                                       float(p["high"]))  # fmt: skip
            elif strip.node == "meegxer.nodes.notch":
                harmonics = int(p.get("harmonics", 1))
                expected += notch_filters(float(p["freq"]), harmonics, 250.0)
        shown = win._offline_window("tap_ica1__ics", 10.0)
        assert shown is not None
        assert shown.filters == tuple(expected)
        assert shown.filters[-2:] == (("highpass", 1.0, 0.0), ("lowpass", 0.0, 40.0))
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)


def test_feature_view_labels_the_value_axis_with_the_unit(qapp: QApplication) -> None:
    """Spec 0.2.19: the value axis names the unit; a spectrum converted to dB
    names its reference; *Show* names each measure's unit when they differ."""
    view = FeatureView(load_theme("dark"))
    left = view.plot.getPlotItem().getAxis("left")
    psd = FeatureSeries("tap_psd1", ("channel", "frequency"),
                        {"channel": ("Fz",), "frequency": (1.0, 2.0)},
                        np.array([2.0]), np.full((1, 1, 2), 1e-12),
                        unit="V^2/Hz")  # fmt: skip
    view.refresh(psd, "PSD", 0.0, 60.0)
    assert left.labelText == "dB (re 1 µV²/Hz)"  # volts shown as microvolts
    in_db = FeatureSeries("tap_psd1", psd.dims, psd.coords, psd.t,
                          np.full((1, 1, 2), 3.0), unit="dB")  # fmt: skip
    view.refresh(in_db, "PSD", 0.0, 60.0)
    y = next(c for c in view.plot._curves if c.isVisible()).getData()[1]
    assert left.labelText == "value (dB)" and y[0] == 3.0  # not converted again

    fooof = FeatureSeries("tap_sp1", ("channel", "measure"),
                          {"channel": ("Fz",), "measure": ("exponent", "cf")},
                          np.arange(3.0), np.ones((3, 1, 2)),
                          unit_dim="measure", units=("1", "Hz"))  # fmt: skip
    view.refresh(fooof, "Specparam", 0.0, 10.0)
    assert [view.slice.itemText(i) for i in range(view.slice.count())] == [
        "exponent", "cf (Hz)"]  # fmt: skip
    assert left.labelText == "value"
    view.slice.setCurrentIndex(1)
    view.refresh(fooof, "Specparam", 0.0, 10.0)
    assert left.labelText == "value (Hz)"
    dispose(qapp, view)


def test_nearest_point_picks_the_line_within_reach() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    low, high = (x, np.zeros(4)), (x, np.full(4, 10.0))
    # one data unit is 10 px on both axes: the pointer 0.5 above the low line
    assert nearest_point([low, high], 1.4, 0.5, (0.1, 0.1)) == (0, 1)
    assert nearest_point([low, high], 2.6, 9.6, (0.1, 0.1)) == (1, 3)
    assert nearest_point([low, high], 1.5, 5.0, (0.1, 0.1)) is None  # 50 px off
    # near the middle of a long segment, far from both its ends
    sparse = (np.array([0.0, 10.0]), np.array([0.0, 0.0]))
    assert nearest_point([sparse], 4.0, 0.5, (0.1, 0.1)) == (0, 0)
    gaps = (np.array([0.0, 1.0, 2.0]), np.array([np.nan, 1.0, np.nan]))
    assert nearest_point([gaps], 1.0, 1.0, (0.1, 0.1)) == (0, 1)


def test_hovering_a_line_names_its_point_and_value(qapp: QApplication) -> None:
    """Item 5: near a line, it is drawn bold, its nearest point is marked and
    labelled with the line's name, where it is and its value with the unit."""
    view = FeatureView(load_theme("dark"))
    view.resize(800, 400)
    view.show()
    values = np.array([[[1.0], [4.0]], [[2.0], [5.0]], [[3.0], [6.0]]]) * 1e-12
    series = FeatureSeries("tap_bp1", ("channel", "band"),
                           {"channel": ("Fz", "Cz"), "band": ("alpha",)},
                           np.array([1.0, 2.0, 3.0]), values, unit="V^2")  # fmt: skip
    view.refresh(series, "Band power", 0.0, 4.0)
    qapp.processEvents()
    vb = view.plot.getViewBox()
    vb.setRange(xRange=(0.0, 4.0), yRange=(0.0, 7.0), padding=0)
    qapp.processEvents()
    width = view.plot._curves[1].opts["pen"].widthF()
    view._mouse_moved(vb.mapViewToScene(QPointF(2.05, 5.05)))
    assert view.hovered == ("Cz", 2.0, 5.0)
    assert view.hover_text() == "Cz · t = 2.00 s: 5 µV²"  # V² shown as µV²
    assert view._marker.isVisible() and view._tag.isVisible()
    assert view.plot._curves[1].opts["pen"].widthF() > width  # the line in bold

    view.refresh(series, "Band power", 0.0, 4.0)  # a new frame keeps the hover
    assert view.hovered == ("Cz", 2.0, 5.0)
    view._mouse_moved(vb.mapViewToScene(QPointF(2.0, 3.5)))  # between the lines
    assert view.hovered is None and not view._marker.isVisible()
    assert view.plot._curves[1].opts["pen"].widthF() == width

    psd = FeatureSeries("tap_psd1", ("channel", "frequency"),
                        {"channel": ("Fz",), "frequency": (1.0, 2.0, 3.0)},
                        np.array([1.0]), np.full((1, 1, 3), 1e-11),
                        unit="V^2/Hz")  # fmt: skip
    view.refresh(psd, "PSD", 0.0, 60.0)
    vb.setRange(xRange=(0.0, 4.0), yRange=(0.0, 20.0), padding=0)
    qapp.processEvents()
    view._mouse_moved(vb.mapViewToScene(QPointF(2.0, 10.0)))
    assert view.hover_text() == "Fz · 2 Hz: 10 dB"  # re 1 µV²/Hz
    dispose(qapp, view)


def test_connectivity_draws_lines_from_the_seed(qapp: QApplication) -> None:
    """Item 3: channel x target windows add a head map with a line from the seed
    (the channel chosen in *Show*, or a clicked electrode) to every other one,
    coloured and thickened by its value."""
    from meegxer.gui.headplot import HeadMap

    view = FeatureView(load_theme("dark"))
    labels = ("Fz", "Cz", "Pz", "O1")
    conn = np.array([[1.0, 0.8, 0.2, 0.1], [0.8, 1.0, 0.5, 0.3],
                     [0.2, 0.5, 1.0, 0.6], [0.1, 0.3, 0.6, 1.0]])  # fmt: skip
    series = FeatureSeries("tap_conn1", ("channel", "target"),
                           {"channel": labels, "target": labels},
                           np.array([2.0]), conn[None], unit="1")  # fmt: skip
    view.refresh(series, "Connectivity", 0.0, 10.0)
    assert not view.head.isHidden()
    m = view.head.map
    assert isinstance(m, HeadMap) and m.caption.startswith("from Fz")
    assert len(m.links) == 3  # Fz to each other channel
    widths = {round(w, 6) for *_, w in m.links}
    assert max(widths) == 4.0 and min(widths) == 1.0  # strongest Cz, weakest O1
    assert m.links[-1][3] == 4.0  # the strongest is drawn last, on top
    by = {e.label: e for e in m.electrodes}
    assert by["Fz"].shown and by["Cz"].text == "Fz–Cz: 0.8"  # noqa: RUF001
    assert (m.scale[0], m.scale[1]) == ("0.1", "0.8")

    view.head.clicked.emit("Pz")  # a click makes Pz the seed
    assert view.slice.currentText() == "Pz"
    view.refresh(series, "Connectivity", 0.0, 10.0)
    m = view.head.map
    assert isinstance(m, HeadMap) and m.caption.startswith("from Pz")

    signed = FeatureSeries("tap_conn1", series.dims, series.coords, series.t,
                           (conn - 0.5)[None])  # fmt: skip
    view.refresh(signed, "Connectivity", 0.0, 10.0)  # both signs: diverging
    m = view.head.map
    assert isinstance(m, HeadMap) and m.scale[0].startswith("-")

    bands = FeatureSeries("tap_bp1", ("channel", "band"),
                          {"channel": labels, "band": ("alpha",)},
                          np.array([1.0]), np.ones((1, 4, 1)))  # fmt: skip
    view.refresh(bands, "Band power", 0.0, 10.0)
    assert view.head.isHidden()  # not connectivity: no map
    dispose(qapp, view)


def test_an_epochs_rail_tap_draws_without_error(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    raw = mne.io.RawArray(np.random.default_rng(2).standard_normal((3, 250 * 20))
                          * 1e-5, info, verbose="error")  # fmt: skip
    raw.set_annotations(mne.Annotations([2.0, 6.0, 10.0], [0.0] * 3, ["go"] * 3))
    rec = tmp_path / "erp_raw.fif"
    raw.save(rec, verbose="error")
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        win.insert_strip(len(win.doc.strips), "meegxer.nodes.epochs")
        win.start()
        wait_for(qapp, lambda: win.offline_state in ("done", "error"), 60)
        assert win.offline_state == "done", win.logbox.toPlainText()
        win.click_tap("epochs1:epochs", False)
        win.redraw()
        view = win.erp_view  # an epochs rail: its ERP (M10.3)
        assert win.views.currentWidget() is view
        assert "3 epochs (offline run)" in view.legend.text()
        assert view.reset.isHidden()  # nothing to reset offline
        assert view.channel.currentText() == "Cz"
        assert [name for name, _, _ in view.curves()] == ["go (n=3)"]
        view.condition.setCurrentText("go")  # one condition: the channels
        assert [name for name, _, _ in view.curves()] == ["Fz", "Cz", "Pz"]
        assert view.channel.isHidden() and not view.channels.isHidden()
    finally:
        win.close()


def test_a_live_erp_tap_counts_and_resets(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0\tgo\n"
                      "1.5\t0\tstop\n2.0\t0\tgo\n")  # fmt: skip
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        assert win.doc.source is not None
        win.doc.source.recording = {"events": {"file": str(events)}}
        win.insert_strip(len(win.doc.strips), "meegxer.nodes.epochs")
        win.click_tap("epochs1:epochs", False)
        win.start()
        assert win.running, win.logbox.toPlainText()

        def counts() -> tuple[int, ...]:
            frame = win.erp_frames.get("tap_epochs1__epochs")
            return () if frame is None else frame.counts

        wait_for(qapp, lambda: sum(counts()) == 3, 30)
        win.redraw()
        view = win.erp_view
        assert win.views.currentWidget() is view and not view.reset.isHidden()
        assert [name for name, _, _ in view.curves()] == ["go (n=2)", "stop (n=1)"]
        view.reset.click()  # start over: no epochs come after the third
        wait_for(qapp, lambda: counts() == (), 10)
        win.redraw()
        assert view.curves() == [] and "0 epochs" in view.legend.text()
        win.stop()
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)
