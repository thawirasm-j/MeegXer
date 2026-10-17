"""Signal view mouse behaviour and the stacked window."""

import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from meegxer.gui.signal_view import (
    Panner,
    SignalView,
    StackedWindow,
    TracePlot,
    Window,
)
from meegxer.gui.theme import load_theme

CHANNELS = ["Fp1", "Fp2", "Cz", "O1", "EOG"]


class Wheel:
    def __init__(self, delta: int) -> None:
        self._delta = delta

    def delta(self) -> int:
        return self._delta

    def pos(self) -> QPointF:
        return QPointF(10, 10)

    def accept(self) -> None: ...

    def ignore(self) -> None: ...


def window(t0: float = 0.0, seconds: float = 10.0, scale: Any = 1.0) -> Window:
    t = t0 + np.arange(int(seconds * 50)) / 50
    rng = np.random.default_rng(0)
    mid = rng.standard_normal((len(CHANNELS), len(t))) * 1e-5
    mid *= np.asarray(scale, dtype=float).reshape(-1, 1)
    return Window("tap", tuple(CHANNELS), t, mid - 1e-6, mid + 1e-6,
                  np.zeros(len(t), dtype=bool), 50.0)  # fmt: skip


def dispose(qapp: QApplication, *widgets: QWidget) -> None:
    """Delete parentless widgets now: left to interpreter exit, plot widgets
    can outlive the application and crash the process on the way out."""
    for w in widgets:
        w.close()
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_mouse_moves_y_only_and_double_click_fits_y(qapp: QApplication) -> None:
    plot = TracePlot(load_theme("dark"))
    vb = plot.getViewBox()
    vb.setRange(xRange=(0, 10), yRange=(-50, 50), padding=0)
    (x0, y0) = vb.viewRange()
    vb.wheelEvent(Wheel(120))
    (x1, y1) = vb.viewRange()
    assert x1 == pytest.approx(x0)
    assert (y1[1] - y1[0]) < (y0[1] - y0[0])  # zoomed in on Y
    assert vb.state["mouseEnabled"] == [False, True]  # time is the panner's

    class Click:
        def accept(self) -> None: ...

    vb.enableAutoRange(enable=False)
    vb.mouseDoubleClickEvent(Click())
    assert not vb.autoRangeEnabled()[0] and vb.autoRangeEnabled()[1]
    dispose(qapp, plot)


def test_time_axis_shows_exactly_the_window(qapp: QApplication) -> None:
    """No blank margins left or right: X is the window's span, edge to edge."""
    view = SignalView(load_theme("dark"))
    view.refresh(lambda p, s: window(t0=-10.0), "tap", [])
    (x0, x1), _ = view.plot.getViewBox().viewRange()
    assert x0 == pytest.approx(-10.0)
    assert x1 == pytest.approx(0.0)
    dispose(qapp, view)


def test_stacked_has_own_channels_all_by_default_and_normalizes(
    qapp: QApplication,
) -> None:
    view = SignalView(load_theme("dark"))
    view.types = {"EOG": "eog"}
    view.refresh(lambda p, s: window(), "tap", [])
    assert "EOG" in view.channels.hidden  # the signal view hides non-EEG
    stacked = StackedWindow(load_theme("dark"))
    stacked.show()
    scale = [1, 1, 1, 1, 100]  # EOG a hundred times larger
    stacked.refresh(view, lambda p, s: window(scale=scale), "tap", [])
    assert not stacked.channels.hidden  # all channels by default
    assert stacked.channels.labels == tuple(CHANNELS)

    def spreads() -> list[float]:
        curves = [c for c in stacked.plot._curves if c.isVisible()]
        return [float(np.ptp(c.getData()[1])) for c in curves]

    raw = spreads()
    assert raw[-1] > 20 * raw[0]
    stacked.normalize.setChecked(True)
    assert not stacked.spacing.isEnabled()
    stacked.refresh(view, lambda p, s: window(scale=scale), "tap", [])
    norm = spreads()
    assert norm[-1] < 2 * norm[0]  # comparable rows after normalizing
    dispose(qapp, stacked, view)


def test_live_panner_follows_until_moved_back(qapp: QApplication) -> None:
    panner = Panner()
    panner.set_extent(100, 1000, 100, live=True)
    assert panner.position() == 1000
    assert not panner.latest.isEnabled()
    panner.set_extent(110, 1010, 100, live=True)
    assert panner.position() == 1010  # follows the stream
    panner.bar.setSliderPosition(500)
    panner._moved(500)
    panner.set_extent(120, 1020, 100, live=True)
    assert panner.position() == 500  # stays where the user looks
    assert panner.latest.isEnabled()
    panner.follow()  # "Latest"
    panner.set_extent(130, 1030, 100, live=True)
    assert panner.position() == 1030
    dispose(qapp, panner)


def test_stacked_rests_the_signal_view_and_scrolls_the_recording(
    qapp: QApplication,
    daemon: tuple[str, subprocess.Popen[bytes]],
    tmp_path: Path,
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    names = ["Fp1", "Fp2", "Cz", "O1"]
    data = np.random.default_rng(1).standard_normal((4, 250 * 60)) * 1e-5
    info = mne.create_info(names, 250.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    rec = tmp_path / "rec_raw.fif"
    raw.save(rec, verbose="error")
    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        win.start()
        deadline = time.monotonic() + 60
        while win.offline_state != "done":
            assert win.offline_state != "error", win.logbox.toPlainText()
            assert time.monotonic() < deadline
            qapp.processEvents()
            time.sleep(0.05)
        qapp.processEvents()
        win.show_stacked()
        assert not win.view.active and not win.timeline.isEnabled()
        win.redraw()
        bar = win.stacked.panner.bar
        assert bar.maximum() == 250 * 50  # 60 s recording, 10 s shown
        bar.setValue(250 * 30)
        win.redraw()
        x = next(c for c in win.stacked.plot._curves if c.isVisible()).getData()[0]
        assert x[0] == pytest.approx(30.0, abs=0.1)  # seconds into the recording
        win.stacked.close()
        assert win.view.active and win.timeline.isEnabled()
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)


def test_comparison_stays_a_comparison_while_main_has_no_data(
    qapp: QApplication,
) -> None:
    """Main still calibrating (no data), a comparison tap has data: draw it
    thin and faded and say Main has no data yet, never as the Main trace."""
    view = SignalView(load_theme("dark"))
    windows = {"tap_hp": window()}
    view.refresh(lambda p, s: windows.get(p), "tap_ica", ["tap_hp"])
    curves = [c for c in view.plot._curves if c.isVisible()]
    assert curves and all(c.opts["pen"].widthF() == 1.0 for c in curves)
    assert "Main: tap_ica (no data yet)" in view.legend.text()
    windows["tap_ica"] = window()  # Main arrives: drawn thick
    view.refresh(lambda p, s: windows.get(p), "tap_ica", ["tap_hp"])
    widths = {c.opts["pen"].widthF() for c in view.plot._curves if c.isVisible()}
    assert widths == {1.0, 2.0}

    stacked = StackedWindow(load_theme("dark"))
    stacked.show()
    del windows["tap_ica"]
    stacked.refresh(view, lambda p, s: windows.get(p), "tap_ica", ["tap_hp"])
    curves = [c for c in stacked.plot._curves if c.isVisible()]
    assert curves and all(c.opts["pen"].widthF() == 1.0 for c in curves)
    dispose(qapp, stacked, view)


def test_offline_strips_fit_on_the_whole_recording(qapp: QApplication) -> None:
    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import StripWidget
    from meegxer.gui.pipeline_doc import PipelineDoc

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.file")
    strip = doc.insert(1, "meegxer.nodes.ica")

    def texts(mode: str) -> set[str]:
        doc.input_mode = mode  # type: ignore[assignment]
        widget = StripWidget(doc, strip, load_theme("dark"), status="stopped",
                             advanced=False, selected=False)  # fmt: skip
        from PySide6.QtWidgets import QLabel, QPushButton

        found = {w.text() for w in widget.findChildren(QPushButton)} | {
            w.text() for w in widget.findChildren(QLabel)
        }
        widget.deleteLater()  # parentless: delete while the app runs, not at exit
        qapp.processEvents()
        return found

    assert "Calibrate" not in texts("offline")
    assert "fits on the whole recording" in texts("offline")
    assert "Calibrate" in texts("live")


def test_start_without_positions_opens_the_channels_tab(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.insert_strip(len(win.doc.strips), "meegxer.nodes.ica")
        win.start()
        assert not win.running
        assert "Cannot start: ica1 needs electrode positions" in (
            win.logbox.toPlainText()
        )
        assert win.recording_window.isVisible()
        assert win.recording_window.tabs.currentIndex() == 0  # Channels
        win.recording_window.close()
        assert win.doc.source is not None
        win.doc.source.recording = {"montage": "colin27_1020"}
        win.start()
        assert win.running
        win.stop()
    finally:
        win.close()
        dispose(qapp, win.stacked, win.recording_window, win)


def test_chain_puts_every_strip_on_the_signal_and_rails_above(
    qapp: QApplication,
) -> None:
    """docs/RAILS.md: one chain; a feature strip starts a rail, the output that
    reads it draws from it; CSP without events is marked."""
    from meegxer.core.control import manifest_to_dict
    from meegxer.core.registry import Registry
    from meegxer.gui.chain import ChainView, RailBoard, StripWidget
    from meegxer.gui.pipeline_doc import PipelineDoc

    registry = Registry.from_entry_points()
    doc = PipelineDoc([manifest_to_dict(registry.get(n).manifest)
                       for n in registry.names()])  # fmt: skip
    doc.set_source("meegxer.core.synthetic")
    for node in ["meegxer.nodes.rereference", "meegxer.nodes.band_power",
                 "meegxer.core.lsl_features_out", "meegxer.nodes.ica",
                 "meegxer.nodes.csp"]:  # fmt: skip
        doc.insert(len(doc.strips), node)
    chain = ChainView()
    chain.rebuild(doc, load_theme("dark"), statuses={}, taps={}, advanced=False,
                  selected=None)  # fmt: skip
    board = chain.widget()
    assert isinstance(board, RailBoard)
    strips = chain.findChildren(StripWidget)
    assert len(strips) == len(doc.strips)
    chain.resize(1800, 400)
    chain.show()
    qapp.processEvents()
    tops = {w.strip_id: w.mapTo(board, QPoint(0, 0)).y() for w in strips}
    on_signal = {tops[s] for s in ("input", "rereferenc1", "ica1")}
    lowered = {tops[s] for s in ("bandpower1", "lslfeature1", "csp1")}
    assert len(on_signal) == 1 and len(lowered) == 1  # one chain, none forked
    assert min(lowered) > max(on_signal)  # features and outputs sit lower
    assert [(r.label, r.readers, r.lane) for r in board.rails] == [
        ("Band power", ["lslfeature1"], 1), ("CSP", [], 0)]  # fmt: skip
    assert board.needs == {"csp1": "Events"}
    assert board.lanes == 3  # a row per rail, one for Events
    labels = {
        w.strip_id: label.text()
        for w in strips
        if (label := w.findChild(QLabel, "group")) is not None
    }
    assert labels["bandpower1"] == "◆ FEATURES"
    assert labels["ica1"] == "● PREPROCESSING"
    chain.deleteLater()
    qapp.processEvents()


def test_themes_without_group_colours_get_the_defaults(tmp_path: Path) -> None:
    import json

    from meegxer.gui.theme import DEFAULT_GROUPS

    data = json.loads((Path(__file__).parents[1] / "src" / "meegxer" / "gui"
                       / "themes" / "dark.json").read_text())  # fmt: skip
    del data["groups"]
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(data))
    assert load_theme(path).group_color("features") == DEFAULT_GROUPS["features"]
    assert load_theme("light").group_color("features") != DEFAULT_GROUPS["features"]


def test_live_legend_shows_each_tap_delay(qapp: QApplication) -> None:
    view = SignalView(load_theme("dark"))
    view.names = {"tap": "after Notch", "tap_in": "Input"}
    view.delays = {"tap": 0.75, "tap_in": 0.0}
    view.delay_note = "Estimated delay behind the input"
    view.refresh(lambda p, s: replace(window(), probe=p), "tap", ["tap_in"])
    assert view.legend.text() == "Main: after Notch +0.8 ms · Input"
    assert view.legend.toolTip() == "Estimated delay behind the input"
    view.delays = {}  # offline or stopped: names only
    view.refresh(lambda p, s: replace(window(), probe=p), "tap", ["tap_in"])
    assert view.legend.text() == "Main: after Notch · Input"
    dispose(qapp, view)


def test_the_live_window_is_always_the_full_span(qapp: QApplication) -> None:
    """A stream younger than the span: the window still covers the whole span,
    blank before the first data, so the time axis does not change size."""
    from types import SimpleNamespace

    from meegxer.core.display import DisplayFrame, Envelope
    from meegxer.gui.buffers import ProbeBuffer
    from meegxer.gui.main_window import MainWindow

    buf = ProbeBuffer()
    mins = np.zeros((2, 250))  # 2 s at 125 bins per second
    buf.add(DisplayFrame("tap", 0, ("Cz", "Pz"), 250.0, Envelope(0, 2, mins, mins)))
    host = SimpleNamespace(buffers={"tap": buf}, _live_reference=lambda b: b,
                           _since_start=lambda: False)  # fmt: skip
    win = MainWindow._live_window(host, "tap", 10.0)  # type: ignore[arg-type]
    assert win is not None
    assert win.t[0] == pytest.approx(-10.0) and len(win.t) == 1250
    assert np.isnan(win.mins[:, :1000]).all() and not np.isnan(win.mins[:, -250:]).any()


def test_the_live_time_axis_can_count_from_the_stream_start(
    qapp: QApplication,
) -> None:
    """Setting ``time_axis: since_start``: time is the sample index on the
    source's clock, so the ticks run on; the default counts back from now."""
    from types import SimpleNamespace

    from meegxer.core.display import DisplayFrame, Envelope
    from meegxer.gui.buffers import ProbeBuffer
    from meegxer.gui.main_window import MainWindow

    buf = ProbeBuffer()
    mins = np.zeros((1, 1500))  # bins 1000-2500 at 125 per second: 8 s to 20 s
    buf.add(DisplayFrame("tap", 0, ("Cz",), 250.0, Envelope(1000, 2, mins, mins)))
    for since, first, last in ((False, -10.0, -0.008), (True, 10.0, 19.992)):
        host = SimpleNamespace(buffers={"tap": buf}, _live_reference=lambda b: b,
                               _since_start=lambda since=since: since)  # fmt: skip
        win = MainWindow._live_window(host, "tap", 10.0)  # type: ignore[arg-type]
        assert win is not None
        assert win.t[0] == pytest.approx(first) and win.t[-1] == pytest.approx(last)

    panner = Panner()
    panner.set_extent(0, 100, 10, live=True)
    panner.following = False
    panner.show_span(-6.0, -1.0)
    assert panner.where.text() == "1.0 s behind"
    panner.since_start = True
    panner.show_span(12.0, 17.0)
    assert panner.where.text() == "12.0 – 17.0 s"  # noqa: RUF001
    panner.deleteLater()
