"""Signal view mouse behaviour and the stacked window."""

import subprocess
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from meegxer.gui.signal_view import SignalView, StackedWindow, TracePlot, Window
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


def test_wheel_scales_y_only_and_double_click_restores(qapp: QApplication) -> None:
    plot = TracePlot(load_theme("dark"))
    vb = plot.getViewBox()
    vb.setRange(xRange=(0, 10), yRange=(-50, 50), padding=0)
    (x0, y0) = vb.viewRange()
    vb.wheelEvent(Wheel(120))
    (x1, y1) = vb.viewRange()
    assert x1 == pytest.approx(x0)
    assert (y1[1] - y1[0]) < (y0[1] - y0[0])  # zoomed in on Y
    assert vb.state["mouseEnabled"] == [True, True]  # dragging pans X and Y

    class Click:
        def accept(self) -> None: ...

    vb.enableAutoRange(enable=False)
    vb.mouseDoubleClickEvent(Click())
    assert all(vb.autoRangeEnabled())


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
    stacked.close()


def test_live_scrollbar_follows_until_moved_back(qapp: QApplication) -> None:
    stacked = StackedWindow(load_theme("dark"))
    stacked.set_extent(100, 1000, 100, live=True)
    assert stacked.position() == 1000
    stacked.set_extent(110, 1010, 100, live=True)
    assert stacked.position() == 1010  # follows the stream
    stacked.scrollbar.setSliderPosition(500)
    stacked._moved(500)
    stacked.set_extent(120, 1020, 100, live=True)
    assert stacked.position() == 500  # stays where the user looks


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
        bar = win.stacked.scrollbar
        assert bar.maximum() == 250 * 50  # 60 s recording, 10 s shown
        bar.setValue(250 * 30)
        win.redraw()
        x = next(c for c in win.stacked.plot._curves if c.isVisible()).getData()[0]
        assert x[0] == pytest.approx(30.0, abs=0.1)  # seconds into the recording
        win.stacked.close()
        assert win.view.active and win.timeline.isEnabled()
    finally:
        win.close()
