"""M5 exit (docs/WORKPLAN.md): build Synthetic -> HP -> probe -> notch -> probe, see
the 50 Hz line in difference mode, and kill the GUI without interrupting
processing."""

import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from meegxer.core.control import ControlClient


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_difference_mode_shows_the_removed_50_hz_line(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]]
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow
    from meegxer.gui.signal_view import _psd

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        assert [s.node for s in win.doc.strips] == [
            "meegxer.core.synthetic",
            "meegxer.sterile.iir_filter",
            "meegxer.sterile.notch",
        ]
        win.start()
        win.click_tap("notch1", make_main=True)
        win.click_tap("input", make_main=False)  # hide the input tap
        win.click_tap("iirfilter1", make_main=False)  # compare with the high-pass
        main, comps = win._shown()
        assert main == "tap_notch1" and comps == ["tap_iirfilter1"]
        pump(qapp, 5.0)
        a = win._live_window("tap_notch1", 4.0)
        b = win._live_window("tap_iirfilter1", 4.0)
        assert a is not None and b is not None
        diff = ((a.mins + a.maxs) - (b.mins + b.maxs))[0] / 2
        f, p = _psd(diff[-500:], a.bin_rate)
        assert f[np.argmax(p)] == pytest.approx(50.0, abs=1.5)
        win.view._set_mode("difference")
        pump(qapp, 0.3)
        assert "what this stretch removed" in win.view.legend.text()
    finally:
        win.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_killing_the_gui_does_not_interrupt_processing(
    daemon: tuple[str, subprocess.Popen[bytes]],
) -> None:
    control, _ = daemon
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    gui = subprocess.Popen(
        [sys.executable, "-m", "meegxer.gui.app", "--connect", control, "--start"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = ControlClient(control, timeout=5)
    try:
        deadline = time.monotonic() + 30
        while not client.call("status").get("running"):
            assert time.monotonic() < deadline, "the GUI did not start the pipeline"
            time.sleep(0.2)
        os.kill(gui.pid, signal.SIGKILL)
        gui.wait(5)
        before = client.call("status")["samples"]
        time.sleep(1.0)
        status = client.call("status")
        assert status["running"] and status["samples"] > before + 300
    finally:
        if gui.poll() is None:
            gui.kill()
        client.close()
