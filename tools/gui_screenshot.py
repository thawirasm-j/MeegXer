"""Render the GUI offscreen against a fresh daemon and save screenshots.

Used for the M5 report and visual checks: ``uv run python tools/gui_screenshot.py
OUT_DIR``. Starts a daemon on free ports, builds the default chain (Synthetic ->
high-pass -> notch), runs it live for a few seconds and grabs the window in the
dark and light themes, the difference and spectrum modes, advanced mode, an
incompatible strip, the stacked window, the chain's rails (docs/RAILS.md) and the
settings pages. Uses a throwaway
config folder, so your own settings are neither used nor changed.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["MEEGXER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="meegxer-shots-")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from meegxer.gui.connection import DaemonLink, spawn_daemon
from meegxer.gui.theme import load_theme


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        # widgets replaced with deleteLater go away in a real event loop
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.01)


def demo_recording(path: Path) -> Path:
    """A 2-minute 32-channel recording with alpha, a 50 Hz line, events and one
    channel stored under a non-standard name (to rename in the window)."""
    import mne
    import numpy as np

    names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "PO3",
        "PO4",
        "O1",
        "Oz",
        "O2",
        "HEOG",
        "VEOG",
    ]
    sfreq, n = 250.0, 250 * 120
    t = np.arange(n) / sfreq
    rng = np.random.default_rng(1)
    data = rng.standard_normal((len(names), n)) * 8e-6
    data += (
        12e-6 * np.sin(2 * np.pi * 10 * t) * np.linspace(0.3, 1, len(names))[:, None]
    )
    data += 4e-6 * np.sin(2 * np.pi * 50 * t)
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, "eeg"), verbose="error")
    onsets = np.arange(4.0, 118.0, 3.5)
    raw.set_annotations(
        mne.Annotations(
            onsets,
            0.0,
            ["target" if i % 3 == 0 else "standard" for i in range(len(onsets))],
        )
    )
    raw.save(path, overwrite=True, verbose="error")
    return path


def main() -> int:
    # it closes the window while running: no question box to wait on offscreen
    Path(os.environ["MEEGXER_CONFIG_DIR"], "settings.json").write_text(
        '{"general": {"confirm_quit_running": false}}', encoding="utf-8"
    )
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots")
    out.mkdir(parents=True, exist_ok=True)
    control = f"tcp://127.0.0.1:{free_port()}"
    display = f"tcp://127.0.0.1:{free_port()}"
    proc = spawn_daemon(control, display)
    app = QApplication(sys.argv[:1])
    from meegxer.gui.main_window import MainWindow

    win = MainWindow(DaemonLink(control), load_theme("dark"), owns_daemon=True)
    win.resize(1440, 900)
    win.show()
    try:
        win.start()
        pump(app, 4.0)
        win.select("iirfilter1")
        pump(app, 0.5)
        win.grab().save(str(out / "01-live-dark.png"))
        win.view._set_mode("difference")
        pump(app, 1.0)
        win.grab().save(str(out / "02-difference-dark.png"))
        win.view._set_mode("spectrum")
        win.view.span.setCurrentIndex(3)
        pump(app, 1.0)
        win.grab().save(str(out / "03-spectrum-dark.png"))
        win.view._set_mode("overlay")
        win.view.span.setCurrentIndex(2)
        win.set_advanced(True)
        win.apply_theme(load_theme("light"))
        pump(app, 1.0)
        win.grab().save(str(out / "04-live-light-advanced.png"))
        win.stop()
        win.insert_strip(3, "meegxer.nodes.iir_filter_zerophase")
        win.apply_theme(load_theme("dark"))
        pump(app, 0.5)
        win.grab().save(str(out / "05-incompatible-dark.png"))
        win.remove_strip("iirfilterz1")
        win.toggle_mode()  # offline
        assert win.doc.source is not None
        win.doc.source.params.update({"duration": 30.0})
        win.start()
        pump(app, 4.0)
        win.grab().save(str(out / "06-offline-dark.png"))
        win.toggle_mode()
        win.start()
        pump(app, 3.0)
        win.show_stacked()
        pump(app, 1.0)
        win.stacked.grab().save(str(out / "07-stacked-dark.png"))
        win.stacked.close()
        win.stop()
        # a recorded file: Recording window, then the offline view with events
        rec = demo_recording(out / "demo_raw.fif")
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        win.open_recording()
        rw = win.recording_window
        rw.resize(1000, 640)
        rw.montage.setCurrentText("colin27_1005")
        rw.apply()
        pump(app, 0.5)
        rw.grab().save(str(out / "08-recording-channels.png"))
        rw.tabs.setCurrentIndex(1)
        pump(app, 0.5)
        rw.grab().save(str(out / "09-recording-montage.png"))
        rw.tabs.setCurrentIndex(2)
        pump(app, 0.3)
        rw.grab().save(str(out / "10-recording-events.png"))
        rw.close()
        win.start()
        pump(app, 3.0)
        win.timeline.bar.setValue(250 * 8)
        pump(app, 1.0)
        win.grab().save(str(out / "11-offline-events-dark.png"))
        # rails: feature strips start named rails, outputs pick one, CSP reads Events
        for node in [
            "meegxer.nodes.psd",
            "meegxer.core.feature_tsv",
            "meegxer.nodes.band_power",
            "meegxer.nodes.csp",
            "meegxer.core.lsl_features_out",
            "meegxer.core.bids_features",
        ]:
            win.insert_strip(len(win.doc.strips), node)
        ids = [s.id for s in win.doc.strips]
        bids = next(i for i in ids if i.startswith("bidsfeatur"))
        win.change_input(bids, next(i for i in ids if i.startswith("bandpower")))
        win.select(bids)
        win.resize(1900, 900)
        pump(app, 0.5)
        win.grab().save(str(out / "13-rails-dark.png"))
        win.apply_theme(load_theme("light"))
        pump(app, 0.5)
        win.grab().save(str(out / "14-rails-light.png"))
        win.apply_theme(load_theme("dark"))
        win.resize(1440, 900)
        from meegxer.gui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(win.settings, win.theme, win)
        dialog.show()
        for page in range(dialog.pages.count()):
            dialog.pages.setCurrentRow(page)
            pump(app, 0.3)
            dialog.grab().save(str(out / f"12-settings-{page + 1}.png"))
        dialog.reject()
        # feature views: a feature strip's tap (spectrum, over time), the ICs rail
        for sid in [s.id for s in win.doc.strips[3:]]:
            win.remove_strip(sid)
        for node in ["meegxer.nodes.ica", "meegxer.nodes.psd",
                     "meegxer.nodes.band_power"]:  # fmt: skip
            win.insert_strip(len(win.doc.strips), node)
        assert win.doc.source is not None
        win.doc.source.recording = {
            "montage": "colin27_1005",
            "channels": {"HEOG": {"type": "eog"}, "VEOG": {"type": "eog"}},
        }
        win.start()
        pump(app, 12.0)
        for key, name in [("psd1", "15-features-spectrum"),
                          ("bandpower1", "16-features-over-time"),
                          ("ica1:ics", "17-ics-classes")]:  # fmt: skip
            win.click_tap(key, False)
            win.feature_view.span.setCurrentIndex(1)  # 30 s
            pump(app, 1.0)
            win.grab().save(str(out / f"{name}-dark.png"))
        win.click_tap("ica1:ics", False)
        print("\n".join(win.logbox.toPlainText().splitlines()[-12:]))
    finally:
        win.close()
        pump(app, 0.3)
        if proc.poll() is None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
