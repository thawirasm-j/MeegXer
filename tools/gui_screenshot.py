"""Render the GUI offscreen against a fresh daemon and save screenshots.

Used for the M5 report and visual checks: ``uv run python tools/gui_screenshot.py
OUT_DIR``. Starts a daemon on free ports, builds the default chain (Synthetic ->
high-pass -> notch), runs it live for a few seconds and grabs the window in the
dark and light themes, the difference and spectrum modes, advanced mode, an
incompatible strip and the stacked window.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
        time.sleep(0.01)


def main() -> int:
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
        win.insert_strip(3, "meegxer.sterile.iir_filter_zerophase")
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
        print("\n".join(win.logbox.toPlainText().splitlines()[-12:]))
    finally:
        win.close()
        pump(app, 0.3)
        if proc.poll() is None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
