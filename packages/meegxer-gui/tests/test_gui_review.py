"""The review window (docs/REVIEW.md): the offline run pauses at a review strip,
the marks made on the signal view are saved next to the recording, and the run
goes on."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
from PySide6.QtWidgets import QApplication

from meegxer.core.review import Mark, load_review


def pump_until(
    app: QApplication, done: Callable[[], bool], seconds: float = 30
) -> None:
    end = time.monotonic() + seconds
    while not done():
        assert time.monotonic() < end, "timed out"
        app.processEvents()
        time.sleep(0.02)


def test_the_run_pauses_for_marks_and_goes_on(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    info = mne.create_info(["Fz", "Cz", "Pz"], 250.0, "eeg")
    data = np.random.default_rng(0).standard_normal((3, 250 * 20)) * 1e-5
    rec = tmp_path / "sub-01_raw.fif"
    mne.io.RawArray(data, info, verbose="error").save(rec, verbose="error")

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        hp = win.doc.insert(len(win.doc.strips), "meegxer.nodes.iir_filter")
        review = win.doc.insert(len(win.doc.strips), "meegxer.core.review")
        review.params["labels"] = "BAD_muscle, BAD_eyes"
        win._structure_changed("added")
        win.click_tap(hp.id, False)  # see the signal before the strip
        win.start()
        pump_until(qapp, lambda: win.review_window is not None)
        window = win.review_window
        assert window is not None and win.offline_state == "review"
        assert not win.start_btn.isEnabled()  # the run is not over
        labels = [window.label.itemText(i) for i in range(window.label.count())]
        assert labels[:2] == ["BAD_muscle", "BAD_eyes"]

        window.label.setCurrentText("BAD_eyes")
        window.selection.setRegion((2.0, 3.5))
        window.mark_selection()
        window.label.setCurrentText("drowsy")  # a new label, typed in
        window.selection.setRegion((10.0, 12.0))
        window.mark_selection()
        assert window.count.text() == "2 marks"
        assert sum(s.isVisible() for s in window._spans) == 2
        window.submit()
        pump_until(qapp, lambda: win.offline_state == "done")
        assert win.review_window is None
        assert window.selection not in win.view.plot.getPlotItem().items

        saved = load_review(tmp_path / f"sub-01_raw_desc-{review.id}_events.tsv")
        assert saved is not None
        assert saved.marks == (Mark(2.0, 1.5, "BAD_eyes"), Mark(10.0, 2.0, "drowsy"))
        win.start()  # reviewed on this input: runs through without asking
        pump_until(qapp, lambda: win.offline_state == "done")
        assert win.review_window is None
    finally:
        win.close()
