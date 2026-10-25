"""The review window (docs/REVIEW.md): the offline run pauses at a review strip,
the marks made on the signal view are saved next to the recording, and the run
goes on."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from meegxer.core.review import Mark, load_review
from meegxer.gui.review_window import BAD_STYLE


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
        hp_tap = win._shown()[0]
        asked: list[bool] = []

        def ask_reviewer() -> str:
            asked.append(True)
            return "C. Rater"

        win.ask_reviewer = ask_reviewer  # type: ignore[method-assign]
        win.start()
        pump_until(qapp, lambda: win.review_window is not None)
        window = win.review_window
        assert window is not None and win.offline_state == "review"
        assert not win.start_btn.isEnabled()  # the run is not over
        labels = [window.label.itemText(i) for i in range(window.label.count())]
        assert labels[:2] == ["BAD_muscle", "BAD_eyes"]
        assert window.reject_words == ("BAD",) and window.duration == 20.0
        # the stacked traces fill, and what they show counts as looked at
        pump_until(qapp, lambda: bool(window.viewed))
        assert window.viewed[0][0] == 0.0 and window.coverage.shown_span is not None

        window.label.setCurrentText("drowsy")  # a new label, typed in
        window.selection.setRegion((2.0, 3.5))
        window.reject()  # not a reject word: the strip's first, BAD
        window.selection.setRegion((4.0, 5.0))
        window.annotate()
        assert [m["label"] for m in window.marks] == ["BAD", "drowsy"]
        assert [window.is_reject(m) for m in window.marks] == [True, False]
        assert "2 marks" in window.progress.text()
        assert "1.5 s rejected" in window.progress.text()
        assert sum(s.isVisible() for s in window._spans) == 2

        # the display: average reference and clinical units by default; the
        # core computes the page through the montage
        assert window.display() == {
            "montage": "average",
            "highpass": 0.0,
            "lowpass": 0.0,
            "notch": 0.0,
            "bads": [],
        }
        assert window.clinical and window.fixed_spacing(3) is not None
        assert window.source_labels == ("Fz", "Cz", "Pz")
        window.montage.setCurrentIndex(window.montage.findData("as recorded"))
        window.filters["highpass"].setValue(1.0)
        win.redraw()
        cached = win._offline_cache[f"{hp_tap}#review"]
        assert "'highpass': 1.0" in cached[0][3] and cached[1] is not None
        # palette keys, bad channels and undo
        assert window.palette_hint.text().startswith("Keys: 1 BAD_muscle · 2 BAD_eyes")
        window.selection.setRegion((6.0, 7.0))
        window.mark_with(1)  # key 2
        assert window.marks[-1]["label"] == "BAD_eyes"
        window.set_bad("Cz", True)
        assert window.display()["bads"] == ["Cz"]
        assert window.restyle("Cz", window.restyle("Fz", BAD_STYLE)) == BAD_STYLE
        assert window.bad_button.text() == "Bad channels (1)"
        window.undo()  # the bad channel
        window.undo()  # the BAD_eyes mark
        assert window.bads == [] and len(window.marks) == 2
        window.set_bad("Pz", True)

        window.page(1)  # the next page is shown, then counted
        pump_until(qapp, lambda: window.viewed[-1][1] >= 15.0)
        window.seek(17.0)
        pump_until(qapp, lambda: window.viewed[-1][1] >= 20.0)
        assert window.viewed == [(0.0, 20.0)]
        window.submit()  # all looked at: no question
        pump_until(qapp, lambda: win.offline_state == "done")
        assert win.review_window is None

        saved = load_review(tmp_path / f"sub-01_raw_desc-{review.id}_events.tsv")
        assert saved is not None
        assert saved.marks == (Mark(2.0, 1.5, "BAD"), Mark(4.0, 1.0, "drowsy"))
        assert saved.viewed == ((0.0, 20.0),) and saved.bads == ("Pz",)
        assert saved.reviewer == "C. Rater" and len(asked) == 1  # once per run
        win.start()  # reviewed on this input: runs through without a review
        pump_until(qapp, lambda: win.offline_state == "done")
        assert win.review_window is None
        win.ask_reviewer = lambda: None  # type: ignore[method-assign]
        win.start()  # no name given: the run does not start
        assert win.offline_state == "done"
    finally:
        win.close()


def test_coverage_helpers() -> None:
    from meegxer.gui.review_window import covered, first_gap, merge

    spans = merge([(5.0, 8.0), (0.0, 2.0), (1.5, 3.0)])
    assert spans == [(0.0, 3.0), (5.0, 8.0)]
    assert covered(spans, 10.0) == pytest.approx(0.6)
    assert first_gap(spans, 10.0) == 3.0
    assert first_gap(spans, 10.0, after=6.0) == 8.0
    assert first_gap([(0.0, 3.0), (5.0, 10.0)], 10.0, after=6.0) == 3.0  # wraps
    assert first_gap([(0.0, 10.0)], 10.0) is None
