"""The review window (docs/REVIEW.md): the offline run pauses at a review strip,
the marks made on the signal view are saved next to the recording, and the run
goes on."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import mne
import numpy as np
import pyqtgraph as pg
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

        assert not window.has_selection()  # nothing until a span is dragged
        window.annotate()  # so nothing is marked
        assert window.marks == [] and "Drag on the traces" in window.progress.text()
        window.label.setCurrentText("drowsy")  # a new label, typed in
        window.select(2.0, 3.5)
        window.reject()  # not a reject word: the strip's first, BAD
        window.select(4.0, 5.0)
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
        window.select(6.0, 7.0)
        window.mark_with(1)  # key 2
        assert window.marks[-1]["label"] == "BAD_eyes"
        assert window.editing is window.marks[-1]  # the span stays on it
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


class FakeDrag:
    """Enough of a pyqtgraph mouse drag or click event for the handlers."""

    def __init__(self, start: object, now: object, *, first: bool = False,
                 last: bool = False) -> None:  # fmt: skip
        self._start, self._now, self._first, self._last = start, now, first, last
        self.accepted = False

    def isStart(self) -> bool:
        return self._first

    def isFinish(self) -> bool:
        return self._last

    def buttonDownScenePos(self) -> object:
        return self._start

    def scenePos(self) -> object:
        return self._now

    def button(self) -> object:
        from PySide6.QtCore import Qt

        return Qt.MouseButton.LeftButton

    def accept(self) -> None:
        self.accepted = True


def test_spans_are_dragged_adjusted_and_channels_clicked(qapp: QApplication) -> None:
    from meegxer.gui.review_window import ReviewWindow
    from meegxer.gui.theme import load_theme

    request = {
        "node": "rev",
        "reason": "not reviewed",
        "labels": ["BAD", "spike"],
        "duration_s": 60.0,
        "reject": ["BAD"],
        "marks": [],
        "viewed": [],
    }
    sent: list[tuple[object, ...]] = []
    window = ReviewWindow(
        request,
        load_theme("dark"),
        submit=lambda *a: sent.append(a),
        cancel=lambda: None,
    )
    window.resize(900, 600)
    window.show()
    qapp.processEvents()
    window.plot.set_time(0.0, 10.0)
    window.shown(0.0, 10.0)
    box = window.plot.getViewBox()

    def scene(t: float) -> object:
        return box.mapViewToScene(pg.Point(t, 0.0))

    # drag from the start of a span to its end (right to left works too)
    for ev in (FakeDrag(scene(5.0), scene(4.0), first=True),
               FakeDrag(scene(5.0), scene(3.0), last=True)):  # fmt: skip
        assert box.drag_handler is not None and box.drag_handler(ev) and ev.accepted
    s0, s1 = window.selection.getRegion()
    assert s0 == pytest.approx(3.0, abs=0.05) and s1 == pytest.approx(5.0, abs=0.05)
    window.mark_with(1)
    mark = window.marks[0]
    assert mark["label"] == "spike" and window.editing is mark
    # its edges follow the span; a label key relabels it
    window.selection.setRegion((3.0, 6.0))
    window._region_moved()
    assert mark["duration"] == pytest.approx(3.0)
    window.mark_with(0)
    assert [m["label"] for m in window.marks] == ["BAD"]
    # a span longer than the page (the mark follows) shows where it ends
    window.selection.setRegion((3.0, 25.0))
    window._place_overlays()
    left, right = window._edge_tabs
    assert right.isVisible() and not left.isVisible()
    assert "25.0 s" in right.textItem.toPlainText()
    # Escape lets go; a click on the mark selects it again; Delete removes it
    window.clear_selection()
    assert window.editing is None and not window.has_selection()
    click = FakeDrag(scene(4.0), scene(4.0))
    assert box.click_handler is not None and box.click_handler(click)
    assert window.editing is window.marks[0]
    window.remove_selected()
    assert window.marks == []
    window.undo()
    assert len(window.marks) == 1
    # a click away from any mark picks an instant
    box.click_handler(FakeDrag(scene(28.0), scene(28.0)))  # past the mark
    s0, s1 = window.selection.getRegion()
    assert window.editing is None and s0 == pytest.approx(s1)

    # channel names: a pill per row; a click toggles bad, a pair asks which
    from meegxer.gui.colors import ChannelStyle

    window.source_labels = ("Fp1", "F7", "Cz")
    window.rows_drawn([("Fp1", 200.0, ChannelStyle((200, 0, 0))),
                       ("Cz", 100.0, ChannelStyle((0, 200, 0)))])  # fmt: skip
    tags = [t for t in window._channel_tags if t.isVisible()]
    assert [t.label for t in tags] == ["Fp1", "Cz"]
    window.channel_clicked("Cz")
    assert window.bads == ["Cz"]
    window.channel_clicked("Cz")
    assert window.bads == []
    assert window.electrodes("Fp1-F7") == ["Fp1", "F7"]
    menu = window.channel_menu("Fp1-F7")
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts[:2] == ["Fp1 is bad", "F7 is bad"] and "Hide Fp1-F7" in texts
    menu.actions()[1].setChecked(True)  # F7 bad, from the menu
    assert window.bads == ["F7"]
    window.answered = True
    window.close()


def test_review_points_are_stepped_through_and_answered(qapp: QApplication) -> None:
    from meegxer.gui.review_window import ReviewWindow
    from meegxer.gui.theme import load_theme

    points = [{"onset": 5.0, "duration": 1.0, "label": "beta"},
              {"onset": 20.0, "duration": 0.5, "label": "beta"},
              {"onset": 40.0, "duration": 0.8, "label": "beta"}]  # fmt: skip
    request = {
        "node": "rev",
        "reason": "not reviewed",
        "duration_s": 60.0,
        "labels": ["spindle", "not_spindle"],
        "reject": ["BAD"],
        "marks": [],
        "viewed": [],
        "points": points,
        "display": {"montage": "longitudinal bipolar", "lowpass": 30.0},
    }
    sent: list[tuple[object, ...]] = []
    window = ReviewWindow(
        request,
        load_theme("dark"),
        submit=lambda *a: sent.append(a),
        cancel=lambda: None,
    )
    window.seek_rate = 100.0
    # the strip's display is where the window starts; the reviewer can change it
    assert window.montage.currentData() == "longitudinal bipolar"
    assert window.montage.isEnabled() and window.filters["lowpass"].isEnabled()
    assert window.filters["lowpass"].value() == 30.0
    assert window.point_label.isVisibleTo(window)
    assert window.point_label.text() == "Point –/3 · 0 answered"  # noqa: RUF001

    window.step(1)  # → : the first point, selected
    assert window.point_index == 0 and window.selection.getRegion() == (5.0, 6.0)
    window.selection.setRegion((4.8, 6.3))  # the physician retimes it
    window.mark_with(0)  # spindle; then on to the next point
    first = window.marks[0]
    assert (first["label"], first["onset"], first["point"]) == ("spindle", 4.8, 5.0)
    assert window.point_index == 1 and window.point_label.text().endswith("1 answered")
    window.auto_next.setChecked(False)
    window.mark_with(1)  # not a spindle; stays at this point
    assert window.marks[1]["point"] == 20.0 and window.point_index == 1
    window.step(-1)  # back to the first: its answer is selected to change
    assert window.editing is first
    window.mark_with(1)  # relabelled
    assert first["label"] == "not_spindle" and len(window.marks) == 2
    assert window.coverage.points == [(5.0, True), (20.0, True), (40.0, False)]
    window.page(1)  # away from the points: a new span answers none
    window.select(50.0, 51.0)
    window.mark_with(0)
    assert "point" not in window.marks[-1]

    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []

    def answer(*args: object) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.Yes

    original = QMessageBox.question
    QMessageBox.question = answer  # type: ignore[assignment]
    try:
        window.submit()
    finally:
        QMessageBox.question = original  # type: ignore[method-assign]
    assert asked and "1 of 3 review points" in asked[0]
    marks = sent[0][0]
    assert isinstance(marks, list) and [m.get("point") for m in marks] == [
        5.0,
        20.0,
        None,
    ]


def test_a_recording_is_reviewed_at_import(
    qapp: QApplication, daemon: tuple[str, subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    from meegxer.core.import_review import import_review_path
    from meegxer.gui.connection import DaemonLink
    from meegxer.gui.main_window import MainWindow

    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.random.default_rng(1).standard_normal((3, 2000)) * 1e-5,
                          info, verbose="error")  # fmt: skip
    raw.set_annotations(mne.Annotations([2.0], [0.0], ["stim"]))
    rec = tmp_path / "sub-02_raw.fif"
    raw.save(rec, verbose="error")

    control, _ = daemon
    win = MainWindow(DaemonLink(control))
    try:
        win.change_source("meegxer.core.file")
        assert win.doc.source is not None
        win.doc.source.params["path"] = str(rec)
        win._structure_changed("input")
        win.ask_reviewer = lambda: "I. Rater"  # type: ignore[method-assign]
        entry = {"node": "meegxer.core.file", "params": {"path": str(rec)}}
        win.recording_window.load(entry)
        assert win.recording_window.review_btn.isEnabled()
        win.recording_window.review_requested.emit(entry)  # Review… clicked
        window = win.import_window
        assert window is not None and window.import_mode
        assert [m["label"] for m in window.marks] == ["stim"]  # the file's own
        pump_until(qapp, lambda: bool(window.viewed))  # the raw traces fill
        assert window.source_labels == ("Fz", "Cz", "Pz")

        window.set_crop(1.0, 15.0)
        assert window.crop_span() == (1.0, 15.0)
        window.select(5.0, 6.0)
        window.label.setCurrentText("BAD_move")
        window.annotate()
        window.set_bad("Cz", True)
        window.submit()  # no question about coverage at import
        pump_until(qapp, lambda: win.import_window is not None
                   and not window.isVisible())  # fmt: skip

        path = import_review_path(_file_source(rec))  # type: ignore[arg-type]
        assert path is not None
        saved = load_review(path)
        assert saved is not None and saved.crop == (1.0, 15.0)
        assert saved.bads == ("Cz",) and saved.reviewer == "I. Rater"
        assert [m.label for m in saved.marks] == ["stim", "BAD_move"]
        # the Recording window now shows the reviewed events
        win.recording_window.load(entry)
        assert win.recording_window.current["events_origin"] == "import review"
        assert win.recording_window.current["duration_s"] == pytest.approx(14.0)
    finally:
        win.close()


def _file_source(path: Path) -> object:
    from meegxer.core.graph import Graph
    from meegxer.core.manifest import SPEC_VERSION
    from meegxer.core.registry import Registry

    entry = {"id": "src", "node": "meegxer.core.file", "params": {"path": str(path)}}
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [entry]},
                            Registry.from_entry_points())  # fmt: skip
    return graph.source.impl
