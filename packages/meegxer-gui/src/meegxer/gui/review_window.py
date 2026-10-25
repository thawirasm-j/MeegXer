"""The review window (docs/REVIEW.md): the stacked view of the recording, with
the tools to mark it below, while the offline run waits at a review strip.

The traces are the stacked view's (the probes chosen as Main and aux, every
channel by default, its own span, spacing and panner) over everything the run
has computed so far: the taps before the strip. Below them:

- the **coverage bar**: the whole recording, the parts already looked at in
  green, rejected spans in red, other marks in the accent colour, the shown
  span outlined; a click moves there;
- the **tools**: a band on the traces picks a span; *Annotate* (A) labels it
  with the chosen label (the strip's, one of the input's event types, or a new
  one typed in), *Reject* (R) labels it as rejected (a label starting with one
  of the strip's reject words, `BAD` by default; a Reject segments strip
  downstream cuts it out); pages move with ←/→ (or
  PgUp/PgDn), *Next unreviewed* (N) jumps to the first part not looked at yet;
- the **marks** table: double-click shows a mark, Delete removes it.

Above the traces, the **display**: the montage (monopolar against the average
reference by default), display filters, and the scale: sensitivity in µV/mm and
speed in mm/s on the screen's real size, or channel spacing in µV and seconds
per page (Settings, *Review*). The core computes each page through the montage
and filters from the raw samples. What the Review strip sets (its display
presets) is fixed for the reviewer. *Bad channels* marks channels bad: they are
drawn grey and dashed, left out of the average and the Laplacian, and become
the stream's bad channels after the strip. Keys 1 to 9 mark the band with the
first nine labels; Ctrl+Z undoes the last change to the marks or channels.

*Save marks and continue* sends the marks and the spans looked at (saved next
to the recording; the run goes on), asking first when part of the recording
was never shown. *Cancel run*, or closing the window, stops the run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from meegxer.core import settings as core_settings
from meegxer.core.montage import MONTAGES
from meegxer.gui.colors import ChannelStyle
from meegxer.gui.signal_view import StackedWindow
from meegxer.gui.theme import Theme

REASONS = {
    "not reviewed": "This recording has not been reviewed at this strip yet.",
    "input changed": "Something before this strip changed since the review: "
    "check the marks below and confirm them, or change them.",
}
DEFAULT_SPAN_S = 1.0
FULL = 0.995  # viewed fraction counted as the whole recording
SENSITIVITIES = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200)  # µV/mm
SPEEDS = (10, 15, 30, 60)  # mm/s
BIPOLAR = ("longitudinal bipolar", "transverse bipolar")  # labels are pairs
BAD_STYLE = ChannelStyle((128, 128, 128), dashed=True)
PRESET_TIP = "Set by the pipeline's Review strip"

Span = tuple[float, float]


def note_color(theme: Theme) -> str:
    """Annotations (not rejections): apart from the selection band's accent."""
    return theme.status_color("bypassed")


def merge(spans: list[Span]) -> list[Span]:
    """Sorted spans, overlapping or touching ones merged."""
    out: list[Span] = []
    for lo, hi in sorted(s for s in spans if s[1] > s[0]):
        if out and lo <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def covered(spans: list[Span], duration: float) -> float:
    """The fraction of ``[0, duration)`` the spans cover."""
    if duration <= 0:
        return 1.0
    inside = sum(min(hi, duration) - max(lo, 0.0) for lo, hi in spans
                 if hi > 0 and lo < duration)  # fmt: skip
    return min(1.0, inside / duration)


def first_gap(spans: list[Span], duration: float, after: float = 0.0) -> float | None:
    """The start of the first part of ``[after, duration)`` not covered (None:
    all of it is), wrapping round to the start once."""
    for start in (after, 0.0):
        t = start
        for lo, hi in spans:
            if hi <= t:
                continue
            if lo > t + 1e-6:
                return t
            t = hi
        if t < duration - 1e-6:
            return t
    return None


class CoverageBar(QWidget):
    """The whole recording at a glance: looked at, marked, shown now."""

    seek = Signal(float)  # seconds

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.duration = 0.0
        self.viewed: list[Span] = []
        self.marks: list[tuple[float, float, bool]] = []  # onset, duration, reject
        self.shown_span: Span | None = None
        self.setMinimumHeight(22)
        self.setMaximumHeight(22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Green: looked at. Red: rejected. Click to go there.")

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(self.theme.surface_alt))
        if self.duration <= 0:
            return
        scale = w / self.duration

        def box(lo: float, hi: float, color: QColor, top: float, height: float) -> None:
            painter.fillRect(QRectF(lo * scale, top, max(1.0, (hi - lo) * scale),
                                    height), color)  # fmt: skip

        seen = QColor(self.theme.status_color("ready"))
        seen.setAlpha(150)
        for lo, hi in self.viewed:
            box(lo, hi, seen, 0, h)
        reject = QColor(self.theme.status_color("error"))
        note = QColor(note_color(self.theme))
        for onset, length, rejected in self.marks:
            box(onset, onset + length, reject if rejected else note, h * 0.55, h * 0.45)
        if self.shown_span is not None:
            painter.setPen(QColor(self.theme.text))
            lo, hi = self.shown_span
            painter.drawRect(QRectF(lo * scale, 0.5, max(2.0, (hi - lo) * scale),
                                    h - 1.0))  # fmt: skip

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.duration > 0 and self.width() > 0:
            self.seek.emit(event.position().x() / self.width() * self.duration)


class ReviewWindow(StackedWindow):
    """The stacked view plus the review tools, for one review request
    (``offline.review`` of the daemon status)."""

    def __init__(
        self,
        request: dict[str, Any],
        theme: Theme,
        submit: Callable[[list[dict[str, Any]], list[list[float]], list[str]], None],
        cancel: Callable[[], None],
        recording: str = "",
        settings: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(theme)
        self.request = request
        self._submit, self._cancel = submit, cancel
        # the reviewer's Settings, under the Review strip's presets
        self.prefs = {**core_settings.defaults()["review"], **(settings or {})}
        self.presets: dict[str, Any] = dict(request.get("display") or {})
        self.bads: list[str] = [str(b) for b in request.get("bads", [])]
        self.source_labels: tuple[str, ...] = ()  # the channels (set by the GUI)
        self._history: list[tuple[list[dict[str, Any]], list[str]]] = []
        self.on_display: Callable[[], None] = lambda: None  # the GUI redraws
        self.answered = False
        self.duration = float(request.get("duration_s") or 0.0)
        self.reject_words = tuple(str(w) for w in request.get("reject", ())) or ()
        self.marks: list[dict[str, Any]] = [dict(m) for m in request.get("marks", [])]
        self.viewed: list[Span] = merge(
            [(float(a), float(b)) for a, b in request.get("viewed", [])]
        )
        self.seek_rate = 0.0  # samples per second of the panner (set by the GUI)
        where = f" · {recording}" if recording else ""
        self.setWindowTitle(f"MeegXer · review {request['node']}{where}")
        self.resize(1100, 820)
        lay = self.layout()
        assert isinstance(lay, QVBoxLayout)
        head = QLabel(f"<b>{REASONS.get(str(request.get('reason')), '')}</b> "
                      "The run waits here; the traces are the taps before this "
                      "strip (choose Main and aux in the main window).")  # fmt: skip
        head.setWordWrap(True)
        lay.insertWidget(0, head)
        lay.insertLayout(1, self._display_bar())

        self.coverage = CoverageBar(theme)
        self.coverage.duration = self.duration
        self.coverage.seek.connect(self.seek)
        lay.addWidget(self.coverage)

        tools = QHBoxLayout()
        tools.addWidget(QLabel("Label"))
        self.label = QComboBox()
        self.label.setEditable(True)  # a new label is typed in
        self.label.addItems([str(x) for x in request.get("labels", [])])
        self.label.setMinimumWidth(160)
        tools.addWidget(self.label)
        annotate = QPushButton("Annotate (A)")
        annotate.setToolTip("Label the span under the band")
        annotate.clicked.connect(self.annotate)
        tools.addWidget(annotate)
        reject = QPushButton("Reject (R)")
        words = ", ".join(self.reject_words) or "nothing (no reject labels set)"
        reject.setToolTip(f"Label the span under the band as rejected (labels "
                          f"starting with {words}); a Reject segments strip "
                          "cuts it out")  # fmt: skip
        reject.setEnabled(bool(self.reject_words))
        reject.clicked.connect(self.reject)
        tools.addWidget(reject)
        tools.addSpacing(16)
        back = QPushButton("◀")
        back.setToolTip("Previous page (← or PgUp)")
        back.clicked.connect(lambda: self.page(-1))
        ahead = QPushButton("▶")
        ahead.setToolTip("Next page (→ or PgDn)")
        ahead.clicked.connect(lambda: self.page(1))
        unseen = QPushButton("Next unreviewed (N)")
        unseen.clicked.connect(self.next_unreviewed)
        for b in (back, ahead, unseen):
            tools.addWidget(b)
        undo = QPushButton("Undo")
        undo.setToolTip("Undo the last change to the marks or bad channels (Ctrl+Z)")
        undo.clicked.connect(self.undo)
        tools.addWidget(undo)
        tools.addStretch(1)
        self.progress = QLabel("")
        self.progress.setObjectName("muted")
        tools.addWidget(self.progress)
        lay.addLayout(tools)
        self.palette_hint = QLabel("")
        self.palette_hint.setObjectName("muted")
        lay.addWidget(self.palette_hint)
        self._update_palette()

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Onset (s)", "Duration (s)", "Label"])
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(170)
        self.table.cellDoubleClicked.connect(self._show_mark)
        lay.addWidget(self.table)

        bar = QHBoxLayout()
        remove = QPushButton("Remove selected (Del)")
        remove.clicked.connect(self.remove_selected)
        bar.addWidget(remove)
        bar.addStretch(1)
        cancel_btn = QPushButton("Cancel run")
        cancel_btn.setToolTip("Stop the offline run; nothing is saved")
        cancel_btn.clicked.connect(self.cancel)
        go = QPushButton("Save marks and continue")
        go.setObjectName("primary")
        go.setToolTip(f"Save the marks to {request.get('file', '')} and run on")
        go.clicked.connect(self.submit)
        bar.addWidget(cancel_btn)
        bar.addWidget(go)
        lay.addLayout(bar)

        for keys, slot in (("A", self.annotate), ("R", self.reject),
                           ("N", self.next_unreviewed),
                           ("Delete", self.remove_selected),
                           ("Right", lambda: self.page(1)),
                           ("PgDown", lambda: self.page(1)),
                           ("Left", lambda: self.page(-1)),
                           ("PgUp", lambda: self.page(-1)),
                           ("Ctrl+Z", self.undo)):  # fmt: skip
            QShortcut(QKeySequence(keys), self).activated.connect(slot)
        for k in range(9):  # the label palette
            QShortcut(QKeySequence(str(k + 1)), self).activated.connect(
                lambda k=k: self.mark_with(k)
            )

        accent = QColor(theme.accent)
        accent.setAlpha(60)
        self.selection = pg.LinearRegionItem(brush=accent, movable=True)
        self.selection.setZValue(5)
        self.plot.addItem(self.selection)
        self.plot.getViewBox().sigYRangeChanged.connect(lambda *_: self._draw_spans())
        self._spans: list[pg.LinearRegionItem] = []
        self._tags: list[pg.TextItem] = []  # each mark's label, at the top
        self._view: Span | None = None
        self._refresh()

    # -- display ----------------------------------------------------------------------

    def _pref(self, key: str) -> Any:
        return self.presets.get(key, self.prefs[key])

    def _display_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Montage"))
        self.montage = QComboBox()
        for name in MONTAGES:
            self.montage.addItem(name.capitalize(), name)
        self.montage.setCurrentIndex(
            max(0, self.montage.findData(self._pref("montage")))
        )
        self.montage.setToolTip("How the channels are shown (display only)")
        self.montage.currentIndexChanged.connect(lambda _: self._display_changed())
        bar.addWidget(self.montage)
        self.filters: dict[str, QDoubleSpinBox] = {}
        for key, text in (("highpass", "HP"), ("lowpass", "LP"), ("notch", "Notch")):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000.0)
            spin.setDecimals(1)
            spin.setSuffix(" Hz")
            spin.setSpecialValueText("off")
            spin.setValue(float(self._pref(key)))
            spin.setToolTip(f"Display {text.lower()} filter (display only; 0: off)")
            spin.valueChanged.connect(lambda _: self._display_changed())
            bar.addWidget(QLabel(text))
            bar.addWidget(spin)
            self.filters[key] = spin
        self.clinical = (
            self.prefs["units"] == "uv_per_mm" or "sensitivity" in self.presets
        )
        self.sensitivity = QComboBox()
        self.speed = QComboBox()
        scales = ((self.sensitivity, SENSITIVITIES, "µV/mm", "sensitivity"),
                  (self.speed, SPEEDS, "mm/s", "speed"))  # fmt: skip
        for box, values, unit, key in scales:
            want = float(self._pref(key))
            for v in sorted({*map(float, values), want}):
                box.addItem(f"{v:g} {unit}", v)
            box.setCurrentIndex(box.findData(want))
            box.currentIndexChanged.connect(lambda _: self._display_changed())
            box.setVisible(self.clinical)
            bar.addWidget(box)
        self.speed.setVisible(self.clinical and "page" not in self.presets)
        uv_controls = (self.span, self.spacing_label, self.spacing, self.normalize)
        for control in uv_controls:
            control.setVisible(not self.clinical)
        if not self.clinical:
            for box, key, unit in ((self.span, "page", "s"),
                                   (self.spacing, "spacing", "µV")):  # fmt: skip
                want = float(self._pref(key))
                if box.findData(want) < 0:
                    box.addItem(f"{want:g} {unit}", want)
                box.setCurrentIndex(box.findData(want))
        self.bad_button = QToolButton()
        self.bad_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bad_menu = QMenu(self.bad_button)
        bad_menu.aboutToShow.connect(self._fill_bad_menu)
        self.bad_button.setMenu(bad_menu)
        bar.addWidget(self.bad_button)
        bar.addStretch(1)
        fixed: tuple[tuple[str, QWidget], ...] = (
            ("montage", self.montage), ("highpass", self.filters["highpass"]),
            ("lowpass", self.filters["lowpass"]), ("notch", self.filters["notch"]),
            ("sensitivity", self.sensitivity), ("page", self.span),
        )  # fmt: skip
        for key, widget in fixed:
            if key in self.presets:
                widget.setEnabled(False)
                widget.setToolTip(PRESET_TIP)
        self._update_bad_button()
        return bar

    def display(self) -> dict[str, Any]:
        """What the core computes each page through (``offline_window``)."""
        return {"montage": str(self.montage.currentData()),
                **{k: float(s.value()) for k, s in self.filters.items()},
                "bads": list(self.bads)}  # fmt: skip

    def _display_changed(self) -> None:
        self._update_bad_button()
        self.on_display()

    def _px_per_mm(self) -> float:
        screen = self.screen()
        dpi = screen.physicalDotsPerInch() if screen is not None else 96.0
        ratio = screen.devicePixelRatio() if screen is not None else 1.0
        return max(1.0, float(dpi) / 25.4 / max(float(ratio), 1e-6))

    def seconds(self) -> float:
        if "page" in self.presets:
            return float(self.presets["page"])
        if self.clinical:  # the page is the plot's width at the speed
            width_mm = max(10.0, float(self.plot.width()) / self._px_per_mm())
            return width_mm / float(self.speed.currentData())
        return super().seconds()

    def fixed_spacing(self, n_rows: int) -> float | None:
        """At a sensitivity S µV/mm a mm of screen is S µV: the rows share the
        plot's height."""
        if not self.clinical or n_rows <= 0:
            return None
        height_mm = max(10.0, float(self.plot.height()) / self._px_per_mm())
        return height_mm * float(self.sensitivity.currentData()) / n_rows

    def restyle(self, label: str, style: ChannelStyle) -> ChannelStyle:
        """Bad channels, and bipolar pairs with a bad electrode, in grey."""
        bad = label in self.bads or any(p in self.bads for p in label.split("-"))
        return BAD_STYLE if bad else style

    # -- bad channels -----------------------------------------------------------------

    def _fill_bad_menu(self) -> None:
        menu = self.bad_button.menu()
        assert menu is not None
        menu.clear()
        for label in self.source_labels:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(label in self.bads)
            action.toggled.connect(lambda on, lab=label: self.set_bad(lab, on))

    def set_bad(self, label: str, bad: bool) -> None:
        if (label in self.bads) == bad:
            return
        self._snapshot()
        if bad:
            self.bads.append(label)
        else:
            self.bads.remove(label)
        self._display_changed()

    def _update_bad_button(self) -> None:
        n = len(self.bads)
        self.bad_button.setText(f"Bad channels ({n})")
        bipolar = self.montage.currentData() in BIPOLAR
        self.bad_button.setToolTip(
            "Mark channels bad: grey and dashed here, left out of the average "
            "and the Laplacian, and the stream's bad channels after this strip"
            + (" (marked bad by electrode, not by pair)" if bipolar else "")
        )

    # -- undo -------------------------------------------------------------------------

    def _snapshot(self) -> None:
        self._history.append(([dict(m) for m in self.marks], list(self.bads)))
        del self._history[:-100]

    def undo(self) -> None:
        if not self._history:
            return
        marks, bads = self._history.pop()
        display = bads != self.bads
        self.marks, self.bads = marks, bads
        self._refresh()
        if display:
            self._display_changed()

    # -- what is shown ----------------------------------------------------------------

    def shown(self, t0: float, t1: float) -> None:
        """The span drawn: counted as looked at, and the band kept in it."""
        self._view = (t0, t1)
        hi = min(t1, self.duration) if self.duration else t1
        if hi > max(t0, 0.0):
            self.viewed = merge([*self.viewed, (max(t0, 0.0), hi)])
        s0, s1 = self.selection.getRegion()
        if s1 < t0 or s0 > t1 or s1 - s0 <= 0:
            mid, half = (t0 + t1) / 2, min(DEFAULT_SPAN_S, (t1 - t0) / 2) / 2
            self.selection.setRegion((max(0.0, mid - half), mid + half))
        self._update_coverage()

    def seek(self, t: float) -> None:
        """Show the page starting near ``t`` seconds (the GUI redraws)."""
        if self.seek_rate <= 0:
            return
        start = max(0.0, t - self.seconds() * 0.1)  # a little context before
        self.panner.bar.setValue(round(start * self.seek_rate))

    def page(self, step: int) -> None:
        bar = self.panner.bar
        bar.setValue(bar.value() + step * bar.pageStep())

    def next_unreviewed(self) -> None:
        after = self._view[1] if self._view is not None else 0.0
        gap = first_gap(self.viewed, self.duration, after)
        if gap is None:
            self.progress.setText("Everything has been looked at.")
            return
        self.seek(gap)

    # -- marks ------------------------------------------------------------------------

    def annotate(self) -> None:
        label = self.label.currentText().strip()
        if not label:
            self.label.setFocus()
            return
        self._mark(label)

    def reject(self) -> None:
        if not self.reject_words:
            return
        label = self.label.currentText().strip()
        if not label.startswith(self.reject_words):
            label = self.reject_words[0]
        self._mark(label)

    def mark_with(self, k: int) -> None:
        """Mark the band with the palette's label ``k`` (keys 1 to 9)."""
        if k < self.label.count():
            self.label.setCurrentIndex(k)
            self._mark(self.label.itemText(k))

    def _update_palette(self) -> None:
        keys = [f"{k + 1} {self.label.itemText(k)}"
                for k in range(min(9, self.label.count()))]  # fmt: skip
        self.palette_hint.setText("Keys: " + " · ".join(keys) if keys else "")

    def _mark(self, label: str) -> None:
        s0, s1 = self.selection.getRegion()
        onset = max(0.0, float(s0))
        self.add_mark(onset, max(0.0, float(s1) - onset), label)
        if self.label.findText(label) < 0:
            self.label.addItem(label)
            self._update_palette()

    def add_mark(self, onset: float, duration: float, label: str) -> None:
        self._snapshot()
        self.marks.append({"onset": round(onset, 6), "duration": round(duration, 6),
                           "label": label})  # fmt: skip
        self.marks.sort(key=lambda m: (m["onset"], m["label"]))
        self._refresh()

    def is_reject(self, mark: dict[str, Any]) -> bool:
        return bool(self.reject_words) and str(mark["label"]).startswith(
            self.reject_words
        )

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if rows:
            self._snapshot()
        for row in rows:
            del self.marks[row]
        self._refresh()

    def _show_mark(self, row: int, _column: int) -> None:
        m = self.marks[row]
        self.seek(float(m["onset"]))
        self.selection.setRegion((m["onset"], m["onset"] + m["duration"]))

    def _refresh(self) -> None:
        self.table.setRowCount(len(self.marks))
        reject = QColor(self.theme.status_color("error"))
        for row, m in enumerate(self.marks):
            texts = (f"{m['onset']:.3f}", f"{m['duration']:.3f}", str(m["label"]))
            for col, text in enumerate(texts):
                item = QTableWidgetItem(text)
                if self.is_reject(m):
                    item.setForeground(reject)
                self.table.setItem(row, col, item)
        self._draw_spans()
        self._update_coverage()

    def _draw_spans(self) -> None:
        while len(self._spans) < len(self.marks):
            span = pg.LinearRegionItem(movable=False)
            span.setZValue(-5)
            for line in span.lines:
                line.setPen(pg.mkPen(None))
            self.plot.addItem(span)
            self._spans.append(span)
        while len(self._tags) < len(self.marks):
            tag = pg.TextItem(anchor=(0, 0))
            tag.setZValue(6)
            self.plot.addItem(tag, ignoreBounds=True)  # Y fits the traces
            self._tags.append(tag)
        top = self.plot.viewRange()[1][1]
        for span, tag, m in zip(self._spans, self._tags, self.marks, strict=False):
            hue = (self.theme.status_color("error") if self.is_reject(m)
                   else note_color(self.theme))  # fmt: skip
            color = QColor(hue)
            color.setAlpha(55)
            span.setBrush(color)
            # a mark without duration still shows, as a thin span
            span.setRegion((m["onset"], m["onset"] + max(m["duration"], 0.01)))
            span.setVisible(True)
            tag.setText(str(m["label"]), color=hue)
            tag.setPos(float(m["onset"]), top)
            tag.setVisible(True)
        for item in [*self._spans[len(self.marks) :], *self._tags[len(self.marks) :]]:
            item.setVisible(False)

    def _update_coverage(self) -> None:
        bar = self.coverage
        bar.viewed = self.viewed
        bar.marks = [(float(m["onset"]), float(m["duration"]), self.is_reject(m))
                     for m in self.marks]  # fmt: skip
        bar.shown_span = self._view
        bar.update()
        n = len(self.marks)
        cut = sum(float(m["duration"]) for m in self.marks if self.is_reject(m))
        self.progress.setText(
            f"Reviewed {covered(self.viewed, self.duration):.0%} · {n} mark"
            f"{'s' if n != 1 else ''} · {cut:.1f} s rejected"
        )

    # -- answer -----------------------------------------------------------------------

    def submit(self) -> None:
        part = covered(self.viewed, self.duration)
        if part < FULL:
            ask = QMessageBox.question(
                self, "Not all reviewed",
                f"Only {part:.0%} of the recording has been shown. Save the "
                "marks and continue anyway?",
            )  # fmt: skip
            if ask != QMessageBox.StandardButton.Yes:
                return
        self.answered = True
        self._submit([dict(m) for m in self.marks],
                     [[round(a, 6), round(b, 6)] for a, b in self.viewed],
                     list(self.bads))  # fmt: skip
        self.close()

    def cancel(self) -> None:
        self.answered = True
        self._cancel()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.answered:
            ask = QMessageBox.question(
                self, "Stop the run?",
                "The offline run waits for this review. Closing stops the run "
                "(nothing is saved). Stop it?",
            )  # fmt: skip
            if ask != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.answered = True
            self._cancel()
        super().closeEvent(event)
