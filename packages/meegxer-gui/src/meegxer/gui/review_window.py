"""The review window (docs/REVIEW.md): marks for a review strip, made while the
offline run waits at it.

The main window's views keep showing the taps the run has filled so far (every
tap before the strip), with the probes chosen as Main and aux, and Features too.
A selection band on the signal view picks a span; *Mark selection* labels it
with a label of the strip's, one of the input's event types, or a new one typed
in. Marks show as shaded spans on the signal view. *Continue run* sends them
(the daemon saves them next to the recording and the run goes on); *Cancel run*
stops the run without saving.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.signal_view import SignalView
from meegxer.gui.theme import Theme

REASONS = {
    "not reviewed": "This recording has not been reviewed at this strip yet.",
    "input changed": "Something before this strip changed since the review: "
    "check the marks below and confirm them, or change them.",
}
DEFAULT_SPAN_S = 1.0


class ReviewWindow(QWidget):
    """Marks for one review request (``offline.review`` of the daemon status)."""

    def __init__(
        self,
        request: dict[str, Any],
        view: SignalView,
        theme: Theme,
        submit: Callable[[list[dict[str, Any]]], None],
        cancel: Callable[[], None],
        recording: str = "",
    ) -> None:
        super().__init__()
        self.request = request
        self.view = view
        self.theme = theme
        self._submit, self._cancel = submit, cancel
        self.answered = False
        self.marks: list[dict[str, Any]] = [dict(m) for m in request.get("marks", [])]
        self.setWindowTitle(f"MeegXer · review {request['node']}")
        self.setWindowFlag(Qt.WindowType.Tool, True)  # stays above the main window
        self.resize(460, 520)
        lay = QVBoxLayout(self)
        where = f" of {recording}" if recording else ""
        title = QLabel(f"<b>Review at “{request['node']}”{where}</b>")
        lay.addWidget(title)
        reason = QLabel(REASONS.get(str(request.get("reason")), ""))
        reason.setWordWrap(True)
        lay.addWidget(reason)
        hint = QLabel(
            "The run waits here. The views show the taps before this strip: "
            "choose one as Main or aux to see it. Drag the band on the signal "
            "view over a span, pick or type a label, and mark it."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Label"))
        self.label = QComboBox()
        self.label.setEditable(True)  # a new label is typed in
        self.label.addItems([str(x) for x in request.get("labels", [])])
        row.addWidget(self.label, 1)
        mark = QPushButton("Mark selection")
        mark.setToolTip("Label the span under the band on the signal view")
        mark.clicked.connect(self.mark_selection)
        row.addWidget(mark)
        lay.addLayout(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Onset (s)", "Duration (s)", "Label"])
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._show_mark)
        lay.addWidget(self.table, 1)
        tools = QHBoxLayout()
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self.remove_selected)
        tools.addWidget(remove)
        tools.addStretch(1)
        self.count = QLabel("")
        self.count.setObjectName("muted")
        tools.addWidget(self.count)
        lay.addLayout(tools)

        bar = QHBoxLayout()
        cancel_btn = QPushButton("Cancel run")
        cancel_btn.setToolTip("Stop the offline run; nothing is saved")
        cancel_btn.clicked.connect(self.cancel)
        go = QPushButton("Save marks and continue")
        go.setObjectName("primary")
        go.setToolTip(f"Save the marks to {request.get('file', '')} and run on")
        go.clicked.connect(self.submit)
        bar.addWidget(cancel_btn)
        bar.addStretch(1)
        bar.addWidget(go)
        lay.addLayout(bar)

        plot = view.plot
        accent = QColor(theme.accent)
        accent.setAlpha(60)
        self.selection = pg.LinearRegionItem(brush=accent, movable=True)
        self.selection.setZValue(5)
        plot.addItem(self.selection)
        self._spans: list[pg.LinearRegionItem] = []
        plot.getViewBox().sigXRangeChanged.connect(self._keep_selection_in_view)
        self._keep_selection_in_view()
        self._refresh()

    # -- marks ------------------------------------------------------------------------

    def _keep_selection_in_view(self, *_: object) -> None:
        """The band follows the view when scrolled out of it."""
        (t0, t1), (s0, s1) = self.view.plot.viewRange()[0], self.selection.getRegion()
        if s1 < t0 or s0 > t1 or s1 - s0 <= 0:
            mid, half = (t0 + t1) / 2, min(DEFAULT_SPAN_S, (t1 - t0) / 2) / 2
            self.selection.setRegion((max(0.0, mid - half), mid + half))

    def mark_selection(self) -> None:
        label = self.label.currentText().strip()
        if not label:
            self.label.setFocus()
            return
        s0, s1 = self.selection.getRegion()
        onset = max(0.0, float(s0))
        self.add_mark(onset, max(0.0, float(s1) - onset), label)
        if self.label.findText(label) < 0:
            self.label.addItem(label)

    def add_mark(self, onset: float, duration: float, label: str) -> None:
        self.marks.append({"onset": round(onset, 6), "duration": round(duration, 6),
                           "label": label})  # fmt: skip
        self.marks.sort(key=lambda m: (m["onset"], m["label"]))
        self._refresh()

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            del self.marks[row]
        self._refresh()

    def _show_mark(self, row: int, _column: int) -> None:
        m = self.marks[row]
        self.selection.setRegion((m["onset"], m["onset"] + m["duration"]))

    def _refresh(self) -> None:
        self.table.setRowCount(len(self.marks))
        for row, m in enumerate(self.marks):
            for col, text in enumerate(
                (f"{m['onset']:.3f}", f"{m['duration']:.3f}", str(m["label"]))
            ):
                self.table.setItem(row, col, QTableWidgetItem(text))
        n = len(self.marks)
        self.count.setText(f"{n} mark{'s' if n != 1 else ''}")
        self._draw_spans()

    def _draw_spans(self) -> None:
        color = QColor(self.theme.text_muted)
        color.setAlpha(45)
        plot = self.view.plot
        while len(self._spans) < len(self.marks):
            span = pg.LinearRegionItem(movable=False, brush=color)
            span.setZValue(-5)
            plot.addItem(span)
            self._spans.append(span)
        for span, m in zip(self._spans, self.marks, strict=False):
            # a mark without duration still shows, as a thin span
            span.setRegion((m["onset"], m["onset"] + max(m["duration"], 0.01)))
            span.setVisible(True)
        for span in self._spans[len(self.marks) :]:
            span.setVisible(False)

    # -- answer -----------------------------------------------------------------------

    def submit(self) -> None:
        self.answered = True
        self._submit([dict(m) for m in self.marks])
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
        self.dispose()
        super().closeEvent(event)

    def dispose(self) -> None:
        """Take the band and the spans off the signal view."""
        plot = self.view.plot
        with contextlib.suppress(RuntimeError, TypeError):  # already done
            plot.getViewBox().sigXRangeChanged.disconnect(self._keep_selection_in_view)
        for item in [self.selection, *self._spans]:
            plot.removeItem(item)
        self._spans = []
