"""The epoch review window (docs/REVIEW.md, step 5): while the offline run waits
at a Review epochs strip, each epoch of the Epochs strip before it is accepted,
rejected or relabelled, with the ERP of what is kept below.

- The **epochs**: a page of epochs side by side, every channel stacked, the
  event at the dashed line. A click selects an epoch; a rejected one is shaded
  red. ←/→ move the selection (the page follows), PgUp/PgDn turn pages.
- **Accept** (A) and **Reject** (R, or Space to toggle) give the selected epoch
  its verdict; **Label** (L) gives it the label chosen (the strip's, the
  epochs' own, or a new one typed in); keys 1 to 9 give the first nine labels.
  Ctrl+Z undoes the last change.
- The **ERP**: the mean of the kept epochs of each label, on the channel
  chosen, updated as the verdicts change.

*Save verdicts and continue* sends them (saved next to the recording; the run
goes on, and later strips read the reviewed epochs). *Cancel run*, or closing
the window, stops the run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.signal_view import HeadChannelMenu
from meegxer.gui.theme import Theme

REASONS = {
    "not reviewed": "These epochs have not been reviewed at this strip yet.",
    "input changed": "Something before this strip changed since the review: "
    "the verdicts made before are filled in for the epochs at the same place; "
    "check and confirm them.",
}
REJECTED = (220, 60, 60)
GAP = 0.08  # between epochs, of an epoch's length
# the ERP's label colours (readable on light and dark plots)
ERP_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
              "#e377c2", "#17becf")  # fmt: skip

Fetch = Callable[[list[int], str], "dict[str, Any] | None"]


class EpochReviewWindow(QWidget):
    def __init__(
        self,
        request: dict[str, Any],
        theme: Theme,
        submit: Callable[[list[dict[str, Any]]], None],
        cancel: Callable[[], None],
        fetch: Fetch,
        recording: str = "",
    ) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.request = request
        self.theme = theme
        self._submit, self._cancel, self._fetch = submit, cancel, fetch
        self.epochs: list[dict[str, Any]] = [dict(v) for v in request.get("epochs", [])]
        self._history: list[list[dict[str, Any]]] = []
        self.selected = 0
        self.first = 0  # the first epoch on the page
        self.answered = False
        self._data: dict[int, Any] = {}  # epoch index -> channels x times (V)
        # kept indices -> the ERP's times (s) and mean (channels x times, V)
        self._means: dict[tuple[int, ...], tuple[Any, Any]] = {}
        self.times: list[float] = []
        self.ch_labels: list[str] = []
        where = f" · {recording}" if recording else ""
        self.setWindowTitle(f"MeegXer · review epochs {request['node']}{where}")
        self.resize(1100, 820)
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>{REASONS.get(str(request.get('reason')), '')}</b> The "
                      "run waits here. Accept, reject or relabel each epoch; "
                      "later strips read the reviewed epochs.")  # fmt: skip
        head.setWordWrap(True)
        lay.addWidget(head)

        tools = QHBoxLayout()
        back = QPushButton("◀")
        back.setToolTip("Previous epoch (←); PgUp: previous page")
        back.clicked.connect(lambda: self.step(-1))
        ahead = QPushButton("▶")
        ahead.setToolTip("Next epoch (→); PgDn: next page")
        ahead.clicked.connect(lambda: self.step(1))
        tools.addWidget(back)
        tools.addWidget(ahead)
        tools.addWidget(QLabel("Per page"))
        self.per_page = QSpinBox()
        self.per_page.setRange(1, 20)
        self.per_page.setValue(5)
        self.per_page.valueChanged.connect(lambda _: self._show_page())
        tools.addWidget(self.per_page)
        tools.addWidget(QLabel("Spacing"))
        self.spacing = QSpinBox()
        self.spacing.setRange(1, 2000)
        self.spacing.setValue(50)
        self.spacing.setSuffix(" µV")
        self.spacing.valueChanged.connect(lambda _: self._draw_epochs())
        tools.addWidget(self.spacing)
        tools.addSpacing(16)
        accept = QPushButton("Accept (A)")
        accept.clicked.connect(lambda: self.set_keep(True))
        reject = QPushButton("Reject (R)")
        reject.setToolTip("Reject the selected epoch (Space toggles)")
        reject.clicked.connect(lambda: self.set_keep(False))
        tools.addWidget(accept)
        tools.addWidget(reject)
        self.label = QComboBox()
        self.label.setEditable(True)  # a new label is typed in
        self.label.addItems([str(x) for x in request.get("labels", [])])
        self.label.setMinimumWidth(140)
        tools.addWidget(self.label)
        relabel = QPushButton("Label (L)")
        relabel.setToolTip("Give the selected epoch the label chosen")
        relabel.clicked.connect(lambda: self.set_label(self.label.currentText()))
        tools.addWidget(relabel)
        undo = QPushButton("Undo")
        undo.setToolTip("Undo the last verdict (Ctrl+Z)")
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

        split = QSplitter(Qt.Orientation.Vertical)
        self.plot = pg.PlotWidget(background=theme.plot_background)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.getPlotItem().getAxis("bottom").setStyle(showValues=False)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        split.addWidget(self.plot)
        erp_box = QWidget()
        erp_lay = QVBoxLayout(erp_box)
        erp_lay.setContentsMargins(0, 0, 0, 0)
        erp_bar = QHBoxLayout()
        erp_bar.addWidget(QLabel("<b>ERP</b> (mean of the kept epochs) · channel"))
        self.channel = HeadChannelMenu(theme)  # one channel, picked on a head
        self.channel.set_single(True)
        self.channel.changed.connect(self._draw_erp)
        erp_bar.addWidget(self.channel)
        erp_bar.addStretch(1)
        erp_lay.addLayout(erp_bar)
        self.erp = pg.PlotWidget(background=theme.plot_background)
        self.erp.setMenuEnabled(False)
        self.erp.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        self.erp.setLabel("bottom", "time from the event", units="ms")
        self.erp.setLabel("left", "µV")
        self.erp.showGrid(x=True, y=True, alpha=0.2)
        self.erp_legend = self.erp.addLegend(offset=(10, 10))
        erp_lay.addWidget(self.erp)
        split.addWidget(erp_box)
        split.setSizes([520, 260])
        lay.addWidget(split, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        cancel_btn = QPushButton("Cancel run")
        cancel_btn.setToolTip("Stop the offline run; nothing is saved")
        cancel_btn.clicked.connect(self.cancel)
        go = QPushButton("Save verdicts and continue")
        go.setObjectName("primary")
        go.setToolTip(f"Save the verdicts to {request.get('file', '')} and run on")
        go.clicked.connect(self.submit)
        bar.addWidget(cancel_btn)
        bar.addWidget(go)
        lay.addLayout(bar)

        for keys, slot in (("A", lambda: self.set_keep(True)),
                           ("R", lambda: self.set_keep(False)),
                           ("Space", self.toggle),
                           ("L", lambda: self.set_label(self.label.currentText())),
                           ("Right", lambda: self.step(1)),
                           ("Left", lambda: self.step(-1)),
                           ("PgDown", lambda: self.page(1)),
                           ("PgUp", lambda: self.page(-1)),
                           ("Ctrl+Z", self.undo)):  # fmt: skip
            QShortcut(QKeySequence(keys), self).activated.connect(slot)
        for k in range(9):  # the label palette
            QShortcut(QKeySequence(str(k + 1)), self).activated.connect(
                lambda k=k: self.label_with(k)
            )
        self._update_palette()
        self._show_page()
        self._draw_erp()

    # -- data ---------------------------------------------------------------------

    def _load(self, indices: list[int]) -> None:
        missing = [i for i in indices if i not in self._data]
        if not missing:
            return
        reply = self._fetch(missing, "none")
        if not reply:
            return
        if not self.times:
            self.times = [float(t) for t in reply.get("times", [])]
            self.ch_labels = [str(c) for c in reply.get("ch_labels", [])]
            labels = tuple(self.ch_labels)
            cz = next((c for c in labels if c.lower() == "cz"),
                      labels[0] if labels else "")  # fmt: skip
            self.channel.blockSignals(True)
            self.channel.set_labels(labels)
            self.channel._set(set(labels) - {cz})
            self.channel.blockSignals(False)
        for entry in reply.get("epochs", []):
            self._data[int(entry["index"])] = np.asarray(entry["data"], dtype=float)

    def kept_by_label(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, v in enumerate(self.epochs):
            if v.get("keep", True):
                out.setdefault(str(v["label"]), []).append(i)
        return out

    # -- verdicts -----------------------------------------------------------------

    def _change(self, **fields: Any) -> None:
        if not self.epochs:
            return
        epoch = self.epochs[self.selected]
        if all(epoch.get(k) == v for k, v in fields.items()):
            return
        self._history.append([dict(v) for v in self.epochs])
        epoch.update(fields)
        self._draw_epochs()
        self._draw_erp()

    def set_keep(self, keep: bool) -> None:
        self._change(keep=keep)

    def toggle(self) -> None:
        if self.epochs:
            self.set_keep(not self.epochs[self.selected].get("keep", True))

    def set_label(self, label: str) -> None:
        label = label.strip()
        if not label:
            return
        if self.label.findText(label) < 0:
            self.label.addItem(label)
            self._update_palette()
        self._change(label=label)

    def label_with(self, k: int) -> None:
        if k < self.label.count():
            self.set_label(self.label.itemText(k))

    def undo(self) -> None:
        if self._history:
            self.epochs = self._history.pop()
            self._draw_epochs()
            self._draw_erp()

    def _update_palette(self) -> None:
        names = [self.label.itemText(k) for k in range(min(9, self.label.count()))]
        self.palette_hint.setText("Keys: " + "  ".join(
            f"{k + 1} {name}" for k, name in enumerate(names)))  # fmt: skip

    # -- paging -------------------------------------------------------------------

    def select(self, index: int) -> None:
        if not self.epochs:
            return
        self.selected = max(0, min(len(self.epochs) - 1, index))
        n = self.per_page.value()
        if not self.first <= self.selected < self.first + n:
            self.first = self.selected - self.selected % n
            self._show_page()
        else:
            self._draw_epochs()

    def step(self, step: int) -> None:
        self.select(self.selected + step)

    def page(self, step: int) -> None:
        n = self.per_page.value()
        last = max(0, len(self.epochs) - 1)
        self.first = max(0, min(last - last % n, self.first + step * n))
        self.selected = self.first
        self._show_page()

    def page_indices(self) -> list[int]:
        n = self.per_page.value()
        self.first -= self.first % n
        return list(range(self.first, min(len(self.epochs), self.first + n)))

    def _show_page(self) -> None:
        indices = self.page_indices()
        if self.selected not in indices and indices:
            self.selected = indices[0]
        self._load(indices)
        self._draw_epochs()

    # -- drawing ------------------------------------------------------------------

    def _layout(self) -> tuple[float, float]:
        """An epoch's length (s) and the step from one to the next on the page."""
        span = (self.times[-1] - self.times[0]) if len(self.times) > 1 else 1.0
        return span, span * (1 + GAP)

    def _draw_epochs(self) -> None:
        self.plot.clear()
        kept = sum(bool(v.get("keep", True)) for v in self.epochs)
        self.progress.setText(f"{len(self.epochs)} epochs · {kept} kept · "
                              f"{len(self.epochs) - kept} rejected")  # fmt: skip
        indices = self.page_indices()
        if not indices or not self.times:
            return
        span, step = self._layout()
        t = np.asarray(self.times) - self.times[0]
        spacing = float(self.spacing.value())
        n_ch = len(self.ch_labels)
        accent = QColor(self.theme.accent)
        for k, i in enumerate(indices):
            x0 = k * step
            verdict = self.epochs[i]
            keep = bool(verdict.get("keep", True))
            if not keep or i == self.selected:
                fill = QColor(*REJECTED) if not keep else QColor(accent)
                fill.setAlpha(45 if not keep else 30)
                box = pg.LinearRegionItem((x0, x0 + span), movable=False, brush=fill)
                box.setZValue(-10)
                for line in box.lines:
                    line.setPen(
                        pg.mkPen(
                            accent if i == self.selected else fill,
                            width=2 if i == self.selected else 0,
                        )
                    )
                self.plot.addItem(box)
            dashed = pg.mkPen(self.theme.text_muted, style=Qt.PenStyle.DashLine)
            self.plot.addItem(pg.InfiniteLine(x0 - self.times[0], 90, pen=dashed))
            data = self._data.get(i)
            pen = pg.mkPen(self.theme.text if keep else self.theme.text_muted, width=1)
            if data is not None:
                for c in range(min(n_ch, data.shape[0])):
                    y = data[c] * 1e6 - c * spacing
                    self.plot.plot(x0 + t[: len(y)], y, pen=pen)
            tag = "rejected" if not keep else str(verdict["label"])
            text = pg.TextItem(
                f"#{i + 1}  {tag}",
                anchor=(0, 1),
                color=REJECTED if not keep else self.theme.text,
            )
            text.setPos(x0, spacing * 0.9)
            self.plot.addItem(text)
        axis = self.plot.getPlotItem().getAxis("left")
        axis.setTicks(
            [[(-c * spacing, label) for c, label in enumerate(self.ch_labels)]]
        )
        self.plot.setXRange(-span * GAP / 2, len(indices) * step - span * GAP / 2,
                            padding=0)  # fmt: skip
        self.plot.setYRange(-(n_ch - 0.5) * spacing, spacing * 1.2, padding=0)

    def epoch_at(self, x: float) -> int | None:
        """The epoch on the page at ``x`` (plot coordinates)."""
        if not self.times:
            return None
        span, step = self._layout()
        k = math.floor(x / step)
        indices = self.page_indices()
        if 0 <= k < len(indices) and x - k * step <= span:
            return indices[k]
        return None

    def _clicked(self, ev: Any) -> None:
        box = self.plot.getPlotItem().getViewBox()
        if not box.sceneBoundingRect().contains(ev.scenePos()):
            return
        index = self.epoch_at(box.mapSceneToView(ev.scenePos()).x())
        if index is not None:
            self.select(index)

    def erp_curves(self) -> dict[str, tuple[int, Any, Any]]:
        """The ERP of each label: how many epochs, the times (s) and the mean
        (channels x times, V), from the kept epochs."""
        out: dict[str, tuple[int, Any, Any]] = {}
        for label, indices in self.kept_by_label().items():
            key = tuple(indices)
            if key not in self._means:
                reply = self._fetch(list(indices), "mean")
                if not reply:
                    continue
                self._means[key] = (
                    np.asarray(reply.get("times", []), dtype=float),
                    np.asarray(reply.get("mean", []), dtype=float),
                )
            out[label] = (len(indices), *self._means[key])
        return out

    def _draw_erp(self) -> None:
        self.erp.clear()
        self.erp_legend.clear()
        shown = self.channel.selected(tuple(self.ch_labels))
        if not shown:
            return
        c = shown[0]
        zero = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(
            self.theme.text_muted, style=Qt.PenStyle.DashLine))  # fmt: skip
        self.erp.addItem(zero)
        curves = sorted(self.erp_curves().items())
        for k, (label, (n, times, mean)) in enumerate(curves):
            if mean.ndim != 2 or c >= mean.shape[0] or len(times) != mean.shape[1]:
                continue
            self.erp.plot(
                times * 1e3,  # ms
                mean[c] * 1e6,
                pen=pg.mkPen(ERP_COLORS[k % len(ERP_COLORS)], width=2),
                name=f"{label} (n={n})",
            )

    # -- answer -------------------------------------------------------------------

    def verdicts(self) -> list[dict[str, Any]]:
        return [dict(v) for v in self.epochs]

    def submit(self) -> None:
        self.answered = True
        self._submit(self.verdicts())
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
