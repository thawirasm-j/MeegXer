"""The ERP view of an epochs tap (M10.3): the running mean of each condition's
epochs, with its standard error, live (kept in the core's ERP tap, so both are
exact however many frames are lost) or over an offline run.

*Show* ``All conditions`` draws one curve per condition on one channel (Cz
first, picked on the head under *Channel*), with each condition's epoch count
in the legend, as the epoch review window does; a single condition draws the
channels picked on the head overlaid (a butterfly plot). *SEM* shades each
curve's mean ± its standard error (the sample SD over √n; none for a single
epoch). Times are in ms from the event. *Reset* starts a live ERP over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.epoch_review_window import ERP_COLORS
from meegxer.gui.signal_view import HeadChannelMenu
from meegxer.gui.theme import Theme

ALL = "All conditions"
SEM_ALPHA = 60  # of 255: the SEM band under its curve


@dataclass(frozen=True, eq=False)
class Erp:
    """The ERPs of one tap: ``values`` (the means) and ``sem`` are conditions
    x channels x times, ``times`` in seconds from the event."""

    conditions: tuple[str, ...]
    counts: tuple[int, ...]
    channels: tuple[str, ...]
    times: np.ndarray
    values: np.ndarray
    unit: str = "V"
    sem: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class Curve:
    """One drawn line: its legend name, times (ms), mean and SEM (the plot's
    unit; the SEM None or NaN where there is none)."""

    name: str
    times: np.ndarray
    mean: np.ndarray
    sem: np.ndarray | None


class ErpView(QWidget):
    """Toolbar and plot for one ERP tap; ``reset_requested`` asks the core to
    start the live averages over."""

    reset_requested = Signal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        caption = QLabel("ERP")
        caption.setObjectName("title")
        bar.addWidget(caption)
        self.show_label = QLabel("Show")
        self.show_label.setObjectName("muted")
        bar.addWidget(self.show_label)
        self.condition = QComboBox()
        self.condition.setToolTip(
            "Every condition on one channel, or one condition on the channels "
            "picked on the head (butterfly)"
        )
        self.condition.currentIndexChanged.connect(lambda _: self._mode())
        bar.addWidget(self.condition)
        self.channels = HeadChannelMenu(theme)
        self.channels.setToolTip("Pick on the head: one channel for every "
                                 "condition, or the channels of one")  # fmt: skip
        self.channels.changed.connect(self._picked)
        bar.addWidget(self.channels)
        self.sem = QCheckBox("SEM")
        self.sem.setToolTip("Shade each curve's mean ± its standard error "
                            "(sample SD over √n)")  # fmt: skip
        self.sem.toggled.connect(lambda _: self._draw())
        bar.addWidget(self.sem)
        self.reset = QPushButton("Reset")
        self.reset.setToolTip("Start the running averages over")
        self.reset.clicked.connect(self.reset_requested.emit)
        bar.addWidget(self.reset)
        bar.addStretch(1)
        self.legend = QLabel("")
        self.legend.setObjectName("muted")
        bar.addWidget(self.legend)
        lay.addLayout(bar)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", "time from the event", units="ms")
        self.key = self.plot.addLegend(offset=(8, 8))
        lay.addWidget(self.plot, 1)
        self.erp: Erp | None = None
        self.channel = ""  # the one channel of every condition (Cz first)
        self.butterfly: set[str] | None = None  # hidden in one condition's view
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.setBackground(theme.plot_background)
        for axis in ("left", "bottom"):
            ax = self.plot.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(theme.grid))
            ax.setTextPen(pg.mkPen(theme.text_muted))
        self.channels.apply_theme(theme)
        self._draw()

    def refresh(self, erp: Erp | None, name: str, live: bool) -> None:
        self.reset.setVisible(live)
        self.erp = erp
        if erp is not None:
            self._choices(erp)
        n = sum(erp.counts) if erp is not None else 0
        what = "live, since the start or the last reset" if live else "offline run"
        self.legend.setText(f"{name} · {n} epochs ({what})" if erp else
                            f"{name} · no epochs yet")  # fmt: skip
        self._draw()

    def one_condition(self) -> bool:
        return self.erp is not None and self.condition.currentText() in (
            self.erp.conditions
        )

    def _choices(self, erp: Erp) -> None:
        """Keep the shown condition and channels when new epochs come in."""
        current = self.condition.currentText()
        items = [ALL, *erp.conditions]
        if [self.condition.itemText(i) for i in range(self.condition.count())] != items:
            self.condition.blockSignals(True)
            self.condition.clear()
            self.condition.addItems(items)
            self.condition.setCurrentIndex(items.index(current)
                                           if current in items else 0)  # fmt: skip
            self.condition.blockSignals(False)
        if self.channel not in erp.channels:
            self.channel = next((c for c in erp.channels if c.lower() == "cz"),
                                erp.channels[0] if erp.channels else "")  # fmt: skip
        if self.channels.labels != erp.channels:
            self.channels.blockSignals(True)
            self.channels.set_labels(erp.channels)
            self.channels.blockSignals(False)
            self._mode(draw=False)

    def _mode(self, draw: bool = True) -> None:
        """Set the head picker for the mode: one channel for every condition,
        or the butterfly's channels for one."""
        labels = self.channels.labels
        one = self.one_condition()
        self.channels.blockSignals(True)
        self.channels.set_single(not one)
        if one:
            self.channels._set(set(self.butterfly or ()) & set(labels))
        else:
            self.channels._set(set(labels) - {self.channel})
        self.channels.blockSignals(False)
        if draw:
            self._draw()

    def _picked(self) -> None:
        if self.one_condition():
            self.butterfly = set(self.channels.hidden)
        else:
            shown = [c for c in self.channels.labels if c not in self.channels.hidden]
            self.channel = shown[0] if shown else self.channel
        self._draw()

    def curves(self) -> list[Curve]:
        """What is drawn, in order."""
        erp = self.erp
        if erp is None or not erp.conditions or not erp.channels:
            return []
        scale = 1e6 if erp.unit == "V" else 1.0
        ms = erp.times * 1e3

        def curve(name: str, k: int, c: int) -> Curve:
            sem = None if erp.sem is None else erp.sem[k, c] * scale
            return Curve(name, ms, erp.values[k, c] * scale, sem)

        if self.one_condition():  # one condition: the picked channels
            k = erp.conditions.index(self.condition.currentText())
            return [curve(erp.channels[c], k, c)
                    for c in self.channels.selected(erp.channels)]  # fmt: skip
        if self.channel not in erp.channels:
            return []
        c = erp.channels.index(self.channel)
        return [curve(f"{label} (n={n})", k, c)
                for k, (label, n) in enumerate(
                    zip(erp.conditions, erp.counts, strict=True))]  # fmt: skip

    def _draw(self) -> None:
        self.plot.clear()
        self.key.clear()
        unit = "µV" if self.erp is None or self.erp.unit == "V" else self.erp.unit
        self.plot.setLabel("left", unit)
        zero = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(
            self.theme.text_muted, style=Qt.PenStyle.DashLine))  # fmt: skip
        self.plot.addItem(zero)
        for k, line in enumerate(self.curves()):
            color = QColor(ERP_COLORS[k % len(ERP_COLORS)])
            if self.sem.isChecked() and line.sem is not None:
                self._band(line, color)
            pen = pg.mkPen(color, width=2)
            self.plot.plot(line.times, line.mean, name=line.name, pen=pen)

    def _band(self, line: Curve, color: QColor) -> None:
        """The mean ± SEM, shaded (left out where there is no SEM: one epoch)."""
        assert line.sem is not None
        if not np.isfinite(line.sem).any():
            return
        sem = np.nan_to_num(line.sem)
        lo = pg.PlotDataItem(line.times, line.mean - sem)
        hi = pg.PlotDataItem(line.times, line.mean + sem)
        fill = QColor(color)
        fill.setAlpha(SEM_ALPHA)
        band = pg.FillBetweenItem(lo, hi, brush=fill)
        band.setZValue(-1)
        self.plot.addItem(band)
