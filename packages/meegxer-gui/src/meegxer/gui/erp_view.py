"""The ERP view of an epochs tap (M10.3): the running mean of each condition's
epochs, live (kept as sums and counts in the core's ERP tap, so it is exact
however many frames are lost) or over an offline run.

*Show* ``All conditions`` draws one curve per condition on the chosen channel
(Cz first), with each condition's epoch count in the legend, as the epoch
review window does; a single condition draws the channels picked under
*Channels* overlaid (a butterfly plot). *Reset* starts a live ERP over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.epoch_review_window import ERP_COLORS
from meegxer.gui.signal_view import ChannelMenu
from meegxer.gui.theme import Theme

ALL = "All conditions"


@dataclass(frozen=True, eq=False)
class Erp:
    """The ERPs of one tap: ``values`` is conditions x channels x times."""

    conditions: tuple[str, ...]
    counts: tuple[int, ...]
    channels: tuple[str, ...]
    times: np.ndarray
    values: np.ndarray
    unit: str = "V"


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
            "picked under Channels (butterfly)"
        )
        self.condition.currentIndexChanged.connect(lambda _: self._draw())
        bar.addWidget(self.condition)
        self.channel = QComboBox()
        self.channel.setToolTip("The channel the conditions are drawn on")
        self.channel.currentIndexChanged.connect(lambda _: self._draw())
        bar.addWidget(self.channel)
        self.channels = ChannelMenu()
        self.channels.changed.connect(self._draw)
        bar.addWidget(self.channels)
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
        self.plot.setLabel("bottom", "time from the event", units="s")
        self.key = self.plot.addLegend(offset=(8, 8))
        lay.addWidget(self.plot, 1)
        self.erp: Erp | None = None
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.setBackground(theme.plot_background)
        for axis in ("left", "bottom"):
            ax = self.plot.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(theme.grid))
            ax.setTextPen(pg.mkPen(theme.text_muted))
        self._draw()

    def refresh(self, erp: Erp | None, name: str, live: bool) -> None:
        self.reset.setVisible(live)
        if erp is not None:
            self._choices(erp)
        self.erp = erp
        n = sum(erp.counts) if erp is not None else 0
        what = "live, since the start or the last reset" if live else "offline run"
        self.legend.setText(f"{name} · {n} epochs ({what})" if erp else
                            f"{name} · no epochs yet")  # fmt: skip
        self._draw()

    def _choices(self, erp: Erp) -> None:
        """Keep the shown condition and channel when new epochs come in."""
        current = self.condition.currentText()
        items = [ALL, *erp.conditions]
        if [self.condition.itemText(i) for i in range(self.condition.count())] != items:
            self.condition.blockSignals(True)
            self.condition.clear()
            self.condition.addItems(items)
            self.condition.setCurrentIndex(max(0, items.index(current))
                                           if current in items else 0)  # fmt: skip
            self.condition.blockSignals(False)
        labels = list(erp.channels)
        if [self.channel.itemText(i) for i in range(self.channel.count())] != labels:
            chosen = self.channel.currentText()
            self.channel.blockSignals(True)
            self.channel.clear()
            self.channel.addItems(labels)
            cz = next((i for i, c in enumerate(labels) if c.lower() == "cz"), 0)
            self.channel.setCurrentIndex(labels.index(chosen)
                                         if chosen in labels else cz)  # fmt: skip
            self.channel.blockSignals(False)
        self.channels.set_labels(erp.channels)

    def curves(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """What is drawn: (legend name, times in s, values in the plot's unit)."""
        erp = self.erp
        if erp is None or not erp.conditions or not erp.channels:
            return []
        scale = 1e6 if erp.unit == "V" else 1.0
        chosen = self.condition.currentText()
        if chosen in erp.conditions:  # one condition: the picked channels
            k = erp.conditions.index(chosen)
            return [(erp.channels[c], erp.times, erp.values[k, c] * scale)
                    for c in self.channels.selected(erp.channels)]  # fmt: skip
        c = max(0, self.channel.currentIndex())
        return [(f"{label} (n={n})", erp.times, erp.values[k, c] * scale)
                for k, (label, n) in enumerate(
                    zip(erp.conditions, erp.counts, strict=True))]  # fmt: skip

    def _draw(self) -> None:
        one = (
            self.erp is not None and self.condition.currentText() in self.erp.conditions
        )
        self.channel.setVisible(not one)
        self.channels.setVisible(one)
        self.plot.clear()
        self.key.clear()
        unit = "µV" if self.erp is None or self.erp.unit == "V" else self.erp.unit
        self.plot.setLabel("left", unit)
        zero = pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(
            self.theme.text_muted, style=Qt.PenStyle.DashLine))  # fmt: skip
        self.plot.addItem(zero)
        for k, (name, times, values) in enumerate(self.curves()):
            pen = pg.mkPen(ERP_COLORS[k % len(ERP_COLORS)], width=2)
            self.plot.plot(times, values, name=name, pen=pen)
