"""Signal view: probes overlaid by bin index (rule 8), plus the stacked window.

One trace area shows every selected channel (all by default, i.e. a butterfly
plot), one colour per channel from the scalp colour wheel. The Main probe is drawn
thick, comparison probes thin and faded. Difference mode shows Main minus the
first comparison (what that stretch of the chain removed); spectrum mode shows
power spectra of the same windows. Bypassed data is hatched.

Mouse: the wheel scales the Y axis only, dragging pans X and Y, a double click
restores the automatic range. While the stacked window is open the signal view
rests; the stacked window has its own channels, span and scrollbar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.colors import ChannelStyle, channel_styles_with
from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]
UV = 1e6  # display in microvolts


@dataclass
class Window:
    """Bins of one probe for the visible span; ``t`` in seconds."""

    probe: str
    labels: tuple[str, ...]
    t: FloatArray
    mins: FloatArray  # (n_channels, n_bins), volts
    maxs: FloatArray
    bypassed: npt.NDArray[np.bool_]
    bin_rate: float


# provider(probe, seconds) -> Window or None
Provider = Callable[[str, float], "Window | None"]


def _pen(color: tuple[int, int, int], width: float, alpha: int, dashed: bool) -> QPen:
    pen = QPen(QColor(*color, alpha))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


def _envelope_xy(
    t: FloatArray, mins: FloatArray, maxs: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Interleave min and max so each bin draws as a vertical stroke."""
    x = np.repeat(t, 2)
    y = np.empty(2 * len(t))
    y[0::2], y[1::2] = mins, maxs
    return x, y


def _psd(
    series: FloatArray, rate: float, seg: int = 256
) -> tuple[FloatArray, FloatArray]:
    """Welch power spectrum (µV²/Hz) of an envelope mid-line."""
    x = series[np.isfinite(series)]
    if len(x) < 32:
        return np.empty(0), np.empty(0)
    seg = min(seg, len(x))
    step = seg // 2
    win = np.hanning(seg)
    scale = rate * (win**2).sum()
    frames = [x[i : i + seg] for i in range(0, len(x) - seg + 1, step)]
    spectra = [np.abs(np.fft.rfft((f - f.mean()) * win)) ** 2 for f in frames]
    psd = 2 * np.mean(spectra, axis=0) / scale
    return np.fft.rfftfreq(seg, 1 / rate), psd


class ChannelMenu(QToolButton):
    """Pick the channels to show; all by default."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setText("Channels: all")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.labels: tuple[str, ...] = ()
        self.defaults: frozenset[str] = frozenset()
        self.hidden: set[str] = set()
        self.setMenu(QMenu(self))

    def set_labels(
        self, labels: tuple[str, ...], hide_by_default: frozenset[str] = frozenset()
    ) -> None:
        """New channels: non-EEG ones (``hide_by_default``) start hidden."""
        if labels == self.labels and hide_by_default == self.defaults:
            return
        self.labels, self.defaults = labels, hide_by_default
        self.hidden = set(hide_by_default) & set(labels)
        menu = QMenu(self)
        menu.addAction("All", lambda: self._set(set()))
        menu.addAction("None", lambda: self._set(set(labels)))
        menu.addSeparator()
        for label in labels:
            action = QAction(label, menu, checkable=True)
            action.setChecked(label not in self.hidden)
            action.toggled.connect(lambda on, lb=label: self._toggle(lb, on))
            menu.addAction(action)
        self.setMenu(menu)
        self._caption()

    def _set(self, hidden: set[str]) -> None:
        self.hidden = hidden
        for action in self.menu().actions():
            if action.isCheckable():
                action.blockSignals(True)
                action.setChecked(action.text() not in hidden)
                action.blockSignals(False)
        self._caption()
        self.changed.emit()

    def _toggle(self, label: str, on: bool) -> None:
        (self.hidden.discard if on else self.hidden.add)(label)
        self._caption()
        self.changed.emit()

    def _caption(self) -> None:
        shown = len(self.labels) - len(self.hidden)
        self.setText(
            "Channels: all"
            if not self.hidden
            else f"Channels: {shown}/{len(self.labels)}"
        )

    def selected(self, labels: tuple[str, ...]) -> list[int]:
        return [i for i, lb in enumerate(labels) if lb not in self.hidden]


class TraceViewBox(pg.ViewBox):
    """The wheel scales the Y axis only; dragging pans X and Y (pyqtgraph's
    default); a double click restores the automatic range."""

    def wheelEvent(self, ev: Any, axis: int | None = None) -> None:
        super().wheelEvent(ev, axis=1)

    def mouseDoubleClickEvent(self, ev: Any) -> None:
        self.enableAutoRange()
        ev.accept()


class TracePlot(pg.PlotWidget):
    """Draws main and comparison windows; reuses curve items between frames."""

    def __init__(self, theme: Theme) -> None:
        super().__init__(viewBox=TraceViewBox())
        self.theme = theme
        self._x_key: object = None
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setMenuEnabled(False)
        self.getPlotItem().hideButtons()
        self._curves: list[pg.PlotDataItem] = []
        self._regions: list[pg.LinearRegionItem] = []
        self._marks: list[pg.InfiniteLine] = []
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.setBackground(theme.plot_background)
        for axis in ("left", "bottom"):
            ax = self.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen(theme.grid))
            ax.setTextPen(pg.mkPen(theme.text_muted))

    def _curve(self, i: int) -> pg.PlotDataItem:
        while len(self._curves) <= i:
            item = pg.PlotDataItem(connect="finite", skipFiniteCheck=True)
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            self.addItem(item)
            self._curves.append(item)
        return self._curves[i]

    def follow_x(self, key: object) -> None:
        """A new stretch of time (or a new axis): fit X again, even after a pan."""
        if key != self._x_key:
            self._x_key = key
            self.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis)

    def draw(self, layers: list[tuple[FloatArray, FloatArray, QPen]]) -> None:
        for i, (x, y, pen) in enumerate(layers):
            curve = self._curve(i)
            curve.setPen(pen)
            curve.setData(x, y)
            curve.setVisible(True)
        for curve in self._curves[len(layers) :]:
            curve.setVisible(False)

    def hatch(self, spans: list[tuple[float, float]]) -> None:
        brush = QBrush(QColor(self.theme.bypass_hatch), Qt.BrushStyle.BDiagPattern)
        while len(self._regions) < len(spans):
            region = pg.LinearRegionItem(movable=False, brush=brush)
            region.setZValue(-10)
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            self.addItem(region)
            self._regions.append(region)
        for region, span in zip(self._regions, spans, strict=False):
            region.setRegion(span)
            region.setBrush(brush)
            region.setVisible(True)
        for region in self._regions[len(spans) :]:
            region.setVisible(False)

    def mark(self, events: list[tuple[float, str]]) -> None:
        """Event markers: a vertical line and the label at each onset (s)."""
        pen = pg.mkPen(
            QColor(self.theme.text_muted), width=1, style=Qt.PenStyle.DashLine
        )
        while len(self._marks) < len(events):
            line = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pen,
                label="",
                labelOpts={"position": 0.95, "color": self.theme.text_muted},
            )
            self.addItem(line)
            self._marks.append(line)
        for line, (t, label) in zip(self._marks, events, strict=False):
            line.setPen(pen)
            line.setPos(t)
            line.label.setFormat(label)
            line.setVisible(True)
        for line in self._marks[len(events) :]:
            line.setVisible(False)


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


class SignalView(QWidget):
    """Toolbar + trace area; ``refresh`` pulls windows from a provider."""

    stacked_requested = Signal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.mode = "overlay"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        caption = QLabel("SIGNAL")
        caption.setObjectName("title")
        bar.addWidget(caption)
        self.group = QButtonGroup(self)
        self.active = True
        for i, name in enumerate(("Overlay", "Difference", "Spectrum")):
            button = QPushButton(name)
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.group.addButton(button)
            mode = name.lower()
            button.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            bar.addWidget(button)
        self.channels = ChannelMenu()
        bar.addWidget(self.channels)
        self.span = QComboBox()
        for seconds in (2, 5, 10, 20, 30):
            self.span.addItem(f"{seconds} s", float(seconds))
        self.span.setCurrentIndex(2)
        bar.addWidget(self.span)
        bar.addStretch(1)
        self.legend = QLabel("")
        self.legend.setObjectName("muted")
        bar.addWidget(self.legend)
        stacked = QPushButton("⧉ Stacked view")
        stacked.setToolTip("Stacked traces in a separate window")
        stacked.clicked.connect(self.stacked_requested)
        bar.addWidget(stacked)
        lay.addLayout(bar)
        self.plot = TracePlot(theme)
        lay.addWidget(self.plot, 1)
        self.styles: dict[tuple[str, ...], list[ChannelStyle]] = {}
        self.wheel: dict[str, tuple[float, float]] = {}
        self.types: dict[str, str] = {}  # label -> channel type, when known
        self.events: list[dict[str, Any]] = []

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.styles.clear()
        self.plot.apply_theme(theme)

    def set_active(self, active: bool) -> None:
        """Rest while the stacked window shows the signal (it is not drawn)."""
        self.active = active
        widgets: list[QWidget] = [*self.group.buttons(), self.channels, self.span]
        for widget in widgets:
            widget.setEnabled(active)
        self.plot.setEnabled(active)
        if not active:
            self.plot.draw([])
            self.plot.hatch([])
            self.plot.mark([])
            self.legend.setText("Shown in the stacked view (close it to return here)")

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        for button in self.group.buttons():
            button.setChecked(button.text().lower() == mode)

    def seconds(self) -> float:
        return float(self.span.currentData())

    def set_wheel(self, wheel: dict[str, tuple[float, float]]) -> None:
        """Colour channels by the recording's own positions."""
        self.wheel = wheel
        self.styles.clear()

    def _styles(self, labels: tuple[str, ...]) -> list[ChannelStyle]:
        if labels not in self.styles:
            self.styles[labels] = channel_styles_with(labels, self.theme, self.wheel)
        return self.styles[labels]

    def refresh(
        self, provider: Provider, main: str | None, comparisons: list[str]
    ) -> None:
        if not self.active:
            return
        main_win = provider(main, self.seconds()) if main else None
        comp = [
            w for p in comparisons if (w := provider(p, self.seconds())) is not None
        ]
        if main_win is None and comp:
            main_win, comp = comp[0], comp[1:]
        if main_win is None:
            self.plot.draw([])
            self.plot.hatch([])
            self.plot.mark([])
            self.legend.setText("No probe shown: click a ◇ tap in the chain")
            return
        self.channels.set_labels(
            main_win.labels,
            frozenset(
                lb
                for lb in main_win.labels
                if self.types.get(lb, "eeg") not in ("eeg", "")
            ),
        )
        chans = self.channels.selected(main_win.labels)
        styles = self._styles(main_win.labels)
        layers: list[tuple[FloatArray, FloatArray, QPen]] = []
        plot = self.plot.getPlotItem()
        names = f"Main: {main_win.probe}" + "".join(f" · {w.probe}" for w in comp)
        if self.mode == "difference":
            other = comp[0] if comp else None
            if other is None:
                self.legend.setText("Difference needs a second tap (click a ◇)")
            else:
                self.legend.setText(
                    f"{main_win.probe} − {other.probe} "  # noqa: RUF001
                    "(what this stretch removed)"
                )
                mid_a = (main_win.mins + main_win.maxs) / 2 * UV
                mid_b = (other.mins + other.maxs) / 2 * UV
                for c in chans:
                    st = styles[c]
                    layers.append(
                        (
                            main_win.t,
                            mid_a[c] - mid_b[c],
                            _pen(st.color, 1.6, 230, st.dashed),
                        )
                    )
            plot.setLabel("left", "µV")
            plot.setLabel("bottom", "time", units="s")
            plot.setLogMode(False, False)
        elif self.mode == "spectrum":
            self.legend.setText(names)
            for win, width, alpha in [(w, 1.0, 90) for w in comp] + [
                (main_win, 2.0, 255)
            ]:
                mid = (win.mins + win.maxs) / 2 * UV
                for c in chans:
                    f, p = _psd(mid[c], win.bin_rate)
                    if len(f) > 1:
                        st = styles[c]
                        layers.append(
                            (
                                f[1:],
                                10 * np.log10(p[1:] + 1e-12),
                                _pen(st.color, width, alpha, st.dashed),
                            )
                        )
            plot.setLabel("left", "power (dB µV²/Hz)")
            plot.setLabel("bottom", "frequency", units="Hz")
        else:
            self.legend.setText(names)
            for win, width, alpha in [(w, 1.0, 80) for w in comp] + [
                (main_win, 2.0, 255)
            ]:
                for c in chans:
                    st = styles[c]
                    x, y = _envelope_xy(win.t, win.mins[c] * UV, win.maxs[c] * UV)
                    layers.append((x, y, _pen(st.color, width, alpha, st.dashed)))
            plot.setLabel("left", "µV")
            plot.setLabel("bottom", "time", units="s")
        self.plot.draw(layers)
        t0, t1 = float(main_win.t[0]), float(main_win.t[-1])
        self.plot.follow_x((self.mode, t0, t1) if self.mode != "spectrum" else "f")
        self.plot.mark(
            []
            if self.mode == "spectrum"
            else [
                (float(e["onset"]), str(e["label"]))
                for e in self.events
                if t0 <= float(e["onset"]) <= t1
            ][:200]
        )
        if self.mode == "spectrum":
            self.plot.hatch([])
        else:
            t = main_win.t
            step = 1 / main_win.bin_rate
            self.plot.hatch(
                [(t[a], t[b - 1] + step) for a, b in _runs(main_win.bypassed)]
            )


class StackedWindow(QWidget):
    """Stacked traces (EEGLAB-scroll style) in their own window, following the
    same probes as the signal view, with its own channel selection (all by
    default), span and scrollbar: through the whole recording offline, through
    the history the GUI keeps live (at the right end it follows the stream)."""

    closed = Signal()

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.setWindowTitle("MeegXer · stacked view")
        self.resize(900, 700)
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.channels = ChannelMenu()
        bar.addWidget(self.channels)
        self.span = QComboBox()
        for seconds in (2, 5, 10, 20, 30, 60):
            self.span.addItem(f"{seconds} s", float(seconds))
        self.span.setCurrentIndex(2)
        bar.addWidget(self.span)
        bar.addWidget(QLabel("Spacing"))
        self.spacing = QComboBox()
        self.spacing.addItem("auto", 0.0)
        for uv in (20, 50, 100, 200, 500):
            self.spacing.addItem(f"{uv} µV", float(uv))
        bar.addWidget(self.spacing)
        self.normalize = QCheckBox("Normalize")
        self.normalize.setToolTip(
            "Scale each channel by its standard deviation in the shown window"
        )
        self.normalize.toggled.connect(lambda on: self.spacing.setEnabled(not on))
        bar.addWidget(self.normalize)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.plot = TracePlot(theme)
        lay.addWidget(self.plot, 1)
        self.scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        lay.addWidget(self.scrollbar)
        self.following = True  # live: the scrollbar sits at the newest data
        self.scrollbar.sliderMoved.connect(self._moved)
        self.scrollbar.actionTriggered.connect(lambda _: self._moved(None))
        self.theme = theme
        self._y_key: object = None

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.apply_theme(theme)

    def seconds(self) -> float:
        return float(self.span.currentData())

    def _moved(self, value: int | None) -> None:
        pos = self.scrollbar.sliderPosition() if value is None else value
        self.following = pos >= self.scrollbar.maximum()

    def set_extent(self, minimum: int, maximum: int, page: int, live: bool) -> None:
        """Scrollbar range (samples offline, display bins live). Live, it keeps
        following the newest data unless moved back from the right end."""
        maximum = max(minimum, maximum)
        self.scrollbar.blockSignals(True)
        self.scrollbar.setRange(minimum, maximum)
        self.scrollbar.setPageStep(max(1, page))
        self.scrollbar.setSingleStep(max(1, page // 10))
        if live and self.following:
            self.scrollbar.setValue(maximum)
        self.scrollbar.blockSignals(False)

    def position(self) -> int:
        return int(self.scrollbar.value())

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
        self.closed.emit()

    def refresh(
        self,
        view: SignalView,
        provider: Provider,
        main: str | None,
        comparisons: list[str],
    ) -> None:
        if not self.isVisible():
            return
        seconds = self.seconds()
        wins = [w for p in [*comparisons, main] if p and (w := provider(p, seconds))]
        if not wins:
            self.plot.draw([])
            return
        top = wins[-1]
        self.channels.set_labels(top.labels)
        chans = self.channels.selected(top.labels)
        styles = view._styles(top.labels)
        normalize = self.normalize.isChecked()
        scale = np.ones(len(top.labels))  # per channel; comparisons use main's
        if normalize:
            sd = np.nanstd((top.mins + top.maxs) / 2 * UV, axis=1)
            ok = np.isfinite(sd) & (sd > 0)
            scale[ok] = 1.0 / sd[ok]
        spacing = 6.0 if normalize else float(self.spacing.currentData())
        if spacing <= 0:
            mid = (top.mins + top.maxs) / 2 * UV
            spread = np.nanstd(mid[chans], axis=1) if chans else np.array([1.0])
            spacing = (
                float(6 * np.nanmedian(spread)) if np.isfinite(spread).any() else 50.0
            )
            spacing = max(spacing, 1.0)
        layers: list[tuple[FloatArray, FloatArray, QPen]] = []
        for k, win in enumerate(wins):
            is_main = k == len(wins) - 1
            same = win.labels == top.labels
            for row, c in enumerate(chans):
                offset = (len(chans) - 1 - row) * spacing
                st = styles[c]
                g = scale[c] if same else 1.0
                x, y = _envelope_xy(
                    win.t, win.mins[c] * UV * g + offset, win.maxs[c] * UV * g + offset
                )
                layers.append(
                    (
                        x,
                        y,
                        _pen(
                            st.color,
                            1.6 if is_main else 1.0,
                            255 if is_main else 80,
                            st.dashed,
                        ),
                    )
                )
        self.plot.draw(layers)
        self.plot.follow_x((float(top.t[0]), float(top.t[-1])))
        y_key = (tuple(chans), spacing)
        if y_key != self._y_key:  # other rows: fit Y again, even after a zoom
            self._y_key = y_key
            self.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plot.getPlotItem().setLabel("bottom", "time", units="s")
        axis = self.plot.getPlotItem().getAxis("left")
        axis.setTicks(
            [
                [
                    ((len(chans) - 1 - row) * spacing, top.labels[c])
                    for row, c in enumerate(chans)
                ]
            ]
        )
