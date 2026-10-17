"""Signal view: probes overlaid by bin index (rule 8), plus the stacked window.

One trace area shows every selected channel (all by default, i.e. a butterfly
plot), one colour per channel from the scalp colour wheel. The Main probe is drawn
thick, comparison probes thin and faded. Difference mode shows Main minus the
first comparison (what that stretch of the chain removed); spectrum mode shows
power spectra of the same windows, with a head plot of the power in a band (the
shaded, draggable region; ``headplot``) and, live, spectra smoothed between
refreshes. Bypassed data is hatched.

Time moves only with the panner under each trace area, and the shown span is
always the chosen one (e.g. 10 s): no margins beyond it, and while a live stream
is younger than the span the part before its first data stays blank. The mouse
works on the Y axis only:
the wheel scales it, dragging moves it, a double click fits it again. While the
stacked window is open the signal view rests; the stacked window has its own
channels, span and panner.
"""

from __future__ import annotations

import time
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

from meegxer.gui.colormaps import Maps
from meegxer.gui.colors import ChannelStyle, channel_styles_with
from meegxer.gui.headplot import (
    BANDS,
    DEFAULT_RANGE,
    RANGES,
    SCHEMES,
    SYMBOLS,
    HeadInput,
    HeadPlot,
    Layout,
    PeakHold,
    Smoother,
    render,
)
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
    f, p = _psd_rows(series[None, :], rate, seg)
    return f, (p[0] if len(f) else p.reshape(-1))


def _psd_rows(
    series: FloatArray, rate: float, seg: int = 256
) -> tuple[FloatArray, FloatArray]:
    """Welch power spectra (µV²/Hz) of each row of envelope mid-lines, over the
    bins where every row has data (live, the start of the span may be blank)."""
    rows = np.any(np.isfinite(series), axis=1)  # a channel without data stays NaN
    x = series[rows][:, np.all(np.isfinite(series[rows]), axis=0)]
    if x.shape[1] < 32:
        return np.empty(0), np.empty((series.shape[0], 0))
    seg = min(seg, x.shape[1])
    step = seg // 2
    win = np.hanning(seg)
    scale = rate * (win**2).sum()
    starts = range(0, x.shape[1] - seg + 1, step)
    frames = np.stack([x[:, i : i + seg] for i in starts], axis=1)  # ch, frame, seg
    frames = frames - frames.mean(axis=-1, keepdims=True)
    spectra = np.abs(np.fft.rfft(frames * win, axis=-1)) ** 2
    psd = np.full((series.shape[0], seg // 2 + 1), np.nan)
    psd[rows] = 2 * spectra.mean(axis=1) / scale
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

    def toggle_channel(self, label: str) -> None:
        """Show or hide one channel (e.g. clicked on the head plot)."""
        hidden = set(self.hidden)
        (hidden.discard if label in hidden else hidden.add)(label)
        self._set(hidden)

    def selected(self, labels: tuple[str, ...]) -> list[int]:
        return [i for i, lb in enumerate(labels) if lb not in self.hidden]


class TraceViewBox(pg.ViewBox):
    """The mouse works on Y only (time belongs to the panner): the wheel scales
    it, dragging moves it, a double click fits it again."""

    def __init__(self) -> None:
        super().__init__()
        self.setMouseEnabled(x=False, y=True)

    def wheelEvent(self, ev: Any, axis: int | None = None) -> None:
        super().wheelEvent(ev, axis=1)

    def mouseDoubleClickEvent(self, ev: Any) -> None:
        self.enableAutoRange(axis=pg.ViewBox.YAxis)
        ev.accept()


class Panner(QWidget):
    """The time control under a trace area: a scrollbar over what can be shown
    (the whole recording offline, the kept history live). Live, it follows the
    newest data until moved back; "Latest" returns to it."""

    def __init__(self) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        caption = QLabel("Time")
        caption.setObjectName("muted")
        lay.addWidget(caption)
        self.bar = QScrollBar(Qt.Orientation.Horizontal)
        self.bar.setToolTip("Drag to move through time")
        lay.addWidget(self.bar, 1)
        self.where = QLabel("")
        self.where.setObjectName("muted")
        self.where.setMinimumWidth(130)
        self.where.setAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.where)
        self.latest = QPushButton("Latest ⇥")
        self.latest.setToolTip("Jump to the newest data and keep following it")
        self.latest.clicked.connect(self.follow)
        self.latest.setVisible(False)
        lay.addWidget(self.latest)
        self.following = True  # live: the panner sits at the newest data
        self.live = False
        self.bar.sliderMoved.connect(self._moved)
        self.bar.actionTriggered.connect(lambda _: self._moved(None))

    def _moved(self, value: int | None) -> None:
        pos = self.bar.sliderPosition() if value is None else value
        self.following = pos >= self.bar.maximum()
        self.latest.setEnabled(not self.following)

    def follow(self) -> None:
        self.following = True
        self.bar.setValue(self.bar.maximum())
        self.latest.setEnabled(False)

    def set_extent(self, minimum: int, maximum: int, page: int, live: bool) -> None:
        """Range (samples offline, display bins live). Live, it keeps following
        the newest data unless moved back from the right end."""
        maximum = max(minimum, maximum)
        self.live = live
        self.bar.blockSignals(True)
        self.bar.setRange(minimum, maximum)
        self.bar.setPageStep(max(1, page))
        self.bar.setSingleStep(max(1, page // 10))
        if live and self.following:
            self.bar.setValue(maximum)
        self.bar.blockSignals(False)
        self.latest.setVisible(live)
        self.latest.setEnabled(not self.following)

    def position(self) -> int:
        return int(self.bar.value())

    def show_span(self, t0: float, t1: float) -> None:
        """Say where the shown span is (live: how far behind the newest data)."""
        if not self.live:
            self.where.setText(f"{t0:.1f} – {t1:.1f} s")  # noqa: RUF001
        elif self.following:
            self.where.setText("newest data")
        else:
            self.where.setText(f"{-t1:.1f} s behind")


class TracePlot(pg.PlotWidget):
    """Draws main and comparison windows; reuses curve items between frames."""

    def __init__(self, theme: Theme) -> None:
        super().__init__(viewBox=TraceViewBox())
        self.theme = theme
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

    def set_time(self, t0: float, t1: float) -> None:
        """Show exactly this span: no margins, so nothing blank at the sides."""
        if t1 > t0:
            self.getViewBox().setXRange(t0, t1, padding=0)

    def fit_x(self) -> None:
        """Spectrum: X follows the data."""
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
        lay.addWidget(self._spectrum_bar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        self.head = HeadPlot(theme)
        self.head.setVisible(False)
        self.head.hovered.connect(self._hover_channel)
        self.head.clicked.connect(self.channels.toggle_channel)
        body.addWidget(self.head)
        self.plot = TracePlot(theme)
        body.addWidget(self.plot, 1)
        lay.addLayout(body, 1)
        self.band = pg.LinearRegionItem(values=BANDS["alpha"], movable=True)
        self.band.setZValue(-5)
        self.band.setVisible(False)
        self.band.setToolTip("The band the head plot shows: drag it or its edges")
        self.band.sigRegionChanged.connect(self._band_dragged)
        self.plot.addItem(self.band, ignoreBounds=True)
        self._style_band()
        self.panner = Panner()
        lay.addWidget(self.panner)
        self.smoother = Smoother(2.0)
        self.peak = PeakHold(2.0)  # the heat scale's top, live
        self.hover: str | None = None  # channel named on the head plot
        self._layouts: dict[tuple[object, ...], Layout] = {}
        self._head_inputs: tuple[object, ...] | None = None  # of the shown map
        self.maps = Maps()  # colour maps (Settings → Visualization)
        self.styles: dict[tuple[str, ...], list[ChannelStyle]] = {}
        self.wheel: dict[str, tuple[float, float]] = {}
        self.types: dict[str, str] = {}  # label -> channel type, when known
        self.events: list[dict[str, Any]] = []
        self.names: dict[str, str] = {}  # probe -> what it shows, e.g. "after Notch"
        self.delays: dict[str, float] = {}  # live: ms behind the input tap
        self.delay_note = ""

    def _spectrum_bar(self) -> QWidget:
        """Spectrum settings: the head plot's band, measure and colours, and the
        smoothing of live spectra. Shown in spectrum mode only."""
        self.spectrum_bar = QWidget()
        row = QHBoxLayout(self.spectrum_bar)
        row.setContentsMargins(0, 0, 0, 0)

        def labelled(text: str, widget: QWidget, tip: str) -> None:
            caption = QLabel(text)
            caption.setObjectName("muted")
            row.addWidget(caption)
            widget.setToolTip(tip)
            row.addWidget(widget)

        self.head_toggle = QPushButton("Head plot")
        self.head_toggle.setCheckable(True)
        self.head_toggle.setChecked(True)
        self.head_toggle.setToolTip("Show the head plot of band power")
        self.head_toggle.toggled.connect(lambda _: self._show_head())
        row.addWidget(self.head_toggle)
        self.band_menu = QComboBox()
        for name, (lo, hi) in BANDS.items():
            text = f"{SYMBOLS[name]} {name.capitalize()} ({lo:g}–{hi:g} Hz)"  # noqa: RUF001
            self.band_menu.addItem(text, (lo, hi))
        self.band_menu.addItem("Custom", None)
        self.band_menu.setCurrentIndex(2)
        self.band_menu.activated.connect(self._band_chosen)
        labelled("Band", self.band_menu, "The band the head plot shows (or drag the "
                 "shaded region on the spectrum)")  # fmt: skip
        self.measure = QComboBox()
        self.measure.addItem("Absolute", False)
        self.measure.addItem("Relative", True)
        labelled("Power", self.measure, "Absolute band power, or relative to the "
                 "power over 1–45 Hz (cancels broadband differences between "  # noqa: RUF001
                 "electrodes, such as impedance)")  # fmt: skip
        self.scheme = QComboBox()
        for key, text in SCHEMES.items():
            self.scheme.addItem(text, key)
        labelled("Map", self.scheme, "Band power: dB below the head's maximum · "
                 "Power change: Main vs. the comparison tap · Peak frequency: "
                 "where the band peaks")  # fmt: skip
        self.db_range = QComboBox()
        for db in RANGES:
            self.db_range.addItem(f"{db:g} dB", db)
        self.db_range.setCurrentIndex(RANGES.index(DEFAULT_RANGE))
        labelled("Range", self.db_range, "How far below the head's maximum the "
                 "band-power map shows (live, the maximum decays with the "
                 "smoothing)")  # fmt: skip
        self.scheme.currentIndexChanged.connect(
            lambda _: self.db_range.setEnabled(self.scheme.currentData() == "heat")
        )
        self.smoothing = QComboBox()
        for tau in (0.0, 1.0, 2.0, 5.0, 10.0):
            self.smoothing.addItem("Off" if tau == 0 else f"{tau:g} s", tau)
        self.smoothing.setCurrentIndex(2)
        self.smoothing.currentIndexChanged.connect(self._smoothing_changed)
        labelled("Smoothing", self.smoothing, "Live: average spectra over time "
                 "(exponential moving average with this time constant), for the "
                 "spectrum and the head plot")  # fmt: skip
        row.addStretch(1)
        self.spectrum_bar.setVisible(False)
        return self.spectrum_bar

    def _style_band(self) -> None:
        accent = QColor(self.theme.accent)
        fill = QColor(accent)
        fill.setAlpha(40)
        self.band.setBrush(fill)
        hover = QColor(accent)
        hover.setAlpha(70)
        self.band.setHoverBrush(hover)
        for line in self.band.lines:
            line.setPen(pg.mkPen(accent, width=1.5))
            line.setHoverPen(pg.mkPen(accent, width=3))

    def band_range(self) -> tuple[float, float]:
        lo, hi = self.band.getRegion()
        return float(min(lo, hi)), float(max(lo, hi))

    def _band_chosen(self, index: int) -> None:
        band = self.band_menu.itemData(index)
        if band is not None:
            self.band.blockSignals(True)
            self.band.setRegion(band)
            self.band.blockSignals(False)

    def _band_dragged(self) -> None:
        lo, hi = self.band_range()
        custom = self.band_menu.count() - 1
        for i in range(custom):
            a, b = self.band_menu.itemData(i)
            if abs(a - lo) < 0.05 and abs(b - hi) < 0.05:
                self.band_menu.setCurrentIndex(i)
                return
        self.band_menu.setItemText(custom, f"Custom ({lo:.1f}–{hi:.1f} Hz)")  # noqa: RUF001
        self.band_menu.setCurrentIndex(custom)

    def _smoothing_changed(self) -> None:
        self.smoother.tau = self.peak.tau = float(self.smoothing.currentData())
        self.smoother.reset()
        self.peak.reset()

    def view_state(self) -> dict[str, Any]:
        """The span and spectrum settings as shown, in the shape of the
        ``visualization`` settings (``span``, ``spectrum``) plus ``head``."""
        lo, hi = self.band_range()
        return {
            "span": int(self.seconds()),
            "spectrum": {
                "band": [round(lo, 2), round(hi, 2)],
                "relative": bool(self.measure.currentData()),
                "scheme": str(self.scheme.currentData()),
                "range": int(self.db_range.currentData()),
                "smoothing": int(self.smoothing.currentData()),
            },
            "head": self.head_toggle.isChecked(),
        }

    def set_view_state(self, state: dict[str, Any]) -> None:
        """Show these settings (any subset of ``view_state``'s keys; values the
        menus do not offer are ignored)."""

        def choose(box: QComboBox, value: Any) -> None:
            for i in range(box.count()):
                data = box.itemData(i)
                if data == value or (
                    isinstance(data, float) and isinstance(value, int | float)
                    and abs(data - value) < 1e-9
                ):  # fmt: skip
                    box.setCurrentIndex(i)
                    return

        if "span" in state:
            choose(self.span, float(state["span"]))
        spectrum = state.get("spectrum", {})
        band = spectrum.get("band")
        if isinstance(band, list | tuple) and len(band) == 2:
            lo, hi = sorted(float(v) for v in band)
            if hi > lo:
                self.band.setRegion((lo, hi))  # the band menu follows
        boxes = (("relative", self.measure), ("scheme", self.scheme),
                 ("range", self.db_range), ("smoothing", self.smoothing))  # fmt: skip
        for key, box in boxes:
            if key in spectrum:
                choose(box, spectrum[key])
        if "head" in state:
            self.head_toggle.setChecked(bool(state["head"]))

    def _hover_channel(self, label: str | None) -> None:
        self.hover = label

    def _head_message(self, text: str) -> None:
        self._head_inputs = None  # the next map is drawn afresh
        self.head.show_map(text)

    def _show_head(self) -> None:
        self.head.setVisible(self.mode == "spectrum" and self.head_toggle.isChecked())
        self._size_head()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._size_head()

    def _size_head(self) -> None:
        """The head plot is square beside the spectrum: as tall as the trace
        area, at most 40 % of the width."""
        side = int(min(self.plot.height() + 44, 0.4 * self.width()))
        self.head.setFixedWidth(max(140, side))

    def name(self, probe: str | None) -> str:
        """The probe's name, with its estimated delay when running live."""
        text = self.names.get(probe or "", probe or "")
        delay = self.delays.get(probe or "")
        if delay is not None and delay > 0:
            text += f" +{delay:.1f} ms" if delay < 10 else f" +{delay:.0f} ms"
        return text

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.styles.clear()
        self.plot.apply_theme(theme)
        self.head.apply_theme(theme)
        self._style_band()

    def set_active(self, active: bool) -> None:
        """Rest while the stacked window shows the signal (it is not drawn)."""
        self.active = active
        widgets: list[QWidget] = [
            *self.group.buttons(),
            self.channels,
            self.span,
            self.panner,
            self.spectrum_bar,
        ]
        for widget in widgets:
            widget.setEnabled(active)
        self.plot.setEnabled(active)
        if not active:
            self.plot.draw([])
            self.plot.hatch([])
            self.plot.mark([])
            self._head_message("")
            self.legend.setText("Shown in the stacked view (close it to return here)")

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        for button in self.group.buttons():
            button.setChecked(button.text().lower() == mode)
        spectrum = mode == "spectrum"
        self.spectrum_bar.setVisible(spectrum)
        self.band.setVisible(spectrum)
        self.smoother.reset()  # spectra from before are stale
        self.peak.reset()
        self._show_head()

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

    def _spectra(
        self, windows: list[Window]
    ) -> list[tuple[Window, FloatArray, FloatArray]]:
        """Each window's spectra (channels x frequencies), smoothed between
        refreshes while live."""
        out = []
        live = self.panner.live
        for win in windows:
            f, p = _psd_rows((win.mins + win.maxs) / 2 * UV, win.bin_rate)
            if live and len(f):
                p = self.smoother.update(win.probe, win.labels, f, p, time.monotonic())
            out.append((win, f, p))
        if not live:
            self.smoother.reset()
        return out

    def _draw_head(
        self,
        spectra: list[tuple[Window, FloatArray, FloatArray]],
        has_comparison: bool,
        waiting: bool,
    ) -> None:
        if not self.head.isVisible():
            return
        if waiting:
            self._head_message("Main has no data yet")
            return
        win, f, p = spectra[-1]
        first = spectra[0] if has_comparison else None
        data = HeadInput(
            labels=win.labels,
            f=f,
            psd=p,
            band=self.band_range(),
            scheme=str(self.scheme.currentData()),
            relative=bool(self.measure.currentData()),
            db_range=float(self.db_range.currentData()),
            hidden=frozenset(self.channels.hidden),
            compare_psd=None if first is None or not len(first[1]) else first[2],
            compare_labels=() if first is None else first[0].labels,
            wheel=self.wheel,
        )
        if first is not None and first[1].shape != f.shape:
            data.compare_psd = None  # taps at different rates: no common grid
        hold = None
        live = self.panner.live
        if live:  # the scale's top decays instead of flickering
            key = (win.probe, data.band, data.relative, data.scheme)
            hold = lambda top: self.peak.update(key, top)  # noqa: E731
        else:  # offline the view often redraws the same spectra: skip the work
            inputs = (data.labels, p.tobytes(), data.band, data.scheme,
                      data.relative, data.db_range, data.hidden,
                      None if data.compare_psd is None else data.compare_psd.tobytes(),
                      id(self.theme), tuple(sorted(self.wheel.items())),
                      self.maps)  # fmt: skip
            if inputs == self._head_inputs:
                return
            self._head_inputs = inputs
        if live:
            self._head_inputs = None
        self.head.show_map(render(data, self.theme, self._layouts, hold, self.maps))

    def refresh(
        self, provider: Provider, main: str | None, comparisons: list[str]
    ) -> None:
        if not self.active:
            return
        main_win = provider(main, self.seconds()) if main else None
        comp = [
            w for p in comparisons if (w := provider(p, self.seconds())) is not None
        ]
        # Main has no data yet (e.g. a strip still calibrating): show the
        # comparisons as comparisons; the first one only lends its channels/time
        waiting = main_win is None and bool(comp)
        if waiting:
            main_win, comp = comp[0], comp[1:]
        if main_win is None:
            self.plot.draw([])
            self.plot.hatch([])
            self.plot.mark([])
            self.legend.setText("Nothing shown: click a ◇ tap in the chain")
            self._head_message("Nothing shown: click a ◇ tap in the chain")
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
        self.legend.setToolTip(self.delay_note)
        layers: list[tuple[FloatArray, FloatArray, QPen]] = []
        plot = self.plot.getPlotItem()
        names = f"Main: {self.name(main_win.probe)}" + "".join(
            f" · {self.name(w.probe)}" for w in comp
        )
        main_pen = (1.0, 80) if waiting else (2.0, 255)
        if waiting:
            names = f"Main: {self.name(main)} (no data yet) · " + " · ".join(
                self.name(w.probe) for w in [main_win, *comp]
            )
        if self.mode == "difference":
            other = comp[0] if comp else None
            if waiting:
                self.legend.setText(
                    f"Difference: Main ({self.name(main)}) has no data yet"
                )
            elif other is None:
                self.legend.setText("Difference needs a second tap (click a ◇)")
            else:
                self.legend.setText(
                    f"{self.name(main_win.probe)} − {self.name(other.probe)} "  # noqa: RUF001
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
            spectra = self._spectra([*comp, main_win])
            pens = [(1.0, 90)] * len(comp) + [(1.0, 90) if waiting else main_pen]
            for (win, f, p), (width, alpha) in zip(spectra, pens, strict=True):
                if len(f) < 2:
                    continue
                hovered = [c for c in chans if win.labels[c] == self.hover]
                for c in [c for c in chans if c not in hovered] + hovered:
                    st = styles[c]
                    bold = bool(hovered) and c == hovered[0] and win is main_win
                    pen = _pen(st.color, width + 2 if bold else width,
                               255 if bold else alpha, st.dashed)  # fmt: skip
                    layers.append((f[1:], 10 * np.log10(p[c, 1:] + 1e-12), pen))
            self._draw_head(spectra, has_comparison=bool(comp), waiting=waiting)
            plot.setLabel("left", "power (dB µV²/Hz)")
            plot.setLabel("bottom", "frequency", units="Hz")
        else:
            self.legend.setText(names)
            for win, width, alpha in [(w, 1.0, 80) for w in comp] + [
                (main_win, *main_pen)
            ]:
                for c in chans:
                    st = styles[c]
                    x, y = _envelope_xy(win.t, win.mins[c] * UV, win.maxs[c] * UV)
                    layers.append((x, y, _pen(st.color, width, alpha, st.dashed)))
            plot.setLabel("left", "µV")
            plot.setLabel("bottom", "time", units="s")
        self.plot.draw(layers)
        t0, t1 = float(main_win.t[0]), float(main_win.t[-1]) + 1 / main_win.bin_rate
        self.panner.show_span(t0, t1)
        if self.mode == "spectrum":
            self.plot.fit_x()
        else:
            self.plot.set_time(t0, t1)
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
    default), span and panner: through the whole recording offline, through
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
        self.panner = Panner()
        lay.addWidget(self.panner)
        self.theme = theme
        self._y_key: object = None

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.apply_theme(theme)

    def seconds(self) -> float:
        return float(self.span.currentData())

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
        main_win = provider(main, seconds) if main else None
        wins = [w for p in comparisons if (w := provider(p, seconds)) is not None]
        has_main = main_win is not None  # else comparisons stay comparisons
        if main_win is not None:
            wins.append(main_win)
        if not wins:
            self.plot.draw([])
            return
        top = wins[-1]  # Main, or a comparison lending its channels and time
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
            is_main = has_main and k == len(wins) - 1
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
        t0, t1 = float(top.t[0]), float(top.t[-1]) + 1 / top.bin_rate
        self.plot.set_time(t0, t1)
        self.panner.show_span(t0, t1)
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
