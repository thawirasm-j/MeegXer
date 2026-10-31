"""Feature view: what a feature strip computes, shown when its rail is tapped.

- **Spectrum** (the windows have a ``frequency`` dimension, e.g. PSD): one
  curve per channel over frequency; live the newest window, offline the mean of
  the windows in the shown span. Positive values are drawn in dB, unless they
  have no unit (``1``: SSVEP detection's correlation scores).
- **Over time** (anything else, e.g. band power, entropy): one line per channel
  (or per feature), a point at the end of each window. Live, time is seconds
  before now or since the start (as in the signal view); offline, recording time.

- **Connectivity** (channel x target, e.g. wPLI): also a head map with a line
  from the channel chosen in *Show* (the seed; or click an electrode) to every
  other, coloured by its value (live the newest window, offline the mean).

Hovering near a line names it: the line is drawn bold, its nearest point is
marked and labelled with the line's name, where it is and its value.

Further dimensions (e.g. the band of band power) are chosen in *Show*. The value
axis names the features' unit when the node states one (spec 0.2.19). The
windows come from the core, decimated and lossy (rule 2): live from the display
stream, offline from the run's feature store.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from meegxer.core.block import unit_at
from meegxer.gui.colormaps import Maps
from meegxer.gui.colors import ChannelStyle, channel_styles_with
from meegxer.gui.headplot import HeadPlot, render_links
from meegxer.gui.signal_view import HeadChannelMenu, Panner, TracePlot, _pen
from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]
SPANS = (10, 30, 60, 120, 300)  # seconds
HOVER_PX = 12.0  # how near (pixels) the pointer must be to a line to name it


@dataclass
class FeatureSeries:
    """Windows of one Features tap: ``values`` is (windows, *dims); ``t`` the
    end of each window in seconds."""

    probe: str
    dims: tuple[str, ...]
    coords: dict[str, tuple[str | float, ...]]
    t: FloatArray
    values: FloatArray
    n_windows: int = 0  # windows behind ``values`` (offline: averaged into one)
    unit: str = ""  # as FeatureBlock
    unit_dim: str = ""
    units: tuple[str, ...] = ()


def pretty_unit(unit: str) -> str:
    """A unit for an axis label: ``V^2/Hz`` as ``V²/Hz``."""
    return unit.replace("^2", "²").replace("^3", "³")


# volt-based units are shown in microvolts, as in the signal and spectrum views
# (pyqtgraph cannot SI-prefix a squared unit: it would show "V² (x1e-12)")
MICRO = {"V": ("µV", 1e6), "V^2": ("µV^2", 1e12), "V^2/Hz": ("µV^2/Hz", 1e12)}


def _slice_unit(series: FeatureSeries, chosen: dict[str, int]) -> str:
    unit = unit_at(series.unit, series.unit_dim, series.units, chosen)
    if series.unit_dim and series.unit_dim not in chosen and len(set(series.units)) > 1:
        unit = ""  # the curves differ in unit
    return unit


def unit_scale(series: FeatureSeries, chosen: dict[str, int]) -> float:
    """What the shown slice's values are multiplied by for display (1e6 for
    volts shown as µV, 1e12 for V² as µV²; else 1)."""
    return MICRO.get(_slice_unit(series, chosen), ("", 1.0))[1]


def hover_unit(series: FeatureSeries, chosen: dict[str, int]) -> str:
    """The unit of a hovered value (as on the axis; "" when none or mixed)."""
    label = value_label(series, chosen)
    return label[len("value (") : -1] if label.startswith("value (") else ""


def value_label(
    series: FeatureSeries, chosen: dict[str, int], spectrum_db: bool = False
) -> str:
    """The value axis label for the shown slice ("value" without a unit; dB with
    its reference when the view converted a power to dB)."""
    unit = _slice_unit(series, chosen)
    unit = MICRO.get(unit, (unit, 1.0))[0]
    if spectrum_db:
        return f"dB (re 1 {pretty_unit(unit)})" if unit not in ("", "1") else "dB"
    return f"value ({pretty_unit(unit)})" if unit not in ("", "1") else "value"


def series_dim(dims: tuple[str, ...]) -> str | None:
    """The dimension drawn as separate curves: channels when there are any."""
    rest = [d for d in dims if d != "frequency"]
    return "channel" if "channel" in rest else (rest[0] if rest else None)


def pick(
    values: FloatArray, dims: tuple[str, ...], chosen: dict[str, int]
) -> FloatArray:
    """(windows, series, frequencies or 1), with ``chosen`` dimensions fixed."""
    series = series_dim(dims)
    index = (slice(None), *(chosen.get(d, slice(None)) for d in dims))
    kept = [d for d in dims if d not in chosen]
    arr = values[index]
    order = [kept.index(d) + 1 for d in (series, "frequency") if d in kept]
    arr = np.transpose(arr, [0, *order])
    if series not in kept:
        arr = arr[:, None]
    if "frequency" not in kept:
        arr = arr[..., None]
    return np.asarray(arr, dtype=float)


def nearest_point(
    lines: list[tuple[FloatArray, FloatArray]],
    x: float,
    y: float,
    pixel: tuple[float, float],
    radius: float = HOVER_PX,
) -> tuple[int, int] | None:
    """The line nearest the pointer at (``x``, ``y``) and its point nearest the
    pointer, as (line, point) indices, when that line passes within ``radius``
    pixels; ``pixel`` is the data size of one pixel (x, y)."""
    px, py = (abs(pixel[0]) or 1.0), (abs(pixel[1]) or 1.0)
    best: tuple[float, int] | None = None
    for i, (lx, ly) in enumerate(lines):
        ok = np.isfinite(lx) & np.isfinite(ly)
        sx, sy = (lx[ok] - x) / px, (ly[ok] - y) / py  # pointer at the origin
        if not sx.size:
            continue
        d = float(np.min(np.hypot(sx, sy)))
        if sx.size > 1:  # distance to each segment, not only its ends
            ax, ay, bx, by = sx[:-1], sy[:-1], sx[1:], sy[1:]
            dx, dy = bx - ax, by - ay
            length = dx * dx + dy * dy
            with np.errstate(all="ignore"):
                t = np.clip(np.where(length > 0, -(ax * dx + ay * dy) / length, 0),
                            0, 1)  # fmt: skip
            d = min(d, float(np.min(np.hypot(ax + t * dx, ay + t * dy))))
        if d <= radius and (best is None or d < best[0]):
            best = (d, i)
    if best is None:
        return None
    lx, ly = lines[best[1]]
    with np.errstate(all="ignore"):
        dist = np.hypot((lx - x) / px, (ly - y) / py)
    return best[1], int(np.nanargmin(np.where(np.isfinite(dist), dist, np.nan)))


def _coords(series: FeatureSeries, dim: str) -> tuple[str | float, ...]:
    if dim in series.coords:
        return tuple(series.coords[dim])
    return tuple(range(series.values.shape[1 + series.dims.index(dim)]))


class FeatureView(QWidget):
    """Toolbar, plot and panner for one Features tap."""

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        caption = QLabel("FEATURES")
        caption.setObjectName("title")
        bar.addWidget(caption)
        self.channels = HeadChannelMenu(theme)  # channels on a head plot
        bar.addWidget(self.channels)
        self.show_label = QLabel("Show")
        self.show_label.setObjectName("muted")
        bar.addWidget(self.show_label)
        self.slice = QComboBox()
        self.slice.setToolTip("Which value of the other dimensions to show")
        bar.addWidget(self.slice)
        self.span = QComboBox()
        for seconds in SPANS:
            self.span.addItem(f"{seconds} s", float(seconds))
        self.span.setCurrentIndex(SPANS.index(60))
        bar.addWidget(self.span)
        bar.addStretch(1)
        self.legend = QLabel("")
        self.legend.setObjectName("muted")
        bar.addWidget(self.legend)
        lay.addLayout(bar)
        row = QHBoxLayout()
        self.head = HeadPlot(theme)  # connectivity map, beside the plot
        self.head.setMinimumWidth(240)
        self.head.setVisible(False)
        self.head.clicked.connect(self._seed_clicked)
        row.addWidget(self.head, 0)
        self.plot = TracePlot(theme)
        row.addWidget(self.plot, 1)
        lay.addLayout(row, 1)
        self.maps = Maps()  # colour maps (Settings → Visualization)
        self.panner = Panner()
        lay.addWidget(self.panner)
        self.wheel: dict[str, tuple[float, float]] = {}
        self._slices: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
        # hover: the drawn lines (name, x, y, pen), how to word a point, the
        # pointer (scene position) and what it names
        self._lines: list[tuple[str, FloatArray, FloatArray, object]] = []
        self._where = ("", "")  # x quantity and unit, e.g. ("t", "s")
        self._value_unit = ""
        self._mouse: QPointF | None = None
        self.hovered: tuple[str, float, float] | None = None  # name, x, y
        self._marker = pg.ScatterPlotItem(size=9)
        self._marker.setZValue(20)
        self._tag = pg.TextItem(anchor=(0, 1))
        self._tag.setZValue(21)
        for item in (self._marker, self._tag):
            item.setVisible(False)
            self.plot.addItem(item, ignoreBounds=True)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.viewport().installEventFilter(self)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.apply_theme(theme)
        self.head.apply_theme(theme)
        self.channels.apply_theme(theme)

    # -- connectivity map ---------------------------------------------------------

    @staticmethod
    def connectivity(series: FeatureSeries) -> bool:
        """Channel-by-channel values (``channel`` x ``target``, same labels)."""
        return {"channel", "target"} <= set(series.dims) and tuple(
            series.coords.get("channel", ())
        ) == tuple(series.coords.get("target", ()))

    def _seed_clicked(self, label: str) -> None:
        """Clicking an electrode makes it the seed: the *Show* entry with that
        target (the other dimensions as they are)."""
        others, names = self._slices
        if "target" not in others:
            return
        current = names[self.slice.currentIndex()].split(" · ") if names else []
        k = others.index("target")
        for i, name in enumerate(names):
            parts = name.split(" · ")
            if parts[k] == label and all(
                p == c for j, (p, c) in enumerate(zip(parts, current, strict=False))
                if j != k
            ):  # fmt: skip
                self.slice.setCurrentIndex(i)
                return

    def _draw_links(
        self, series: FeatureSeries, labels: tuple[str, ...],
        chosen: dict[str, int], data: FloatArray, n: int,
    ) -> None:  # fmt: skip
        seed = str(_coords(series, "target")[chosen["target"]])
        with np.errstate(all="ignore"):
            values = np.nanmean(data[:, :, 0], axis=0)  # live: the newest window
        what = "newest window" if self.panner.live else f"mean of {n} windows"
        self.head.show_map(render_links(
            labels, values, seed, self.theme, self.maps, self.wheel,
            hover_unit(series, chosen), f"from {seed}, {what}"))  # fmt: skip

    # -- hover --------------------------------------------------------------------

    def _mouse_moved(self, pos: QPointF) -> None:
        self._mouse = QPointF(pos)
        self._update_hover()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Leave:
            self._mouse = None
            self._update_hover()
        return super().eventFilter(watched, event)

    def _update_hover(self) -> None:
        """Bold the line nearest the pointer, mark its nearest point and label it
        (or clear all of that when no line is near)."""
        found = None
        if self._mouse is not None and self._lines:
            vb = self.plot.getViewBox()
            if vb.sceneBoundingRect().contains(self._mouse):
                at = vb.mapSceneToView(self._mouse)
                px, py = vb.viewPixelSize()
                found = nearest_point([(x, y) for _, x, y, _ in self._lines],
                                      at.x(), at.y(), (px, py))  # fmt: skip
        for i, (_, _, _, pen) in enumerate(self._lines):  # undo an earlier bold
            if i < len(self.plot._curves):
                self.plot._curves[i].setPen(pen)
        if found is None:
            self.hovered = None
            self._marker.setVisible(False)
            self._tag.setVisible(False)
            return
        i, k = found
        name, x, y, pen = self._lines[i]
        bold = pg.mkPen(pen)
        bold.setWidthF(bold.widthF() + 2)
        self.plot._curves[i].setPen(bold)
        self.hovered = (name, float(x[k]), float(y[k]))
        color = bold.color()
        self._marker.setData([x[k]], [y[k]], brush=pg.mkBrush(color),
                             pen=pg.mkPen(self.theme.text, width=1.5))  # fmt: skip
        self._tag.setText(self.hover_text(), color=self.theme.text)
        self._tag.setPos(float(x[k]), float(y[k]))
        self._marker.setVisible(True)
        self._tag.setVisible(True)

    def hover_text(self) -> str:
        """The hovered point in words: ``Fz · 10 Hz: -112 dB``."""
        if self.hovered is None:
            return ""
        name, x, y = self.hovered
        quantity, unit = self._where
        where = f"{quantity} = {x:.2f} {unit}" if quantity else f"{x:g} {unit}"
        value = f"{y:.4g}" + (f" {self._value_unit}" if self._value_unit else "")
        return f"{name} · {where}: {value}"

    def seconds(self) -> float:
        return float(self.span.currentData())

    @staticmethod
    def spectral(dims: tuple[str, ...]) -> bool:
        return "frequency" in dims

    def _set_slices(self, series: FeatureSeries) -> dict[str, int]:
        """Fill *Show* with the other dimensions' values; the chosen one."""
        sdim = series_dim(series.dims)
        others = tuple(d for d in series.dims if d not in (sdim, "frequency"))
        values = tuple(
            itertools.product(*(range(len(_coords(series, d))) for d in others))
        )
        names = tuple(
            " · ".join(self._entry(series, d, i) for d, i in zip(others, combo,
                                                                 strict=True))
            for combo in values
        )  # fmt: skip
        if (others, names) != self._slices:
            self._slices = (others, names)
            self.slice.blockSignals(True)
            self.slice.clear()
            for name in names:
                self.slice.addItem(name)
            self.slice.blockSignals(False)
        shown = bool(others)
        self.slice.setVisible(shown)
        self.show_label.setVisible(shown)
        if not others:
            return {}
        combo = values[max(0, self.slice.currentIndex())]
        return dict(zip(others, combo, strict=True))

    @staticmethod
    def _entry(series: FeatureSeries, dim: str, i: int) -> str:
        """One entry of *Show*, with its unit when units differ along ``dim``."""
        name = str(_coords(series, dim)[i])
        if dim == series.unit_dim and series.units[i] not in ("", "1"):
            return f"{name} ({pretty_unit(series.units[i])})"
        return name

    def _styles(self, labels: tuple[str, ...], channels: bool) -> list[ChannelStyle]:
        if channels:
            return channel_styles_with(labels, self.theme, self.wheel)
        out = []
        for i in range(len(labels)):
            c = pg.intColor(i, hues=max(1, len(labels)))
            out.append(ChannelStyle((c.red(), c.green(), c.blue())))
        return out

    def clear(self, message: str) -> None:
        self.head.setVisible(False)
        self._lines = []
        self._update_hover()
        self.plot.draw([])
        self.legend.setText(message)

    def refresh(
        self, series: FeatureSeries | None, name: str, t0: float, t1: float
    ) -> None:
        """Draw ``series``; ``t0`` to ``t1`` is the shown span (over time)."""
        if series is None or not len(series.t):
            self.clear(f"{name}: no feature windows yet")
            return
        sdim = series_dim(series.dims)
        labels = tuple(str(c) for c in _coords(series, sdim)) if sdim else (name,)
        self.channels.set_labels(labels)
        chans = self.channels.selected(labels)
        styles = self._styles(labels, sdim == "channel")
        chosen = self._set_slices(series)
        data = pick(series.values, series.dims, chosen) * unit_scale(series, chosen)
        links = self.connectivity(series) and "target" in chosen
        self.head.setVisible(links)
        plot = self.plot.getPlotItem()
        layers = []
        n = series.n_windows or len(series.t)
        count = f"{n} window" + ("" if n == 1 else "s")
        if self.spectral(series.dims):
            freqs = np.asarray(_coords(series, "frequency"), dtype=float)
            with np.errstate(all="ignore"):
                spectrum = np.nanmean(data, axis=0)  # one window live
            finite = spectrum[np.isfinite(spectrum)]
            db = (bool(finite.size) and bool((finite > 0).all())
                  and value_label(series, chosen) != "value (dB)"
                  and _slice_unit(series, chosen) != "1")  # fmt: skip
            shown = 10 * np.log10(spectrum) if db else spectrum
            for c in chans:
                st = styles[c]
                layers.append((freqs, shown[c], _pen(st.color, 1.4, 230, st.dashed)))
            self._where = ("", "Hz")
            self._value_unit = "dB" if db else hover_unit(series, chosen)
            plot.setLabel("left", value_label(series, chosen, spectrum_db=db))
            plot.setLabel("bottom", "frequency", units="Hz")
            what = "newest window" if self.panner.live else "mean over the span"
            self.legend.setText(f"{name}: spectrum ({what}, {count})")
            self.plot.set_amplitude(False)
            self.plot.draw(layers)
            self.plot.fit_x()
        else:
            # an amplitude over time (volts): the setting's fit and y = 0
            self.plot.set_amplitude(_slice_unit(series, chosen) == "V")
            for c in chans:
                st = styles[c]
                layers.append((series.t, data[:, c, 0],
                               _pen(st.color, 1.4, 230, st.dashed)))  # fmt: skip
            self._where = ("t", "s")
            self._value_unit = hover_unit(series, chosen)
            plot.setLabel("left", value_label(series, chosen))
            plot.setLabel("bottom", "time", units="s")
            self.legend.setText(f"{name}: over time ({count})")
            self.plot.draw(layers)
            self.plot.set_time(t0, t1)
        if links:
            if not self.panner.live:  # offline: the mean over the shown span
                data = data[(series.t >= t0) & (series.t <= t1)] if len(data) else data
            self._draw_links(series, labels, chosen, data[-1:] if self.panner.live
                             else data, n)  # fmt: skip
        drawn = zip(chans, layers, strict=True)
        self._lines = [(labels[c], x, y, pen) for c, (x, y, pen) in drawn]
        self._update_hover()  # the pointer may now be near another line or point
        self.panner.show_span(t0, t1)
