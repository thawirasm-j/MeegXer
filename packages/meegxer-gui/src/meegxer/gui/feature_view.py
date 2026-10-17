"""Feature view: what a feature strip computes, shown when its rail is tapped.

- **Spectrum** (the windows have a ``frequency`` dimension, e.g. PSD): one
  curve per channel over frequency; live the newest window, offline the mean of
  the windows in the shown span. Positive values are drawn in dB.
- **Over time** (anything else, e.g. band power, entropy): one line per channel
  (or per feature), a point at the end of each window. Live, time is seconds
  before now or since the start (as in the signal view); offline, recording time.

Further dimensions (e.g. the band of band power) are chosen in *Show*. The
windows come from the core, decimated and lossy (rule 2): live from the display
stream, offline from the run's feature store.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from meegxer.gui.colors import ChannelStyle, channel_styles_with
from meegxer.gui.signal_view import ChannelMenu, Panner, TracePlot, _pen
from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]
SPANS = (10, 30, 60, 120, 300)  # seconds


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
        self.channels = ChannelMenu()
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
        self.plot = TracePlot(theme)
        lay.addWidget(self.plot, 1)
        self.panner = Panner()
        lay.addWidget(self.panner)
        self.wheel: dict[str, tuple[float, float]] = {}
        self._slices: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.plot.apply_theme(theme)

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
            " · ".join(str(_coords(series, d)[i]) for d, i in zip(others, combo,
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

    def _styles(self, labels: tuple[str, ...], channels: bool) -> list[ChannelStyle]:
        if channels:
            return channel_styles_with(labels, self.theme, self.wheel)
        out = []
        for i in range(len(labels)):
            c = pg.intColor(i, hues=max(1, len(labels)))
            out.append(ChannelStyle((c.red(), c.green(), c.blue())))
        return out

    def clear(self, message: str) -> None:
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
        data = pick(series.values, series.dims, self._set_slices(series))
        plot = self.plot.getPlotItem()
        layers = []
        n = series.n_windows or len(series.t)
        count = f"{n} window" + ("" if n == 1 else "s")
        if self.spectral(series.dims):
            freqs = np.asarray(_coords(series, "frequency"), dtype=float)
            with np.errstate(all="ignore"):
                spectrum = np.nanmean(data, axis=0)  # one window live
            finite = spectrum[np.isfinite(spectrum)]
            db = bool(finite.size) and bool((finite > 0).all())
            shown = 10 * np.log10(spectrum) if db else spectrum
            for c in chans:
                st = styles[c]
                layers.append((freqs, shown[c], _pen(st.color, 1.4, 230, st.dashed)))
            plot.setLabel("left", "dB" if db else "value")
            plot.setLabel("bottom", "frequency", units="Hz")
            what = "newest window" if self.panner.live else "mean over the span"
            self.legend.setText(f"{name}: spectrum ({what}, {count})")
            self.plot.draw(layers)
            self.plot.fit_x()
        else:
            for c in chans:
                st = styles[c]
                layers.append((series.t, data[:, c, 0],
                               _pen(st.color, 1.4, 230, st.dashed)))  # fmt: skip
            plot.setLabel("left", "value")
            plot.setLabel("bottom", "time", units="s")
            self.legend.setText(f"{name}: over time ({count})")
            self.plot.draw(layers)
            self.plot.set_time(t0, t1)
        self.panner.show_span(t0, t1)
