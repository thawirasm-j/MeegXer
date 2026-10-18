"""Head plot for the spectrum view: band power across the scalp, as a topomap.

Electrodes sit where the channel colours put them (the azimuthal projection from
Cz of ``colors``) and are drawn in their channel colour inside a black-and-white
ring, so they read apart from the map's colours. The map is a smooth (C1,
Clough-Tocher) interpolation over the electrodes, extrapolated to the head's edge
through virtual points on a ring outside it (each the mean of its nearest
electrodes, as MNE's ``extrapolate="head"`` does), and clipped to the electrodes'
own range so it never overshoots. It is drawn as filled contour bands with lines,
in the colour maps chosen in Settings (``colormaps.Maps``).

Colour schemes (``SCHEMES``):

- ``heat``: band power in dB over a range below the top (``RANGES``; 2 dB bands).
  Live, the top follows the head's maximum: it rises at once and decays with the
  smoothing time constant (``PeakHold``), so the map does not flicker.
- ``change``: Main against the first comparison tap, in dB (diverging; negative =
  this stretch of the chain removed power there; the 0 dB line is bold).
- ``peak``: the frequency of the spectral peak inside the band; electrodes whose
  maximum sits on a band edge have no peak and are left out.

Power is the integral of the PSD over the band (µV²), absolute or relative to
1-45 Hz. Live, spectra are smoothed between refreshes by an exponential moving
average with time constant ``tau`` (``Smoother``), in linear power.

With the aperiodic part removed (``HeadInput.aperiodic``), each channel's
spectrum is first divided by its aperiodic (1/f) fit over 1-45 Hz, the robust
fixed-mode fit of specparam (``aperiodic_fit``): heat shows the band's mean power
above the fit (0 dB = on the 1/f line, below it counts as 0), change compares
those values, and peak frequency looks for the peak above the fit.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QToolTip, QWidget
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay, QhullError, cKDTree

from meegxer.gui.colormaps import Maps
from meegxer.gui.colors import RGB, channel_style, position
from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]

SCHEMES = {
    "heat": "Band power",
    "change": "Power change",
    "peak": "Peak frequency",
}
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
SYMBOLS = dict(zip(BANDS, "δθαβγ", strict=True))
TOTAL = (1.0, 45.0)  # the reference band of relative power
RANGES = (10.0, 20.0, 30.0)  # dB below the top that heat shows
DEFAULT_RANGE = 20.0
HEAT_STEP = 2.0  # dB per heat band
MIN_CHANGE_DB = 3.0  # the change scale spans at least ±3 dB
GRID = 200  # map resolution (pixels per side)
RING = 24  # virtual points on the extrapolation ring
FIT_RANGE = (1.0, 45.0)  # where the aperiodic (1/f) fit is made
MIN_PERIODIC_DB = 3.0  # the scale above the 1/f fit spans at least 3 dB


# -- scales -----------------------------------------------------------------------


def nice_step(x: float) -> float:
    """The smallest 1, 2 or 5 x 10^n that is at least ``x``."""
    if x <= 0:
        return 1.0
    base = 10.0 ** math.floor(math.log10(x))
    return float(next(m * base for m in (1, 2, 5, 10) if m * base >= x * (1 - 1e-9)))


# -- spectra ------------------------------------------------------------------------


def band_power(f: FloatArray, psd: FloatArray, lo: float, hi: float) -> FloatArray:
    """Integral of ``psd`` (rows: channels, µV²/Hz) over ``[lo, hi]`` Hz, in µV²."""
    if f.size < 2:
        return np.full(psd.shape[:-1], np.nan)
    keep = (f >= lo) & (f <= hi)
    if keep.sum() >= 2:
        return np.asarray(np.trapezoid(psd[..., keep], f[keep], axis=-1), dtype=float)
    # a band narrower than the resolution: the nearest bin times its width
    i = int(np.argmin(np.abs(f - (lo + hi) / 2)))
    return np.asarray(psd[..., i] * (f[1] - f[0]), dtype=float)


def peak_frequency(f: FloatArray, psd: FloatArray, lo: float, hi: float) -> FloatArray:
    """Frequency of each row's largest value inside the band; NaN when that
    maximum lies on a band edge (no peak inside)."""
    keep = np.flatnonzero((f >= lo) & (f <= hi))
    out = np.full(psd.shape[0], np.nan)
    if keep.size < 3:
        return out
    seg = psd[:, keep]
    best = np.argmax(seg, axis=1)
    inside = (best > 0) & (best < keep.size - 1)
    out[inside] = f[keep][best[inside]]
    return out


def aperiodic_fit(
    f: FloatArray, psd: FloatArray, lo: float = FIT_RANGE[0], hi: float = FIT_RANGE[1]
) -> tuple[FloatArray, FloatArray]:
    """Each row's aperiodic fit log10(P) = offset - exponent * log10(f) over
    ``[lo, hi]`` Hz, as specparam's robust fit (fixed mode) makes it: a
    least-squares line, then a second one through the points that lie below the
    first (where no peak lifts the spectrum). Vectorized over rows; NaN for rows
    with too few usable points."""
    n_rows = psd.shape[0]
    offset, exponent = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    keep = (f >= lo) & (f <= hi) & (f > 0)
    if keep.sum() < 3:
        return offset, exponent
    x = np.log10(f[keep])
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(psd[:, keep])
    ok = np.all(np.isfinite(y), axis=1)
    if not ok.any():
        return offset, exponent
    y = y[ok]
    a = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(a, y.T, rcond=None)[0]  # (2, rows): intercept, slope
    flat = y - (coef[0][:, None] + coef[1][:, None] * x)
    flat[flat < 0] = 0
    mask = flat <= np.percentile(flat, 0.025, axis=1, keepdims=True)
    n = mask.sum(axis=1)
    sx, sy = (mask * x).sum(axis=1), (mask * y).sum(axis=1)
    sxx, sxy = (mask * x * x).sum(axis=1), (mask * x * y).sum(axis=1)
    denom = n * sxx - sx**2
    good = (n >= 2) & (denom > 1e-12)
    slope = np.where(good, (n * sxy - sx * sy) / np.where(good, denom, 1), coef[1])
    inter = np.where(good, (sy - slope * sx) / np.maximum(n, 1), coef[0])
    offset[ok], exponent[ok] = inter, -slope
    return offset, exponent


def flatten(f: FloatArray, psd: FloatArray) -> FloatArray:
    """Power above each row's aperiodic fit, in dB (NaN at 0 Hz)."""
    offset, exponent = aperiodic_fit(f, psd)
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.where(f > 0, np.log10(np.where(f > 0, f, 1.0)), np.nan)
        fit = offset[:, None] - exponent[:, None] * logf
        return np.asarray(10 * (np.log10(psd) - fit), dtype=float)


def band_mean(f: FloatArray, values: FloatArray, lo: float, hi: float) -> FloatArray:
    """Mean of each row over the bins in ``[lo, hi]`` Hz (the nearest bin for a
    band narrower than the resolution)."""
    keep = (f >= lo) & (f <= hi)
    if not keep.any():
        keep = np.arange(f.size) == int(np.argmin(np.abs(f - (lo + hi) / 2)))
    with np.errstate(invalid="ignore"):
        return np.asarray(np.nanmean(values[:, keep], axis=1), dtype=float)


class Smoother:
    """Exponential moving average of spectra between refreshes, per tap.

    ``tau`` is the time constant in seconds (0: off): a refresh ``dt`` after the
    previous one moves the average by ``1 - exp(-dt / tau)`` towards the new
    spectrum. It starts again when the tap's channels or frequencies change."""

    def __init__(self, tau: float = 2.0) -> None:
        self.tau = tau
        self._held: dict[
            str, tuple[tuple[str, ...], FloatArray, FloatArray, float]
        ] = {}

    def reset(self) -> None:
        self._held.clear()

    def update(
        self,
        key: str,
        labels: tuple[str, ...],
        f: FloatArray,
        psd: FloatArray,
        now: float | None = None,
    ) -> FloatArray:
        now = time.monotonic() if now is None else now
        held = self._held.get(key)
        if (
            self.tau <= 0
            or held is None
            or held[0] != labels
            or held[1].shape != f.shape
            or held[2].shape != psd.shape
        ):
            self._held[key] = (labels, f, psd, now)
            return psd
        dt = max(0.0, now - held[3])
        alpha = 1.0 - math.exp(-dt / self.tau)
        prev = np.where(np.isfinite(held[2]), held[2], psd)
        out = np.asarray(prev + alpha * (psd - prev), dtype=float)
        self._held[key] = (labels, f, out, now)
        return out


class PeakHold:
    """The top of a scale: follows a maximum up at once and decays towards it
    with time constant ``tau`` (0: no hold). Starts again when ``key`` changes."""

    def __init__(self, tau: float = 2.0) -> None:
        self.tau = tau
        self._held: tuple[object, float, float] | None = None

    def reset(self) -> None:
        self._held = None

    def update(self, key: object, value: float, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        held = self._held
        if self.tau <= 0 or held is None or held[0] != key or value >= held[1]:
            self._held = (key, value, now)
            return value
        dt = max(0.0, now - held[2])
        top = value + (held[1] - value) * math.exp(-dt / self.tau)
        self._held = (key, top, now)
        return top


# -- geometry -----------------------------------------------------------------------


def electrode_xy(
    labels: Sequence[str], wheel: Mapping[str, tuple[float, float]] | None = None
) -> dict[str, tuple[float, float]]:
    """Head coordinates (x right, y front; the equator at radius 1) of the
    channels with a known position; bipolar ``A-B`` at the midpoint."""

    def polar(label: str) -> tuple[float, float] | None:
        return (wheel or {}).get(label) or position(label)

    out: dict[str, tuple[float, float]] = {}
    for label in labels:
        pos = polar(label)
        if pos is not None:
            az, r = pos
            out[label] = (
                r * math.cos(math.radians(az)),
                r * math.sin(math.radians(az)),
            )
            continue
        parts = label.split("-")
        if len(parts) == 2:
            a, b = polar(parts[0]), polar(parts[1])
            if a is not None and b is not None:
                pa = [
                    p[1] * f(math.radians(p[0]))
                    for p in (a, b)
                    for f in (math.cos, math.sin)
                ]
                out[label] = ((pa[0] + pa[2]) / 2, (pa[1] + pa[3]) / 2)
    return out


@dataclass
class Layout:
    """The map grid for a set of electrodes and how to interpolate onto it."""

    extent: float  # the grid spans [-extent, extent] on both axes
    radius: float  # the mapped disc (painted with an antialiased clip)
    points: FloatArray  # (n, 2) electrodes
    ring_from: npt.NDArray[np.intp]  # (RING, k): electrodes each ring point averages
    tri: Delaunay | None  # electrodes + ring
    pixels: FloatArray  # (n_pix, 2): the grid points interpolated (disc + margin)
    inside: npt.NDArray[np.bool_]  # (GRID, GRID): where those pixels are
    nearest: npt.NDArray[np.intp]  # (n_pix,): nearest electrode (fallback)

    def interpolate(self, values: FloatArray) -> FloatArray:
        """The map (``GRID`` x ``GRID``, NaN outside the disc), clipped to the
        electrodes' range."""
        grid = np.full(self.inside.shape, np.nan)
        lo, hi = float(np.min(values)), float(np.max(values))
        if self.tri is None:
            got = values[self.nearest]
        else:
            ring = values[self.ring_from].mean(axis=1)
            interp = CloughTocher2DInterpolator(
                self.tri, np.concatenate([values, ring])
            )
            got = interp(self.pixels)
            got = np.where(np.isfinite(got), got, values[self.nearest])
        grid[self.inside] = np.clip(got, lo, hi)
        return grid


def layout(points: FloatArray, n: int = GRID) -> Layout:
    """Smooth interpolation over the electrodes and a ring of virtual points just
    outside the disc that reaches past the outermost electrode."""
    radius = max(1.0, float(np.max(np.hypot(points[:, 0], points[:, 1])))) + 0.04
    extent = radius + 0.18  # room for the nose and ears
    axis = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(axis, axis[::-1])  # row 0 is the front
    margin = 2 * (2 * extent / (n - 1))  # data under the clipped, smoothed edge
    inside = np.hypot(gx, gy) <= radius + margin
    pixels = np.column_stack([gx[inside], gy[inside]])
    tree = cKDTree(points)
    _, nearest = tree.query(pixels)
    angle = np.linspace(0, 2 * np.pi, RING, endpoint=False)
    ring = (radius + 0.15) * np.column_stack([np.cos(angle), np.sin(angle)])
    k = min(3, len(points))
    _, ring_from = tree.query(ring, k=k)
    ring_from = np.asarray(ring_from, dtype=np.intp).reshape(RING, k)
    tri = None
    if len(points) >= 3:
        try:
            tri = Delaunay(np.vstack([points, ring]))
        except QhullError:  # degenerate geometry: nearest electrode only
            tri = None
    return Layout(extent, radius, points, ring_from, tri, pixels, inside,
                  np.asarray(nearest, dtype=np.intp))  # fmt: skip


_PAIRS = {  # edges joined per case (corner bits: top-left 8 ... bottom-left 1)
    ("L", "B"): (1, 14, 10), ("B", "R"): (2, 13, 5), ("L", "R"): (3, 12),
    ("T", "R"): (4, 11, 10), ("T", "B"): (6, 9), ("L", "T"): (7, 8, 5),
}  # fmt: skip
_JOINS = [(e0, e1, np.isin(np.arange(16), cases)) for (e0, e1), cases in _PAIRS.items()]


def contour_segments(grid: FloatArray, level: float) -> FloatArray:
    """Line segments of ``grid == level`` by marching squares, as rows of
    ``(row0, col0, row1, col1)`` in grid coordinates; cells touching NaN skipped."""
    above = np.nan_to_num(grid, nan=-np.inf) > level
    case = (above[:-1, :-1].astype(np.int8) * 8 + above[:-1, 1:] * 4
            + above[1:, 1:] * 2 + above[1:, :-1] * 1)  # fmt: skip
    finite = np.isfinite(grid)
    ok = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, 1:] & finite[1:, :-1]
    r, c = np.nonzero(ok & (case > 0) & (case < 15))  # cells the level crosses
    if r.size == 0:
        return np.empty((0, 4))
    a, b = grid[r, c], grid[r, c + 1]  # top-left, top-right
    d_, d = grid[r + 1, c + 1], grid[r + 1, c]  # bottom-right, bottom-left
    cases = case[r, c]
    rf, cf = r.astype(float), c.astype(float)

    def cross(v0: FloatArray, v1: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (level - v0) / (v1 - v0)
        return np.asarray(np.clip(np.nan_to_num(t, nan=0.5), 0, 1), dtype=float)

    edges = {
        "T": (rf, cf + cross(a, b)),
        "R": (rf + cross(b, d_), cf + 1),
        "B": (rf + 1, cf + cross(d, d_)),
        "L": (rf + cross(a, d), cf),
    }
    out = []
    for e0, e1, table in _JOINS:
        hit = table[cases]
        if hit.any():
            (r0, c0), (r1, c1) = edges[e0], edges[e1]
            out.append(np.column_stack([r0[hit], c0[hit], r1[hit], c1[hit]]))
    return np.vstack(out)


# -- the map ------------------------------------------------------------------------


@dataclass
class Electrode:
    label: str
    x: float
    y: float
    color: RGB  # the channel's own colour
    shown: bool  # in the map (not hidden, has a value)
    text: str  # the hover line


@dataclass
class HeadMap:
    """What the head plot draws: the map image, its contour lines and the
    electrodes on it."""

    extent: float
    radius: float
    rgba: npt.NDArray[np.uint8]  # (GRID, GRID, 4)
    lines: list[tuple[FloatArray, bool]]  # head-coordinate segments, bold?
    electrodes: list[Electrode]
    unplaced: list[str]
    scale: tuple[str, str, list[RGB]]  # low label, high label, band colours
    caption: str
    # lines between electrodes (connectivity): from, to, colour, width
    links: list[tuple[tuple[float, float], tuple[float, float], RGB, float]] = field(
        default_factory=list
    )


@dataclass
class HeadInput:
    """One refresh's spectra and settings."""

    labels: tuple[str, ...]
    f: FloatArray
    psd: FloatArray  # (n_channels, n_freqs), µV²/Hz, Main tap
    band: tuple[float, float]
    scheme: str = "heat"
    relative: bool = False
    aperiodic: bool = False  # remove each channel's 1/f fit first
    db_range: float = DEFAULT_RANGE
    hidden: frozenset[str] = frozenset()  # off the map (not EEG)
    compare_psd: FloatArray | None = None  # first comparison tap, same channels
    compare_labels: tuple[str, ...] = ()
    wheel: Mapping[str, tuple[float, float]] = field(default_factory=dict)


def _power(
    f: FloatArray, psd: FloatArray, band: tuple[float, float], relative: bool
) -> FloatArray:
    p = band_power(f, psd, *band)
    if relative:
        total = band_power(f, psd, TOTAL[0], min(TOTAL[1], float(f[-1])))
        p = p / np.where(total > 0, total, np.nan)
    return p


def _db(p: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(10 * np.log10(np.where(p > 0, p, np.nan)), dtype=float)


def _bands(
    grid: FloatArray, edges: FloatArray
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Band index of each pixel for ascending band ``edges`` (n_bands + 1), and
    which pixels fall inside them."""
    filled = np.isfinite(grid) & (grid >= edges[0] - 1e-9) & (grid <= edges[-1] + 1e-9)
    index = np.clip(np.searchsorted(edges, np.nan_to_num(grid), side="right") - 1,
                    0, len(edges) - 2)  # fmt: skip
    return np.asarray(index, dtype=np.intp), filled


def render(
    data: HeadInput,
    theme: Theme,
    cache: dict[tuple[object, ...], Layout] | None = None,
    hold: Callable[[float], float] | None = None,
    maps: Maps | None = None,
) -> HeadMap | str:
    """The head map for one refresh, or why there is none. ``hold`` maps the
    heat map's maximum (dB) to the top of its scale (live: a ``PeakHold``);
    ``maps`` are the colour maps (Settings → Visualization)."""
    maps = maps or Maps()
    xy = electrode_xy(data.labels, data.wheel)
    unplaced = [lb for lb in data.labels if lb not in xy]
    if len(xy) < 3:
        return "No electrode positions: set a montage in Recording settings"
    lo, hi = sorted(data.band)
    nyquist = float(data.f[-1]) if data.f.size else 0.0
    if data.f.size < 2 or lo >= nyquist:
        return f"The band is beyond this view's {nyquist:.0f} Hz limit"
    hi = min(hi, nyquist)
    index = {lb: i for i, lb in enumerate(data.labels)}
    placed = [lb for lb in data.labels if lb in xy]
    rows = np.array([index[lb] for lb in placed])

    unit = "rel." if data.relative else "µV²"

    def band_db(psd: FloatArray) -> FloatArray:
        if data.aperiodic:  # mean dB above the 1/f fit
            return band_mean(data.f, flatten(data.f, psd), lo, hi)
        return _db(_power(data.f, psd, (lo, hi), data.relative))

    if data.scheme == "change":
        if data.compare_psd is None:
            return "Change needs a second tap (click a ◇ in the chain)"
        cmp_index = {lb: i for i, lb in enumerate(data.compare_labels)}
        main_db, cmp_db = band_db(data.psd), band_db(data.compare_psd)
        value = np.array([main_db[index[lb]] - cmp_db[cmp_index[lb]]
                          if lb in cmp_index else np.nan for lb in placed])  # fmt: skip
    elif data.scheme == "peak":
        psd = 10 ** (flatten(data.f, data.psd) / 10) if data.aperiodic else data.psd
        value = peak_frequency(data.f, np.nan_to_num(psd, nan=-np.inf), lo, hi)[rows]
    else:
        value = band_db(data.psd)[rows]

    valid = np.isfinite(value) & np.array([lb not in data.hidden for lb in placed])
    if valid.sum() == 0:
        return "No channel to map in this band"
    points = np.array([xy[lb] for lb in placed])
    key = (tuple(points[valid].round(6).ravel()),)
    lay = (cache or {}).get(key)
    if lay is None:
        lay = layout(points[valid])
        if cache is not None:
            if len(cache) > 8:
                cache.clear()
            cache[key] = lay
    v = value[valid]
    grid = lay.interpolate(v)

    bold: list[float] = []
    if data.scheme == "change":
        span = max(MIN_CHANGE_DB, float(np.max(np.abs(v))))
        step = nice_step(span / 5)
        n = math.ceil(span / step - 1e-9)
        edges = step * np.arange(-n, n + 1, dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2 / (n * step)
        band_rgb = maps.diverging_rgb(centres, theme)
        bold = [0.0]
        scale = (f"−{n * step:g} dB", f"+{n * step:g} dB")  # noqa: RUF001
        what = "Power change above 1/f" if data.aperiodic else "Power change"
        caption = f"{what} vs. the comparison tap, {step:g} dB steps"
    elif data.scheme == "peak":
        step = nice_step((hi - lo) / 10)
        start = math.floor(lo / step) * step
        edges = np.arange(start, hi + step, step, dtype=float)
        centres = ((edges[:-1] + edges[1:]) / 2 - lo) / (hi - lo)
        band_rgb = maps.peak_rgb(centres, theme)
        scale = (f"{edges[0]:g} Hz", f"{edges[-1]:g} Hz")
        caption = f"Peak frequency in the band, {step:g} Hz steps"
    elif data.aperiodic:  # from the 1/f line (0 dB) up to the held top
        top = max(MIN_PERIODIC_DB, float(np.max(v)))
        top = max(top, hold(top)) if hold is not None else top
        step = nice_step(top / 8)
        n = math.ceil(top / step - 1e-9)
        edges = step * np.arange(n + 1, dtype=float)
        grid = np.maximum(grid, 0.0)  # below the fit: no periodic power
        band_rgb = maps.sequential_rgb((np.arange(n) + 0.5) / n, theme)
        scale = ("0 dB", f"+{n * step:g} dB")
        caption = f"Band power above the aperiodic (1/f) fit, {step:g} dB steps"
    else:
        top = float(np.max(v))
        top = max(top, hold(top)) if hold is not None else top
        n = max(1, round(data.db_range / HEAT_STEP))
        edges = top - HEAT_STEP * np.arange(n, -1, -1, dtype=float)
        band_rgb = maps.sequential_rgb((np.arange(n) + 0.5) / n, theme)
        scale = (f"−{n * HEAT_STEP:g} dB", "max")  # noqa: RUF001
        what = "Relative band power" if data.relative else "Band power"
        caption = f"{what}, {HEAT_STEP:g} dB steps"

    band_index, filled = _bands(grid, edges)
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    rgba[filled, :3] = band_rgb[band_index[filled]]
    rgba[filled, 3] = 255

    to_head = 2 * lay.extent / (grid.shape[0] - 1)
    lines = []
    for level in edges[1:-1]:
        seg = contour_segments(grid, float(level))
        if len(seg):
            head = np.column_stack([
                -lay.extent + seg[:, 1] * to_head, lay.extent - seg[:, 0] * to_head,
                -lay.extent + seg[:, 3] * to_head, lay.extent - seg[:, 2] * to_head,
            ])  # fmt: skip
            lines.append((head, any(abs(level - b) < 1e-9 for b in bold)))

    electrodes = []
    for i, lb in enumerate(placed):
        x, y = xy[lb]
        if data.scheme == "peak":
            text = (f"{lb}: {value[i]:.1f} Hz" if np.isfinite(value[i])
                    else f"{lb}: no peak in the band")  # fmt: skip
        elif data.scheme == "change":
            text = (f"{lb}: {value[i]:+.1f} dB" if np.isfinite(value[i])
                    else f"{lb}: not in the comparison tap")  # fmt: skip
        elif data.aperiodic:
            text = f"{lb}: {value[i]:+.1f} dB above the 1/f fit"
        else:
            p = 10 ** (value[i] / 10) if np.isfinite(value[i]) else float("nan")
            text = f"{lb}: {p:.3g} {unit} ({value[i]:.1f} dB)"
        if lb in data.hidden:
            text += " (not EEG: off the map)"
        color = channel_style(lb, theme, data.wheel).color
        electrodes.append(Electrode(lb, x, y, color, bool(valid[i]), text))
    ramp = [(int(r), int(g), int(b)) for r, g, b in band_rgb]
    return HeadMap(lay.extent, lay.radius, rgba, lines, electrodes, unplaced,
                   (scale[0], scale[1], ramp), caption)  # fmt: skip


def render_links(
    labels: Sequence[str],
    values: FloatArray,
    seed: str,
    theme: Theme,
    maps: Maps,
    wheel: Mapping[str, tuple[float, float]] | None = None,
    unit: str = "",
    caption: str = "",
) -> HeadMap | str:
    """A connectivity map: a line from ``seed`` to every other placed channel,
    coloured (and thickened) by its value in ``values`` (one per label). Values
    of both signs (imaginary coherence) use the diverging map around 0, else
    the sequential map from the lowest to the highest."""
    xy = electrode_xy(labels, wheel)
    if seed not in xy:
        return f"{seed} has no position: set a montage in Recording settings"
    placed = [lb for lb in labels if lb in xy]
    v = np.array([values[labels.index(lb)] for lb in placed], dtype=float)
    others = np.array([lb != seed for lb in placed]) & np.isfinite(v)
    if not others.any():
        return "No other channel to connect"
    lo, hi = float(v[others].min()), float(v[others].max())
    if lo < 0 < hi:
        top = max(-lo, hi)
        s = v / top
        rgb = maps.diverging_rgb(s, theme)
        strength = np.abs(s)
        ramp = maps.diverging_rgb(np.linspace(-1, 1, 11), theme)
        scale = (f"{-top:.3g}", f"{top:.3g}")
    else:
        span = hi - lo or 1.0
        strength = (v - lo) / span
        rgb = maps.sequential_rgb(strength, theme)
        ramp = maps.sequential_rgb(np.linspace(0, 1, 10), theme)
        scale = (f"{lo:.3g}", f"{hi:.3g}")
    suffix = f" {unit}" if unit else ""
    scale = (scale[0] + suffix, scale[1] + suffix)
    sx, sy = xy[seed]
    order = np.argsort(np.where(others, strength, -1))  # strongest drawn last
    links = [((sx, sy), xy[placed[i]], (int(rgb[i][0]), int(rgb[i][1]),
              int(rgb[i][2])), 1.0 + 3.0 * float(strength[i]))
             for i in order if others[i]]  # fmt: skip
    electrodes = []
    for lb, value in zip(placed, v, strict=True):
        x, y = xy[lb]
        text = (f"{lb}: seed" if lb == seed else
                f"{seed}–{lb}: {value:.3g}{suffix}" if np.isfinite(value)  # noqa: RUF001
                else f"{seed}–{lb}: no value")  # noqa: RUF001  # fmt: skip
        color = channel_style(lb, theme, wheel or {}).color
        electrodes.append(Electrode(lb, x, y, color, lb == seed, text))
    blank = np.zeros((GRID, GRID, 4), dtype=np.uint8)
    colours: list[RGB] = [(int(r), int(g), int(b)) for r, g, b in ramp]
    unplaced = [lb for lb in labels if lb not in xy]
    return HeadMap(1.18, 1.0, blank, [], electrodes, unplaced,
                   (scale[0], scale[1], colours), caption, links)  # fmt: skip


# -- the widget ---------------------------------------------------------------------


class HeadPlot(QWidget):
    """Draws a ``HeadMap``: the map, contour lines, head outline, electrodes and
    a colour scale. Hovering an electrode names it (``hovered``); clicking shows
    or hides its channel's curve (``clicked``); the map keeps every EEG channel."""

    hovered = Signal(object)  # label or None
    clicked = Signal(str)

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.map: HeadMap | None = None
        self.message = "Nothing shown yet"
        self._hover: str | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(140, 140)

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def show_map(self, result: HeadMap | str) -> None:
        if isinstance(result, str):
            self.map, self.message = None, result
        else:
            self.map = result
        self.update()

    # geometry: the map square sits on top, the scale and notes below it
    def _square(self) -> QRectF:
        side = max(10.0, min(self.width(), self.height() - 44.0))
        return QRectF((self.width() - side) / 2, 0, side, side)

    def _to_px(self, x: float, y: float) -> QPointF:
        assert self.map is not None
        sq = self._square()
        half = sq.width() / 2
        return QPointF(sq.center().x() + x / self.map.extent * half,
                       sq.center().y() - y / self.map.extent * half)  # fmt: skip

    def electrode_at(self, point: QPointF) -> str | None:
        if self.map is None:
            return None
        best, best_d = None, 9.0**2
        for e in self.map.electrodes:
            p = self._to_px(e.x, e.y)
            d = (p.x() - point.x()) ** 2 + (p.y() - point.y()) ** 2
            if d < best_d:
                best, best_d = e.label, d
        return best

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        label = self.electrode_at(event.position())
        if label != self._hover:
            self._hover = label
            self.hovered.emit(label)
            self.update()
        if label is not None and self.map is not None:
            text = next(e.text for e in self.map.electrodes if e.label == label)
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, event: object) -> None:
        if self._hover is not None:
            self._hover = None
            self.hovered.emit(None)
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        label = self.electrode_at(event.position())
        if label is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(label)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted = QColor(self.theme.text_muted)
        p.fillRect(self.rect(), QColor(self.theme.plot_background))
        if self.map is None:
            p.setPen(muted)
            p.drawText(self.rect().adjusted(8, 8, -8, -8),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       self.message)  # fmt: skip
            return
        m = self.map
        sq = self._square()
        image = QImage(m.rgba.tobytes(), m.rgba.shape[1], m.rgba.shape[0],
                       4 * m.rgba.shape[1], QImage.Format.Format_RGBA8888)  # fmt: skip
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        disc = QPainterPath()
        centre = self._to_px(0, 0)
        reach = self._to_px(m.radius, 0).x() - centre.x()
        disc.addEllipse(centre, reach, reach)
        p.save()
        p.setClipPath(disc)
        p.drawImage(sq, image)
        # contour lines: dark on a light theme, light on a dark one
        ink = QColor(0, 0, 0, 110) if not self.theme.dark else QColor(255, 255, 255, 90)
        for segments, bold in m.lines:
            p.setPen(QPen(ink, 1.8 if bold else 0.8))
            p.drawLines([QLineF(self._to_px(s[0], s[1]), self._to_px(s[2], s[3]))
                         for s in segments])  # fmt: skip
        p.restore()
        # head outline (radius 1), nose and ears
        outline = QPen(QColor(self.theme.text_muted), 1.4)
        p.setPen(outline)
        p.setBrush(Qt.BrushStyle.NoBrush)
        r = self._to_px(1, 0).x() - centre.x()
        p.drawEllipse(centre, r, r)
        nose = QPainterPath(self._to_px(-0.12, 0.99))
        nose.lineTo(self._to_px(0, 1.13))
        nose.lineTo(self._to_px(0.12, 0.99))
        p.drawPath(nose)
        for side in (-1, 1):
            ear = QPainterPath(self._to_px(side * 0.99, 0.12))
            ear.cubicTo(self._to_px(side * 1.10, 0.14), self._to_px(side * 1.12, -0.14),
                        self._to_px(side * 0.99, -0.12))  # fmt: skip
            p.drawPath(ear)
        for (x0, y0), (x1, y1), rgb, width in m.links:  # connectivity
            pen = QPen(QColor(*rgb), width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(self._to_px(x0, y0), self._to_px(x1, y1))
        # electrodes: the channel colour inside a black and a white ring, so it
        # reads apart from the map whatever the colours around it
        dot = max(3.0, sq.width() / 60)
        for e in m.electrodes:
            at = self._to_px(e.x, e.y)
            size = dot * (1.5 if e.label == self._hover else 1.0)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0), 3.2))
            p.drawEllipse(at, size, size)
            p.setPen(QPen(QColor(255, 255, 255), 1.4))
            p.setBrush(QColor(*e.color) if e.shown else Qt.BrushStyle.NoBrush)
            p.drawEllipse(at, size, size)
        # colour scale and notes
        low, high, ramp = m.scale
        font = p.font()
        font.setPointSizeF(max(6.5, font.pointSizeF() * 0.8))
        p.setFont(font)
        metrics = p.fontMetrics()
        left = metrics.horizontalAdvance(low) + 6
        right = metrics.horizontalAdvance(high) + 6
        width = max(10.0, sq.width() - left - right)
        bar = QRectF(sq.left() + left, sq.bottom() + 6, width, 8)
        step = bar.width() / len(ramp)
        p.setPen(Qt.PenStyle.NoPen)
        for i, rgb in enumerate(ramp):
            p.setBrush(QColor(*rgb))
            p.drawRect(
                QRectF(bar.left() + i * step, bar.top(), step + 0.5, bar.height())
            )
        p.setPen(muted)
        middle = Qt.AlignmentFlag.AlignVCenter
        p.drawText(QRectF(sq.left(), bar.top() - 4, left - 4, 16),
                   Qt.AlignmentFlag.AlignRight | middle, low)  # fmt: skip
        p.drawText(QRectF(bar.right() + 4, bar.top() - 4, right, 16),
                   Qt.AlignmentFlag.AlignLeft | middle, high)  # fmt: skip
        note = m.caption
        if m.unplaced:
            shown = ", ".join(m.unplaced[:4]) + ("…" if len(m.unplaced) > 4 else "")
            note += f" · no position: {shown}"
        top = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        p.drawText(QRectF(0, bar.bottom() + 2, self.width(), 18), top, note)
