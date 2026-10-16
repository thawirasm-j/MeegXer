"""Head plot for the spectrum view: band power across the scalp.

Electrodes sit where the channel colours put them (the azimuthal projection from
Cz of ``colors``). Values are interpolated linearly inside the electrodes' Delaunay
triangulation and taken from the nearest electrode outside it (its Voronoi cell),
so the map never invents values beyond the outermost electrodes' own.

Colour schemes (``SCHEMES``):

- ``spatial``: the hue is the scalp colour wheel at each point (so an electrode
  has its spectrum line's colour); band power sets how far the colour stands out
  from the background, over the top ``DB_RANGE`` dB of the map. The blend is
  equalised in OKLab so equal power stands out equally everywhere: the wheel's
  vertex and rim differ in contrast with the background, which would otherwise
  bias the map by position.
- ``heat``: one sequential ramp over the same dB range (the classic topomap).
- ``change``: Main against the first comparison tap, in dB (diverging; negative =
  this stretch of the chain removed power there).
- ``peak``: the frequency of the spectral peak inside the band; electrodes whose
  maximum sits on a band edge have no peak and are left out.

Power is the integral of the PSD over the band (µV²), absolute or relative to
1-45 Hz. Live, spectra are smoothed between refreshes by an exponential moving
average with time constant ``tau`` (``Smoother``), in linear power.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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
from scipy.spatial import Delaunay, QhullError, cKDTree

from meegxer.gui.colors import RGB, position
from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]

SCHEMES = {
    "spatial": "Spatial",
    "heat": "Heat",
    "change": "Change",
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
DB_RANGE = 20.0  # the map shows the top 20 dB
MIN_CHANGE_DB = 3.0  # the change scale spans at least ±3 dB
GRID = 112  # map resolution (pixels per side)


# -- colour -------------------------------------------------------------------------


def srgb_to_oklab(rgb: npt.ArrayLike) -> FloatArray:
    """sRGB (0-255, ``(..., 3)``) to OKLab."""
    c = np.asarray(rgb, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return np.asarray(np.cbrt(lin @ m1.T) @ m2.T, dtype=float)


def oklab_to_srgb(lab: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """OKLab (``(..., 3)``) to sRGB bytes, clipped to the gamut."""
    lab = np.asarray(lab, dtype=float)
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack(
        [
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        ],
        axis=-1,
    ).clip(0.0, 1.0)
    enc = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.asarray(np.round(enc * 255), dtype=np.uint8)


def wheel_oklab(
    azimuth: npt.ArrayLike, radius: npt.ArrayLike, theme: Theme
) -> FloatArray:
    """The scalp colour wheel (``colors.wheel_color``) for arrays of points."""
    wheel = theme.wheel
    r = np.clip(np.asarray(radius, dtype=float), 0.0, 1.0)
    lightness = wheel.vertex_lightness + (
        wheel.rim_lightness - wheel.vertex_lightness
    ) * (1 - (1 - r) ** 2)
    chroma = wheel.chroma * r
    hue = np.radians((60.0 - np.asarray(azimuth, dtype=float)) % 360.0)
    return np.stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1)


def _lch(lightness: float, chroma: float, hue_deg: float) -> FloatArray:
    h = math.radians(hue_deg)
    return np.array([lightness, chroma * math.cos(h), chroma * math.sin(h)])


def _background(theme: Theme) -> FloatArray:
    return srgb_to_oklab(_hex(theme.plot_background))


def _rgb(values: npt.NDArray[np.uint8]) -> RGB:
    return (int(values[0]), int(values[1]), int(values[2]))


def _hex(color: str) -> RGB:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def spatial_colors(
    wheel: FloatArray, strength: FloatArray, background: FloatArray, floor: float
) -> FloatArray:
    """Blend wheel colours over the background so that the OKLab distance from
    the background is ``strength * floor``: equal power stands out equally, and
    the least contrasting colour reaches full colour at full strength."""
    full = np.linalg.norm(wheel - background, axis=-1)
    t = np.clip(strength * floor / np.maximum(full, 1e-9), 0.0, 1.0)
    return np.asarray(background + t[..., None] * (wheel - background), dtype=float)


def heat_colors(s: FloatArray, theme: Theme) -> FloatArray:
    """A sequential ramp from the background: dim violet to bright yellow on a
    dark theme, pale yellow to deep violet on a light one."""
    bg = _background(theme)
    s = np.clip(s, 0.0, 1.0)[..., None]
    lo, hi = (_lch(0.45, 0.12, 300), _lch(0.93, 0.16, 95))
    if not theme.dark:
        lo, hi = (_lch(0.93, 0.10, 95), _lch(0.32, 0.15, 290))
    ramp = lo + s * (hi - lo)
    fade = np.clip(s * 4, 0.0, 1.0)  # the lowest values melt into the background
    return np.asarray(bg + fade * (ramp - bg), dtype=float)


def diverging_colors(s: FloatArray, theme: Theme) -> FloatArray:
    """-1 (blue, removed) through the background (0) to +1 (red, added)."""
    bg = _background(theme)
    light = 0.72 if theme.dark else 0.55
    neg, pos = _lch(light, 0.15, 255), _lch(light, 0.17, 28)
    s = np.clip(s, -1.0, 1.0)[..., None]
    end = np.where(s < 0, neg, pos)
    return np.asarray(bg + np.abs(s) * (end - bg), dtype=float)


def peak_colors(s: FloatArray, theme: Theme) -> FloatArray:
    """0 (the band's low edge, blue) through green and yellow to 1 (red)."""
    light = 0.76 if theme.dark else 0.58
    hue = 255.0 - 225.0 * np.clip(s, 0.0, 1.0)
    h = np.radians(hue)
    return np.stack(
        [np.full_like(h, light), 0.13 * np.cos(h), 0.13 * np.sin(h)], axis=-1
    )


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
    """Pixel grid of a map and its interpolation weights for a set of points."""

    extent: float  # the grid spans [-extent, extent] on both axes
    radius: float  # the mapped disc (painted with an antialiased clip)
    inside: npt.NDArray[np.bool_]  # (GRID, GRID): pixels computed (disc + margin)
    index: npt.NDArray[np.intp]  # (n_inside, 3): points used per pixel
    weight: FloatArray  # (n_inside, 3)
    points: FloatArray  # (n, 2)

    def interpolate(self, values: FloatArray) -> FloatArray:
        """Values at the pixels inside the disc (``inside`` order)."""
        return np.asarray((values[self.index] * self.weight).sum(axis=1), dtype=float)


def layout(points: FloatArray, n: int = GRID) -> Layout:
    """Linear interpolation over the Delaunay triangulation, nearest point (its
    Voronoi cell) outside it, on a disc reaching just past the outermost point."""
    radius = max(1.0, float(np.max(np.hypot(points[:, 0], points[:, 1])))) + 0.04
    extent = radius + 0.18  # room for the nose and ears
    axis = np.linspace(-extent, extent, n)
    gx, gy = np.meshgrid(axis, axis[::-1])  # row 0 is the front
    margin = 2 * (2 * extent / (n - 1))  # data under the clipped, smoothed edge
    inside = np.hypot(gx, gy) <= radius + margin
    pix = np.column_stack([gx[inside], gy[inside]])
    _, nearest = cKDTree(points).query(pix)
    index = np.repeat(np.asarray(nearest, dtype=np.intp)[:, None], 3, axis=1)
    weight = np.zeros((len(pix), 3))
    weight[:, 0] = 1.0
    if len(points) >= 3:
        try:
            tri = Delaunay(points)
        except QhullError:  # all on a line
            tri = None
        if tri is not None:
            simplex = tri.find_simplex(pix)
            hit = simplex >= 0
            trans = tri.transform[simplex[hit]]
            bary = np.einsum("nij,nj->ni", trans[:, :2], pix[hit] - trans[:, 2])
            weight[hit] = np.column_stack([bary, 1 - bary.sum(axis=1)])
            index[hit] = tri.simplices[simplex[hit]]
    return Layout(extent, radius, inside, index, weight, points)


# -- the map ------------------------------------------------------------------------


@dataclass
class Electrode:
    label: str
    x: float
    y: float
    color: RGB
    shown: bool  # in the map (not hidden, has a value)
    text: str  # the hover line


@dataclass
class HeadMap:
    """What the head plot draws: the map image and the electrodes on it."""

    extent: float
    radius: float
    rgba: npt.NDArray[np.uint8]  # (GRID, GRID, 4)
    electrodes: list[Electrode]
    unplaced: list[str]
    scale: tuple[str, str, list[RGB]]  # low label, high label, ramp colours
    caption: str


@dataclass
class HeadInput:
    """One refresh's spectra and settings."""

    labels: tuple[str, ...]
    f: FloatArray
    psd: FloatArray  # (n_channels, n_freqs), µV²/Hz, Main tap
    band: tuple[float, float]
    scheme: str = "spatial"
    relative: bool = False
    hidden: frozenset[str] = frozenset()
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


def render(
    data: HeadInput, theme: Theme, cache: dict[tuple[object, ...], Layout] | None = None
) -> HeadMap | str:
    """The head map for one refresh, or why there is none."""
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
    if data.scheme == "change":
        if data.compare_psd is None:
            return "Change needs a second tap (click a ◇ in the chain)"
        cmp_index = {lb: i for i, lb in enumerate(data.compare_labels)}
        main_db = _db(_power(data.f, data.psd, (lo, hi), data.relative))
        cmp_db = _db(_power(data.f, data.compare_psd, (lo, hi), data.relative))
        value = np.array(
            [
                main_db[index[lb]] - cmp_db[cmp_index[lb]]
                if lb in cmp_index
                else np.nan
                for lb in placed
            ]
        )
    elif data.scheme == "peak":
        value = peak_frequency(data.f, data.psd, lo, hi)[rows]
    else:
        value = _db(_power(data.f, data.psd, (lo, hi), data.relative))[rows]

    valid = np.isfinite(value) & np.array([lb not in data.hidden for lb in placed])
    if valid.sum() == 0:
        return "No channel to map in this band"
    points = np.array([xy[lb] for lb in placed])
    key = (tuple(points[valid].round(6).ravel()),)
    lay = (cache or {}).get(key)
    if lay is None:
        lay = layout(points[valid])
        if cache is not None:
            if len(cache) > 32:
                cache.clear()
            cache[key] = lay
    v = value[valid]
    filled = np.where(np.isfinite(value), value, 0.0)  # left-out dots are grey
    bg = _background(theme)

    if data.scheme == "change":
        span = max(MIN_CHANGE_DB, float(np.max(np.abs(v))))
        s_pix, s_el = lay.interpolate(v / span), filled / span
        lab_pix, lab_el = diverging_colors(s_pix, theme), diverging_colors(s_el, theme)
        ramp = diverging_colors(np.linspace(-1, 1, 24), theme)
        scale = (f"−{span:.1f} dB", f"+{span:.1f} dB")  # noqa: RUF001
        caption = "Change in band power vs. the comparison tap"
    elif data.scheme == "peak":
        s_pix, s_el = lay.interpolate((v - lo) / (hi - lo)), (filled - lo) / (hi - lo)
        lab_pix, lab_el = peak_colors(s_pix, theme), peak_colors(s_el, theme)
        ramp = peak_colors(np.linspace(0, 1, 24), theme)
        scale = (f"{lo:g} Hz", f"{hi:g} Hz")
        caption = "Peak frequency in the band"
    else:
        top = float(np.max(v))
        s_pix = np.clip((lay.interpolate(v) - top) / DB_RANGE + 1, 0, 1)
        s_el = np.clip((filled - top) / DB_RANGE + 1, 0, 1)
        if data.scheme == "heat":
            lab_pix, lab_el = heat_colors(s_pix, theme), heat_colors(s_el, theme)
            ramp = heat_colors(np.linspace(0, 1, 24), theme)
        else:
            axis = np.linspace(-lay.extent, lay.extent, GRID)
            gx, gy = np.meshgrid(axis, axis[::-1])
            px, py = gx[lay.inside], gy[lay.inside]
            pix_wheel = wheel_oklab(
                np.degrees(np.arctan2(py, px)), np.hypot(px, py), theme
            )
            el_wheel = wheel_oklab(
                np.degrees(np.arctan2(points[:, 1], points[:, 0])),
                np.hypot(points[:, 0], points[:, 1]),
                theme,
            )
            floor = float(np.min(np.linalg.norm(el_wheel[valid] - bg, axis=-1)))
            lab_pix = spatial_colors(pix_wheel, s_pix, bg, floor)
            lab_el = spatial_colors(el_wheel, s_el, bg, floor)
            neutral = srgb_to_oklab(_hex(theme.text))
            ramp = spatial_colors(
                np.repeat(neutral[None], 24, axis=0), np.linspace(0, 1, 24), bg,
                float(np.linalg.norm(neutral - bg)),
            )  # fmt: skip
        scale = (f"−{DB_RANGE:.0f} dB", "max")  # noqa: RUF001
        what = "Relative band power" if data.relative else "Band power"
        caption = f"{what}, top {DB_RANGE:.0f} dB"

    rgba = np.zeros((GRID, GRID, 4), dtype=np.uint8)
    rgba[lay.inside, :3] = oklab_to_srgb(lab_pix)
    rgba[lay.inside, 3] = 255
    el_rgb = oklab_to_srgb(lab_el)
    electrodes = []
    for i, lb in enumerate(placed):
        x, y = xy[lb]
        if data.scheme == "peak":
            text = (
                f"{lb}: {value[i]:.1f} Hz"
                if np.isfinite(value[i])
                else f"{lb}: no peak in the band"
            )
        elif data.scheme == "change":
            text = (
                f"{lb}: {value[i]:+.1f} dB"
                if np.isfinite(value[i])
                else f"{lb}: not in the comparison tap"
            )
        else:
            p = 10 ** (value[i] / 10) if np.isfinite(value[i]) else float("nan")
            text = f"{lb}: {p:.3g} {unit} ({value[i]:.1f} dB)"
        if lb in data.hidden:
            text += " (hidden)"
        color = _rgb(el_rgb[i]) if valid[i] else _hex(theme.unknown_channel)
        electrodes.append(Electrode(lb, x, y, color, bool(valid[i]), text))
    ramp_rgb = [_rgb(rgb) for rgb in oklab_to_srgb(ramp)]
    return HeadMap(
        lay.extent, lay.radius, rgba, electrodes, unplaced,
        (scale[0], scale[1], ramp_rgb), caption,
    )  # fmt: skip


# -- the widget ---------------------------------------------------------------------


class HeadPlot(QWidget):
    """Draws a ``HeadMap``: the map, head outline, electrodes and a colour scale.
    Hovering an electrode names it (``hovered``); clicking shows or hides its
    channel (``clicked``)."""

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
        # electrodes
        dot = max(2.5, sq.width() / 70)
        for e in m.electrodes:
            at = self._to_px(e.x, e.y)
            size = dot * (1.6 if e.label == self._hover else 1.0)
            p.setPen(
                QPen(
                    QColor(
                        self.theme.text
                        if e.label == self._hover
                        else self.theme.text_muted
                    ),
                    1.0,
                )
            )
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
