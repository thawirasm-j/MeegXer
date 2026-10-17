"""Colour maps for visualizations, chosen in Settings → Visualization.

Sequential maps (magnitudes, frequencies) come from pyqtgraph's bundled set
(viridis, cividis, inferno, magma, plasma and turbo, as in matplotlib) plus the
MeegXer violet-to-yellow ramp. Each map looks the same in every theme, which keeps
figures comparable; ``flip_on_light`` reverses the sequential and frequency maps on
a light theme, so that more is darker there. Diverging maps (signed changes) run
blue (negative) to red (positive) through white, black (CET-D1A and CET-D4, Kovesi
2015, doi:10.48550/arXiv.1509.03700) or the theme's background (MeegXer); they are
never flipped, since that would swap their meaning.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg

from meegxer.gui.theme import Theme

FloatArray = npt.NDArray[np.float64]
Bytes = npt.NDArray[np.uint8]
RGB = tuple[int, int, int]

SEQUENTIAL = {
    "viridis": "Viridis",
    "cividis": "Cividis (colour-vision friendly)",
    "inferno": "Inferno",
    "magma": "Magma",
    "plasma": "Plasma",
    "turbo": "Turbo (rainbow)",
    "meegxer": "MeegXer (violet to yellow)",
}
DIVERGING = {
    "blue-white-red": "Blue – white – red",  # noqa: RUF001
    "blue-black-red": "Blue – black – red",  # noqa: RUF001
    "meegxer": "Blue – background – red",  # noqa: RUF001
}
_PG = {"blue-white-red": "CET-D1A", "blue-black-red": "CET-D4"}


@dataclass(frozen=True)
class Maps:
    sequential: str = "viridis"
    diverging: str = "blue-white-red"
    peak: str = "plasma"
    flip_on_light: bool = False

    @classmethod
    def from_settings(cls, visualization: Mapping[str, Any]) -> Maps:
        return cls(
            str(visualization.get("sequential", cls.sequential)),
            str(visualization.get("diverging", cls.diverging)),
            str(visualization.get("peak", cls.peak)),
            bool(visualization.get("flip_on_light", cls.flip_on_light)),
        )

    def sequential_rgb(self, s: npt.ArrayLike, theme: Theme) -> Bytes:
        return _sequential(self.sequential, s, self._flip(theme))

    def peak_rgb(self, s: npt.ArrayLike, theme: Theme) -> Bytes:
        return _sequential(self.peak, s, self._flip(theme))

    def diverging_rgb(self, s: npt.ArrayLike, theme: Theme) -> Bytes:
        """``s`` from -1 (blue) through 0 to +1 (red)."""
        x = np.clip(np.asarray(s, dtype=float), -1.0, 1.0)
        if self.diverging in _PG:
            return _pg_map(_PG[self.diverging], (x + 1) / 2)
        bg = background(theme)
        light = 0.72 if theme.dark else 0.55
        neg, pos = lch(light, 0.15, 255), lch(light, 0.17, 28)
        end = np.where(x[..., None] < 0, neg, pos)
        return oklab_to_srgb(bg + np.abs(x)[..., None] * (end - bg))

    def _flip(self, theme: Theme) -> bool:
        return self.flip_on_light and not theme.dark


def _sequential(name: str, s: npt.ArrayLike, flip: bool) -> Bytes:
    x = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    if flip:
        x = 1.0 - x
    if name in SEQUENTIAL and name != "meegxer":
        return _pg_map(name, x)
    lo, hi = lch(0.38, 0.11, 300), lch(0.93, 0.16, 95)  # MeegXer (and fallback)
    return oklab_to_srgb(lo + x[..., None] * (hi - lo))


@cache
def _colormap(name: str) -> Any:
    return pg.colormap.get(name)


def _pg_map(name: str, s: FloatArray) -> Bytes:
    flat = _colormap(name).map(s.reshape(-1), mode="byte")[:, :3]
    return np.asarray(flat, dtype=np.uint8).reshape(*s.shape, 3)


# -- OKLab --------------------------------------------------------------------------


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


def oklab_to_srgb(lab: npt.ArrayLike) -> Bytes:
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


def lch(lightness: float, chroma: float, hue_deg: float) -> FloatArray:
    h = math.radians(hue_deg)
    return np.array([lightness, chroma * math.cos(h), chroma * math.sin(h)])


def hex_rgb(color: str) -> RGB:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def background(theme: Theme) -> FloatArray:
    return srgb_to_oklab(hex_rgb(theme.plot_background))
