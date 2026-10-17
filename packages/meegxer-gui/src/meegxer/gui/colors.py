"""Channel colours: a perceptually uniform colour wheel on the scalp.

Electrodes are projected azimuthally from Cz (``_positions``). Hue follows the
angle around an OKLCH wheel (front magenta-red, right orange, back green, left
blue) and chroma follows the distance from Cz, so the vertex is neutral: white in
the dark theme, near-black in the light theme (theme ``wheel`` tokens). Bipolar
channels (``A-B``) mix both electrodes' colours; unknown or non-EEG channels are
neutral grey and drawn dashed. See docs/DESIGN.md §3.7.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from meegxer.gui._positions import ALIASES, POSITIONS
from meegxer.gui.theme import Theme, Wheel

RGB = tuple[int, int, int]
_UPPER = {k.upper(): k for k in POSITIONS}
_NON_EEG = re.compile(
    r"^(EOG|HEOG|VEOG|ECG|EKG|EMG|RESP|GSR|TRIG|STI|STATUS|AUX)", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class ChannelStyle:
    color: RGB
    dashed: bool = False  # unknown position or not EEG


def position(label: str) -> tuple[float, float] | None:
    """``(azimuth_deg, radius)`` of a 10-05 label (case-insensitive), or None."""
    key = label.strip().upper()
    key = ALIASES.get(key, key).upper()
    name = _UPPER.get(key)
    return POSITIONS[name] if name is not None else None


def _oklab_to_srgb(lightness: float, a: float, b: float) -> RGB:
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    lc, mc, sc = l_**3, m_**3, s_**3
    linear = (
        4.0767416621 * lc - 3.3077115913 * mc + 0.2309699292 * sc,
        -1.2684380046 * lc + 2.6097574011 * mc - 0.3413193965 * sc,
        -0.0041960863 * lc - 0.7034186147 * mc + 1.7076147010 * sc,
    )

    def encode(c: float) -> int:
        c = min(max(c, 0.0), 1.0)
        v = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(v * 255)

    r, g, bl = (encode(c) for c in linear)
    return (r, g, bl)


def _oklab(azimuth: float, radius: float, wheel: Wheel) -> tuple[float, float, float]:
    r = min(max(radius, 0.0), 1.0)
    lightness = wheel.vertex_lightness + (
        wheel.rim_lightness - wheel.vertex_lightness
    ) * (1 - (1 - r) ** 2)
    chroma = wheel.chroma * r
    hue = math.radians((60.0 - azimuth) % 360.0)
    return lightness, chroma * math.cos(hue), chroma * math.sin(hue)


def wheel_color(azimuth: float, radius: float, wheel: Wheel) -> RGB:
    return _oklab_to_srgb(*_oklab(azimuth, radius, wheel))


def _hex(color: str) -> RGB:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


def wheel_positions(
    positions: Mapping[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float]]:
    """``(azimuth_deg, radius)`` from 3-D positions (x right, y front, z up):
    fit a sphere to the electrodes, then project azimuthally from its top, so the
    equator (T7-Fpz-T8-Oz) lands on radius 1 whatever the montage's origin."""
    if not positions:
        return {}
    xyz = np.array(list(positions.values()), dtype=float)
    centre = np.zeros(3)
    if len(xyz) >= 4:
        a = np.column_stack([2 * xyz, np.ones(len(xyz))])
        sol, *_ = np.linalg.lstsq(a, (xyz**2).sum(axis=1), rcond=None)
        if np.all(np.isfinite(sol[:3])):
            centre = sol[:3]
    out: dict[str, tuple[float, float]] = {}
    for label, p in zip(positions, xyz, strict=True):
        v = p - centre
        norm = float(np.linalg.norm(v))
        if norm == 0:
            continue
        theta = math.degrees(math.acos(max(-1.0, min(1.0, v[2] / norm))))
        out[label] = (math.degrees(math.atan2(v[1], v[0])) % 360.0, theta / 90.0)
    return out


def channel_style(
    label: str,
    theme: Theme,
    wheel: Mapping[str, tuple[float, float]] | None = None,
) -> ChannelStyle:
    """Style of one channel: from the recording's positions (``wheel``, see
    ``wheel_positions``) when known, else from the standard 10-05 table."""
    if _NON_EEG.match(label):
        return ChannelStyle(_hex(theme.unknown_channel), dashed=True)
    pos = (wheel or {}).get(label) or position(label)
    if pos is not None:
        return ChannelStyle(wheel_color(*pos, theme.wheel))
    parts = label.split("-")
    if len(parts) == 2:
        a, b = position(parts[0]), position(parts[1])
        if a is not None and b is not None:
            la, lb = _oklab(*a, theme.wheel), _oklab(*b, theme.wheel)
            mixed = tuple((x + y) / 2 for x, y in zip(la, lb, strict=True))
            return ChannelStyle(_oklab_to_srgb(*mixed))
    return ChannelStyle(_hex(theme.unknown_channel), dashed=True)


def channel_styles_with(
    labels: tuple[str, ...] | list[str],
    theme: Theme,
    wheel: Mapping[str, tuple[float, float]] | None,
) -> list[ChannelStyle]:
    return [channel_style(label, theme, wheel) for label in labels]


def channel_styles(
    labels: tuple[str, ...] | list[str], theme: Theme
) -> list[ChannelStyle]:
    return [channel_style(label, theme) for label in labels]


# ICLabel classes (the ICs rail): Okabe-Ito colours, readable on both themes
# and for colour-blind viewers; "other" is neutral and dashed.
IC_CLASS_COLORS: dict[str, str] = {
    "brain": "#009E73",
    "muscle": "#E69F00",
    "eye": "#56B4E9",
    "heart": "#D55E00",
    "line_noise": "#CC79A7",
    "channel_noise": "#0072B2",
}


def class_style(name: str, theme: Theme) -> ChannelStyle:
    """Style of a component by its class (e.g. ICLabel's ``eye``)."""
    colour = IC_CLASS_COLORS.get(name)
    if colour is None:
        return ChannelStyle(_hex(theme.unknown_channel), dashed=True)
    return ChannelStyle(_hex(colour))


def class_label(name: str) -> str:
    """``line_noise`` -> ``line noise``."""
    return name.replace("_", " ")
