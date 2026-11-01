"""Colour-token themes (docs/DESIGN.md §3.7).

A theme is a small JSON file of colour tokens. The same tokens drive the Qt
stylesheet and the plots, so a new theme needs no code. ``dark`` (default) and
``light`` ship with the GUI; ``load_theme`` also accepts a path to a JSON file.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

BUILTIN = ("dark", "light")
# strip groups (input, preprocessing, feature extraction, output) for themes
# that do not set their own
DEFAULT_GROUPS = {
    "input": "#8b92a3",
    "preprocess": "#5b8def",
    "features": "#e8a23b",
    "output": "#2fb89a",
}


@dataclass(frozen=True, slots=True)
class Wheel:
    """Channel colour wheel: OKLCH lightness at the rim and the vertex (Cz), and
    the chroma at the rim."""

    rim_lightness: float
    vertex_lightness: float
    chroma: float


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    dark: bool
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    plot_background: str
    grid: str
    status: dict[str, str]
    incompatible: str
    incompatible_border: str
    bypass_hatch: str
    unknown_channel: str
    wheel: Wheel
    groups: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GROUPS))

    def status_color(self, status: str) -> str:
        return self.status.get(status, self.status["stopped"])

    def group_color(self, group: str) -> str:
        return self.groups.get(group, DEFAULT_GROUPS[group])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Theme:
        fields = dict(data)
        fields["wheel"] = Wheel(**fields["wheel"])
        return cls(**fields)


def load_theme(name: str | Path = "dark") -> Theme:
    """A built-in theme by name, or a theme JSON file by path."""
    if isinstance(name, str) and name in BUILTIN:
        text = resources.files("meegxer.gui").joinpath(f"themes/{name}.json")
        data = json.loads(text.read_text(encoding="utf-8"))
    else:
        data = json.loads(Path(name).read_text(encoding="utf-8"))
    return Theme.from_dict(data)


def _arrow(colour: str, up: bool) -> str:
    """A small arrow image for spin boxes in ``colour`` (Qt draws no arrow on a
    styled spin box without one), written once per colour; its path with
    forward slashes, as a stylesheet's url() takes on every platform."""
    folder = Path(tempfile.gettempdir()) / "meegxer-icons"
    path = folder / f"arrow-{'up' if up else 'down'}-{colour.lstrip('#')}.svg"
    if not path.is_file():
        points = "1,7 5,3 9,7" if up else "1,3 5,7 9,3"
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            'stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/></svg>',
            encoding="utf-8",
        )
    return path.as_posix()


def stylesheet(t: Theme, scale: float = 1.0) -> str:
    """Qt stylesheet for the whole window; ``scale`` sizes the text (Settings →
    Appearance)."""
    base, heading = f"{10 * scale:g}pt", f"{13 * scale:g}pt"
    return f"""
QWidget {{ background: {t.background}; color: {t.text}; font-size: {base}; }}
QFrame#panel, QWidget#panel {{ background: {t.surface}; border: 1px solid {t.border};
    border-radius: 6px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#muted {{ color: {t.text_muted}; }}
QLabel:disabled {{ color: {t.text_muted}; }}
QLabel#title {{ font-weight: 600; }}
QLabel#meter {{ color: {t.text_muted}; padding: 0 6px; }}
QFrame#card {{ background: {t.surface}; border: 1px solid {t.border};
    border-radius: 6px; }}
QWidget#cardbody {{ background: transparent; }}
QLabel#section {{ font-weight: 600; }}
QLabel#heading {{ font-size: {heading}; font-weight: 700; }}
QPushButton, QToolButton {{ background: {t.surface_alt}; border: 1px solid {t.border};
    border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {t.accent}; }}
QPushButton:checked, QToolButton:checked {{ background: {t.accent}; color: {t.surface};
    border-color: {t.accent}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.text_muted}; }}
QToolButton#qt_toolbar_ext_button {{ padding: 2px; }}
QPushButton:checked:disabled {{ color: {t.surface}; }}
QPushButton#primary {{ background: {t.accent}; color: {t.surface};
    border-color: {t.accent}; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {t.surface_alt}; color: {t.text_muted};
    border-color: {t.border}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {t.surface_alt};
    border: 1px solid {t.border}; border-radius: 4px; padding: 2px 6px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 18px; border-left: 1px solid {t.border};
    background: {t.surface_alt}; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right;
    border-top-right-radius: 4px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right;
    border-bottom-right-radius: 4px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {t.border}; }}
QAbstractSpinBox::up-arrow {{ image: url({_arrow(t.text, True)}); width: 9px;
    height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({_arrow(t.text, False)}); width: 9px;
    height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({_arrow(t.text_muted, True)}); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({_arrow(t.text_muted, False)}); }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border};
    selection-background-color: {t.accent}; }}
QMenu {{ background: {t.surface}; border: 1px solid {t.border}; }}
QMenu::item:selected {{ background: {t.accent}; color: {t.surface}; }}
QMenu::item:disabled {{ color: {t.text_muted}; }}
QPlainTextEdit {{ background: {t.surface}; border: 1px solid {t.border};
    border-radius: 6px; color: {t.text_muted}; font-family: monospace; }}
QScrollArea {{ border: none; }}
QCheckBox {{ background: transparent; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t.text_muted};
    border-radius: 3px; background: {t.surface_alt}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; }}
QSplitter::handle {{ background: {t.background}; }}
QScrollBar:horizontal {{ background: {t.surface_alt}; height: 12px; border: none;
    border-radius: 6px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {t.text_muted}; border-radius: 6px;
    min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {t.accent}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent; }}
"""
