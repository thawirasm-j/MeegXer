"""Colour-token themes (docs/DESIGN.md §3.7).

A theme is a small JSON file of colour tokens. The same tokens drive the Qt
stylesheet and the plots, so a new theme needs no code. ``dark`` (default) and
``light`` ship with the GUI; ``load_theme`` also accepts a path to a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

BUILTIN = ("dark", "light")


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

    def status_color(self, status: str) -> str:
        return self.status.get(status, self.status["stopped"])

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


def stylesheet(t: Theme) -> str:
    """Qt stylesheet for the whole window."""
    return f"""
QWidget {{ background: {t.background}; color: {t.text}; font-size: 10pt; }}
QFrame#panel, QWidget#panel {{ background: {t.surface}; border: 1px solid {t.border};
    border-radius: 6px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#muted {{ color: {t.text_muted}; }}
QLabel#title {{ font-weight: 600; }}
QLabel#meter {{ color: {t.text_muted}; padding: 0 6px; }}
QPushButton, QToolButton {{ background: {t.surface_alt}; border: 1px solid {t.border};
    border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {t.accent}; }}
QPushButton:checked, QToolButton:checked {{ background: {t.accent}; color: {t.surface};
    border-color: {t.accent}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.text_muted}; }}
QPushButton#primary {{ background: {t.accent}; color: {t.surface};
    border-color: {t.accent}; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {t.surface_alt}; color: {t.text_muted};
    border-color: {t.border}; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {t.surface_alt};
    border: 1px solid {t.border}; border-radius: 4px; padding: 2px 6px; }}
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
"""
