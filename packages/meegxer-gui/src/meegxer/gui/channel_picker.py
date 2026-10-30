"""Pick the channels to show on a head plot (the signal view's *Channels*):
each electrode at its place on the scalp, filled when shown and hollow when
hidden, a click toggling it; channels without a known place are listed beside.

The channels shown first (``visualization.channels``): the standard 19 of the
10-20 system when a recording has more than 19 EEG channels, else every EEG
channel (``standard_19``); or every EEG channel (``all_eeg``). Other types
(EOG, ECG, …) start hidden either way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui._positions import ALIASES
from meegxer.gui.headplot import electrode_xy
from meegxer.gui.theme import Theme

# the 19 electrodes of the 10-20 system (T7/T8/P7/P8 also as T3/T4/T5/T6)
STANDARD_19 = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
               "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2")  # fmt: skip
_STANDARD = {s.upper() for s in STANDARD_19}
RULES = ("standard_19", "all_eeg")


def is_standard(label: str) -> bool:
    key = label.strip().upper()
    return ALIASES.get(key, key).upper() in _STANDARD


def default_hidden(
    labels: Sequence[str], types: Mapping[str, str], rule: str = "standard_19"
) -> frozenset[str]:
    """The channels hidden at first: every non-EEG one, and with
    ``standard_19`` the EEG channels outside the standard 19 when there are
    more than 19 EEG channels (unless none of them is one of the 19)."""
    eeg = [lb for lb in labels if types.get(lb, "eeg") in ("eeg", "")]
    hidden = set(labels) - set(eeg)
    if rule == "standard_19" and len(eeg) > len(STANDARD_19):
        outside = {lb for lb in eeg if not is_standard(lb)}
        if len(outside) < len(eeg):  # some of the 19 are there
            hidden |= outside
    return frozenset(hidden)


class HeadPicker(QWidget):
    """The head with its electrodes; a click toggles one (``toggled``)."""

    toggled = Signal(str)

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.theme = theme
        self.points: dict[str, tuple[float, float]] = {}
        self.colors: dict[str, QColor] = {}
        self.hidden: set[str] = set()
        self.setMouseTracking(True)
        self.setMinimumSize(260, 260)

    def _square(self) -> QRectF:
        side = min(self.width(), self.height())
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def _to_px(self, x: float, y: float) -> QPointF:
        sq = self._square()
        half = sq.width() / 2 / 1.25  # room for the nose and the ears
        return QPointF(sq.center().x() + x * half, sq.center().y() - y * half)

    def electrode_at(self, point: QPointF) -> str | None:
        best, best_d = None, 11.0**2
        for label, (x, y) in self.points.items():
            p = self._to_px(x, y)
            d = (p.x() - point.x()) ** 2 + (p.y() - point.y()) ** 2
            if d < best_d:
                best, best_d = label, d
        return best

    def mousePressEvent(self, event: QMouseEvent) -> None:
        label = self.electrode_at(event.position())
        if label is not None and event.button() == Qt.MouseButton.LeftButton:
            self.toggled.emit(label)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        label = self.electrode_at(event.position())
        if label is None:
            QToolTip.hideText()
            return
        state = "hidden" if label in self.hidden else "shown"
        QToolTip.showText(event.globalPosition().toPoint(),
                          f"{label} ({state}; click to toggle)", self)  # fmt: skip

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(self.theme.plot_background))
        muted = QColor(self.theme.text_muted)
        p.setPen(QPen(muted, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        centre = self._to_px(0, 0)
        r = self._to_px(1, 0).x() - centre.x()
        p.drawEllipse(centre, r, r)
        p.drawPolyline([self._to_px(-0.12, 0.99), self._to_px(0, 1.13),
                        self._to_px(0.12, 0.99)])  # fmt: skip
        dot = max(4.0, self._square().width() / 45)
        font = p.font()
        font.setPointSizeF(max(6.0, font.pointSizeF() * 0.8))
        p.setFont(font)
        for label, (x, y) in self.points.items():
            at = self._to_px(x, y)
            colour = self.colors.get(label, QColor(self.theme.text))
            if label in self.hidden:  # hollow: hidden
                p.setPen(QPen(muted, 1.4))
                p.setBrush(Qt.BrushStyle.NoBrush)
            else:
                p.setPen(QPen(QColor(self.theme.text), 1.0))
                p.setBrush(colour)
            p.drawEllipse(at, dot, dot)
            p.setPen(muted if label in self.hidden else QColor(self.theme.text))
            p.drawText(QRectF(at.x() - 40, at.y() + dot, 80, 14),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                       label)  # fmt: skip
        p.end()


class ChannelPicker(QWidget):
    """The head picker, the channels without a place as check boxes, and
    quick choices; ``changed`` carries the hidden channels."""

    changed = Signal(object)  # set[str]

    def __init__(self, theme: Theme) -> None:
        super().__init__()
        self.labels: tuple[str, ...] = ()
        self.defaults: frozenset[str] = frozenset()
        self.hidden: set[str] = set()
        self.head = HeadPicker(theme)
        self.head.toggled.connect(self.toggle)
        lay = QHBoxLayout(self)
        lay.addWidget(self.head, 1)
        side = QVBoxLayout()
        buttons = QHBoxLayout()
        for text, slot in (("All", self.show_all), ("None", self.show_none),
                           ("Standard 19", self.show_standard),
                           ("Default", self.show_default)):  # fmt: skip
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, f=slot: f())
            buttons.addWidget(b)
        side.addLayout(buttons)
        self.unplaced = QVBoxLayout()
        side.addLayout(self.unplaced)
        side.addStretch(1)
        lay.addLayout(side)
        self.boxes: dict[str, QCheckBox] = {}

    def set_channels(
        self,
        labels: tuple[str, ...],
        hidden: set[str],
        defaults: frozenset[str],
        *,
        wheel: Mapping[str, tuple[float, float]] | None = None,
        colors: Mapping[str, QColor] | None = None,
        notes: Mapping[str, str] | None = None,
    ) -> None:
        self.labels, self.defaults = labels, defaults
        self.head.points = electrode_xy(labels, wheel)
        self.head.colors = dict(colors or {})
        while self.unplaced.count():
            item = self.unplaced.takeAt(0)
            if item is not None and (w := item.widget()) is not None:
                w.deleteLater()
        self.boxes = {}
        for label in labels:
            if label in self.head.points:
                continue
            box = QCheckBox((notes or {}).get(label, label))
            box.toggled.connect(lambda on, lb=label: self._box(lb, on))
            self.unplaced.addWidget(box)
            self.boxes[label] = box
        self.head.setVisible(bool(self.head.points))
        self._apply(set(hidden), emit=False)

    def _apply(self, hidden: set[str], emit: bool = True) -> None:
        self.hidden = hidden & set(self.labels)
        self.head.hidden = self.hidden
        for label, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(label not in self.hidden)
            box.blockSignals(False)
        self.head.update()
        if emit:
            self.changed.emit(set(self.hidden))

    def _box(self, label: str, on: bool) -> None:
        hidden = set(self.hidden)
        (hidden.discard if on else hidden.add)(label)
        self._apply(hidden)

    def toggle(self, label: str) -> None:
        hidden = set(self.hidden)
        (hidden.discard if label in hidden else hidden.add)(label)
        self._apply(hidden)

    def show_all(self) -> None:
        self._apply(set())

    def show_none(self) -> None:
        self._apply(set(self.labels))

    def show_standard(self) -> None:
        self._apply({lb for lb in self.labels if not is_standard(lb)})

    def show_default(self) -> None:
        self._apply(set(self.defaults))
