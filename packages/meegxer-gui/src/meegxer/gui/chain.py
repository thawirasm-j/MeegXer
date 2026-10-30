"""The chain of strips: Input strip, processor and sink strips, taps and gaps.

One chain on one thick line, the signal (docs/RAILS.md). Every strip sits on it in
order; a feature strip passes the signal on unchanged. Above the strips, thinner
parallel rails show what else travels with the signal: each feature strip starts
a rail named after it, the input's events start the Events rail, and the strips
that read a rail (an output, a classifier, CSP) draw from it. A rail ends at its
last reader, and rails share rows where they do not overlap, so the band stays
low. Each strip's top band, shape and label give its group: ■ input,
● preprocessing, ◆ feature extraction, ▶ output.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QFont,
    QFontMetrics,
    QHelpEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.labels import param_label
from meegxer.gui.names import badge, title
from meegxer.gui.pipeline_doc import PipelineDoc, Rail, StripNode
from meegxer.gui.theme import Theme

MIME = "application/x-meegxer-strip"
STRIP_WIDTH = 158
TAP_STATES = ("off", "comparison", "main")
RAIL_Y = 26  # the signal line's height in the chain, px below the strips' top
SIGNAL_WIDTH = 6  # the signal is the thick main line
GAP_WIDTH = 34  # between two strips
LANE_H = 15  # rows of the parallel rails above the strips
TOP = 6  # margin above the rails
PORT = 22  # a strip reads a rail near its left edge, writes near its right
LOWER = 52  # feature and output strips sit this far below the others
TAP_SIZE = 22
GROUP_LABELS = {
    "input": "■ INPUT",
    "preprocess": "● PREPROCESSING",
    "features": "◆ FEATURES",
    "output": "▶ OUTPUT",
}


def _led(color: str, size: int = 10) -> QLabel:
    led = QLabel()
    led.setFixedSize(size, size)
    led.setStyleSheet(f"background: {color}; border-radius: {size // 2}px;")
    return led


class StripWidget(QFrame):
    """One node: title, status LED, the parameters chosen for the strip."""

    clicked = Signal(str)
    remove_requested = Signal(str)
    calibrate_requested = Signal(str)

    def __init__(
        self,
        doc: PipelineDoc,
        strip: StripNode,
        theme: Theme,
        *,
        status: str,
        advanced: bool,
        selected: bool,
    ) -> None:
        super().__init__()
        self.strip_id = strip.id
        self._press: QPoint | None = None
        self.setObjectName("strip")
        self.setFixedWidth(STRIP_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        problem = doc.incompatibility(strip)
        bg = theme.incompatible if problem else theme.surface
        border = (
            theme.incompatible_border
            if problem
            else theme.accent
            if selected
            else theme.border
        )
        group = doc.group(strip)
        band = theme.group_color(group)
        self.setStyleSheet(
            f"QFrame#strip {{ background: {bg}; border: {2 if selected else 1}px "
            f"solid {border}; border-top: 4px solid {band}; border-radius: 8px; }}"
        )
        manifest = doc.manifest(strip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(3)
        text = GROUP_LABELS[group]
        if doc.is_custom(strip):  # "● CUSTOM PREPROCESSING"
            text = f"{text[:2]}CUSTOM {text[2:]}"
        label = QLabel(text)
        label.setObjectName("group")
        label.setStyleSheet(
            f"color: {band}; font-size: 7pt; font-weight: 700; letter-spacing: 1px;"
        )
        lay.addWidget(label)

        head = QHBoxLayout()
        head.addWidget(_led(theme.status_color(status)))
        full = (
            "Input" if manifest["role"] == "source" else title(strip.node, strip.params)
        )
        name = QLabel()
        name.setObjectName("title")
        name.ensurePolished()  # measure in the title's (bold) font, not the default
        name.setText(
            name.fontMetrics().elidedText(
                full, Qt.TextElideMode.ElideRight, STRIP_WIDTH - 60
            )
        )
        name.setToolTip(full)
        head.addWidget(name, 1)
        if manifest["role"] != "source":
            close = QToolButton()
            close.setText("✕")
            close.setToolTip("Remove this strip")
            close.setStyleSheet(
                "QToolButton { border: none; padding: 0 2px; background: transparent; }"
            )
            close.clicked.connect(lambda: self.remove_requested.emit(self.strip_id))
            head.addWidget(close)
        lay.addLayout(head)

        if advanced:  # the id is for pipeline files and the command line
            sub = QLabel(strip.id)
            sub.setObjectName("muted")
            lay.addWidget(sub)
        note = problem or badge(strip.node, strip.params, doc.mode)
        if not problem and strip.channel_types and strip.channel_types != ["eeg"]:
            types = "+".join(t.upper() for t in strip.channel_types)
            note = f"{note} · {types}" if note else types
        if note:
            chip = QLabel(note)
            colour = theme.incompatible_border if problem else theme.text_muted
            chip.setStyleSheet(f"color: {colour}; font-size: 8pt; font-weight: 600;")
            lay.addWidget(chip)
        lay.addSpacing(4)
        width = STRIP_WIDTH - 24  # inside the margins (10 + 10) and the border
        gap = 4  # between a name and its value
        for pname in doc.shown_params(strip, advanced):
            pretty = param_label(pname)
            text = doc.format_param(strip, pname)
            key, value = QLabel(), QLabel()
            key.setObjectName("muted")
            key.ensurePolished()  # measure both in the fonts they are drawn in
            value.ensurePolished()
            fm = value.fontMetrics()
            needed = key.fontMetrics().horizontalAdvance(pretty) + gap
            if needed + fm.horizontalAdvance(text) <= width:  # both on one line
                key.setText(pretty)
                value.setText(text)
                row = QHBoxLayout()
                row.setSpacing(gap)
                row.addWidget(key, 1)
                row.addWidget(value)
                lay.addLayout(row)
            else:  # the value on a line of its own, under its name
                key.setText(key.fontMetrics().elidedText(
                    pretty, Qt.TextElideMode.ElideRight, width))  # fmt: skip
                value.setText(fm.elidedText(text, Qt.TextElideMode.ElideRight, width))
                lay.addWidget(key)
                lay.addWidget(value, alignment=Qt.AlignmentFlag.AlignRight)
            key.setToolTip(f"{pretty}: {text}")
            value.setToolTip(f"{pretty}: {text}")
        lay.addStretch(1)
        if manifest.get("requires_fit") and doc.mode == "live":
            calib = QPushButton("Calibrate")
            calib.clicked.connect(lambda: self.calibrate_requested.emit(self.strip_id))
            lay.addWidget(calib)
        elif manifest.get("requires_fit"):
            fit_note = QLabel("fits on the whole recording")
            fit_note.setObjectName("muted")
            fit_note.setWordWrap(True)  # a strip is narrower than the note
            lay.addWidget(fit_note)
        if problem:
            self.setToolTip(
                f"{problem}: this strip cannot run in "
                f"{doc.mode} mode; the pipeline will not start."
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press is None or self.objectName() == "input":
            return
        if (event.position().toPoint() - self._press).manhattanLength() < 8:
            return
        self._press = None
        mime = QMimeData()
        mime.setData(MIME, self.strip_id.encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is not None:
            self.clicked.emit(self.strip_id)
        self._press = None
        super().mouseReleaseEvent(event)


class InputStrip(StripWidget):
    """The source strip: picks the input and, with it, live or offline."""

    source_changed = Signal(str)
    mode_toggled = Signal()
    recording_requested = Signal()

    def __init__(
        self,
        doc: PipelineDoc,
        strip: StripNode,
        theme: Theme,
        *,
        status: str,
        advanced: bool,
        selected: bool,
    ) -> None:
        super().__init__(
            doc, strip, theme, status=status, advanced=advanced, selected=selected
        )
        self.setObjectName("input")
        self.setStyleSheet(self.styleSheet().replace("QFrame#strip", "QFrame#input"))
        lay = self.layout()
        assert isinstance(lay, QVBoxLayout)
        combo = QComboBox()
        for m in doc.sources():
            combo.addItem(title(m["name"]), m["name"])
        combo.setCurrentIndex(combo.findData(strip.node))
        combo.currentIndexChanged.connect(
            lambda _: self.source_changed.emit(combo.currentData())
        )
        lay.insertWidget(1, combo)
        mode = QPushButton("● LIVE" if doc.mode == "live" else "◼ OFFLINE")
        mode.setCheckable(True)
        mode.setChecked(doc.mode == "live")
        mode.setEnabled(not doc.mode_is_fixed())
        mode.setToolTip(
            "LSL input is always live"
            if doc.mode_is_fixed()
            else "Play this input live, or process it offline"
        )
        mode.clicked.connect(self.mode_toggled)
        lay.insertWidget(2, mode)
        rec = QPushButton("Recording…" + (" ●" if strip.recording else ""))
        rec.setToolTip(
            "Channel names, types, bad channels, positions and events"
            + (" (edited)" if strip.recording else "")
        )
        rec.clicked.connect(self.recording_requested)
        lay.addWidget(rec)


class Gap(QWidget):
    """Between two strips: the signal line, a probe tap and an insert button."""

    insert_requested = Signal(int)
    moved_here = Signal(str, int)
    tap_clicked = Signal(str, bool)  # strip id, make main

    def __init__(
        self,
        index: int,
        theme: Theme,
        tap_of: str | None,
        tap_state: str,
    ) -> None:
        super().__init__()
        self.index = index
        self.theme = theme
        self.rail = signal_colour(theme)
        self.setAcceptDrops(True)
        self.setFixedWidth(GAP_WIDTH)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, RAIL_Y - TAP_SIZE // 2, 0, 0)
        lay.setSpacing(4)
        if tap_of is None:
            lay.addSpacing(TAP_SIZE)
        else:
            tap = QToolButton()  # sits on the rail
            tap.setFixedSize(TAP_SIZE, TAP_SIZE)
            tap.setText({"main": "◆", "comparison": "◆", "off": "◇"}[tap_state])
            colour = {
                "main": theme.accent,
                "comparison": theme.text_muted,
                "off": theme.border,
            }[tap_state]
            size = 13 if tap_state == "main" else 10  # pt: px leaves pointSize -1
            tap.setStyleSheet(
                f"QToolButton {{ border: none; background: {theme.background}; "
                f"color: {colour}; font-size: {size}pt; padding: 0; }}"
            )
            tap.setToolTip(
                f"Probe tap after {tap_of} ({tap_state}).\nClick: show or "
                "hide in the signal view. Right-click: make it Main."
            )
            tap.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            tap.clicked.connect(lambda: self.tap_clicked.emit(tap_of, False))
            tap.customContextMenuRequested.connect(
                lambda _: self.tap_clicked.emit(tap_of, True)
            )
            lay.addWidget(tap, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)
        add = QToolButton()
        add.setText("+")
        add.setToolTip("Insert a strip here")
        add.setStyleSheet(
            f"QToolButton {{ border: 1px dashed {theme.border}; "
            "border-radius: 9px; padding: 0 4px; }"
        )
        add.clicked.connect(lambda: self.insert_requested.emit(self.index))
        lay.addWidget(add, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)

    def paintEvent(self, event: Any) -> None:
        """The rail board draws the signal; a gap only lights it up while a
        strip is dragged over it."""
        if self.rail == signal_colour(self.theme):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(self.rail), SIGNAL_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(0, RAIL_Y, self.width(), RAIL_Y)
        painter.end()

    def _line_colour(self, colour: str) -> None:
        self.rail = colour
        self.update()

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(MIME):
            self._line_colour(self.theme.accent)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:
        self._line_colour(signal_colour(self.theme))

    def dropEvent(self, event: Any) -> None:
        self._line_colour(signal_colour(self.theme))
        strip_id = bytes(event.mimeData().data(MIME)).decode()
        self.moved_here.emit(strip_id, self.index)
        event.acceptProposedAction()


def signal_colour(theme: Theme) -> str:
    """The signal line: neutral grey, so Signal rails (the ICs) stand apart."""
    return theme.text_muted


def rail_colour(theme: Theme, kind: str) -> str:
    """Events violet, Signal rails (e.g. ICs) in the preprocessing colour,
    feature rails in the feature strips' colour."""
    if kind == "events":
        return "#b18cff" if theme.dark else "#7448c2"
    if kind == "signal":
        return theme.group_color("preprocess")
    return theme.group_color("features")


def _toward_text(
    colour: str, theme: Theme, amount: float = 0.45, alpha: float = 0.5
) -> QColor:
    """A rail's colour moved toward the text colour (whiter on dark themes,
    darker on light ones) and half transparent: its continuation past the
    last reader."""
    a, b = QColor(colour), QColor(theme.text)
    mix = [round(x + (y - x) * amount) for x, y in
           zip((a.red(), a.green(), a.blue()), (b.red(), b.green(), b.blue()),
               strict=True)]  # fmt: skip
    out = QColor(*mix)
    out.setAlphaF(alpha)
    return out


def _head(
    painter: QPainter, colour: QColor, tip: QPointF, ux: float, uy: float
) -> None:
    """An arrowhead at ``tip`` pointing along (ux, uy)."""
    head = QPainterPath(tip)
    head.lineTo(tip - QPointF(7 * ux - 4 * uy, 7 * uy + 4 * ux))
    head.lineTo(tip - QPointF(7 * ux + 4 * uy, 7 * uy - 4 * ux))
    head.closeSubpath()
    painter.fillPath(head, colour)


class Lowered(QWidget):
    """A feature or output strip, set below the signal line: the signal passes
    over it (the rail board draws the signal and the arrows into and out of
    it)."""

    def __init__(self, strip: StripWidget, theme: Theme) -> None:
        super().__init__()
        self.strip = strip
        self.setFixedWidth(STRIP_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, LOWER, 0, 0)
        lay.addWidget(strip)


class RailBoard(QWidget):
    """Holds the chain and paints the parallel rails above it (docs/RAILS.md).

    - A feature strip, set below the signal, gets a down arrow from the signal
      (and from each rail it reads) and an up arrow into the rail it starts,
      whose start is a solid node.
    - An output strip, also below, gets a down arrow from what it reads: the
      signal or a feature rail.
    - Each rail is solid up to its last reader, then runs on, thinner, dashed
      and in a fainter colour, to the end of the chain with an arrow.
    - A strip that needs a rail the chain lacks gets a red dashed arrow."""

    tap_clicked = Signal(str, bool)  # a feature rail or rail output, make main

    def __init__(
        self, doc: PipelineDoc, theme: Theme, taps: dict[str, str] | None = None
    ) -> None:
        super().__init__()
        self.theme = theme
        self.taps = taps or {}
        self.rails, self.needs = doc.rails()
        self.lanes = doc.rail_lanes()
        self.strips: dict[str, QWidget] = {}
        self.lowered = {s.id for s in doc.strips if doc.lowered(s)}
        self.pull_signal = {s.id for s in doc.strips
                            if doc.lowered(s) and doc.reads_signal(s)}  # fmt: skip
        self.font_ = QFont(self.font())
        self.font_.setPointSizeF(max(7.0, self.font_.pointSizeF() * 0.82))
        self.font_.setBold(True)
        self.setMouseTracking(True)

    @property
    def band(self) -> int:
        """Height of the rails above the strips."""
        return TOP + self.lanes * LANE_H + (4 if self.lanes else 0)

    @property
    def signal_y(self) -> float:
        return float(self.band + 2 + RAIL_Y)

    def lane_y(self, lane: int) -> float:
        return TOP + (self.lanes - 1 - lane) * LANE_H + LANE_H / 2

    def _rect(self, strip_id: str) -> QRectF | None:
        w = self.strips.get(strip_id)
        if w is None:
            return None
        return QRectF(w.mapTo(self, QPoint(0, 0)).toPointF(), w.size().toSizeF())

    def _reads(self, strip_id: str) -> list[str]:
        """The rails a strip reads, left to right as their arrows are drawn."""
        return [r.id for r in self.rails if strip_id in r.readers]

    def _read_x(self, strip_id: str, rail_id: str | None) -> float | None:
        """Where a strip's arrow from the signal (None) or a rail comes in."""
        rect = self._rect(strip_id)
        if rect is None:
            return None
        slots: list[str | None] = [None] if strip_id in self.pull_signal else []
        slots += self._reads(strip_id)
        slots += [f"needs:{n}" for s, n in self.needs.items() if s == strip_id]
        return rect.left() + PORT + 14 * slots.index(rail_id)

    def end_x(self) -> float:
        rights = [r.right() for sid in self.strips if (r := self._rect(sid))]
        return (max(rights) if rights else 0.0) + 28

    def spans(self) -> list[tuple[Rail, float, float]]:
        """Each drawn rail: where it starts and where its solid part ends."""
        out = []
        produced: dict[str, int] = {}  # a strip writing several rails: side by side
        for rail in self.rails:
            rect = self._rect(rail.producer)
            if rect is None:
                continue
            k = produced[rail.producer] = produced.get(rail.producer, -1) + 1
            x0 = rect.right() - PORT - 14 * k
            xs = [x for r in rail.readers if (x := self._read_x(r, rail.id))]
            xs += [x for w in rail.writers if (x := self._write_x(w))]
            out.append((rail, x0, max([x0, *xs])))
        return out

    def _write_x(self, strip_id: str) -> float | None:
        """Where a strip adding to a rail (the Events rail) pushes up into it."""
        rect = self._rect(strip_id)
        return None if rect is None else rect.right() - PORT

    def _vertical(self, painter: QPainter, x: float, y0: float, y1: float,
                  crossings: list[tuple[float, float]]) -> None:  # fmt: skip
        """A line from ``y0`` to ``y1``, hopping over ``crossings`` (each a
        line's y and the hop's radius)."""
        lo, hi = min(y0, y1), max(y0, y1)
        path = QPainterPath(QPointF(x, lo))
        for c, r in sorted((c, r) for c, r in crossings if lo + r + 1 < c < hi - r - 1):
            path.lineTo(x, c - r)
            path.arcTo(QRectF(x - r, c - r, 2 * r, 2 * r), 90, -180)
        path.lineTo(x, hi)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font_)
        fm = QFontMetrics(self.font_)
        spans = self.spans()
        end = self.end_x()
        sig = QColor(signal_colour(self.theme))
        signal_y = self.signal_y

        def crossings(x: float, below_lane: int) -> list[tuple[float, float]]:
            """Rails under ``below_lane`` that a vertical at ``x`` passes, and
            the signal (a vertical into a lowered strip crosses it)."""
            rails = [(self.lane_y(r.lane), 4.0) for r, x0, _ in spans
                     if r.lane < below_lane and x0 < x]  # fmt: skip
            return [*rails, (signal_y, SIGNAL_WIDTH / 2 + 3)]

        # the signal first, under every arrow (strips on it hide it)
        rects = [r for sid in self.strips if (r := self._rect(sid))]
        if rects:
            pen = QPen(sig, SIGNAL_WIDTH)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            right = max(r.right() for r in rects) + GAP_WIDTH
            painter.drawLine(QPointF(min(r.left() for r in rects), signal_y),
                             QPointF(right, signal_y))  # fmt: skip

        # down arrows from the signal into lowered strips
        for sid in self.pull_signal:
            rect, x = self._rect(sid), self._read_x(sid, None)
            if rect is None or x is None:
                continue
            painter.setPen(QPen(sig, 1.6))
            painter.drawLine(QPointF(x, signal_y + SIGNAL_WIDTH / 2),
                             QPointF(x, rect.top() - 6))  # fmt: skip
            _head(painter, sig, QPointF(x, rect.top() - 1), 0, 1)

        for rail, x0, x1 in spans:
            colour = QColor(rail_colour(self.theme, rail.kind))
            y = self.lane_y(rail.lane)
            # solid to the last reader, then dashed and fainter to the end
            painter.setPen(QPen(colour, 2))
            painter.drawLine(QPointF(x0, y), QPointF(x1, y))
            faint = _toward_text(colour.name(), self.theme)
            painter.setPen(QPen(faint, 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x1, y), QPointF(end - 6, y))
            _head(painter, faint, QPointF(end, y), 1, 0)
            # up arrow from the producer into its start
            prod = self._rect(rail.producer)
            assert prod is not None
            painter.setPen(QPen(colour, 1.6))
            self._vertical(painter, x0, y + 9, prod.top(), crossings(x0, rail.lane))
            _head(painter, colour, QPointF(x0, y + 4), 0, -1)
            painter.setPen(QPen(colour, 2))
            painter.setBrush(colour)
            painter.drawEllipse(QPointF(x0, y), 3.5, 3.5)
            # up arrows from the strips adding to it (reviews, detectors)
            for writer in rail.writers:
                rect, x = self._rect(writer), self._write_x(writer)
                if rect is None or x is None:
                    continue
                painter.setPen(QPen(colour, 1.6))
                self._vertical(painter, x, y + 9, rect.top(), crossings(x, rail.lane))
                _head(painter, colour, QPointF(x, y + 4), 0, -1)
                painter.setPen(QPen(colour, 1.6))  # ○: adds to a rail it did not start
                painter.setBrush(QColor(self.theme.background))
                painter.drawEllipse(QPointF(x, y), 3.5, 3.5)
            # down arrows into the readers
            for reader in rail.readers:
                rect, x = self._rect(reader), self._read_x(reader, rail.id)
                if rect is None or x is None:
                    continue
                painter.setPen(QPen(colour, 1.6))
                self._vertical(painter, x, y, rect.top() - 6, crossings(x, rail.lane))
                _head(painter, colour, QPointF(x, rect.top() - 1), 0, 1)
            # its name above its producer; a rail with a view has a tap mark
            box = self.label_box(rail, x0, fm)
            label = fm.elidedText(rail.label, Qt.TextElideMode.ElideRight,
                                  int(box.width()))  # fmt: skip
            if self.tappable(rail):
                state = self.taps.get(rail.id, "off")
                label = ("◆ " if state == "main" else "◇ ") + label
                if state == "main":
                    colour = QColor(self.theme.accent)
            painter.setPen(colour)
            painter.drawText(
                box, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label
            )

        warn = QColor(self.theme.incompatible_border)
        for strip_id, needed in self.needs.items():
            rect = self._rect(strip_id)
            x = self._read_x(strip_id, f"needs:{needed}")
            if rect is None or x is None:
                continue
            y = self.lane_y(self.lanes - 1)
            painter.setPen(QPen(warn, 1.5, Qt.PenStyle.DashLine))
            self._vertical(painter, x, y + 4, rect.top() - 6,
                           crossings(x, self.lanes - 1))  # fmt: skip
            _head(painter, warn, QPointF(x, rect.top() - 1), 0, 1)
            painter.setPen(QPen(warn, 2))
            painter.setBrush(QColor(self.theme.background))
            painter.drawEllipse(QPointF(x, y), 3.5, 3.5)
            text = f"needs {needed}"
            box = QRectF(x + 8, y - fm.height() / 2, fm.horizontalAdvance(text) + 4,
                         fm.height())  # fmt: skip
            painter.setPen(warn)
            painter.drawText(
                box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
            )
        painter.end()

    @staticmethod
    def tappable(rail: Rail) -> bool:
        """Feature rails and Signal rail outputs (the ICs) have a view."""
        return rail.kind in ("features", "signal")

    def label_box(self, rail: Rail, x0: float, fm: QFontMetrics) -> QRectF:
        room = STRIP_WIDTH - 2 * PORT - 30
        return QRectF(x0 - 8 - room, self.lane_y(rail.lane) - fm.height() / 2, room,
                      fm.height())  # fmt: skip

    def tap_at(self, point: QPointF) -> Rail | None:
        """The rail whose name or start is at ``point``, when it has a view."""
        fm = QFontMetrics(self.font_)
        for rail, x0, _ in self.spans():
            if not self.tappable(rail):
                continue
            box = self.label_box(rail, x0, fm)
            text = fm.horizontalAdvance("◇ " + rail.label) + 6
            box.setLeft(max(box.left(), box.right() - text))
            if box.adjusted(0, -2, 8, 2).contains(point):
                return rail
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        rail = self.tap_at(event.position())
        if rail is None:
            super().mousePressEvent(event)
            return
        right = event.button() == Qt.MouseButton.RightButton
        self.tap_clicked.emit(rail.id, right)

    def rail_at(self, point: QPointF) -> Rail | None:
        end = self.end_x()
        for rail, x0, _ in self.spans():
            y = self.lane_y(rail.lane)
            if x0 - 4 <= point.x() <= end and abs(point.y() - y) <= 5:
                return rail
        return None

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            assert isinstance(event, QHelpEvent)
            point = QPointF(event.pos())
            rail = self.rail_at(point) or self.tap_at(point)
            if rail is None:
                QToolTip.hideText()
            else:
                readers = ", ".join(rail.readers) or "nothing yet"
                tip = f"{rail.label} rail, from {rail.producer}\nRead by: {readers}"
                if rail.writers:
                    tip += f"\nAdded to by: {', '.join(rail.writers)}"
                if self.tappable(rail):
                    tip += "\nClick its name: show it in the view above"
                QToolTip.showText(event.globalPos(), tip, self)
            return True
        return super().event(event)


class ChainView(QScrollArea):
    """Horizontal chain; rebuilt whenever the pipeline or its status changes."""

    selected = Signal(str)
    remove_requested = Signal(str)
    insert_requested = Signal(int, str)
    script_requested = Signal(int)  # add a custom script, then insert it here
    move_requested = Signal(str, int)
    tap_clicked = Signal(str, bool)
    calibrate_requested = Signal(str)
    source_changed = Signal(str)
    mode_toggled = Signal()
    recording_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(240)
        self._doc: PipelineDoc | None = None

    def rebuild(
        self,
        doc: PipelineDoc,
        theme: Theme,
        *,
        statuses: dict[str, str],
        taps: dict[str, str],
        advanced: bool,
        selected: str | None,
    ) -> None:
        self._doc = doc
        self._theme = theme
        scroll = self.horizontalScrollBar().value()
        inner = RailBoard(doc, theme, taps)
        inner.tap_clicked.connect(self.tap_clicked)
        lay = QHBoxLayout(inner)
        lay.setContentsMargins(8, inner.band + 2, 8, 8)
        lay.setSpacing(0)
        self.setMinimumHeight(232 + inner.band)
        tappable = {s.id for s in doc.tappable()}
        for i, strip in enumerate(doc.strips):
            status = statuses.get(strip.id, "stopped")
            chosen = strip.id == selected
            widget: StripWidget
            if i == 0:
                widget = InputStrip(
                    doc, strip, theme, status=status, advanced=advanced, selected=chosen
                )
                widget.source_changed.connect(self.source_changed)
                widget.mode_toggled.connect(self.mode_toggled)
                widget.recording_requested.connect(self.recording_requested)
            else:
                widget = StripWidget(
                    doc, strip, theme, status=status, advanced=advanced, selected=chosen
                )
            widget.clicked.connect(self.selected)
            widget.remove_requested.connect(self.remove_requested)
            widget.calibrate_requested.connect(self.calibrate_requested)
            lay.addWidget(Lowered(widget, theme) if doc.lowered(strip) else widget)
            inner.strips[strip.id] = widget
            tap_of = strip.id if strip.id in tappable else None
            gap = Gap(i + 1, theme, tap_of, taps.get(strip.id, "off"))
            gap.insert_requested.connect(self._palette)
            gap.moved_here.connect(self.move_requested)
            gap.tap_clicked.connect(self.tap_clicked)
            lay.addWidget(gap)
        add = QPushButton("+  Add strip")
        add.setFixedWidth(120)
        add.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        add.setStyleSheet(
            f"QPushButton {{ border: 1px dashed {theme.border}; "
            f"border-radius: 8px; color: {theme.text_muted}; }}"
        )
        add.clicked.connect(lambda: self._palette(len(doc.strips)))
        lay.addWidget(add)
        lay.addStretch(1)
        # A rebuild usually runs inside an event handler of one of the old
        # widgets (a strip clicked, a strip removed): setWidget would delete them
        # while Qt is still delivering that event, so delete them later instead.
        # takeWidget leaves them without a parent, a top-level window (on Windows
        # a visible one flashes up): hide first, keep a hidden child until deleted.
        old = self.widget()
        if old is not None:
            old.hide()
        old = self.takeWidget()
        self.setWidget(inner)
        if old is not None:
            old.setParent(self)
            old.deleteLater()
        self.horizontalScrollBar().setValue(scroll)

    def _palette(self, index: int) -> None:
        """Pick a node to insert; nodes that cannot run in this mode are greyed."""
        menu = self.build_palette(index)
        if menu is not None:
            menu.exec(self.mapToGlobal(self.rect().center()))

    def build_palette(self, index: int) -> QMenu | None:
        """The add-strip menu: each group by category (submenus), the user's own
        nodes marked ✎ (unsandboxed) or ⬡ (container plugin)."""
        doc = self._doc
        if doc is None:
            return None
        menu = QMenu(self)
        marks = {"Preprocessing": "●", "Feature extraction": "◆", "Output": "▶"}
        for i, (heading, categories) in enumerate(doc.palette_sections()):
            if i:
                menu.addSeparator()
            # a bold, inert heading (menu sections do not show text in every style)
            head = menu.addAction(f"{marks.get(heading, '')} {heading}".strip())
            font = head.font()
            font.setBold(True)
            head.setFont(font)
            head.setEnabled(False)
            for label, names in categories:
                sub = menu.addMenu(label.replace("&", "&&"))  # "&" marks a shortcut
                sub.setToolTipsVisible(True)
                for name in names:
                    m = doc.manifest(name)
                    problem = doc.incompatibility(name)
                    text = f"{doc.mark(name)} {title(name)}".strip()
                    action = sub.addAction(
                        text + (f"   ({problem})" if problem else "")
                    )
                    tip = m.get("description", "")
                    if doc.mark(name) == "✎":
                        tip += "\n✎ added by you: runs on this computer, unsandboxed"
                    elif doc.mark(name):
                        tip += "\n⬡ plugin: runs in an isolated container"
                    action.setToolTip(tip.strip())
                    action.setEnabled(problem is None)
                    action.triggered.connect(
                        lambda _=False, n=name: self.insert_requested.emit(index, n)
                    )
        menu.addSeparator()
        script = menu.addAction("✎ Add a custom script…")
        script.setToolTip(
            "An MNE-Python script: process(raw, ...) for offline runs, or "
            "transform(data, state, ...) for live and offline, or features(data, "
            "...) per window (runs on your computer, unsandboxed)"
        )
        script.triggered.connect(lambda: self.script_requested.emit(index))
        menu.setToolTipsVisible(True)
        return menu
