"""The chain of strips: Input strip, processor and sink strips, taps and gaps.

One lane. The main line is the signal path (input and preprocessing strips) and
runs along the rail; feature extraction and outputs fork off it: they sit lower,
with an arrow down from the rail, and the rail passes over them to the next
preprocessing strip (a feature strip between re-reference and ICA reads from
re-reference, and ICA still does too). Each strip's top band, shape and label
give its group: ■ input, ● preprocessing, ◆ feature extraction, ▶ output.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QMouseEvent, QPainter, QPainterPath, QPen
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
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.labels import param_label
from meegxer.gui.names import badge, title
from meegxer.gui.pipeline_doc import PipelineDoc, StripNode
from meegxer.gui.theme import Theme

MIME = "application/x-meegxer-strip"
STRIP_WIDTH = 158
TAP_STATES = ("off", "comparison", "main")
RAIL_Y = 26  # the main line's height in the chain, px from the top
BRANCH_TOP = 50  # where forked strips start
BRANCH_LINK_Y = BRANCH_TOP + 26  # links between forked strips
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
        for pname in doc.shown_params(strip, advanced):
            row = QHBoxLayout()
            pretty = param_label(pname)
            text = doc.format_param(strip, pname)
            value = QLabel()
            value.setText(
                value.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, 88)
            )
            room = STRIP_WIDTH - 28 - value.sizeHint().width()
            key = QLabel(
                value.fontMetrics().elidedText(
                    pretty, Qt.TextElideMode.ElideRight, max(room, 30)
                )
            )
            key.setObjectName("muted")
            key.setToolTip(f"{pretty}: {text}")
            value.setToolTip(f"{pretty}: {text}")
            row.addWidget(key, 1)
            row.addWidget(value)
            lay.addLayout(row)
        lay.addStretch(1)
        if manifest.get("requires_fit") and doc.mode == "live":
            calib = QPushButton("Calibrate")
            calib.clicked.connect(lambda: self.calibrate_requested.emit(self.strip_id))
            lay.addWidget(calib)
        elif manifest.get("requires_fit"):
            fit_note = QLabel("fits on the whole recording")
            fit_note.setObjectName("muted")
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
    """Between two strips: the main line (rail), a probe tap and an insert
    button; a link at the forked level when the next strip reads from the
    previous forked one (band power into LSL features)."""

    insert_requested = Signal(int)
    moved_here = Signal(str, int)
    tap_clicked = Signal(str, bool)  # strip id, make main

    def __init__(
        self,
        index: int,
        theme: Theme,
        tap_of: str | None,
        tap_state: str,
        branch_link: str | None = None,
    ) -> None:
        super().__init__()
        self.index = index
        self.theme = theme
        self.rail = theme.text_muted
        self.branch_link = branch_link  # colour of the forked link, if any
        self.setAcceptDrops(True)
        self.setFixedWidth(34)
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(self.rail), 2))
        painter.drawLine(0, RAIL_Y, self.width(), RAIL_Y)
        if self.branch_link:
            start = QPointF(0, BRANCH_LINK_Y)
            end = QPointF(self.width(), BRANCH_LINK_Y)
            _arrow(painter, QColor(self.branch_link), start, end)
        painter.end()

    def _line_colour(self, colour: str) -> None:
        self.rail = colour
        self.update()

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(MIME):
            self._line_colour(self.theme.accent)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:
        self._line_colour(self.theme.text_muted)

    def dropEvent(self, event: Any) -> None:
        self._line_colour(self.theme.text_muted)
        strip_id = bytes(event.mimeData().data(MIME)).decode()
        self.moved_here.emit(strip_id, self.index)
        event.acceptProposedAction()


def _arrow(painter: QPainter, colour: QColor, start: QPointF, end: QPointF) -> None:
    """A straight arrow from ``start`` to ``end`` (horizontal or vertical)."""
    painter.setPen(QPen(colour, 2))
    painter.drawLine(start, end)
    d = end - start
    length = max(1.0, (d.x() ** 2 + d.y() ** 2) ** 0.5)
    ux, uy = d.x() / length, d.y() / length
    head = QPainterPath(end)
    head.lineTo(end - QPointF(7 * ux - 4 * uy, 7 * uy + 4 * ux))
    head.lineTo(end - QPointF(7 * ux + 4 * uy, 7 * uy - 4 * ux))
    head.closeSubpath()
    painter.fillPath(head, colour)


class Fork(QWidget):
    """A forked strip (feature extraction or output): the rail passes over it,
    and an arrow comes down from the rail into it when it reads from the main
    line (rather than from the forked strip before it)."""

    def __init__(
        self, strip: StripWidget, theme: Theme, colour: str, from_rail: bool
    ) -> None:
        super().__init__()
        self.strip = strip
        self.rail = QColor(theme.text_muted)
        self.colour = QColor(colour)
        self.from_rail = from_rail
        self.setFixedWidth(STRIP_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, BRANCH_TOP, 0, 0)
        lay.addWidget(strip)
        self.setToolTip(
            "Forks off the main line: reads the signal here, the main line "
            "continues past it"
            if from_rail
            else "Reads the strip before it"
        )

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.rail, 2))
        painter.drawLine(0, RAIL_Y, self.width(), RAIL_Y)
        if self.from_rail:  # ↘: from the rail down into the strip
            x = 14.0
            painter.setBrush(self.colour)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, RAIL_Y), 4, 4)
            _arrow(painter, self.colour, QPointF(x, RAIL_Y + 4),
                   QPointF(x, BRANCH_TOP - 1))  # fmt: skip
        painter.end()


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
        self.setMinimumHeight(220 + BRANCH_TOP)
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
        inner = QWidget()
        lay = QHBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(0)
        tappable = {s.id for s in doc.tappable()}
        main_line = {s.id for s in doc.strips if doc.on_main_line(s)}
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
            colour = theme.group_color(doc.group(strip))
            if strip.id in main_line:
                lay.addWidget(widget)
            else:
                source = doc.input_of(strip)
                lay.addWidget(Fork(widget, theme, colour, source in main_line))
            tap_of = strip.id if strip.id in tappable else None
            after = doc.strips[i + 1] if i + 1 < len(doc.strips) else None
            linked = (
                after is not None
                and strip.id not in main_line
                and doc.input_of(after) == strip.id
            )
            gap = Gap(
                i + 1,
                theme,
                tap_of,
                taps.get(strip.id, "off"),
                theme.group_color(doc.group(strip)) if linked else None,
            )
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
        old = self.takeWidget()
        self.setWidget(inner)
        if old is not None:
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
