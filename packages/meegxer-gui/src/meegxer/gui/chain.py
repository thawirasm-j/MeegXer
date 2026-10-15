"""The chain of strips: Input strip, processor and sink strips, taps and gaps."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QMouseEvent
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

from meegxer.gui.names import badge, title
from meegxer.gui.pipeline_doc import PipelineDoc, StripNode
from meegxer.gui.theme import Theme

MIME = "application/x-meegxer-strip"
STRIP_WIDTH = 158
TAP_STATES = ("off", "comparison", "main")


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
        self.setStyleSheet(
            f"QFrame#strip {{ background: {bg}; border: {2 if selected else 1}px "
            f"solid {border}; border-radius: 8px; }}"
        )
        manifest = doc.manifest(strip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)

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

        sub = QLabel(strip.id)
        sub.setObjectName("muted")
        lay.addWidget(sub)
        note = problem or badge(strip.node, strip.params, doc.mode)
        if note:
            chip = QLabel(note)
            colour = theme.incompatible_border if problem else theme.text_muted
            chip.setStyleSheet(f"color: {colour}; font-size: 8pt; font-weight: 600;")
            lay.addWidget(chip)
        lay.addSpacing(4)
        for pname in doc.shown_params(strip, advanced):
            row = QHBoxLayout()
            key = QLabel(pname.replace("_", " "))
            key.setObjectName("muted")
            row.addWidget(key)
            row.addStretch(1)
            row.addWidget(QLabel(doc.format_param(strip, pname)))
            lay.addLayout(row)
        lay.addStretch(1)
        if manifest.get("requires_fit"):
            calib = QPushButton("Calibrate")
            calib.clicked.connect(lambda: self.calibrate_requested.emit(self.strip_id))
            lay.addWidget(calib)
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
        self, index: int, theme: Theme, tap_of: str | None, tap_state: str
    ) -> None:
        super().__init__()
        self.index = index
        self.theme = theme
        self.setAcceptDrops(True)
        self.setFixedWidth(34)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addStretch(1)
        self.line = QFrame()
        self.line.setFixedHeight(2)
        self._line_colour(theme.border)
        lay.addWidget(self.line)
        if tap_of is not None:
            tap = QToolButton()
            tap.setText({"main": "◆", "comparison": "◆", "off": "◇"}[tap_state])
            colour = {
                "main": theme.accent,
                "comparison": theme.text_muted,
                "off": theme.border,
            }[tap_state]
            size = 18 if tap_state == "main" else 14
            tap.setStyleSheet(
                f"QToolButton {{ border: none; background: transparent; "
                f"color: {colour}; font-size: {size}px; }}"
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

    def _line_colour(self, colour: str) -> None:
        self.line.setStyleSheet(f"background: {colour}; border: none;")

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat(MIME):
            self._line_colour(self.theme.accent)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event: Any) -> None:
        self._line_colour(self.theme.border)

    def dropEvent(self, event: Any) -> None:
        self._line_colour(self.theme.border)
        strip_id = bytes(event.mimeData().data(MIME)).decode()
        self.moved_here.emit(strip_id, self.index)
        event.acceptProposedAction()


class ChainView(QScrollArea):
    """Horizontal chain; rebuilt whenever the pipeline or its status changes."""

    selected = Signal(str)
    remove_requested = Signal(str)
    insert_requested = Signal(int, str)
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
        self.setMinimumHeight(220)
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
            lay.addWidget(widget)
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
        self.setWidget(inner)
        self.horizontalScrollBar().setValue(scroll)

    def _palette(self, index: int) -> None:
        """Pick a node to insert; nodes that cannot run in this mode are greyed."""
        doc = self._doc
        if doc is None:
            return
        menu = QMenu(self)
        groups = {"processor": "Processing", "sink": "Outputs"}
        for role, heading in groups.items():
            section = menu.addSection(heading)
            section.setEnabled(False)
            for m in doc.nodes_for_palette():
                if m["role"] != role:
                    continue
                problem = doc.incompatibility(m["name"])
                label = title(m["name"])
                action = menu.addAction(label + (f"   ({problem})" if problem else ""))
                action.setToolTip(m.get("description", ""))
                action.setEnabled(problem is None)
                action.triggered.connect(
                    lambda _=False, n=m["name"]: self.insert_requested.emit(index, n)
                )
        menu.setToolTipsVisible(True)
        menu.exec(self.mapToGlobal(self.rect().center()))
