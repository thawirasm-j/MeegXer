"""Inspector: every parameter of the selected strip, generated from its manifest.

Laid out as cards: a header (title, group, what the strip does), the settings
(main ones first, on/off switches in a two-column grid), the advanced settings
(folded away in friendly mode), then calibration, the channel types the strip
processes and the values shown on the strip.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.labels import choice_label, param_label, unit_label
from meegxer.gui.names import title
from meegxer.gui.pipeline_doc import MAX_STRIP_PARAMS, PipelineDoc, StripNode
from meegxer.gui.theme import DEFAULT_GROUPS, Theme

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "misc")
CHANNEL_TYPE_NAMES = {
    "eeg": "EEG",
    "eog": "Eye (EOG)",
    "emg": "Muscle (EMG)",
    "ecg": "Heart (ECG)",
    "resp": "Breathing",
    "misc": "Other",
}
GROUP_NAMES = {
    "input": "Input",
    "preprocess": "Preprocessing",
    "features": "Feature extraction",
    "output": "Output",
}
STATUS_NAMES = {
    "ready": "calibrated and running",
    "uncalibrated": "not calibrated yet",
    "calibrating": "calibrating…",
    "needs-refit": "needs calibrating again (something before it changed)",
    "bypassed": "passing data through uncalibrated",
    "stopped": "pipeline stopped",
    "error": "error (see the log)",
}


class _Number(QDoubleSpinBox):
    """A float field without trailing zeros (``500 Hz``, not ``500.000 Hz``)."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")


def _editor(spec: dict[str, Any], value: Any) -> tuple[QWidget, Any]:
    """A control for one parameter and the signal that reports a final value."""
    kind = spec["type"]
    if kind == "bool":
        box = QCheckBox(param_label(spec["name"], spec))
        box.setChecked(bool(value))
        return box, box.toggled
    if kind == "enum":
        combo = QComboBox()
        for choice in spec["choices"]:
            combo.addItem(choice_label(spec["name"], choice), choice)
        combo.setCurrentIndex(max(0, combo.findData(value)))
        return combo, combo.currentIndexChanged
    if kind == "string":
        line = QLineEdit(str(value))
        return line, line.editingFinished
    spin: QSpinBox | QDoubleSpinBox
    if kind == "int":
        spin = QSpinBox()
        spin.setRange(
            int(spec["min"] if spec["min"] is not None else -(10**9)),
            int(spec["max"] if spec["max"] is not None else 10**9),
        )
        spin.setValue(int(value))
    else:
        spin = _Number()
        spin.setDecimals(3)
        spin.setRange(
            spec["min"] if spec["min"] is not None else -1e12,
            spec["max"] if spec["max"] is not None else 1e12,
        )
        spin.setValue(float(value))
    if spec.get("unit"):
        spin.setSuffix(f" {unit_label(spec['unit'])}")
    if "0 streams forever" in (spec.get("description") or ""):
        spin.setSpecialValueText("forever")
    spin.setKeyboardTracking(False)  # report when typing is done
    return spin, spin.valueChanged


def _value(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QSpinBox | QDoubleSpinBox):
        return widget.value()
    raise TypeError(widget)


def _card(heading: str, note: str = "") -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 8, 10, 10)
    lay.setSpacing(6)
    head = QLabel(heading)
    head.setObjectName("section")
    lay.addWidget(head)
    if note:
        text = QLabel(note)
        text.setObjectName("muted")
        text.setWordWrap(True)
        lay.addWidget(text)
    return card, lay


def _grid(widgets: list[QWidget], columns: int = 2) -> QGridLayout:
    grid = QGridLayout()
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(4)
    for i, w in enumerate(widgets):
        grid.addWidget(w, i // columns, i % columns)
    for c in range(columns):
        grid.setColumnStretch(c, 1)
    return grid


class Inspector(QScrollArea):
    param_changed = Signal(str, str, object)  # strip id, name, value
    strip_params_changed = Signal(str, list)
    channel_types_changed = Signal(str, list)
    pass_uncalibrated_changed = Signal(str, bool)
    clear_calibration = Signal(str)
    calibrate = Signal(str, float)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setObjectName("panel")
        self.theme: Theme | None = None
        self.advanced_open: bool | None = None  # None: follow the mode
        self._advanced_mode = False
        self.show_empty()

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme

    def show_empty(self) -> None:
        hint = QLabel("Select a strip to see all of its settings.")
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._replace(hint)

    def _replace(self, page: QWidget) -> None:
        # Editing a value rebuilds the inspector while Qt is still delivering that
        # event (wheel, Enter) to the old editor: setWidget would delete the editor
        # under it, so delete the old page later instead.
        old = self.takeWidget()
        self.setWidget(page)
        if old is not None:
            old.deleteLater()

    def show_strip(
        self,
        doc: PipelineDoc,
        strip: StripNode,
        *,
        advanced: bool,
        running: bool,
        node_status: dict[str, Any],
    ) -> None:
        if advanced != self._advanced_mode:  # a mode switch resets the fold
            self._advanced_mode, self.advanced_open = advanced, None
        manifest = doc.manifest(strip)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)
        lay.addWidget(self._header(doc, strip, manifest, advanced))

        live = set(manifest.get("live_params", []))
        is_processor = manifest["role"] == "processor"
        main_specs = [p for p in manifest["params"] if not p.get("advanced")]
        main_specs.sort(key=lambda p: not p.get("primary"))  # main ones first
        more_specs = [p for p in manifest["params"] if p.get("advanced")]

        def restart_note(name: str) -> str:
            if running and not is_processor:
                return "Changing this restarts the pipeline."
            if running and name not in live:
                return "Changing this restarts this strip's filter state."
            return ""

        if main_specs:
            card, card_lay = _card("Settings")
            self._fill(card_lay, strip, main_specs, restart_note)
            lay.addWidget(card)
        if more_specs:
            lay.addWidget(self._advanced(strip, more_specs, restart_note))
        if not manifest["params"]:
            note = QLabel("This strip has no settings.")
            note.setObjectName("muted")
            lay.addWidget(note)

        if manifest.get("requires_fit") and doc.mode == "live":
            lay.addWidget(self._calibration(strip, node_status, running))
        elif manifest.get("requires_fit"):
            card, _ = _card(
                "Calibration",
                "Offline runs fit this strip on the whole recording. The fit is "
                "reused while the input and the strips before it stay the same.",
            )
            lay.addWidget(card)
        if advanced and is_processor:
            lay.addWidget(self._channel_types(strip))
        if advanced and manifest["params"]:
            lay.addWidget(self._strip_picker(doc, strip))
        lay.addStretch(1)
        self._replace(page)

    def _header(
        self,
        doc: PipelineDoc,
        strip: StripNode,
        manifest: dict[str, Any],
        advanced: bool,
    ) -> QWidget:
        group = doc.group(strip)
        band = self.theme.group_color(group) if self.theme else DEFAULT_GROUPS[group]
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(f"QFrame#card {{ border-left: 4px solid {band}; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 8, 10, 10)
        lay.setSpacing(3)
        is_source = manifest["role"] == "source"
        name = QLabel(
            f"Input: {title(strip.node, strip.params)}"
            if is_source
            else title(strip.node, strip.params)
        )
        name.setObjectName("heading")
        lay.addWidget(name)
        sub = GROUP_NAMES[group]
        if advanced:
            sub += f" · {strip.id} · {strip.node} {manifest['version']}"
        else:
            sub += f" · version {manifest['version']}"
        kind = QLabel(sub)
        kind.setStyleSheet(f"color: {band}; font-weight: 600;")
        lay.addWidget(kind)
        if manifest.get("description"):
            desc = QLabel(manifest["description"])
            desc.setObjectName("muted")
            desc.setWordWrap(True)
            lay.addWidget(desc)
        return card

    def _fill(
        self,
        lay: QVBoxLayout,
        strip: StripNode,
        specs: list[dict[str, Any]],
        restart_note: Any,
    ) -> None:
        """Values in a form; on/off switches in a two-column grid below it."""
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        switches: list[QWidget] = []
        for spec in specs:
            name = spec["name"]
            editor, changed = _editor(spec, strip.params.get(name, spec["default"]))
            changed.connect(
                lambda *_, n=name, w=editor: self.param_changed.emit(
                    strip.id, n, _value(w)
                )
            )
            tip = "\n".join(
                t for t in (spec.get("description") or "", restart_note(name)) if t
            )
            editor.setToolTip(tip)
            if spec["type"] == "bool":
                switches.append(editor)
                continue
            if spec["type"] == "string":
                editor.setMinimumWidth(260)
            else:
                editor.setFixedWidth(180)
            label = QLabel(param_label(name, spec))
            label.setToolTip(tip)
            form.addRow(label, editor)
        if form.rowCount():
            lay.addLayout(form)
        if switches:
            lay.addLayout(_grid(switches))

    def _advanced(
        self, strip: StripNode, specs: list[dict[str, Any]], restart_note: Any
    ) -> QWidget:
        """Advanced settings, folded away in friendly mode (open in advanced)."""
        is_open = (
            self._advanced_mode if self.advanced_open is None else self.advanced_open
        )
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 6, 10, 8)
        toggle = QToolButton()
        toggle.setObjectName("section")
        toggle.setCheckable(True)
        toggle.setChecked(is_open)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(
            Qt.ArrowType.DownArrow if is_open else Qt.ArrowType.RightArrow
        )
        toggle.setText(f"Advanced settings ({len(specs)})")
        colour = f"color: {self.theme.text};" if self.theme else ""
        toggle.setStyleSheet(
            "QToolButton, QToolButton:checked { border: none; background: "
            f"transparent; {colour} font-weight: 600; padding: 0; }}"
        )
        lay.addWidget(toggle)
        body = QWidget()
        body.setObjectName("cardbody")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        self._fill(body_lay, strip, specs, restart_note)
        body.setVisible(is_open)
        lay.addWidget(body)

        def flip(on: bool) -> None:
            self.advanced_open = on
            body.setVisible(on)
            toggle.setArrowType(
                Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow
            )

        toggle.toggled.connect(flip)
        return card

    def _calibration(
        self, strip: StripNode, node_status: dict[str, Any], running: bool
    ) -> QWidget:
        status = node_status.get("status", "stopped")
        card, lay = _card(
            "Calibration",
            "Records this much input, fits the strip on it, then processes with "
            "the fit.",
        )
        state = QLabel(f"Status: {STATUS_NAMES.get(status, status)}")
        if self.theme:
            state.setStyleSheet(f"color: {self.theme.status_color(status)};")
        lay.addWidget(state)
        row = QHBoxLayout()
        row.addWidget(QLabel("Record"))
        seconds = _Number()
        seconds.setDecimals(1)
        seconds.setRange(1, 3600)
        seconds.setValue(60)
        seconds.setSuffix(" s")
        row.addWidget(seconds)
        go = QPushButton("Calibrate")
        go.setObjectName("primary")
        go.setEnabled(running)
        go.setToolTip(
            "Record this much input, fit, then run"
            if running
            else "Start the pipeline to calibrate"
        )
        go.clicked.connect(lambda: self.calibrate.emit(strip.id, seconds.value()))
        row.addWidget(go)
        row.addStretch(1)
        clear = QPushButton("Clear calibration")
        clear.clicked.connect(lambda: self.clear_calibration.emit(strip.id))
        row.addWidget(clear)
        lay.addLayout(row)
        passthrough = QCheckBox("Pass data through while not calibrated")
        passthrough.setChecked(strip.pass_uncalibrated)
        passthrough.setToolTip(
            "Off (default): nothing passes this strip until it is "
            "calibrated.\nOn: data passes unprocessed and is "
            "marked as bypassed."
        )
        passthrough.toggled.connect(
            lambda on: self.pass_uncalibrated_changed.emit(strip.id, on)
        )
        lay.addWidget(passthrough)
        return card

    def _channel_types(self, strip: StripNode) -> QWidget:
        """Which channel types this strip processes; the rest pass through it."""
        card, lay = _card(
            "Channels processed", "Other channel types pass through untouched."
        )
        chosen = strip.channel_types or ["eeg"]
        boxes: list[tuple[str, QCheckBox]] = []

        def changed() -> None:
            picked = [t for t, b in boxes if b.isChecked()]
            if not picked:  # at least one type: keep EEG
                boxes[0][1].setChecked(True)
                return
            self.channel_types_changed.emit(strip.id, picked)

        for kind in CHANNEL_TYPES:
            b = QCheckBox(CHANNEL_TYPE_NAMES[kind])
            b.setChecked(kind in chosen)
            b.toggled.connect(lambda _: changed())
            boxes.append((kind, b))
        lay.addLayout(_grid([b for _, b in boxes], columns=3))
        return card

    def _strip_picker(self, doc: PipelineDoc, strip: StripNode) -> QWidget:
        card, lay = _card(
            "Shown on the strip", f"Pick up to {MAX_STRIP_PARAMS} values."
        )
        shown = doc.shown_params(strip, advanced=True)
        boxes: list[tuple[str, QCheckBox]] = []

        def changed() -> None:
            chosen = [name for name, b in boxes if b.isChecked()]
            for _, b in boxes:
                b.setEnabled(b.isChecked() or len(chosen) < MAX_STRIP_PARAMS)
            self.strip_params_changed.emit(strip.id, chosen)

        for spec in doc.manifest(strip)["params"]:
            b = QCheckBox(param_label(spec["name"], spec))
            b.setChecked(spec["name"] in shown)
            b.toggled.connect(lambda _: changed())
            boxes.append((spec["name"], b))
        for _, b in boxes:
            b.setEnabled(b.isChecked() or len(shown) < MAX_STRIP_PARAMS)
        lay.addLayout(_grid([b for _, b in boxes]))
        return card
