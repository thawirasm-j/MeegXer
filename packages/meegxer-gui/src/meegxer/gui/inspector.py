"""Inspector: every parameter of the selected strip, generated from its manifest."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.names import title
from meegxer.gui.pipeline_doc import MAX_STRIP_PARAMS, PipelineDoc, StripNode

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "misc")


def _editor(spec: dict[str, Any], value: Any) -> tuple[QWidget, Any]:
    """A control for one parameter and the signal that reports a final value."""
    kind = spec["type"]
    if kind == "bool":
        box = QCheckBox()
        box.setChecked(bool(value))
        return box, box.toggled
    if kind == "enum":
        combo = QComboBox()
        combo.addItems(list(spec["choices"]))
        combo.setCurrentText(str(value))
        return combo, combo.currentTextChanged
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
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(
            spec["min"] if spec["min"] is not None else -1e12,
            spec["max"] if spec["max"] is not None else 1e12,
        )
        spin.setValue(float(value))
    if spec.get("unit"):
        spin.setSuffix(f" {spec['unit']}")
    spin.setKeyboardTracking(False)  # report when typing is done
    return spin, spin.valueChanged


def _value(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QSpinBox | QDoubleSpinBox):
        return widget.value()
    raise TypeError(widget)


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
        self.show_empty()

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
        manifest = doc.manifest(strip)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(12, 10, 12, 10)
        head = QLabel(
            f"<b>{title(strip.node, strip.params)}</b> &nbsp; "
            f"<span style='opacity:.7'>{strip.id} · {strip.node} "
            f"{manifest['version']}</span>"
        )
        lay.addWidget(head)
        if manifest.get("description"):
            desc = QLabel(manifest["description"])
            desc.setObjectName("muted")
            desc.setWordWrap(True)
            lay.addWidget(desc)
        live = set(manifest.get("live_params", []))
        is_processor = manifest["role"] == "processor"
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        hidden = 0
        for spec in manifest["params"]:
            if spec.get("advanced") and not advanced:
                hidden += 1
                continue
            editor, changed = _editor(
                spec, strip.params.get(spec["name"], spec["default"])
            )
            name = spec["name"]
            changed.connect(
                lambda *_, n=name, w=editor: self.param_changed.emit(
                    strip.id, n, _value(w)
                )
            )
            tips = [spec.get("description") or ""]
            if running and not is_processor:
                tips.append("Changing this restarts the pipeline.")
            elif running and name not in live:
                tips.append("Changing this restarts the node's filter state.")
            editor.setToolTip("\n".join(t for t in tips if t))
            label = name.replace("_", " ") + (" ★" if spec.get("primary") else "")
            form.addRow(label, editor)
        lay.addLayout(form)
        if hidden:
            note = QLabel(f"{hidden} advanced setting(s) hidden in friendly mode.")
            note.setObjectName("muted")
            lay.addWidget(note)

        if manifest.get("requires_fit"):
            lay.addWidget(self._calibration(strip, node_status, running))
        if advanced and is_processor:
            lay.addWidget(self._channel_types(strip))
        if advanced and manifest["params"]:
            lay.addWidget(self._strip_picker(doc, strip))
        lay.addStretch(1)
        self._replace(page)

    def _calibration(
        self, strip: StripNode, node_status: dict[str, Any], running: bool
    ) -> QWidget:
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        status = node_status.get("status", "stopped")
        lay.addWidget(QLabel(f"<b>Calibration</b> &nbsp; {status}"))
        row = QHBoxLayout()
        seconds = QDoubleSpinBox()
        seconds.setRange(1, 3600)
        seconds.setValue(60)
        seconds.setSuffix(" s")
        go = QPushButton("Calibrate")
        go.setEnabled(running)
        go.setToolTip(
            "Record this much input, fit, then run"
            if running
            else "Start the pipeline to calibrate"
        )
        go.clicked.connect(lambda: self.calibrate.emit(strip.id, seconds.value()))
        clear = QPushButton("Clear calibration")
        clear.clicked.connect(lambda: self.clear_calibration.emit(strip.id))
        row.addWidget(seconds)
        row.addWidget(go)
        row.addWidget(clear)
        lay.addLayout(row)
        passthrough = QCheckBox("Pass data through while uncalibrated")
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
        return box

    def _channel_types(self, strip: StripNode) -> QWidget:
        """Which channel types this strip processes; the rest pass through it."""
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(
            QLabel("<b>Processes channels</b> (others pass through untouched)")
        )
        chosen = strip.channel_types or ["eeg"]
        row = QHBoxLayout()
        boxes: list[tuple[str, QCheckBox]] = []

        def changed() -> None:
            picked = [t for t, b in boxes if b.isChecked()]
            if not picked:  # at least one type: keep EEG
                boxes[0][1].setChecked(True)
                return
            self.channel_types_changed.emit(strip.id, picked)

        for kind in CHANNEL_TYPES:
            b = QCheckBox(kind.upper())
            b.setChecked(kind in chosen)
            b.toggled.connect(lambda _: changed())
            boxes.append((kind, b))
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        return box

    def _strip_picker(self, doc: PipelineDoc, strip: StripNode) -> QWidget:
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(QLabel(f"<b>Show on strip</b> (up to {MAX_STRIP_PARAMS})"))
        shown = doc.shown_params(strip, advanced=True)
        boxes: list[tuple[str, QCheckBox]] = []

        def changed() -> None:
            chosen = [name for name, b in boxes if b.isChecked()]
            for _, b in boxes:
                b.setEnabled(b.isChecked() or len(chosen) < MAX_STRIP_PARAMS)
            self.strip_params_changed.emit(strip.id, chosen)

        grid = QHBoxLayout()
        for spec in doc.manifest(strip)["params"]:
            b = QCheckBox(spec["name"].replace("_", " "))
            b.setChecked(spec["name"] in shown)
            b.toggled.connect(lambda _: changed())
            boxes.append((spec["name"], b))
            grid.addWidget(b)
        grid.addStretch(1)
        for _, b in boxes:
            b.setEnabled(b.isChecked() or len(shown) < MAX_STRIP_PARAMS)
        lay.addLayout(grid)
        return box
