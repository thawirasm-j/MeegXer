"""Inspector: every parameter of the selected strip, generated from its manifest.

Laid out as cards: a header (title, group, what the strip does), the settings
(main ones first, on/off switches in a two-column grid), the advanced settings
(folded away in friendly mode), then calibration, the channel types the strip
processes and the values shown on the strip.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from meegxer.gui.labels import (
    SAVE_ENDINGS,
    SHOWN_IN_MS,
    choice_label,
    param_label,
    unit_label,
)
from meegxer.gui.names import title
from meegxer.gui.pipeline_doc import MAX_STRIP_PARAMS, PipelineDoc, StripNode
from meegxer.gui.theme import DEFAULT_GROUPS, Theme

# greyed text in an empty field: what leaving it empty means
PLACEHOLDERS = {("meegxer.core.bids", "subject"): "All subjects"}

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "misc")
CHANNEL_TYPE_NAMES = {
    "eeg": "EEG",
    "eog": "Eye (EOG)",
    "emg": "Muscle (EMG)",
    "ecg": "Heart (ECG)",
    "resp": "Breathing",
    "misc": "Other",
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
        text = f"{value:.{self.decimals()}f}"
        # only decimals lose their trailing zeros: 200 ms stays 200, not 2
        return text.rstrip("0").rstrip(".") if "." in text else text


class PathEdit(QWidget):
    """A path parameter (manifest ``path``, spec 0.2.20): its text box and a
    Browse… button that opens the matching dialog (a file to read, a folder, or
    a file to write). ``finished`` reports a final value, typed or browsed."""

    finished = Signal()

    def __init__(
        self, value: str, kind: str, extensions: list[str], ending: str = ""
    ) -> None:
        super().__init__()
        self.kind, self.extensions, self.ending = kind, extensions, ending
        self.line = QLineEdit(value)
        self.button = QPushButton("Browse…")
        self.button.setToolTip({"file": "Choose a file", "folder": "Choose a folder",
                                "save": "Choose where to save"}[kind])  # fmt: skip
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.line, 1)
        lay.addWidget(self.button)
        self.line.editingFinished.connect(self.finished)
        self.button.clicked.connect(self.browse)

    def text(self) -> str:
        return self.line.text()

    def _start(self) -> str:
        current = Path(self.line.text()) if self.line.text() else None
        if current is None:
            return ""
        return str(current if self.kind == "folder" or current.is_dir()
                   else current.parent)  # fmt: skip

    def complete(self, name: str) -> str:
        """A name to save as, with the extension the writer takes, or the
        ending it insists on (``ending``: clean, clean.fif -> clean-ica.fif;
        another extension it takes, such as .set, is kept)."""
        suffix = Path(name).suffix.lower()
        if self.ending:
            tail = Path(self.ending).suffix.lower()  # .fif
            if name.lower().endswith(self.ending.lower()):
                return name
            if (
                suffix
                and suffix != tail
                and suffix in {e.lower() for e in self.extensions}
            ):
                return name
            return (name[: -len(tail)] if suffix == tail else name) + self.ending
        if self.extensions and suffix not in {e.lower() for e in self.extensions}:
            return name + self.extensions[0]
        return name

    def _filter(self) -> str:
        if not self.extensions:
            return "All files (*)"
        patterns = " ".join(f"*{e}" for e in self.extensions)
        return f"Supported files ({patterns});;All files (*)"

    def browse(self) -> None:
        if self.kind == "folder":
            chosen = QFileDialog.getExistingDirectory(self, "Choose a folder",
                                                      self._start())  # fmt: skip
        elif self.kind == "save":
            chosen, _ = QFileDialog.getSaveFileName(self, "Save as", self._start(),
                                                    self._filter())  # fmt: skip
            if chosen:
                chosen = self.complete(chosen)
        else:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Choose a file", self._start(), self._filter()
            )
        if chosen:
            self.line.setText(str(Path(chosen)))
            self.finished.emit()


def _nice(x: float) -> float:
    """The nearest of 1, 2 or 5 times a power of ten (0.1, 0.2, 0.5, 1, 2, ...)."""
    exponent = math.floor(math.log10(x))
    mantissa = x / 10**exponent
    nice = min((1, 2, 5, 10), key=lambda m: abs(math.log10(m / mantissa)))
    return float(nice * 10.0**exponent)


def param_step(spec: dict[str, Any], value: Any) -> float:
    """How much one step (arrow key, wheel) changes a number: the manifest's
    ``step`` if given, else about a tenth of the value (rounded to 1, 2 or 5 x 10^n),
    no more than a tenth of the allowed range (a 0-1 threshold steps by 0.1)."""
    if spec.get("step"):
        return float(spec["step"])
    lo, hi = spec.get("min"), spec.get("max")
    span = hi - lo if lo is not None and hi is not None and hi > lo else None
    ref = abs(float(value or spec.get("default") or 0))
    if ref > 0:
        step = _nice(ref / 10)
    elif span:
        step = _nice(span / 10)
    else:
        step = 0.1
    if span:
        step = min(step, _nice(span / 10))
    if spec["type"] == "int":
        return float(max(1, round(step)) if ref >= 100 else 1)
    return step


def _editor(
    spec: dict[str, Any], value: Any, ms_step: float | None = None, ending: str = ""
) -> tuple[QWidget, Any]:
    """A control for one parameter and the signal that reports a final value.
    With ``ms_step`` a parameter in seconds is shown and edited in ms (the
    field keeps its ``scale``; ``_value`` turns it back into seconds). With
    ``ending`` a file to save is named with it (``-ica.fif``)."""
    kind = spec["type"]
    if kind == "float" and ms_step and spec.get("unit") == "s":
        ms = _Number()
        ms.setDecimals(0)
        ms.setRange(
            spec["min"] * 1e3 if spec["min"] is not None else -1e12,
            spec["max"] * 1e3 if spec["max"] is not None else 1e12,
        )
        ms.setValue(float(value) * 1e3)
        ms.setSingleStep(ms_step)
        ms.setSuffix(" ms")
        ms.setProperty("scale", 1e3)
        ms.setKeyboardTracking(False)
        return ms, ms.valueChanged
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
    if kind == "string" and spec.get("path"):
        edit = PathEdit(str(value), spec["path"], list(spec.get("extensions") or []),
                        ending)  # fmt: skip
        return edit, edit.finished
    if kind == "string":
        line = QLineEdit(str(value))
        # wide enough for what it holds (event labels such as
        # stimulus/compatible/target_left, comma-separated)
        room = line.fontMetrics().horizontalAdvance(str(value)) + 24
        line.setMinimumWidth(min(max(line.sizeHint().width(), room, 220), 480))
        return line, line.editingFinished
    spin: QSpinBox | QDoubleSpinBox
    if kind == "int":
        spin = QSpinBox()
        spin.setRange(
            int(spec["min"] if spec["min"] is not None else -(10**9)),
            int(spec["max"] if spec["max"] is not None else 10**9),
        )
        spin.setValue(int(value))
        spin.setSingleStep(int(param_step(spec, value)))
    else:
        spin = _Number()
        spin.setDecimals(3)
        spin.setRange(
            spec["min"] if spec["min"] is not None else -1e12,
            spec["max"] if spec["max"] is not None else 1e12,
        )
        spin.setValue(float(value))
        step = param_step(spec, value)
        spin.setSingleStep(step)
        spin.setDecimals(max(3, -math.floor(math.log10(step)) + 1))
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
    if isinstance(widget, QLineEdit | PathEdit):
        return widget.text()
    if isinstance(widget, QSpinBox | QDoubleSpinBox):
        scale = widget.property("scale")  # shown in ms, kept in seconds
        return widget.value() / scale if scale else widget.value()
    raise TypeError(widget)


def _reference_text(ref: dict[str, str]) -> str:
    if ref.get("doi"):
        return f"{ref['citation']} https://doi.org/{ref['doi']}"
    if ref.get("url"):
        return f"{ref['citation']} {ref['url']}"
    return ref["citation"]


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
    input_changed = Signal(str, str)  # strip id, the strip it reads ("" = nearest)
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
        self._shown: str | None = None  # the strip on show, to keep the scroll
        self.show_empty()

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme

    def show_empty(self) -> None:
        self._shown = None
        hint = QLabel("Select a strip to see all of its settings.")
        hint.setObjectName("muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._replace(hint)

    def _replace(self, page: QWidget) -> None:
        # Editing a value rebuilds the inspector while Qt is still delivering that
        # event (wheel, Enter) to the old editor: setWidget would delete the editor
        # under it, so delete the old page later instead. takeWidget leaves it
        # without a parent, a top-level window (on Windows a visible one flashes
        # up): hide it first and keep it a hidden child until it is deleted.
        old = self.widget()
        if old is not None:
            old.hide()
        old = self.takeWidget()
        self.setWidget(page)
        if old is not None:
            old.setParent(self)
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
        # the same strip again (a value changed): stay where the user was
        keep = self.verticalScrollBar().value() if strip.id == self._shown else 0
        self._shown = strip.id
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

        if manifest["inputs"] and (manifest["inputs"][0]["type"] == "features"
                                   or doc.rail_choices(strip)):  # fmt: skip
            lay.addWidget(self._reads(doc, strip))
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
        if manifest.get("authors") or manifest.get("references"):
            lay.addWidget(self._credits(manifest))
        if advanced and is_processor:
            lay.addWidget(self._channel_types(strip, doc.processed_types(strip)))
        if advanced and manifest["params"]:
            lay.addWidget(self._strip_picker(doc, strip))
        lay.addStretch(1)
        self._replace(page)
        if keep:
            page.adjustSize()  # lay the new page out now, so the range is known
            self.verticalScrollBar().setValue(keep)

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
        sub = doc.group_label(strip)
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
            key = (strip.node, name)
            editor, changed = _editor(spec, strip.params.get(name, spec["default"]),
                                      SHOWN_IN_MS.get(key),
                                      SAVE_ENDINGS.get(key, ""))  # fmt: skip
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
                hint = PLACEHOLDERS.get((strip.node, name))
                if hint and isinstance(editor, QLineEdit):
                    editor.setPlaceholderText(hint)
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

    def _credits(self, manifest: dict[str, Any]) -> QWidget:
        """Who made the strip and what to cite when using it."""
        card, lay = _card("Credits")
        refs = manifest.get("references") or []
        lines = []
        if manifest.get("authors"):
            lines.append("By " + html.escape(", ".join(manifest["authors"])))
        if manifest.get("license"):
            lines.append("Licence: " + html.escape(manifest["license"]))
        if lines:
            who = QLabel("<br>".join(lines))
            who.setWordWrap(True)
            lay.addWidget(who)
        if refs:
            heading = QLabel("Please cite:")
            heading.setObjectName("muted")
            lay.addWidget(heading)
            colour = f" style='color:{self.theme.accent}'" if self.theme else ""
            items = []
            for ref in refs:
                text = html.escape(ref["citation"])
                if ref.get("doi"):
                    link, shown = f"https://doi.org/{ref['doi']}", f"doi:{ref['doi']}"
                elif ref.get("url"):
                    link = shown = ref["url"]
                else:
                    link = shown = ""
                if link:
                    href = html.escape(link)
                    text += f' <a href="{href}"{colour}>{html.escape(shown)}</a>'
                items.append(f"<li>{text}</li>")
            listing = QLabel(
                "<ol style='margin-left:-20px'>" + "".join(items) + "</ol>"
            )
            listing.setWordWrap(True)
            listing.setOpenExternalLinks(True)
            listing.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction
            )
            lay.addWidget(listing)
            copy = QPushButton("Copy references")
            copy.setToolTip("Copy the references as text, for a methods section")
            copy.clicked.connect(
                lambda: QApplication.clipboard().setText(
                    "\n".join(_reference_text(r) for r in refs)
                )
            )
            row = QHBoxLayout()
            row.addWidget(copy)
            row.addStretch(1)
            lay.addLayout(row)
        return card

    def _channel_types(self, strip: StripNode, chosen: list[str]) -> QWidget:
        """Which channel types this strip processes; the rest pass through it."""
        card, lay = _card(
            "Channels processed", "Other channel types pass through untouched."
        )
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

    def _reads(self, doc: PipelineDoc, strip: StripNode) -> QWidget:
        """The line this strip draws from (docs/RAILS.md): a feature rail, or for
        a strip on the signal the signal or a rail output such as the ICs."""
        on_signal = doc.manifest(strip)["inputs"][0]["type"] == "signal"
        what = "line" if on_signal else "feature rail"
        card, lay = _card("Reads", f"The {what} this strip draws from.")
        choices = doc.rail_choices(strip) if on_signal else doc.input_choices(strip)
        box = QComboBox()
        box.setObjectName("reads")
        if not choices:
            box.addItem("No feature rail before this strip", "")
            box.setEnabled(False)
        else:
            if on_signal:
                box.addItem("The signal", "")
            else:
                box.addItem(f"Nearest ({doc.choice_label(choices[0])})", "")
            for sid in choices:
                box.addItem(doc.choice_label(sid), sid)
            box.setCurrentIndex(max(0, box.findData(strip.input)) if strip.input else 0)
        box.currentIndexChanged.connect(
            lambda _: self.input_changed.emit(strip.id, str(box.currentData()))
        )
        lay.addWidget(box)
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
