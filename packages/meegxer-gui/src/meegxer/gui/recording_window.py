"""Recording window: how the input describes its channels and events.

Opened from the Input strip. Shows what the daemon reads from the recording
(``describe_source``) and edits the input's ``recording`` overrides (spec 0.2.3):
channel names, types, bad flags and positions, the montage, and events. Nothing
here changes the samples; Apply hands the overrides to the main window, which
restarts a running pipeline with them.
"""

from __future__ import annotations

import contextlib
import copy
import math
from collections.abc import Callable
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from meegxer.gui.colors import channel_style, wheel_positions
from meegxer.gui.theme import Theme

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "stim", "misc")
INHERITED = Qt.ItemDataRole.UserRole + 1  # position cell shows a montage value
COLS = ("Channel", "Label", "Type", "Bad", "x (m)", "y (m)", "z (m)")
Call = Callable[..., Any]


def _item(text: str, editable: bool = True) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


# inputs with a file to review at import (meegxer.core.import_review)
REVIEWABLE = ("meegxer.core.file", "meegxer.core.bids")


class RecordingWindow(QWidget):
    applied = Signal(dict)  # the new ``recording`` overrides
    review_requested = Signal(dict)  # the source entry, to review at import

    def __init__(self, call: Call, theme: Theme) -> None:
        super().__init__()
        self.call = call
        self.theme = theme
        self.start_dir: Callable[[], str] = lambda: ""  # where file pickers open
        self.setWindowTitle("MeegXer · recording")
        self.resize(980, 640)
        self.entry: dict[str, Any] = {}
        self.base: dict[str, Any] = {}  # as the source reports it
        self.current: dict[str, Any] = {}  # with the overrides
        self.recording: dict[str, Any] = {}
        lay = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setObjectName("muted")
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._channels_tab(), "Channels")
        self.tabs.addTab(self._montage_tab(), "Montage")
        self.tabs.addTab(self._events_tab(), "Events")
        self.tabs.addTab(self._info_tab(), "Info")
        lay.addWidget(self.tabs, 1)
        bar = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("muted")
        bar.addWidget(self.status, 1)
        self.review_btn = QPushButton("Review…")
        self.review_btn.setToolTip(
            "Review the recording itself, before any pipeline: events, labelled "
            "spans, bad channels and a crop, saved with the recording and used "
            "by every pipeline (the raw file is not changed)"
        )
        self.review_btn.clicked.connect(lambda: self.review_requested.emit(self.entry))
        bar.addWidget(self.review_btn)
        revert = QPushButton("Revert")
        revert.clicked.connect(lambda: self.load(self.entry))
        apply = QPushButton("Apply")
        apply.setObjectName("primary")
        apply.clicked.connect(self.apply)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for b in (revert, apply, close):
            bar.addWidget(b)
        lay.addLayout(bar)

    # -- tabs -------------------------------------------------------------------------

    def _channels_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        tools = QHBoxLayout()
        tools.addWidget(QLabel("Selected rows:"))
        self.bulk_type = QComboBox()
        self.bulk_type.addItems(CHANNEL_TYPES)
        set_type = QPushButton("Set type")
        set_type.clicked.connect(self._bulk_type)
        mark_bad = QPushButton("Mark bad")
        mark_bad.clicked.connect(lambda: self._bulk_bad(True))
        mark_good = QPushButton("Mark good")
        mark_good.clicked.connect(lambda: self._bulk_bad(False))
        for w in (self.bulk_type, set_type, mark_bad, mark_good):
            tools.addWidget(w)
        tools.addStretch(1)
        hint = QLabel("Positions left empty come from the montage or the file.")
        hint.setObjectName("muted")
        tools.addWidget(hint)
        lay.addLayout(tools)
        lookup = QHBoxLayout()
        lookup.addWidget(QLabel("Look up positions by channel name in"))
        self.lookup_source = QComboBox()
        self.lookup_source.setEditable(True)
        self.lookup_source.setMinimumWidth(220)
        self.lookup_source.setToolTip(
            "A standard montage, or a montage file (.elc, .sfp, .loc, .ced, .bvef, "
            "BIDS electrodes.tsv). Names like 'EEG Fp1-REF' and old ones (T3) match."
        )
        lookup.addWidget(self.lookup_source)
        browse = QPushButton("File…")
        browse.clicked.connect(self._browse_lookup)
        lookup.addWidget(browse)
        self.lookup_empty_only = QCheckBox("only channels without a position")
        self.lookup_empty_only.setChecked(True)
        lookup.addWidget(self.lookup_empty_only)
        go = QPushButton("Look up")
        go.clicked.connect(self.look_up_positions)
        lookup.addWidget(go)
        lookup.addStretch(1)
        lay.addLayout(lookup)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._cell_edited)
        lay.addWidget(self.table, 1)
        return page

    def _montage_tab(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Montage</b>"))
        self.montage = QComboBox()
        self.montage.setEditable(True)
        left.addWidget(self.montage)
        browse = QPushButton("Load montage file…")
        browse.setToolTip(
            ".elc, .sfp, .loc, .locs, .bvef, .csd, .txt, BIDS electrodes.tsv"
        )
        browse.clicked.connect(self._browse_montage)
        left.addWidget(browse)
        note = QLabel(
            "A montage gives positions to channels whose labels it knows. "
            "Positions set on the Channels tab win. Apply to preview."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        left.addWidget(note)
        left.addStretch(1)
        lay.addLayout(left, 1)
        self.head = pg.PlotWidget()
        self.head.setAspectLocked(True)
        self.head.hideAxis("left")
        self.head.hideAxis("bottom")
        self.head.setMenuEnabled(False)
        lay.addWidget(self.head, 2)
        return page

    def _events_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Events file"))
        self.events_file = QLineEdit()
        self.events_file.setPlaceholderText(
            "none: use the events stored in the recording"
        )
        row.addWidget(self.events_file, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_events)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.events_file.setText(""))
        row.addWidget(browse)
        row.addWidget(clear)
        lay.addLayout(row)
        self.events_note = QLabel("")
        self.events_note.setObjectName("muted")
        lay.addWidget(self.events_note)
        self.labels = QTableWidget(0, 4)
        self.labels.setHorizontalHeaderLabels(("Event", "Count", "Rename to", "Drop"))
        self.labels.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.labels.verticalHeader().setVisible(False)
        lay.addWidget(self.labels, 1)
        self.event_list = QTableWidget(0, 3)
        self.event_list.setHorizontalHeaderLabels(
            ("Onset (s)", "Duration (s)", "Event")
        )
        self.event_list.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.event_list.verticalHeader().setVisible(False)
        self.event_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.event_list, 1)
        return page

    def _info_tab(self) -> QWidget:
        page = QWidget()
        self.info = QFormLayout(page)
        return page

    # -- loading ----------------------------------------------------------------------

    def load(self, entry: dict[str, Any]) -> bool:
        """Describe ``entry`` (a source node) and fill the tabs. False on failure."""
        self.entry = copy.deepcopy(entry)
        self.recording = copy.deepcopy(entry.get("recording", {}))
        plain = {k: v for k, v in entry.items() if k != "recording"}
        base = self.call("describe_source", node=plain)
        current = self.call("describe_source", node=entry) if self.recording else base
        if base is None or current is None:
            self.status.setText("Could not read the recording (see the log).")
            return False
        self.base, self.current = base, current
        self.review_btn.setEnabled(entry.get("node") in REVIEWABLE)
        n = len(current["channels"])
        dur = current.get("duration_s")
        self.summary.setText(
            f"{entry['node']} · {n} channels · {current['sfreq']:g} Hz"
            + (f" · {dur:.1f} s" if dur is not None else " · live stream")
        )
        self._fill_channels()
        self._fill_montage()
        self._fill_events()
        self._fill_info()
        self.status.setText("")
        return True

    def _fill_channels(self) -> None:
        edits = self.recording.get("channels", {})
        rows = self.current["channels"]
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, ch in enumerate(rows):
            self.table.setItem(r, 0, _item(ch["original"], editable=False))
            self.table.setItem(r, 1, _item(ch["label"]))
            combo = QComboBox()
            combo.addItems(("", *CHANNEL_TYPES))
            combo.setCurrentText(ch["type"])
            self.table.setCellWidget(r, 2, combo)
            bad = QCheckBox()
            bad.setChecked(bool(ch["bad"]))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(bad, 0, Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(r, 3, holder)
            own = edits.get(ch["original"], {}).get("position")
            shown = own or ch["position"]
            for k in range(3):
                item = _item(f"{shown[k]:.4f}" if shown else "")
                if shown and not own:  # from the montage or the file: greyed
                    item.setForeground(QColor(self.theme.text_muted))
                    item.setData(INHERITED, True)
                    item.setToolTip("From the montage or the file; type to override.")
                self.table.setItem(r, 4 + k, item)

        self.table.blockSignals(False)

    def _cell_edited(self, item: QTableWidgetItem) -> None:
        """Typing into a greyed (inherited) position makes the row's position
        an explicit override."""
        if item.column() >= 4 and item.data(INHERITED):
            for k in range(3):
                cell = self.table.item(item.row(), 4 + k)
                if cell is not None:
                    cell.setData(INHERITED, False)
                    cell.setForeground(QColor(self.theme.text))

    def _fill_montage(self) -> None:
        self.montage.clear()
        names = self.call("list_montages") or []
        chosen = self.lookup_source.currentText()
        self.lookup_source.clear()
        self.lookup_source.addItems(names)
        self.lookup_source.setCurrentText(
            chosen or ("standard_1005" if "standard_1005" in names else "colin27_1005")
        )
        self.montage.addItem("")
        self.montage.addItems(names)
        self.montage.setCurrentText(self.recording.get("montage", ""))
        self._draw_head()

    def _draw_head(self) -> None:
        self.head.clear()
        t = self.theme
        self.head.setBackground(t.plot_background)
        ring = [
            (math.cos(a / 50 * math.tau), math.sin(a / 50 * math.tau))
            for a in range(51)
        ]
        self.head.plot(
            [p[0] for p in ring],
            [p[1] for p in ring],
            pen=pg.mkPen(t.border, width=1.5),
        )
        self.head.plot(
            [-0.1, 0, 0.1], [0.99, 1.12, 0.99], pen=pg.mkPen(t.border, width=1.5)
        )
        positions = {
            c["label"]: tuple(c["position"])
            for c in self.current["channels"]
            if c["position"]
        }
        wheel = wheel_positions(positions)
        xs, ys, brushes = [], [], []
        for ch in self.current["channels"]:
            pos = wheel.get(ch["label"])
            if pos is None:
                continue
            az, r = pos
            x, y = r * math.cos(math.radians(az)), r * math.sin(math.radians(az))
            xs.append(x)
            ys.append(y)
            brushes.append(pg.mkBrush(*channel_style(ch["label"], t, wheel).color))
            label = pg.TextItem(ch["label"], color=t.text_muted, anchor=(0.5, -0.3))
            label.setPos(x, y)
            self.head.addItem(label)
        self.head.addItem(
            pg.ScatterPlotItem(xs, ys, size=14, brush=brushes, pen=pg.mkPen(None))
        )
        missing = len(self.current["channels"]) - len(xs)
        self.head.setTitle(
            f"{len(xs)} positioned, {missing} without a position"
            if missing
            else f"{len(xs)} channels positioned"
        )

    def _fill_events(self) -> None:
        events = self.recording.get("events", {})
        self.events_file.setText(events.get("file", ""))
        origin = self.current.get("events_origin", "none")
        shown = self.current["events"]
        self.events_note.setText(
            {
                "file": "Events come from the events file.",
                "recording": "Events stored in the recording.",
                "import review": "Events of the review at import (Review…).",
                "none": "This input has no events.",
            }[origin]
            + f" {len(shown)} shown after renaming and dropping."
        )
        counts: dict[str, int] = {}
        # count from the unrenamed list: describe with renames removed is not needed,
        # the labels column lists the original event names
        for label in self.current.get("event_labels", []):
            counts[label] = 0
        rename = events.get("rename", {})
        drop = set(events.get("drop", []))
        inverse = {v: k for k, v in rename.items()}
        for e in shown:
            original = inverse.get(e["label"], e["label"])
            counts[original] = counts.get(original, 0) + 1
        self.labels.setRowCount(len(counts))
        for r, (label, count) in enumerate(sorted(counts.items())):
            self.labels.setItem(r, 0, _item(label, editable=False))
            self.labels.setItem(
                r, 1, _item("dropped" if label in drop else str(count), editable=False)
            )
            self.labels.setItem(r, 2, _item(rename.get(label, "")))
            box = QCheckBox()
            box.setChecked(label in drop)
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(box, 0, Qt.AlignmentFlag.AlignCenter)
            self.labels.setCellWidget(r, 3, holder)
        self.event_list.setRowCount(min(len(shown), 2000))
        for r, e in enumerate(shown[:2000]):
            self.event_list.setItem(r, 0, _item(f"{e['onset']:.3f}"))
            self.event_list.setItem(r, 1, _item(f"{e['duration']:.3f}"))
            self.event_list.setItem(r, 2, _item(str(e["label"])))

    def _fill_info(self) -> None:
        while self.info.rowCount():
            self.info.removeRow(0)
        c = self.current
        dur = c.get("duration_s")
        rows = [
            ("Input", self.entry.get("node", "")),
            ("Stream", c.get("stream_id", "")),
            ("Sampling rate", f"{c['sfreq']:g} Hz (from the recording)"),
            ("Channels", str(len(c["channels"]))),
            ("Samples", "-" if c.get("n_samples") is None else str(c["n_samples"])),
            ("Duration", "live stream" if dur is None else f"{dur:.2f} s"),
        ]
        for key, value in rows:
            self.info.addRow(key, QLabel(value))

    # -- editing ----------------------------------------------------------------------

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _bulk_type(self) -> None:
        for r in self._selected_rows():
            combo = self.table.cellWidget(r, 2)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(self.bulk_type.currentText())

    def _bad_box(self, table: QTableWidget, r: int, col: int) -> QCheckBox | None:
        holder = table.cellWidget(r, col)
        return holder.findChild(QCheckBox) if holder is not None else None

    def _bulk_bad(self, bad: bool) -> None:
        for r in self._selected_rows():
            box = self._bad_box(self.table, r, 3)
            if box is not None:
                box.setChecked(bad)

    def _browse_lookup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Channel locations file",
                                              self.start_dir(),
                                              "Locations (*)")  # fmt: skip
        if path:
            self.lookup_source.setCurrentText(path)

    def look_up_positions(self) -> None:
        """Fill x, y, z from the chosen montage or file by channel name (like
        EEGLAB's channel-location lookup). Filled positions become this
        recording's own; Apply keeps them."""
        source = self.lookup_source.currentText().strip()
        if not source:
            self.status.setText("Choose a montage or a locations file to look up.")
            return
        rows: dict[str, int] = {}
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 1)
            cells = [self.table.item(r, 4 + k) for k in range(3)]
            has_pos = all(c is not None and c.text().strip() for c in cells)
            if item is not None and not (
                self.lookup_empty_only.isChecked() and has_pos
            ):
                rows[item.text().strip()] = r
        if not rows:
            self.status.setText("Every channel already has a position.")
            return
        reply = self.call("lookup_positions", labels=list(rows), montage=source)
        if reply is None:
            self.status.setText("Lookup failed (see the log).")
            return
        self.table.blockSignals(True)
        for label, xyz in reply["positions"].items():
            r = rows[label]
            for k in range(3):
                item = _item(f"{xyz[k]:.4f}")
                item.setData(INHERITED, False)
                self.table.setItem(r, 4 + k, item)
        self.table.blockSignals(False)
        missing = reply["unmatched"]
        self.status.setText(
            f"Found {len(reply['positions'])} of {len(rows)} channels in {source}."
            + (f" No match: {', '.join(missing[:12])}"
               + ("…" if len(missing) > 12 else "") if missing else "")
            + " Apply to keep them."
        )  # fmt: skip

    def _browse_montage(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Montage file", self.start_dir(), "Montage (*)"
        )
        if path:
            self.montage.setCurrentText(path)

    def _browse_events(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Events file", self.start_dir(), "Events (*.tsv *.csv *.txt)"
        )
        if path:
            self.events_file.setText(path)

    def overrides(self) -> dict[str, Any]:
        """The overrides the tabs describe, relative to what the source reports."""
        base = {c["original"]: c for c in self.base["channels"]}
        channels: dict[str, Any] = {}
        for r in range(self.table.rowCount()):
            orig_item = self.table.item(r, 0)
            if orig_item is None:
                continue
            original = orig_item.text()
            src = base.get(original, {})
            edit: dict[str, Any] = {}
            label_item = self.table.item(r, 1)
            label = label_item.text().strip() if label_item is not None else original
            if label and label != original:
                edit["rename"] = label
            combo = self.table.cellWidget(r, 2)
            if (
                isinstance(combo, QComboBox)
                and combo.currentText()
                and combo.currentText() != src.get("type", "")
            ):
                edit["type"] = combo.currentText()
            box = self._bad_box(self.table, r, 3)
            if box is not None and box.isChecked() != bool(src.get("bad")):
                edit["bad"] = box.isChecked()
            cells = [self.table.item(r, 4 + k) for k in range(3)]
            explicit = [
                c
                for c in cells
                if c is not None and not c.data(INHERITED) and c.text().strip()
            ]
            if len(explicit) == 3:
                with contextlib.suppress(ValueError):  # leave half-typed rows out
                    edit["position"] = [float(c.text()) for c in explicit]
            if edit:
                channels[original] = edit
        out: dict[str, Any] = {}
        if channels:
            out["channels"] = channels
        if self.montage.currentText().strip():
            out["montage"] = self.montage.currentText().strip()
        events: dict[str, Any] = {}
        if self.events_file.text().strip():
            events["file"] = self.events_file.text().strip()
        rename, drop = {}, []
        for r in range(self.labels.rowCount()):
            name_item, to_item = self.labels.item(r, 0), self.labels.item(r, 2)
            if name_item is None:
                continue
            if to_item is not None and to_item.text().strip():
                rename[name_item.text()] = to_item.text().strip()
            box = self._bad_box(self.labels, r, 3)
            if box is not None and box.isChecked():
                drop.append(name_item.text())
        if rename:
            events["rename"] = rename
        if drop:
            events["drop"] = drop
        if events:
            out["events"] = events
        return out

    def apply(self) -> None:
        recording = self.overrides()
        entry = {**{k: v for k, v in self.entry.items() if k != "recording"}}
        if recording:
            entry["recording"] = recording
        if self.call("describe_source", node=entry) is None:
            self.status.setText(
                "Not applied: the daemon refused these settings (see the log)."
            )
            return
        self.load(entry)  # refresh first: listeners read ``current``
        self.applied.emit(recording)
        self.status.setText("Applied.")
