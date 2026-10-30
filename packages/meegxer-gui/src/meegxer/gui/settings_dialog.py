"""Settings dialog: General, Appearance, Visualization, Data & storage, Plugins &
privacy (``meegxer.core.settings``; docs/SETTINGS.md).

Every change previews at once (``changed`` carries the settings being edited);
OK keeps them, Cancel goes back to what they were. "Reset to defaults" restores
the defaults in the dialog; "Forget remembered view" clears what MeegXer
remembered between launches (``forget_requested``). The plugin list is read with
``meegxer-plugin list --json`` in its own process: the GUI never imports plugin
code (rule 9).
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from meegxer.core import settings as core_settings
from meegxer.core.montage import MONTAGES
from meegxer.core.notices import LOCAL_CODE_NOTICE
from meegxer.gui.colormaps import DIVERGING, SEQUENTIAL, Maps
from meegxer.gui.headplot import BANDS, RANGES, SCHEMES, SYMBOLS
from meegxer.gui.theme import Theme, load_theme

PAGES = (
    "General",
    "Appearance",
    "Visualization",
    "Review",
    "Data & storage",
    "Plugins & privacy",
)
THEMES = {"dark": "Dark", "light": "Light", "system": "Follow the system"}
SCALES = (90, 100, 110, 125, 150)
SPANS = (2, 5, 10, 20, 30)
SMOOTHING = (0, 1, 2, 5, 10)

# runs ``meegxer-plugin ARGS``: (exit code, stdout, stderr)
PluginCli = Callable[[list[str]], tuple[int, str, str]]


def plugin_cli(args: list[str]) -> tuple[int, str, str]:
    done = subprocess.run(
        [sys.executable, "-m", "meegxer.plugins.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done.returncode, done.stdout, done.stderr


def _ramp_icon(colors: np.ndarray) -> QIcon:
    """A small gradient swatch of a colour map, for its menu entry."""
    strip = np.ascontiguousarray(np.repeat(colors[None, :, :], 12, axis=0))
    image = QImage(
        strip.tobytes(),
        strip.shape[1],
        12,
        3 * strip.shape[1],
        QImage.Format.Format_RGB888,
    )
    return QIcon(QPixmap.fromImage(image.copy()))


class SettingsDialog(QDialog):
    changed = Signal(dict)  # the settings as edited (preview)
    forget_requested = Signal()  # clear remembered view state
    add_script_requested = Signal()

    def __init__(
        self,
        current: dict[str, Any],
        theme: Theme,
        parent: QWidget | None = None,
        plugins: PluginCli = plugin_cli,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(720, 520)
        self.theme = theme
        self.original = copy.deepcopy(current)
        self.working = copy.deepcopy(current)
        self.plugins = plugins
        self._loaders: list[Callable[[], None]] = []
        self._loading = False

        lay = QVBoxLayout(self)
        body = QHBoxLayout()
        self.pages = QListWidget()
        self.pages.setFixedWidth(170)
        self.stack = QStackedWidget()
        for name, build in zip(
            PAGES,
            (
                self._general,
                self._appearance,
                self._visualization,
                self._review,
                self._storage,
                self._privacy,
            ),
            strict=True,
        ):
            self.pages.addItem(name)
            page = QWidget()
            build(page)
            self.stack.addWidget(page)
        self.pages.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.pages.setCurrentRow(0)
        body.addWidget(self.pages)
        body.addWidget(self.stack, 1)
        lay.addLayout(body, 1)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset to defaults")
        reset.setToolTip("Put every setting back to its default (OK keeps it)")
        reset.clicked.connect(self.reset_defaults)
        forget = QPushButton("Forget remembered view")
        forget.setToolTip(
            "Forget the window size, last band, span and folders "
            "MeegXer remembered between launches"
        )
        forget.clicked.connect(self._forget)
        buttons.addWidget(reset)
        buttons.addWidget(forget)
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.setObjectName("primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        lay.addLayout(buttons)
        self._load()

    # -- the edited settings ----------------------------------------------------

    def settings(self) -> dict[str, Any]:
        return copy.deepcopy(self.working)

    def _set(self, key: str, value: Any) -> None:
        if self._loading:
            return
        core_settings.put(self.working, key, value)
        self.changed.emit(self.settings())

    def _load(self) -> None:
        self._loading = True
        try:
            for load in self._loaders:
                load()
        finally:
            self._loading = False

    def reset_defaults(self) -> None:
        self.working = core_settings.defaults()
        self._load()
        self.changed.emit(self.settings())

    def reject(self) -> None:
        self.changed.emit(copy.deepcopy(self.original))
        super().reject()

    def _forget(self) -> None:
        self.forget_requested.emit()
        self.forget_note.setText("Remembered view forgotten.")

    # -- controls bound to a setting --------------------------------------------

    def _combo(self, key: str, items: dict[Any, str], tip: str = "") -> QComboBox:
        box = QComboBox()
        for value, text in items.items():
            box.addItem(text, value)
        box.setToolTip(tip)

        def load() -> None:
            i = box.findData(core_settings.get(self.working, key))
            box.setCurrentIndex(max(i, 0))

        box.currentIndexChanged.connect(lambda _: self._set(key, box.currentData()))
        self._loaders.append(load)
        return box

    def _check(self, key: str, text: str, tip: str = "") -> QCheckBox:
        box = QCheckBox(text)
        box.setToolTip(tip)
        box.toggled.connect(lambda on: self._set(key, bool(on)))
        self._loaders.append(
            lambda: box.setChecked(bool(core_settings.get(self.working, key)))
        )
        return box

    def _folder(self, key: str, placeholder: str, title: str) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(lambda: self._set(key, edit.text().strip()))
        browse = QPushButton("Browse…")

        def pick() -> None:
            start = edit.text() or os.getcwd()
            path = QFileDialog.getExistingDirectory(self, title, start)
            if path:
                edit.setText(path)
                self._set(key, path)

        browse.clicked.connect(pick)
        lay.addWidget(edit, 1)
        lay.addWidget(browse)
        self._loaders.append(
            lambda: edit.setText(str(core_settings.get(self.working, key)))
        )
        return row

    @staticmethod
    def _form(page: QWidget, intro: str) -> QFormLayout:
        outer = QVBoxLayout(page)
        note = QLabel(intro)
        note.setObjectName("muted")
        note.setWordWrap(True)
        outer.addWidget(note)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)
        return form

    # -- pages ------------------------------------------------------------------

    def _general(self, page: QWidget) -> None:
        form = self._form(page, "How MeegXer starts and stops.")
        form.addRow(
            "Start in",
            self._combo(
                "general.start_mode",
                {"live": "Live", "offline": "Offline"},
                "The input mode a new pipeline starts in",
            ),
        )
        form.addRow(
            "",
            self._check(
                "general.advanced", "Start in advanced mode (every setting shown)"
            ),
        )
        form.addRow(
            "",
            self._check(
                "general.reopen_last",
                "Reopen the last pipeline",
                "When the engine has no pipeline loaded",
            ),
        )
        form.addRow(
            "",
            self._check(
                "general.confirm_quit_running",
                "Ask before closing while a pipeline runs",
            ),
        )
        jobs = QSpinBox()
        jobs.setRange(0, 256)
        jobs.setSpecialValueText("One per CPU")
        jobs.setToolTip(
            "Recordings an all-subjects BIDS run processes at once, each in "
            "its own process (1: one after another)"
        )
        jobs.valueChanged.connect(lambda v: self._set("general.batch_jobs", int(v)))
        self._loaders.append(
            lambda: jobs.setValue(int(core_settings.get(self.working,
                                                        "general.batch_jobs")))
        )  # fmt: skip
        form.addRow("Workers (all subjects)", jobs)
        daemon = QLineEdit()
        daemon.setToolTip(
            "The processing engine's control address; MeegXer "
            "starts one there when none answers"
        )
        self.daemon_note = QLabel("")
        self.daemon_note.setObjectName("muted")

        def edited() -> None:
            text = daemon.text().strip()
            if text.split("://")[0] in ("tcp", "ipc", "inproc") and "://" in text:
                self.daemon_note.setText("Used the next time MeegXer starts.")
                self._set("general.daemon", text)
            else:
                self.daemon_note.setText("Use an address like tcp://127.0.0.1:5555")

        daemon.editingFinished.connect(edited)
        self._loaders.append(
            lambda: daemon.setText(core_settings.get(self.working, "general.daemon"))
        )
        form.addRow("Engine address", daemon)
        form.addRow("", self.daemon_note)

    def _appearance(self, page: QWidget) -> None:
        form = self._form(page, "Theme and text size.")
        theme = QComboBox()
        for value, text in THEMES.items():
            theme.addItem(text, value)
        theme.addItem("Custom theme file…", "custom")

        def load() -> None:
            name = core_settings.get(self.working, "appearance.theme")
            i = theme.findData(name)
            if i < 0:  # a file
                theme.setItemText(
                    theme.count() - 1, f"Custom: {os.path.basename(name)}"
                )
                i = theme.count() - 1
            theme.setCurrentIndex(i)

        def chosen(index: int) -> None:
            value = theme.currentData()
            if value == "custom":
                if self._loading:
                    return
                path, _ = QFileDialog.getOpenFileName(
                    self, "Theme file", "", "MeegXer theme (*.json)"
                )
                try:
                    if not path:
                        raise ValueError("no file chosen")
                    load_theme(path)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    if path:
                        QMessageBox.warning(self, "Theme", f"Not a theme file:\n{exc}")
                    self._load()  # back to the current choice
                    return
                theme.setItemText(
                    theme.count() - 1, f"Custom: {os.path.basename(path)}"
                )
                self._set("appearance.theme", path)
            else:
                self._set("appearance.theme", value)

        theme.currentIndexChanged.connect(chosen)
        self._loaders.append(load)
        form.addRow("Theme", theme)
        form.addRow(
            "Text size", self._combo("appearance.scale", {s: f"{s} %" for s in SCALES})
        )

    def _number(self, key: str, unit: str, top: float, tip: str = "",
                off: bool = False) -> QDoubleSpinBox:  # fmt: skip
        spin = QDoubleSpinBox()
        spin.setRange(0.0 if off else 0.1, top)
        spin.setDecimals(1)
        spin.setSuffix(f" {unit}")
        if off:
            spin.setSpecialValueText("off")
        spin.setToolTip(tip)
        spin.valueChanged.connect(lambda v: self._set(key, float(v)))
        self._loaders.append(
            lambda: spin.setValue(float(core_settings.get(self.working, key)))
        )
        return spin

    def _review(self, page: QWidget) -> None:
        form = self._form(
            page,
            "How the review window shows a recording. A pipeline's Review "
            "strip can fix any of these for its reviewers (its display "
            "presets); the rest come from here.",
        )
        form.addRow("Units", self._combo(
            "review.units",
            {"uv_per_mm": "Clinical: µV/mm and mm/s",
             "uv": "Channel spacing in µV, seconds per page"},
            "Clinical units use the screen's real size: 7 µV/mm and 30 mm/s "
            "look as they do on paper",
        ))  # fmt: skip
        form.addRow("Sensitivity", self._number("review.sensitivity", "µV/mm", 1000))
        form.addRow("Speed", self._number("review.speed", "mm/s", 1000))
        form.addRow("Channel spacing", self._number("review.spacing", "µV", 10000))
        form.addRow("Page", self._number("review.page", "s", 3600))
        form.addRow("Montage", self._combo(
            "review.montage", {m: m.capitalize() for m in MONTAGES},
            "Average: each channel against the mean of the good EEG channels",
        ))  # fmt: skip
        for key, text in (("highpass", "High-pass"), ("lowpass", "Low-pass"),
                          ("notch", "Notch")):  # fmt: skip
            form.addRow(
                text,
                self._number(
                    f"review.{key}", "Hz", 1000, "Display only (0: off)", off=True
                ),
            )

    def _visualization(self, page: QWidget) -> None:
        form = self._form(
            page,
            "Colour maps are the same in every theme unless "
            "flipped below. These defaults are used until you "
            "change a view; MeegXer then remembers your last "
            "choice (Forget remembered view returns to them).",
        )
        sequential = self._combo(
            "visualization.sequential",
            SEQUENTIAL,
            "Magnitudes: the head plot's band-power map",
        )
        peak = self._combo(
            "visualization.peak",
            SEQUENTIAL,
            "Frequencies: the head plot's peak-frequency map",
        )
        diverging = self._combo(
            "visualization.diverging",
            DIVERGING,
            "Signed changes: the head plot's power-change map",
        )
        ramp = np.linspace(0, 1, 64)
        for box in (sequential, peak, diverging):
            box.setIconSize(QSize(64, 10))
        for box in (sequential, peak):
            for i in range(box.count()):
                tone = Maps(sequential=str(box.itemData(i)))
                box.setItemIcon(i, _ramp_icon(tone.sequential_rgb(ramp, self.theme)))
        for i in range(diverging.count()):
            tone = Maps(diverging=str(diverging.itemData(i)))
            colors = tone.diverging_rgb(ramp * 2 - 1, self.theme)
            diverging.setItemIcon(i, _ramp_icon(colors))
        form.addRow("Sequential map", sequential)
        form.addRow("Diverging map", diverging)
        form.addRow("Frequency map", peak)
        form.addRow(
            "",
            self._check(
                "visualization.flip_on_light",
                "Reverse sequential maps on a light theme (more is darker)",
            ),
        )
        form.addRow(
            "Time span", self._combo("visualization.span", {s: f"{s} s" for s in SPANS})
        )
        form.addRow(
            "Live time axis",
            self._combo(
                "visualization.time_axis",
                {"before_now": "Seconds before now", "since_start": "Time since start"},
                "Seconds before now keeps the ticks still (0 is the newest data); "
                "time since start counts from the stream's first sample, as offline",
            ),
        )

        band = QWidget()
        row = QHBoxLayout(band)
        row.setContentsMargins(0, 0, 0, 0)
        preset = QComboBox()
        for name, (lo, hi) in BANDS.items():
            text = f"{SYMBOLS[name]} {name.capitalize()} ({lo:g}–{hi:g} Hz)"  # noqa: RUF001
            preset.addItem(text, [lo, hi])
        preset.addItem("Custom", None)
        low, high = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (low, high):
            spin.setRange(0.0, 500.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" Hz")

        def load() -> None:
            lo, hi = core_settings.get(self.working, "visualization.spectrum.band")
            low.setValue(lo)
            high.setValue(hi)
            i = preset.findData([float(lo), float(hi)])
            preset.setCurrentIndex(i if i >= 0 else preset.count() - 1)

        def edited() -> None:
            lo, hi = sorted((low.value(), high.value()))
            if hi - lo < 0.5:
                return
            self._set("visualization.spectrum.band", [lo, hi])
            i = preset.findData([lo, hi])
            preset.blockSignals(True)
            preset.setCurrentIndex(i if i >= 0 else preset.count() - 1)
            preset.blockSignals(False)

        def picked(_: int) -> None:
            value = preset.currentData()
            if value is not None and not self._loading:
                low.setValue(value[0])
                high.setValue(value[1])
                edited()

        low.editingFinished.connect(edited)
        high.editingFinished.connect(edited)
        preset.currentIndexChanged.connect(picked)
        self._loaders.append(load)
        row.addWidget(preset, 1)
        row.addWidget(low)
        row.addWidget(QLabel("to"))
        row.addWidget(high)
        form.addRow("Head plot band", band)
        form.addRow(
            "Power",
            self._combo(
                "visualization.spectrum.relative",
                {False: "Absolute", True: "Relative (to 1–45 Hz)"},  # noqa: RUF001
            ),
        )
        form.addRow(
            "",
            self._check(
                "visualization.spectrum.aperiodic",
                "Remove the aperiodic (1/f) fit",
                "Flatten each spectrum by its 1/f fit; the head plot shows the "
                "band's power above it",
            ),
        )
        form.addRow(
            "Head plot map", self._combo("visualization.spectrum.scheme", SCHEMES)
        )
        form.addRow(
            "Power range",
            self._combo(
                "visualization.spectrum.range", {int(r): f"{r:g} dB" for r in RANGES}
            ),
        )
        form.addRow(
            "Smoothing (live)",
            self._combo(
                "visualization.spectrum.smoothing",
                {s: "Off" if s == 0 else f"{s} s" for s in SMOOTHING},
                "Time constant of the moving average of live spectra",
            ),
        )

    def _storage(self, page: QWidget) -> None:
        form = self._form(
            page, "Where MeegXer keeps files and where file dialogs start."
        )
        form.addRow(
            "Offline run files",
            self._folder(
                "storage.offline_dir", "System temp folder", "Offline run folder"
            ),
        )
        if os.environ.get("MEEGXER_OFFLINE_DIR"):
            note = QLabel("MEEGXER_OFFLINE_DIR is set and wins over this setting.")
            note.setObjectName("muted")
            form.addRow("", note)
        cache = QDoubleSpinBox()
        cache.setRange(0.0, 10000.0)
        cache.setDecimals(1)
        cache.setSingleStep(1.0)
        cache.setSuffix(" GB")
        cache.setToolTip(
            "Cap of the cache of processor outputs between offline runs (0: no cache)"
        )
        cache.valueChanged.connect(lambda v: self._set("storage.cache_gb", float(v)))
        self._loaders.append(
            lambda: cache.setValue(core_settings.get(self.working, "storage.cache_gb"))
        )
        form.addRow("Offline cache", cache)
        form.addRow(
            "Pipelines and sessions",
            self._folder("storage.output_dir", "Last-used folder", "Default folder"),
        )
        form.addRow(
            "Recordings",
            self._folder("storage.data_dir", "Last-used folder", "Recordings folder"),
        )
        form.addRow(
            "Sample data",
            self._folder(
                "storage.sample_dir", "MNE's data folder", "Sample data folder"
            ),
        )

    def _privacy(self, page: QWidget) -> None:
        lay = QVBoxLayout(page)
        note = QLabel(
            "Plugins and custom scripts you added. Local ones run on this "
            "computer with your permissions (network, files)."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.plugin_list = QListWidget()
        lay.addWidget(self.plugin_list, 1)
        row = QHBoxLayout()
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_plugin)
        add = QPushButton("Add a custom script…")
        add.clicked.connect(self.add_script_requested)
        notice = QPushButton("Read the local-code notice")
        notice.clicked.connect(
            lambda: QMessageBox.information(self, "Local code", LOCAL_CODE_NOTICE)
        )
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_plugins)
        for b in (remove, add, notice):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(refresh)
        lay.addLayout(row)
        self.forget_note = QLabel("")
        self.forget_note.setObjectName("muted")
        lay.addWidget(self.forget_note)
        self.refresh_plugins()

    def refresh_plugins(self) -> None:
        import json

        self.plugin_list.clear()
        try:
            code, out, err = self.plugins(["list", "--json"])
            entries = json.loads(out) if code == 0 else None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            entries, err = None, str(exc)
        if entries is None:
            self.plugin_list.addItem(f"Could not list plugins: {err.strip()}")
            return
        if not entries:
            self.plugin_list.addItem("No plugins or custom scripts added yet.")
            return
        for e in entries:
            where = "local, unsandboxed" if e["unsandboxed"] else "container"
            item = QListWidgetItem(f"{e['name']}  {e['version']}  ({where})")
            item.setToolTip(f"{e.get('description', '')}\n{e['where']}".strip())
            item.setData(Qt.ItemDataRole.UserRole, e["name"])
            self.plugin_list.addItem(item)

    def _remove_plugin(self) -> None:
        item = self.plugin_list.currentItem()
        name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not name:
            return
        if QMessageBox.question(self, "Remove plugin", f"Remove {name}?") != (
            QMessageBox.StandardButton.Yes
        ):
            return
        code, _, err = self.plugins(["remove", name])
        if code != 0:
            QMessageBox.warning(self, "Remove plugin", err.strip() or "Failed.")
        self.refresh_plugins()
