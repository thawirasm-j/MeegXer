"""The MeegXer window: top bar, signal view, chain of strips, inspector and log.

A pure client of the daemon (docs/DESIGN.md §3.7): every action is a control
call; the signal view draws what arrives on the display stream (live) or what the
daemon serves from an offline run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from meegxer.core import settings as core_settings
from meegxer.core.block import units_from_wire
from meegxer.core.control import RpcError
from meegxer.core.display import DisplayFrame, FeatureFrame
from meegxer.core.epoch_review import EPOCH_REVIEW_NODE
from meegxer.core.graph import positions_message
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.notices import LOCAL_CODE_NOTICE
from meegxer.core.passband import filters_from_wire
from meegxer.core.review import REVIEW_NODE
from meegxer.core.session import (
    free_text,
    is_session,
    read_session,
    share_safe,
    state_from_json,
    write_session,
)
from meegxer.gui.buffers import FeatureBuffer, ProbeBuffer
from meegxer.gui.chain import ChainView
from meegxer.gui.colormaps import Maps
from meegxer.gui.colors import wheel_positions
from meegxer.gui.connection import DaemonLink
from meegxer.gui.epoch_review_window import EpochReviewWindow
from meegxer.gui.feature_view import FeatureSeries, FeatureView
from meegxer.gui.inspector import Inspector
from meegxer.gui.labels import param_label
from meegxer.gui.names import title
from meegxer.gui.pipeline_doc import LIVE_ONLY_SOURCES, PipelineDoc, tap_id
from meegxer.gui.recording_window import RecordingWindow
from meegxer.gui.review_window import ReviewWindow
from meegxer.gui.settings_dialog import SettingsDialog
from meegxer.gui.signal_view import Panner, SignalView, StackedWindow, Window
from meegxer.gui.theme import Theme, load_theme, stylesheet
from meegxer.gui.wheel import install_wheel_guard

DEFAULT_CHAIN: list[tuple[str, dict[str, Any]]] = [
    ("meegxer.nodes.iir_filter", {"kind": "highpass", "low": 1.0}),
    ("meegxer.nodes.notch", {}),
]

EPOCH_POINTS = 400  # samples per epoch and channel the epoch review draws
IMPORT_TAP = "import_review"  # the tap of a review at import's run


def resolve_theme(name: str) -> Theme:
    """A theme setting: ``dark``, ``light``, ``system`` (the operating system's
    choice) or a theme file (dark when it cannot be read)."""
    if name == "system":
        hints = QGuiApplication.styleHints()
        name = "light" if hints.colorScheme() == Qt.ColorScheme.Light else "dark"
    try:
        return load_theme(name)
    except (OSError, ValueError, KeyError, TypeError):
        return load_theme("dark")


class MainWindow(QMainWindow):
    def __init__(
        self,
        link: DaemonLink,
        theme: Theme | None = None,
        *,
        owns_daemon: bool = False,
        settings: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.link = link
        self.owns_daemon = owns_daemon
        # preferences (settings.json) and what was remembered (state.json)
        self.settings = (
            settings if settings is not None else core_settings.load_settings()
        )
        self.state = state if state is not None else core_settings.load_state()
        self.scale = self.settings["appearance"]["scale"] / 100
        self.theme = theme or resolve_theme(self.settings["appearance"]["theme"])
        self.doc = PipelineDoc(link.call("list_nodes"))
        self.buffers: dict[str, ProbeBuffer] = {}
        self.feature_buffers: dict[str, FeatureBuffer] = {}
        self.taps: dict[str, str] = {}
        # a tapped feature strip or rail output (e.g. "ica1:ics"): shown alone,
        # in place of the signal taps, until tapped again or a signal tap is
        self.view_tap: str | None = None
        self.selected: str | None = None
        self.advanced = False
        self.running = False
        self.statuses: dict[str, str] = {}
        self.node_status: dict[str, dict[str, Any]] = {}
        self.offline_probes: dict[str, dict[str, Any]] = {}
        self.offline_features: dict[str, dict[str, Any]] = {}
        self.offline_sfreq = 0.0
        self.offline_total = 0
        self.offline_state = ""
        self.review_window: ReviewWindow | None = None  # a review strip waits
        self.epoch_window: EpochReviewWindow | None = None  # an epoch review waits
        self.import_window: ReviewWindow | None = None  # a review at import
        self._answered: dict[str, Any] | None = None  # until the run moves on
        self.offline_recordings: dict[str, Any] = {}  # an all-subjects BIDS run
        self._feature_cache: dict[str, tuple[tuple[Any, ...], FeatureSeries]] = {}
        self._offline_cache: dict[str, tuple[tuple[Any, ...], Window | None]] = {}
        # node -> (what it was read for, when, its sensors and mixing matrix)
        self._mixing: dict[str, tuple[tuple[Any, ...], float, Any]] = {}
        app = QApplication.instance()
        if isinstance(app, QApplication):
            install_wheel_guard(app)  # wheel scrolls panels, not unfocused values
        self.setWindowTitle("MeegXer")
        self.resize(1440, 900)
        self._build()
        self._apply_view_settings()
        self._load_initial()
        QTimer.singleShot(0, self.offer_recovery)  # once the window is up
        link.status.connect(self.on_status)
        link.frame.connect(self.on_frame)
        link.lost.connect(
            lambda msg: self.log(f"The processing engine is not answering: {msg}")
        )
        self._render = QTimer(self)
        self._render.timeout.connect(self.redraw)
        self._render.start(50)
        self.apply_theme(self.theme)

    # -- layout ---------------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)
        lay.addWidget(self._top_bar())

        splitter = QSplitter(Qt.Orientation.Vertical)
        signal_panel = QFrame()
        signal_panel.setObjectName("panel")
        sp = QVBoxLayout(signal_panel)
        sp.setContentsMargins(10, 8, 10, 8)
        self.view = SignalView(self.theme)
        self.view.stacked_requested.connect(self.show_stacked)
        self.feature_view = FeatureView(self.theme)
        self.views = QStackedWidget()  # the signal view, or a feature tap's view
        self.views.addWidget(self.view)
        self.views.addWidget(self.feature_view)
        sp.addWidget(self.views, 1)
        self.timeline = self.view.panner
        splitter.addWidget(signal_panel)

        self.chain = ChainView()
        self.chain.selected.connect(self.select)
        self.chain.remove_requested.connect(self.remove_strip)
        self.chain.insert_requested.connect(self.insert_strip)
        self.chain.script_requested.connect(self.add_script)
        self.chain.move_requested.connect(self.move_strip)
        self.chain.tap_clicked.connect(self.click_tap)
        self.chain.calibrate_requested.connect(lambda sid: self.calibrate(sid, 60.0))
        self.chain.source_changed.connect(self.change_source)
        self.chain.mode_toggled.connect(self.toggle_mode)
        self.chain.recording_requested.connect(self.open_recording)
        splitter.addWidget(self.chain)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.inspector = Inspector()
        self.inspector.param_changed.connect(self.change_param)
        self.inspector.strip_params_changed.connect(self.change_strip_params)
        self.inspector.input_changed.connect(self.change_input)
        self.inspector.channel_types_changed.connect(self.change_channel_types)
        self.inspector.pass_uncalibrated_changed.connect(self.change_passthrough)
        self.inspector.clear_calibration.connect(self.clear_calibration)
        self.inspector.calibrate.connect(self.calibrate)
        bottom.addWidget(self.inspector)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        bottom.addWidget(self.logbox)
        bottom.setSizes([760, 520])
        splitter.addWidget(bottom)
        splitter.setSizes([430, 240, 200])
        splitter.setStretchFactor(0, 3)
        lay.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.stacked = StackedWindow(self.theme)
        self.stacked.closed.connect(self._stacked_closed)
        self.recording_window = RecordingWindow(self.call, self.theme)
        self.recording_window.start_dir = lambda: self._start_dir("data_dir")
        QShortcut(QKeySequence("Ctrl+,"), self, self.open_settings)
        self.recording_window.applied.connect(self.apply_recording)
        self.recording_window.review_requested.connect(self.review_at_import)

    def _top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("panel")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        brand = QLabel("<b>MeegXer</b>")
        lay.addWidget(brand)
        lay.addSpacing(12)
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.clicked.connect(self.stop)
        lay.addWidget(self.start_btn)
        lay.addWidget(self.stop_btn)
        lay.addSpacing(8)
        for text, slot in (
            ("Open…", self.open_pipeline),
            ("Save…", self.save_pipeline),
            ("Save session…", self.save_session),
            ("Share session…", self.share_session),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            lay.addWidget(b)
        lay.addSpacing(12)
        self.mode_group = QButtonGroup(self)
        for i, text in enumerate(("Friendly", "Advanced")):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.clicked.connect(lambda _=False, adv=bool(i): self.set_advanced(adv))
            self.mode_group.addButton(b)
            lay.addWidget(b)
        lay.addStretch(1)
        self.meters: dict[str, QLabel] = {}
        for key in ("state", "load", "latency", "overruns", "display"):
            label = QLabel("")
            label.setObjectName("meter")
            self.meters[key] = label
            lay.addWidget(label)
        self.theme_btn = QPushButton("☀")
        self.theme_btn.setToolTip("Switch between dark and light themes")
        self.theme_btn.clicked.connect(self.toggle_theme)
        lay.addWidget(self.theme_btn)
        settings_btn = QPushButton("⚙")
        settings_btn.setToolTip("Settings (Ctrl+,)")
        settings_btn.clicked.connect(self.open_settings)
        lay.addWidget(settings_btn)
        return bar

    # -- helpers ----------------------------------------------------------------------

    def log(self, text: str) -> None:
        self.logbox.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def call(self, method: str, **params: Any) -> Any:
        try:
            return self.link.call(method, **params)
        except (RpcError, TimeoutError) as exc:
            self.log(f"{method} failed: {exc}")
            return None

    def apply_theme(self, theme: Theme) -> None:
        self.theme = theme
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setStyleSheet(stylesheet(theme, self.scale))
        self.view.apply_theme(theme)
        self.feature_view.apply_theme(theme)
        self.inspector.apply_theme(theme)
        self.stacked.apply_theme(theme)
        self.stacked.setStyleSheet(stylesheet(theme, self.scale))
        self.recording_window.theme = theme
        self.recording_window.setStyleSheet(stylesheet(theme, self.scale))
        self.theme_btn.setText("☀" if theme.dark else "☾")
        self.refresh_chain()

    def toggle_theme(self) -> None:
        """The quick switch: the same as choosing the theme in Settings."""
        name = "light" if self.theme.dark else "dark"
        self.settings["appearance"]["theme"] = name
        try:
            core_settings.save_settings(self.settings)
        except (OSError, ValueError) as exc:
            self.log(f"Could not save the theme: {exc}")
        self.apply_theme(load_theme(name))

    # -- settings ---------------------------------------------------------------------

    def _apply_view_settings(self) -> None:
        """Defaults from the settings, then what was remembered on top."""
        vis = self.settings["visualization"]
        self.view.maps = Maps.from_settings(vis)
        self.feature_view.maps = self.view.maps
        self.view.set_view_state({"span": vis["span"], "spectrum": vis["spectrum"]})
        self.view.set_view_state(self.state.get("view", {}))
        self.set_advanced(
            bool(self.state.get("advanced", self.settings["general"]["advanced"]))
        )
        geometry = self.state.get("geometry")
        if isinstance(geometry, str):
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.theme, self)
        dialog.setStyleSheet(stylesheet(self.theme, self.scale))
        dialog.changed.connect(self._preview_settings)
        dialog.forget_requested.connect(self.forget_view)
        dialog.add_script_requested.connect(
            lambda: self.add_script(len(self.doc.strips))
        )
        if dialog.exec():
            self.accept_settings(dialog.settings())

    def _preview_settings(self, settings: dict[str, Any]) -> None:
        """Show appearance and colour choices while the dialog is open."""
        self.scale = settings["appearance"]["scale"] / 100
        self.apply_theme(resolve_theme(settings["appearance"]["theme"]))
        self.view.maps = Maps.from_settings(settings["visualization"])
        self.feature_view.maps = self.view.maps

    def accept_settings(self, settings: dict[str, Any]) -> None:
        """Keep new settings: save them, and let a changed default replace what
        was remembered for it."""
        old, self.settings = self.settings, settings
        try:
            path = core_settings.save_settings(settings)
            self.log(f"Settings saved to {path}.")
        except (OSError, ValueError) as exc:
            self.log(f"Could not save the settings: {exc}")
        self._preview_settings(settings)
        if settings["general"]["advanced"] != old["general"]["advanced"]:
            self.state.pop("advanced", None)
            self.set_advanced(settings["general"]["advanced"])
        vis, before = settings["visualization"], old["visualization"]
        view = self.state.setdefault("view", {})
        if vis["span"] != before["span"]:
            view.pop("span", None)
            self.view.set_view_state({"span": vis["span"]})
        changed = {k: v for k, v in vis["spectrum"].items()
                   if before["spectrum"].get(k) != v}  # fmt: skip
        if changed:
            for key in changed:
                view.get("spectrum", {}).pop(key, None)
            self.view.set_view_state({"spectrum": changed})

    def forget_view(self) -> None:
        """Forget what was remembered and go back to the settings' defaults."""
        self.state = {}
        core_settings.clear_state()
        vis = self.settings["visualization"]
        self.view.set_view_state({"span": vis["span"], "spectrum": vis["spectrum"],
                                  "head": True})  # fmt: skip
        self.set_advanced(self.settings["general"]["advanced"])
        self.log("Forgot the remembered view; using the settings' defaults.")

    def _start_dir(self, kind: str = "output_dir") -> str:
        """Where a file dialog opens: the last folder used, else the setting."""
        storage = self.settings["storage"]
        if kind == "data_dir" and storage.get("data_dir"):
            return str(self.state.get("last_data_dir") or storage["data_dir"])
        return str(self.state.get("last_dir") or storage.get("output_dir") or "")

    def _remember_file(self, path: str, pipeline: bool = False) -> None:
        self.state["last_dir"] = str(Path(path).parent)
        if pipeline:
            self.state["last_pipeline"] = str(Path(path).resolve())

    def save_state(self) -> None:
        self.state["view"] = self.view.view_state()
        self.state["advanced"] = self.advanced
        self.state["geometry"] = bytes(self.saveGeometry().toBase64().data()).decode()
        try:
            core_settings.save_state(self.state)
        except OSError as exc:
            print(f"MeegXer: could not remember the view: {exc}", file=sys.stderr)

    def chain_taps(self) -> dict[str, str]:
        """Taps as the chain shows them: a tapped feature strip or rail output
        is Main and the signal taps rest meanwhile."""
        if self.view_tap is None:
            return dict(self.taps)
        return {**{k: "off" for k in self.taps}, self.view_tap: "main"}

    def refresh_chain(self) -> None:
        self.chain.rebuild(
            self.doc,
            self.theme,
            statuses=self.statuses,
            taps=self.chain_taps(),
            advanced=self.advanced,
            selected=self.selected,
        )
        self.view.names = self._probe_names()
        self.refresh_inspector()
        self._transport()

    def _probe_names(self) -> dict[str, str]:
        """Legend names: "Input", "after Notch" (the id is added when two strips
        share a title, or in advanced mode)."""
        titles = {s.id: self._name(s.id) for s in self.doc.tappable()}
        repeated = {t for t in titles.values() if list(titles.values()).count(t) > 1}
        names = {}
        for sid, name in titles.items():
            text = "Input" if name == "the input" else f"after {name}"
            if self.advanced or name in repeated:
                text += f" ({sid})"
            names[tap_id(sid)] = text
        for strip in self.doc.feature_taps():
            text = self.doc.rail_name(strip)
            names[tap_id(strip.id)] = text + (f" ({strip.id})" if self.advanced else "")
        for key in self.doc.rail_taps():
            names[tap_id(key)] = self.doc.choice_label(key)
        return names

    def refresh_inspector(self) -> None:
        if self.selected is None:
            self.inspector.show_empty()
            return
        try:
            strip = self.doc.get(self.selected)
        except KeyError:
            self.selected = None
            self.inspector.show_empty()
            return
        self.inspector.show_strip(
            self.doc,
            strip,
            advanced=self.advanced,
            running=self.running,
            node_status=self.node_status.get(strip.id, {}),
        )

    def _transport(self) -> None:
        live = self.doc.mode == "live"
        self.start_btn.setText("▶  Start" if live else "▶  Run offline")
        busy = self.offline_state in ("running", "review")
        self.start_btn.setEnabled(not self.running and not busy)
        self.stop_btn.setVisible(live)
        self.stop_btn.setEnabled(self.running)

    def _default_taps(self) -> None:
        self.view_tap = None
        tappable = [s.id for s in self.doc.tappable()]
        self.taps = {sid: "off" for sid in tappable}
        if tappable:
            self.taps[tappable[-1]] = "main"
        if len(tappable) > 1:
            self.taps[tappable[0]] = "comparison"

    def _load_initial(self) -> None:
        pipeline = self.call("get_pipeline")
        if pipeline:
            try:
                self.doc.load_pipeline(pipeline)
                self.log("Showing the pipeline loaded in the daemon.")
            except (KeyError, ValueError) as exc:
                self.log(f"Could not show the daemon's pipeline: {exc}")
                pipeline = None
        if not pipeline:
            last = self.state.get("last_pipeline")
            if (
                self.settings["general"]["reopen_last"]
                and last
                and Path(last).is_file()
            ):
                try:
                    data = json.loads(Path(last).read_text(encoding="utf-8"))
                    self.doc.load_pipeline(data.get("pipeline", data)
                                           if is_session(data) else data)  # fmt: skip
                    pipeline = data
                    self.log(f"Reopened {last}.")
                except (OSError, ValueError, KeyError) as exc:
                    self.log(f"Could not reopen {last}: {exc}")
        if not pipeline:
            self.doc.set_source("meegxer.core.synthetic")
            for node, params in DEFAULT_CHAIN:
                self.doc.insert(len(self.doc.strips), node).params.update(params)
            self.doc.input_mode = self.settings["general"]["start_mode"]
        self._default_taps()
        self.refresh_chain()

    # -- chain edits ------------------------------------------------------------------

    def _structure_changed(self, what: str) -> None:
        for sid in [s.id for s in self.doc.tappable()]:
            self.taps.setdefault(sid, "off")
        self.taps = {
            k: v
            for k, v in self.taps.items()
            if k in {s.id for s in self.doc.tappable()}
        }
        if "main" not in self.taps.values() and self.taps:
            self.taps[list(self.taps)[-1]] = "main"
        views = {s.id for s in self.doc.feature_taps()} | set(self.doc.rail_taps())
        if self.view_tap not in views:
            self.view_tap = None
        self.log(what)
        self.refresh_chain()
        if self.running:
            self.log("Restarting to apply the change (fitted models are kept).")
            self.restart()

    def _name(self, strip_id: str) -> str:
        """A strip's title for messages (its id when it is no longer there)."""
        try:
            strip = self.doc.get(strip_id)
        except KeyError:
            return strip_id
        if self.doc.role(strip) == "source":
            return "the input"
        return title(strip.node, strip.params)

    def select(self, strip_id: str) -> None:
        self.selected = strip_id
        self.refresh_chain()

    def add_script(self, index: int, path: str | None = None) -> None:
        """Add a custom script as a strip: pick the file, accept the local-code
        notice, install it with ``meegxer-plugin`` (a separate process: the GUI
        never imports plugin code, rule 9), then insert it at ``index``."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Add a custom script", self._start_dir(), "Python scripts (*.py)"
            )
            if not path:
                return
            self._remember_file(path)
        if not self._accept_local_code(Path(path).name):
            self.log("Custom script not added (the notice was not accepted).")
            return
        before = set(self.doc.manifests)
        result = subprocess.run(
            [sys.executable, "-m", "meegxer.plugins.cli", "add", "--script", path,
             "--accept-risk"],
            capture_output=True, text=True, check=False, timeout=120,
        )  # fmt: skip
        if result.returncode != 0:
            reason = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
            self.log(f"Could not add {Path(path).name}: {reason}")
            QMessageBox.warning(self, "Custom script", f"Could not add it:\n{reason}")
            return
        nodes = self.call("rescan_nodes")
        if nodes is None:
            return
        self.doc.set_manifests(nodes)
        added = [n for n in self.doc.manifests if n not in before]
        name = added[0] if added else result.stdout.split()[1]
        self.log(f"Added custom script {Path(path).name} as {name} (unsandboxed).")
        if name in self.doc.manifests:
            self.insert_strip(index, name)

    def _accept_local_code(self, what: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Run code outside the sandbox?")
        box.setText(f"<b>{what}</b> will run on your computer.")
        box.setInformativeText(LOCAL_CODE_NOTICE)
        accept = box.addButton(
            "I understand, add it", QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return box.clickedButton() is accept

    def insert_strip(self, index: int, node: str) -> None:
        strip = self.doc.insert(index, node)
        self.selected = strip.id
        self._structure_changed(f"Added {self._name(strip.id)}.")

    def remove_strip(self, strip_id: str) -> None:
        name = self._name(strip_id)
        self.doc.remove(strip_id)
        self._structure_changed(f"Removed {name}.")

    def move_strip(self, strip_id: str, index: int) -> None:
        try:
            self.doc.move(strip_id, index)
        except (KeyError, ValueError) as exc:
            self.log(str(exc))
            return
        self._structure_changed(f"Moved {self._name(strip_id)}.")

    def open_recording(self) -> None:
        if self.recording_window.load(self.doc.source_entry()):
            self.recording_window.show()
            self.recording_window.raise_()

    def apply_recording(self, recording: dict[str, Any]) -> None:
        assert self.doc.source is not None
        self.doc.source.recording = recording
        self.use_recording_info(self.recording_window.current)
        self._structure_changed("Recording settings changed.")

    def use_recording_info(self, info: dict[str, Any] | None) -> None:
        """Colour channels by the recording's positions; mark its events on the
        offline view (live views have no absolute time axis) and start the
        chain's Events rail with them (live playback carries them too)."""
        events = len(info["events"]) if info else 0
        if events != self.doc.input_events:
            self.doc.input_events = events
            self.refresh_chain()
        if info is None:
            self.view.set_wheel({})
            self.feature_view.wheel = {}
            self.view.types = {}
            self.view.events = []
            return
        self.view.types = {c["label"]: c["type"] for c in info["channels"]}
        positions = {
            c["label"]: tuple(c["position"]) for c in info["channels"] if c["position"]
        }
        self.view.set_wheel(wheel_positions(positions))
        self.feature_view.wheel = self.view.wheel
        self.view.events = list(info["events"]) if self.doc.mode == "offline" else []

    def _describe_input(self) -> dict[str, Any] | None:
        """Read the input's description before a run (not for LSL, whose stream
        the run itself opens; the daemon checks it then)."""
        src = self.doc.source
        if src is None or src.node in LIVE_ONLY_SOURCES:
            self.use_recording_info(None)
            return None
        info: dict[str, Any] | None = self.call(
            "describe_source", node=self.doc.source_entry()
        )
        self.use_recording_info(info)
        return info

    def _positions_ready(self, info: dict[str, Any] | None) -> bool:
        """Strips that need electrode positions (ICA's ICLabel) get them before
        anything runs: if the input lacks some, say which and open the Recording
        window on its Channels tab instead of starting."""
        missing = self.doc.missing_positions(info["channels"]) if info else {}
        if not missing:
            return True
        self.log("Cannot start: " + positions_message(missing))
        self.open_recording()
        self.recording_window.tabs.setCurrentIndex(0)  # Channels
        return False

    def change_source(self, node: str) -> None:
        if self.doc.source is not None and node == self.doc.source.node:
            return
        self.doc.set_source(node)
        self.use_recording_info(None)
        self.offline_probes.clear()
        self._structure_changed(f"Input is now {title(node)} ({self.doc.mode}).")

    def toggle_mode(self) -> None:
        if self.running:
            self.stop()
        self.doc.input_mode = "offline" if self.doc.mode == "live" else "live"
        self.log(f"Mode: {self.doc.mode}.")
        self.refresh_chain()

    def click_tap(self, strip_id: str, make_main: bool) -> None:
        """A signal tap is Main, a comparison or off. A feature strip's or a
        rail output's tap (``ica1:ics``) is shown alone, or off."""
        if self.doc.tap_kind(strip_id) != "signal":
            self.view_tap = None if self.view_tap == strip_id else strip_id
            self.refresh_chain()
            return
        if self.view_tap is not None:  # back to the signal: this tap is Main
            self.view_tap = None
            make_main = True
        state = self.taps.get(strip_id, "off")
        if make_main or (state == "off" and "main" not in self.taps.values()):
            for k, v in self.taps.items():
                if v == "main":
                    self.taps[k] = "comparison"
            self.taps[strip_id] = "main"
        elif state == "off":
            self.taps[strip_id] = "comparison"
        else:
            self.taps[strip_id] = "off"
        self.refresh_chain()

    def set_advanced(self, advanced: bool) -> None:
        self.advanced = advanced
        self.mode_group.buttons()[int(advanced)].setChecked(True)
        self.refresh_chain()

    # -- parameters -------------------------------------------------------------------

    def change_param(self, strip_id: str, name: str, value: Any) -> None:
        strip = self.doc.get(strip_id)
        old = strip.params.get(name)
        if old == value:
            return
        strip.params[name] = value
        if self.running:
            if self.doc.role(strip) == "processor":
                if self.call("set_params", node=strip_id, params={name: value}) is None:
                    strip.params[name] = old  # refused (e.g. zero-phase live)
            else:
                self.log(
                    f"{self._name(strip_id)}: {param_label(name)} changed, restarting."
                )
                self.restart()
        self.refresh_chain()

    def change_strip_params(self, strip_id: str, names: list[str]) -> None:
        self.doc.get(strip_id).strip_params = names
        if self.running:
            self.call("set_strip_params", node=strip_id, params=names)
        self.chain.rebuild(
            self.doc,
            self.theme,
            statuses=self.statuses,
            taps=self.chain_taps(),
            advanced=self.advanced,
            selected=self.selected,
        )

    def change_input(self, strip_id: str, source: str) -> None:
        """Which rail (feature strip) a strip reads; "" for the nearest one."""
        strip = self.doc.get(strip_id)
        if strip.input == source:
            return
        self.doc.set_input(strip_id, source)
        what = self.doc.rail_name(self.doc.get(source)) if source else "the nearest"
        self._structure_changed(f"{self._name(strip_id)} reads {what}.")

    def change_channel_types(self, strip_id: str, types: list[str]) -> None:
        strip = self.doc.get(strip_id)
        if self.doc.processed_types(strip) == types:
            return
        strip.channel_types = [] if types == self.doc.default_types(strip) else types
        self._structure_changed(
            f"{self._name(strip_id)} processes: "
            f"{', '.join(t.upper() for t in types)} channels."
        )

    def change_passthrough(self, strip_id: str, enabled: bool) -> None:
        self.doc.get(strip_id).pass_uncalibrated = enabled
        if self.running:
            self.call("set_pass_uncalibrated", node=strip_id, enabled=enabled)
        self.log(
            f"{self._name(strip_id)}: pass data through while uncalibrated "
            f"{'on' if enabled else 'off'}."
        )

    def calibrate(self, strip_id: str, seconds: float) -> None:
        if not self.running:
            self.log("Start the pipeline before calibrating.")
            return
        if self.call("calibrate", node=strip_id, seconds=seconds) is not None:
            self.log(f"Calibrating {self._name(strip_id)} on {seconds:g} s of input…")

    def clear_calibration(self, strip_id: str) -> None:
        if self.call("clear_calibration", node=strip_id) is not None:
            self.log(f"Cleared the calibration of {self._name(strip_id)}.")

    # -- transport --------------------------------------------------------------------

    def ask_reviewer(self) -> str | None:
        """Who reviews at the pipeline's review strips, asked once per run and
        saved with each review (None: the run is not started)."""
        name, ok = QInputDialog.getText(
            self, "Review", "This pipeline has a review step. Your name, saved "
            "with each review:", text=str(self.state.get("reviewer", "")),
        )  # fmt: skip
        if not ok:
            return None
        self.state["reviewer"] = name.strip()
        return name.strip()

    def start(self) -> None:
        problems = self.doc.problems()
        if problems:
            self.log("Cannot start: " + "; ".join(problems))
            return
        if not self._positions_ready(self._describe_input()):
            return
        if self.doc.mode == "offline":
            reviewer = ""
            if any(s.node in (REVIEW_NODE, EPOCH_REVIEW_NODE) for s in self.doc.strips):
                asked = self.ask_reviewer()
                if asked is None:
                    return
                reviewer = asked
            pipeline = self.doc.to_pipeline(tap_keep_seconds=0.0)
            result = self.call("run_offline", pipeline=pipeline, reviewer=reviewer)
            if result is not None:
                self.offline_state = "running"
                self.offline_probes.clear()
                self.offline_features.clear()
                self._offline_cache.clear()
                self._feature_cache.clear()
                self.offline_recordings = {}
                self.log("Offline run started.")
            self._transport()
            return
        if self.call("load_pipeline", pipeline=self.doc.to_pipeline()) is None:
            return
        self.buffers.clear()
        self.feature_buffers.clear()
        if self.call("start") is not None:
            self.running = True
            self.log("Started.")
        self.refresh_chain()

    def stop(self) -> None:
        self.call("stop")
        self.running = False
        self.log("Stopped.")
        self.refresh_chain()

    def restart(self) -> None:
        self.call("stop")
        self.running = False
        self.start()

    def open_pipeline(self) -> None:
        """Open a pipeline, or a session (``*.meegxer.json``) with its models."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open pipeline or session", self._start_dir(),
            "Pipeline or session (*.json);;MeegXer session (*.meegxer.json)",
        )  # fmt: skip
        if not path:
            return
        self._remember_file(path, pipeline=True)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if is_session(data):
                self.open_session(path)
                return
            self.doc.load_pipeline(data)
        except (OSError, ValueError, KeyError) as exc:
            self.log(f"Could not open {path}: {exc}")
            return
        self._default_taps()
        self._structure_changed(f"Opened {path}.")

    def open_session(self, path: str) -> None:
        """Load a session: its pipeline here, its fitted models in the daemon
        (used while they match the input; otherwise the node needs calibrating)."""
        if self.running:
            self.log("Stop the pipeline before opening a session.")
            return
        session = read_session(path)
        try:
            reply = self.link.call("load_session", session=session)
        except (RpcError, TimeoutError) as exc:
            if "allow the mismatch" not in str(exc) or not self._ask(
                "Different node versions",
                f"{exc}\n\nLoad it anyway? Those nodes' models are refitted.",
            ):
                self.log(f"Could not open {path}: {exc}")
                return
            reply = self.call("load_session", session=session,
                              allow_version_mismatch=True)  # fmt: skip
            if reply is None:
                return
        try:
            self.doc.load_pipeline(session["pipeline"])
        except (ValueError, KeyError) as exc:
            self.log(f"Could not show the session's pipeline: {exc}")
            return
        self._default_taps()
        for warning in reply.get("warnings", []):
            self.log(f"Session: {warning}")
        models = ", ".join(reply.get("models", [])) or "none"
        self._structure_changed(f"Opened session {path} (fitted models: {models}).")

    def offer_recovery(self) -> None:
        """Offer to save each recording a crashed run left unfinished (the file
        writers spool to disk until the run ends; control API 0.2.6)."""
        for rec in self.call("unfinished_recordings") or []:
            choice = self.recovery_choice(rec)
            if choice == "later":
                self.log(f"Unfinished recording {rec['path']} kept for later "
                         "(or: meegxer recover).")  # fmt: skip
                continue
            reply = self.call("recover_recording", path=rec["path"],
                              discard=choice == "discard")  # fmt: skip
            if reply is None:
                continue
            if reply["path"] is None:
                self.log(f"Deleted the unfinished recording {rec['path']}.")
            else:
                self.log(f"Recovered the unfinished recording to {reply['path']}.")

    def recovery_choice(self, rec: dict[str, Any]) -> str:
        """Asks what to do with an unfinished recording: ``recover``,
        ``discard`` or ``later``."""
        minutes, seconds = divmod(round(rec["seconds"]), 60)
        when = datetime.fromisoformat(rec["modified"]).astimezone()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unfinished recording")
        box.setText(
            f"A run ended before it saved {Path(rec['path']).name}.\n\n"
            f"{rec['channels']} channels, {minutes} min {seconds} s, last written "
            f"{when:%d %b %Y %H:%M}\nin {Path(rec['path']).parent}\n\n"
            "Recover it now? (A file already there is kept: the recording is "
            "saved next to it as …_recovered.)"
        )
        recover = box.addButton("Recover", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Delete it", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Ask me later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(recover)
        box.exec()
        clicked = box.clickedButton()
        return (
            "recover"
            if clicked is recover
            else "discard"
            if clicked is discard
            else "later"
        )

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    def share_session(self) -> None:
        """Save a session that names no files or folders (``share_safe``), to
        send to others; asks about the free text it would keep."""
        self.save_session(share=True)

    def keep_free_text(self, text: dict[str, str]) -> bool:
        """Whether a shared session keeps the text the user wrote (asked)."""
        shown = "\n".join(f"{k.capitalize()}: {v}" for k, v in text.items())
        box = QMessageBox(self)
        box.setWindowTitle("Share session")
        box.setText("The shared session keeps the text you wrote. Check that it "
                    "names no one:\n\n" + shown)  # fmt: skip
        keep = box.addButton("Keep it", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Remove it", QMessageBox.ButtonRole.DestructiveRole)
        box.exec()
        return box.clickedButton() is keep

    def save_session(self, share: bool = False, path: str | None = None) -> None:
        """Save what is shown as a session, with the models fitted by the last
        offline run (offline mode) or calibrated live; with ``share``, without
        any file or folder names (``meegxer.core.session.share_safe``)."""
        title = "Share session" if share else "Save session"
        default = "shared.meegxer.json" if share else "session.meegxer.json"
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, title, str(Path(self._start_dir()) / default),
                "MeegXer session (*.meegxer.json)",
            )  # fmt: skip
        if not path:
            return
        self._remember_file(path, pipeline=True)
        if path.endswith(".meegxer"):
            path += ".json"
        elif not path.endswith(".json"):
            path += ".meegxer.json"
        offline = self.doc.mode == "offline" and self.offline_state == "done"
        session = self.call(
            "get_session",
            name=Path(path).name.removesuffix(".json").removesuffix(".meegxer"),
            pipeline=self.doc.to_pipeline(taps=False),
            offline=offline,
        )
        if session is None:
            return
        if share:
            session = share_safe(session)
            text = free_text(session)
            if text and not self.keep_free_text(text):
                for key in text:
                    session.pop(key, None)
        try:
            write_session(path, session)
        except OSError as exc:
            self.log(f"Could not save {path}: {exc}")
            return
        models = ", ".join(session.get("models", {})) or "none"
        outputs = len(session.get("outputs", []))
        shared = "; no file or folder names" if share else ""
        self.log(f"Saved session {path} (fitted models: {models}; "
                 f"output files: {outputs}{shared}).")  # fmt: skip

    def save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save pipeline",
            str(Path(self._start_dir()) / "pipeline.json"),
            "Pipeline (*.json)",
        )
        if path:
            self._remember_file(path, pipeline=True)
            text = json.dumps(self.doc.to_pipeline(taps=False), indent=2)
            Path(path).write_text(text + "\n", encoding="utf-8")
            self.log(f"Saved {path}.")

    # -- daemon events ----------------------------------------------------------------

    def on_status(self, status: dict[str, Any]) -> None:
        running = bool(status.get("running"))
        statuses: dict[str, str] = {}
        self.node_status = {n["id"]: n for n in status.get("nodes", [])}
        for n in status.get("nodes", []):
            s = n.get("status", "stopped")
            fits = "pass_uncalibrated" in n  # only nodes that need calibration
            if not running:
                if fits:
                    s = (
                        "ready"
                        if n.get("calibrated") or s == "ready"
                        else "uncalibrated"
                    )
                else:
                    s = "stopped"
            elif n.get("role") != "processor":
                s = "ready"
            elif n.get("pass_uncalibrated") and s != "ready":
                s = "bypassed"
            statuses[n["id"]] = s
        offline = status.get("offline")
        if self.doc.mode == "offline":  # the last offline run, not the live graph
            fitted = set((offline or {}).get("models", []))
            statuses = {s.id: "ready" for s in self.doc.strips if s.id in fitted}
        for err in status.get("errors", [])[len(getattr(self, "_errors", [])) :]:
            self.log(f"Engine error: {err}")
        self._errors = list(status.get("errors", []))
        if offline:
            self._offline_status(offline)
        if not running and self.running:
            self.log("The pipeline stopped (end of input or an error).")
        changed = running != self.running or statuses != self.statuses
        self.running, self.statuses = running, statuses
        self._meters(status)
        self.view.delays, self.view.delay_note = self._tap_delays(status)
        if changed:
            self.refresh_chain()

    def _tap_delays(self, status: dict[str, Any]) -> tuple[dict[str, float], str]:
        """Live only: how far each tap runs behind the input tap (ms), estimated
        from the strips on its path: their measured computing time plus any
        built-in delay their manifests declare (``latency_samples``). Display
        only: probes still line up by sample index (rule 8)."""
        if self.doc.mode != "live" or not status.get("running"):
            return {}, ""
        timing = {
            n["id"]: (n.get("timing") or {}).get("mean_ms", 0.0)
            for n in status.get("nodes", [])
        }
        sfreq = next((b.sfreq for b in self.buffers.values() if b.sfreq), 0.0)
        delays = {
            tap_id(sid): ms for sid, ms in self.doc.tap_delays(timing, sfreq).items()
        }
        blocks, samples = status.get("blocks") or 0, status.get("samples") or 0
        note = (
            "Estimated delay behind the input: computing time of the strips "
            "before each tap, plus any built-in delay (e.g. look-ahead) they "
            "declare. Traces still line up by sample."
        )
        if blocks and samples and sfreq:
            per_block = samples / blocks
            note += (
                f"\nEvery tap also waits for its block of data: about "
                f"{1000 * per_block / sfreq:.0f} ms ({per_block:.0f} samples)."
            )
        p95 = (status.get("latency") or {}).get("p95_ms")
        if p95 is not None:
            note += f"\nInput to output, 95% of blocks: {p95:.0f} ms."
        return delays, note

    def _offline_status(self, offline: dict[str, Any]) -> None:
        """Probes become scrollable as soon as they have data (the run writes
        them as it goes); the scrollbar spans the whole recording."""
        self.offline_probes = offline.get("probes", {})
        self.offline_features = offline.get("features", {})
        self.offline_sfreq = float(offline.get("sfreq") or 0.0)
        total = self.offline_total = int(offline.get("n_samples") or 0)
        # the scrollbar counts the input's samples (taps after a resampler map)
        sfreq = self.offline_sfreq or next(
            (p["sfreq"] for p in self.offline_probes.values()), 0.0
        )
        span = int(self.view.seconds() * sfreq)
        self.timeline.set_extent(0, max(0, total - span), span, live=False)
        recordings = offline.get("recordings") or {}
        if recordings.get("index") != self.offline_recordings.get("index"):
            self._offline_cache.clear()  # the views now show another recording
            self._feature_cache.clear()
            if recordings.get("current"):
                self.log(f"Recording {recordings['index']}/{recordings['total']}: "
                         f"{recordings['current']}")  # fmt: skip
        self.offline_recordings = recordings
        state = offline.get("state", "")
        self._review(offline.get("review") if state == "review" else None)
        if state == self.offline_state:
            return
        self.offline_state = state
        if state in ("done", "review"):  # the strips' events too (a review's)
            reply = self.call("offline_events")
            if reply is not None:
                self.view.events = list(reply["events"])
        if state == "done":
            self.log(f"Offline run done in {offline.get('elapsed_s', 0):.1f} s.")
            if recordings:
                done, skipped = recordings["done"], recordings["skipped"]
                self.log(f"Recordings: done {len(done)}, skipped {len(skipped)} "
                         f"(already done), failed {len(recordings['failed'])}. "
                         f"Showing {recordings['current']}.")  # fmt: skip
                for failure in recordings["failed"]:
                    self.log(f"Failed: {failure}")
            for note in offline.get("skipped", []):
                self.log(f"Not saved: {note}")
        elif state == "error":
            self.log(f"Offline run failed: {offline.get('error')}")
        self._transport()

    def _review(self, request: dict[str, Any] | None) -> None:
        """Open the review window while the offline run waits at a review strip
        (docs/REVIEW.md), or the epoch review window at a Review epochs strip;
        close it when the run no longer waits."""
        window: ReviewWindow | EpochReviewWindow | None = (
            self.review_window or self.epoch_window)  # fmt: skip
        if request is not None and request == self._answered:
            return  # answered; the run has not picked the answer up yet
        if request is None:
            self._answered = None
            if window is not None:
                window.answered = True  # answered elsewhere, or the run ended
                window.close()
                self.review_window = self.epoch_window = None
            return
        if window is not None and window.request == request:
            return
        if window is not None:  # the next strip, or the next recording
            window.answered = True
            window.close()
            self.review_window = self.epoch_window = None
        current = self.offline_recordings.get("current", "")
        self.log(f"Review needed at {request['node']!r}"
                 + (f" for {current}" if current else "")
                 + f" ({request['reason']}); the run waits.")  # fmt: skip
        if request.get("kind") == "epochs":
            epochs = self.epoch_window = EpochReviewWindow(
                request, self.theme,
                submit=lambda verdicts: self._answer_review(
                    "submit_review", marks=[], epochs=verdicts),
                cancel=lambda: self._answer_review("cancel_review"),
                fetch=lambda indices, reduce: self.call(
                    "offline_review_epochs", indices=indices, reduce=reduce,
                    max_samples=EPOCH_POINTS if reduce == "none" else 1000),
                recording=current,
            )  # fmt: skip
            epochs.setStyleSheet(stylesheet(self.theme, self.scale))
            epochs.show()
            epochs.raise_()
            return
        window = self.review_window = ReviewWindow(
            request, self.theme,
            submit=lambda marks, viewed, bads: self._answer_review(
                "submit_review", marks=marks, viewed=viewed, bads=bads),
            cancel=lambda: self._answer_review("cancel_review"),
            recording=current,
            settings=self.settings.get("review"),
        )  # fmt: skip
        window.on_display = self.redraw
        window.setStyleSheet(stylesheet(self.theme, self.scale))
        window.show()
        window.raise_()

    def review_at_import(self, entry: dict[str, Any]) -> None:
        """Review the recording itself, before any pipeline (docs/REVIEW.md):
        an offline run of the raw input shows it; the review is saved with the
        recording and every pipeline reading it uses it."""
        if self.running or self.offline_state in ("running", "review"):
            self.log("Stop the running pipeline before reviewing at import.")
            return
        node = {"node": entry["node"], "params": dict(entry.get("params", {}))}
        request = self.call("import_review_open", node=node)
        if request is None:
            return
        reviewer = self.ask_reviewer()
        if reviewer is None:
            return
        raw = {"id": "input", "node": node["node"],
               "params": {**node["params"], "import_review": False}}  # fmt: skip
        tap = {"id": IMPORT_TAP, "node": "meegxer.core.probe", "input": "input",
               "params": {"keep_seconds": 0.0}}  # fmt: skip
        pipeline = {"spec_version": SPEC_VERSION, "nodes": [raw, tap]}
        if self.call("run_offline", pipeline=pipeline) is None:
            return
        self.offline_state = "running"
        self.offline_probes.clear()
        self._offline_cache.clear()
        if self.import_window is not None:
            self.import_window.answered = True
            self.import_window.close()
        name = Path(str(node["params"].get("path") or request["file"])).name
        window = self.import_window = ReviewWindow(
            request, self.theme,
            submit=lambda marks, viewed, bads: self._save_import_review(
                node, marks, viewed, bads, reviewer),
            cancel=lambda: None, recording=name,
            settings=self.settings.get("review"), import_mode=True,
        )  # fmt: skip
        window.probe = IMPORT_TAP
        window.on_display = self.redraw
        window.setStyleSheet(stylesheet(self.theme, self.scale))
        window.show()
        window.raise_()
        self.log(f"Reviewing {name} at import (the raw recording).")

    def _save_import_review(self, node: dict[str, Any], marks: list[dict[str, Any]],
                            viewed: list[list[float]], bads: list[str],
                            reviewer: str) -> None:  # fmt: skip
        window = self.import_window
        crop = window.crop_span() if window is not None else None
        reply = self.call("save_import_review", node=node, marks=marks, bads=bads,
                          crop=None if crop is None else list(crop),
                          viewed=viewed, reviewer=reviewer)  # fmt: skip
        if reply is None:
            return
        self.log(f"Review at import saved to {reply['file']}; every pipeline "
                 "reading this recording uses it.")  # fmt: skip
        if self.recording_window.isVisible():
            self.recording_window.load(self.recording_window.entry)
        self.use_recording_info(self._describe_input())

    def _answer_review(self, method: str, **params: Any) -> None:
        window = self.review_window or self.epoch_window
        if window is not None:
            self._answered = window.request
        self.review_window = self.epoch_window = None
        if self.call(method, **params) is not None:
            n = len(params.get("marks", []))
            saved = f"Review saved ({n} mark{'s' if n != 1 else ''}); the run goes on."
            if "epochs" in params:
                kept = sum(bool(v.get("keep", True)) for v in params["epochs"])
                saved = (f"Epoch review saved ({kept} of {len(params['epochs'])} "
                         "kept); the run goes on.")  # fmt: skip
            self.log(saved if method == "submit_review" else "Review cancelled.")

    def _meters(self, status: dict[str, Any]) -> None:
        m = self.meters
        if self.doc.mode == "offline":
            rec = self.offline_recordings
            where = (f" {rec['index']}/{rec['total']} {rec['current']}"
                     if rec.get("current") and self.offline_state == "running"
                     else "")  # fmt: skip
            m["state"].setText(f"offline: {self.offline_state or 'ready'}{where}")
            for key in ("load", "latency", "overruns", "display"):
                m[key].setText("")
            return
        m["state"].setText("● running" if status.get("running") else "○ stopped")
        if not status.get("running"):
            for key in ("load", "latency", "overruns", "display"):
                m[key].setText("")
            return
        blocks = status.get("blocks") or 0
        uptime = status.get("uptime_s") or 0
        if blocks and uptime:
            busy = sum(
                (n.get("timing") or {}).get("mean_ms", 0.0)
                for n in status.get("nodes", [])
            )
            m["load"].setText(
                f"processing load {100 * busy / (1000 * uptime / blocks):.0f}%"
            )
        p95 = (status.get("latency") or {}).get("p95_ms")
        m["latency"].setText(
            f"delay {p95:.0f} ms (95% of blocks)" if p95 is not None else ""
        )
        if "overruns" in status:
            m["overruns"].setText(f"late blocks {status['overruns']}")
        shown = sum(b.dropped for b in self.buffers.values())
        shown += sum(b.dropped for b in self.feature_buffers.values())
        m["display"].setText(f"display frames skipped {shown}" if self.buffers else "")

    def on_frame(self, frame: DisplayFrame | FeatureFrame) -> None:
        if isinstance(frame, FeatureFrame):
            features = self.feature_buffers.get(frame.probe)
            if features is None:
                features = self.feature_buffers[frame.probe] = FeatureBuffer()
            features.add(frame)
            return
        buf = self.buffers.get(frame.probe)
        if buf is None:
            buf = self.buffers[frame.probe] = ProbeBuffer()
        buf.add(frame)

    # -- drawing ----------------------------------------------------------------------

    def _shown(self) -> tuple[str | None, list[str]]:
        if self.view_tap is not None:  # a rail output (features have their view)
            return tap_id(self.view_tap), []
        main = next((tap_id(k) for k, v in self.taps.items() if v == "main"), None)
        comps = [tap_id(k) for k, v in self.taps.items() if v == "comparison"]
        return main, comps

    def _live_reference(self, buf: ProbeBuffer) -> ProbeBuffer:
        """The Main probe's buffer, so all probes line up by index (a tap after
        a resampler counts other bins: its own)."""
        main, _ = self._shown()
        ref = self.buffers.get(main) if main else None
        same = ref is not None and ref.bin_size == buf.bin_size
        if ref is not None and same and ref.bin_rate == buf.bin_rate:
            return ref
        return buf

    def _live_window(
        self, probe: str, seconds: float, end: int | None = None
    ) -> Window | None:
        """The window ending at bin ``end`` (default: the newest); time in
        seconds before the newest data, or since the stream started (setting
        ``visualization.time_axis``). Always the full span: while the history
        is shorter, the part before the first data stays blank."""
        buf = self.buffers.get(probe)
        if buf is None or not buf.labels or not buf.bin_rate:
            return None
        ref = self._live_reference(buf)
        if end is not None:  # the panner counts the Main probe's bins
            main, _ = self._shown()
            panned = self.buffers.get(main) if main else None
            if panned is not None and panned.bin_rate != buf.bin_rate:
                end = round(end * buf.bin_rate / panned.bin_rate)  # the same time
        newest = ref.end_bin
        end = newest if end is None else min(end, newest)
        n = max(2, int(seconds * buf.bin_rate))
        mins, maxs, bypassed = buf.window(end, n)
        origin = 0 if self._since_start() else newest  # bin 0 = the source's sample 0
        t = (np.arange(end - n, end) - origin) / buf.bin_rate
        return Window(probe, buf.labels, t, mins, maxs, bypassed, buf.bin_rate,
                      buf.classes, buf.scores, buf.filters)  # fmt: skip

    def _offline_window(
        self,
        probe: str,
        seconds: float,
        offset: int | None = None,
        rate: float | None = None,
        display: dict[str, Any] | None = None,
    ) -> Window | None:
        """The window starting ``offset`` samples, counted at ``rate``, into the
        recording (default: where the panner is, at the input's rate). A tap
        after a resampler is at another rate: the same time (rule 8). With
        ``display`` (the review window's), through a display montage and
        filter computed in the core."""
        info = self.offline_probes.get(probe)
        if info is None or not info.get("n_filled"):
            return None
        n = max(2, int(seconds * info["sfreq"]))
        if offset is None:
            offset, rate = self.timeline.position(), self.offline_sfreq
        if rate and rate != info["sfreq"]:
            offset = round(offset * info["sfreq"] / rate)
        start = info["first_sample"] + offset
        key = (start, n, int(info["n_filled"]), repr(display))
        slot = probe if display is None else f"{probe}#review"  # apart from the view
        cached = self._offline_cache.get(slot)
        if cached is not None and cached[0] == key:
            return cached[1]
        options = {"display": display} if display else {}
        reply = self.call("offline_window", probe=probe, start=start, n_samples=n,
                          max_bins=2000, **options)  # fmt: skip
        if reply is None:
            self._offline_cache[slot] = (key, None)  # not asked again each redraw
            return None
        mins = np.array(reply["mins"], dtype=float).reshape(len(reply["ch_labels"]), -1)
        maxs = np.array(reply["maxs"], dtype=float).reshape(mins.shape)
        size = reply["bin_size"]
        t = (reply["start_bin"] + np.arange(mins.shape[1])) * size / info["sfreq"]
        bypassed = np.full(mins.shape[1], bool(info.get("bypassed")))
        win = Window(
            probe,
            tuple(reply["ch_labels"]),
            t,
            mins,
            maxs,
            bypassed,
            info["sfreq"] / size,
            tuple(info.get("ch_classes", ())),
            tuple(info.get("ch_scores", ())),
            filters_from_wire(info.get("filters")),
        )
        self._offline_cache[slot] = (key, win)
        return win

    def redraw(self) -> None:
        features = self.view_tap is not None and (
            self.doc.tap_kind(self.view_tap) == "features"
        )
        self.views.setCurrentIndex(1 if features else 0)
        if features:
            self._draw_features()
            return
        main, comps = self._shown()
        review = self.review_window
        if review is None or not review.isVisible():  # else a review at import
            review = self.import_window
            if review is not None and review.probe:
                main, comps = review.probe, []
        if review is not None and review.isVisible():  # the run waits for it
            info = self.offline_probes.get(main) if main else None
            info = info or next(iter(self.offline_probes.values()), None)
            review.seek_rate = float(info["sfreq"]) if info else 0.0
            review.source_labels = tuple(info.get("ch_labels", ())) if info else ()
            provider = self._panned_provider(review.panner, review.seconds(), main,
                                             review.display())  # fmt: skip
            review.refresh(self.view, provider, main, comps)
        if self.stacked.isVisible():  # the signal view rests meanwhile
            self.stacked.refresh(self.view, self._stacked_provider(main), main, comps)
            return
        provider = self._panned_provider(self.timeline, self.view.seconds(), main)
        self.view.mixing = self._mixing_for(self.view_tap)
        self.view.refresh(provider, main, comps)

    def _mixing_for(
        self, tap: str | None
    ) -> tuple[tuple[str, ...], npt.NDArray[np.float64]] | None:
        """For a rail of components in the spectrum view (``ica1:ics``): the
        sensors and mixing matrix of the node writing it (ICA's ``winv``), so
        the head plot maps the selected components back on the electrodes.
        Read from the core when the run changes, and live every 2 s (a
        calibration replaces the model)."""
        if tap is None or ":" not in tap or self.view.mode != "spectrum":
            return None
        node = tap.split(":", 1)[0]
        offline = self.doc.mode == "offline"
        key = (offline, self.offline_state if offline else "")
        now = time.monotonic()
        cached = self._mixing.get(node)
        if cached is not None and cached[0] == key and (offline or now - cached[1] < 2):
            return cached[2]  # type: ignore[no-any-return]
        mixing = None
        try:
            reply = self.link.call("get_model", node=node, offline=offline)
        except (RpcError, TimeoutError):
            reply = None  # e.g. nothing loaded yet: no map until there is a model
        model = state_from_json(reply["model"]) if reply and reply["model"] else None
        if isinstance(model, dict) and "winv" in model and "channels" in model:
            mixing = (tuple(model["channels"]), np.asarray(model["winv"], dtype=float))
        self._mixing[node] = (key, now, mixing)
        return mixing

    def _draw_features(self) -> None:
        """The tapped feature strip's windows over the feature view's span:
        live from the display stream, offline from the run's feature store."""
        assert self.view_tap is not None
        probe = tap_id(self.view_tap)
        view = self.feature_view
        name = self.view.names.get(probe, self.view_tap)
        seconds = view.seconds()
        if self.doc.mode == "live":
            series, t0, t1 = self._live_features(probe, seconds)
        else:
            series, t0, t1 = self._offline_features(probe, seconds)
        view.refresh(series, name, t0, t1)

    def _live_features(
        self, probe: str, seconds: float
    ) -> tuple[FeatureSeries | None, float, float]:
        buf = self.feature_buffers.get(probe)
        panner = self.feature_view.panner
        if buf is None or not buf.windows or not buf.sfreq:
            panner.set_extent(0, 0, 1, live=True)
            return None, -seconds, 0.0
        ends = np.array([w[1] for w in buf.windows])
        newest, n = int(ends[-1]), max(1, int(seconds * buf.sfreq))
        panner.since_start = self._since_start()
        panner.set_extent(min(int(ends[0]) + n, newest), newest, n, live=True)
        end = panner.position()
        idx = np.flatnonzero((ends > end - n) & (ends <= end))
        if FeatureView.spectral(buf.dims) and idx.size:
            idx = idx[-1:]  # live spectrum: the newest window shown
        origin = 0 if self._since_start() else newest
        t0, t1 = (end - n - origin) / buf.sfreq, (end - origin) / buf.sfreq
        if not idx.size:
            return None, t0, t1
        values = np.stack([buf.values[i] for i in idx])
        t = (ends[idx] - origin) / buf.sfreq
        series = FeatureSeries(probe, buf.dims, buf.coords, t, values, 0, *buf.units)
        return series, t0, t1

    def _offline_features(
        self, probe: str, seconds: float
    ) -> tuple[FeatureSeries | None, float, float]:
        info = self.offline_features.get(probe)
        panner, sfreq = self.feature_view.panner, self.offline_sfreq
        if info is None or not sfreq:
            return None, 0.0, seconds
        n = max(1, int(seconds * sfreq))
        panner.set_extent(0, max(0, self.offline_total - n), n, live=False)
        start = panner.position()
        t0, t1 = start / sfreq, (start + n) / sfreq
        dims = tuple(info["dims"])
        reduce = "mean" if FeatureView.spectral(dims) else "none"
        key = (start, n, reduce, int(info["n_windows"]))
        cached = self._feature_cache.get(probe)
        if cached is not None and cached[0] == key:
            return cached[1], t0, t1
        reply = self.call("offline_features", probe=probe, start=start,
                          stop=start + n, max_windows=2000, reduce=reduce)  # fmt: skip
        if reply is None or not reply["windows"]:
            return None, t0, t1
        coords = {k: tuple(v) for k, v in reply["coords"].items()}
        values = np.array(reply["values"], dtype=float)
        stops = np.array([w[1] for w in reply["windows"]], dtype=float)
        series = FeatureSeries(probe, dims, coords, stops / sfreq, values,
                               int(reply["n_windows"]),
                               *units_from_wire(reply))  # fmt: skip
        self._feature_cache[probe] = (key, series)
        return series, t0, t1

    def _stacked_provider(self, main: str | None) -> Callable[..., Window | None]:
        return self._panned_provider(self.stacked.panner, self.stacked.seconds(), main)

    def _since_start(self) -> bool:
        return bool(self.settings["visualization"].get("time_axis") == "since_start")

    def _panned_provider(
        self,
        panner: Panner,
        seconds: float,
        main: str | None,
        display: dict[str, Any] | None = None,
    ) -> Callable[..., Window | None]:
        """Windows at a panner's position, after setting its range: the whole
        recording offline, the kept history live."""
        if self.doc.mode == "live":
            buf = self.buffers.get(main) if main else None
            buf = buf or next(iter(self.buffers.values()), None)
            if buf is None or not buf.bin_rate:
                return self._live_window
            ref = self._live_reference(buf)
            newest, oldest = ref.end_bin, ref.oldest_bin
            n = int(seconds * buf.bin_rate)
            panner.since_start = self._since_start()
            panner.set_extent(min(oldest + n, newest), newest, n, live=True)
            end = panner.position()
            return lambda p, s: self._live_window(p, s, end=end)
        info = self.offline_probes.get(main) if main else None
        info = info or next(iter(self.offline_probes.values()), None)
        if info is None:
            return self._offline_window
        span = int(seconds * info["sfreq"])
        total = int(info.get("n_samples") or 0)
        panner.set_extent(0, max(0, total - span), span, live=False)
        offset, rate = panner.position(), info["sfreq"]
        return lambda p, s: self._offline_window(p, s, offset=offset, rate=rate,
                                                 display=display)  # fmt: skip

    def show_stacked(self) -> None:
        self.stacked.show()
        self.stacked.raise_()
        self.view.set_active(False)
        self.timeline.setEnabled(False)

    def _stacked_closed(self) -> None:
        self.view.set_active(True)
        self.timeline.setEnabled(True)

    def _confirm_close(self) -> str:
        """While running: "stop", "keep" (running in the daemon) or "cancel"."""
        box = QMessageBox(self)
        box.setWindowTitle("Close MeegXer")
        box.setText("The pipeline is running.")
        box.setInformativeText("Stop it, or keep it running in the processing "
                               "engine after the window closes?")  # fmt: skip
        stop = box.addButton("Stop and close", QMessageBox.ButtonRole.AcceptRole)
        keep = box.addButton("Keep running", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "stop" if clicked is stop else "keep" if clicked is keep else "cancel"

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.running and self.settings["general"]["confirm_quit_running"]:
            choice = self._confirm_close()
            if choice == "cancel":
                event.ignore()
                return
            if choice == "stop":
                self.stop()
        self.save_state()
        self._render.stop()
        self.stacked.close()
        self.recording_window.close()
        if self.owns_daemon and not self.running:
            self.call("shutdown")
        elif self.running:
            print(
                "MeegXer: the pipeline keeps running in the daemon "
                "(stop it with `meegxer stop`)."
            )
        self.link.close()
        super().closeEvent(event)
