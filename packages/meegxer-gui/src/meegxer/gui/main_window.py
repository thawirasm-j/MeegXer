"""The MeegXer window: top bar, signal view, chain of strips, inspector and log.

A pure client of the daemon (docs/DESIGN.md §3.7): every action is a control
call; the signal view draws what arrives on the display stream (live) or what the
daemon serves from an offline run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from meegxer.core.control import RpcError
from meegxer.core.display import DisplayFrame
from meegxer.core.session import is_session, read_session, write_session
from meegxer.gui.buffers import ProbeBuffer
from meegxer.gui.chain import ChainView
from meegxer.gui.colors import wheel_positions
from meegxer.gui.connection import DaemonLink
from meegxer.gui.inspector import Inspector
from meegxer.gui.pipeline_doc import LIVE_ONLY_SOURCES, PipelineDoc, tap_id
from meegxer.gui.recording_window import RecordingWindow
from meegxer.gui.signal_view import SignalView, StackedWindow, Window
from meegxer.gui.theme import Theme, load_theme, stylesheet
from meegxer.gui.wheel import install_wheel_guard

DEFAULT_CHAIN: list[tuple[str, dict[str, Any]]] = [
    ("meegxer.sterile.iir_filter", {"kind": "highpass", "low": 1.0}),
    ("meegxer.sterile.notch", {}),
]


class MainWindow(QMainWindow):
    def __init__(
        self, link: DaemonLink, theme: Theme | None = None, *, owns_daemon: bool = False
    ) -> None:
        super().__init__()
        self.link = link
        self.owns_daemon = owns_daemon
        self.theme = theme or load_theme("dark")
        self.doc = PipelineDoc(link.call("list_nodes"))
        self.buffers: dict[str, ProbeBuffer] = {}
        self.taps: dict[str, str] = {}
        self.selected: str | None = None
        self.advanced = False
        self.running = False
        self.statuses: dict[str, str] = {}
        self.node_status: dict[str, dict[str, Any]] = {}
        self.offline_probes: dict[str, dict[str, Any]] = {}
        self.offline_state = ""
        self._offline_cache: dict[str, tuple[tuple[int, int, int], Window]] = {}
        app = QApplication.instance()
        if isinstance(app, QApplication):
            install_wheel_guard(app)  # wheel scrolls panels, not unfocused values
        self.setWindowTitle("MeegXer")
        self.resize(1440, 900)
        self._build()
        self._load_initial()
        link.status.connect(self.on_status)
        link.frame.connect(self.on_frame)
        link.lost.connect(lambda msg: self.log(f"Daemon not answering: {msg}"))
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
        sp.addWidget(self.view, 1)
        self.timeline = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline.setVisible(False)
        self.timeline.valueChanged.connect(lambda _: self._offline_cache.clear())
        sp.addWidget(self.timeline)
        splitter.addWidget(signal_panel)

        self.chain = ChainView()
        self.chain.selected.connect(self.select)
        self.chain.remove_requested.connect(self.remove_strip)
        self.chain.insert_requested.connect(self.insert_strip)
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
        self.recording_window.applied.connect(self.apply_recording)

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
            app.setStyleSheet(stylesheet(theme))
        self.view.apply_theme(theme)
        self.stacked.apply_theme(theme)
        self.stacked.setStyleSheet(stylesheet(theme))
        self.recording_window.theme = theme
        self.recording_window.setStyleSheet(stylesheet(theme))
        self.theme_btn.setText("☀" if theme.dark else "☾")
        self.refresh_chain()

    def toggle_theme(self) -> None:
        self.apply_theme(load_theme("light" if self.theme.dark else "dark"))

    def refresh_chain(self) -> None:
        self.chain.rebuild(
            self.doc,
            self.theme,
            statuses=self.statuses,
            taps=self.taps,
            advanced=self.advanced,
            selected=self.selected,
        )
        self.refresh_inspector()
        self._transport()

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
        self.start_btn.setEnabled(not self.running and self.offline_state != "running")
        self.stop_btn.setVisible(live)
        self.stop_btn.setEnabled(self.running)
        self.timeline.setVisible(not live and bool(self.offline_probes))

    def _default_taps(self) -> None:
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
            self.doc.set_source("meegxer.core.synthetic")
            for node, params in DEFAULT_CHAIN:
                self.doc.insert(len(self.doc.strips), node).params.update(params)
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
        self.log(what)
        self.refresh_chain()
        if self.running:
            self.log("Restarting to apply the change (fitted models are kept).")
            self.restart()

    def select(self, strip_id: str) -> None:
        self.selected = strip_id
        self.refresh_chain()

    def insert_strip(self, index: int, node: str) -> None:
        strip = self.doc.insert(index, node)
        self.selected = strip.id
        self._structure_changed(f"Added {strip.id}.")

    def remove_strip(self, strip_id: str) -> None:
        self.doc.remove(strip_id)
        self._structure_changed(f"Removed {strip_id}.")

    def move_strip(self, strip_id: str, index: int) -> None:
        try:
            self.doc.move(strip_id, index)
        except (KeyError, ValueError) as exc:
            self.log(str(exc))
            return
        self._structure_changed(f"Moved {strip_id}.")

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
        offline view (live views have no absolute time axis)."""
        if info is None:
            self.view.set_wheel({})
            self.view.types = {}
            self.view.events = []
            return
        self.view.types = {c["label"]: c["type"] for c in info["channels"]}
        positions = {
            c["label"]: tuple(c["position"]) for c in info["channels"] if c["position"]
        }
        self.view.set_wheel(wheel_positions(positions))
        self.view.events = list(info["events"]) if self.doc.mode == "offline" else []

    def _describe_input(self) -> None:
        """Read the input's description before a run (not for LSL, whose stream
        the run itself opens)."""
        src = self.doc.source
        if src is None or src.node in LIVE_ONLY_SOURCES:
            self.use_recording_info(None)
            return
        self.use_recording_info(
            self.call("describe_source", node=self.doc.source_entry())
        )

    def change_source(self, node: str) -> None:
        if self.doc.source is not None and node == self.doc.source.node:
            return
        self.doc.set_source(node)
        self.use_recording_info(None)
        self.offline_probes.clear()
        self._structure_changed(f"Input is now {node} ({self.doc.mode}).")

    def toggle_mode(self) -> None:
        if self.running:
            self.stop()
        self.doc.input_mode = "offline" if self.doc.mode == "live" else "live"
        self.log(f"Mode: {self.doc.mode}.")
        self.refresh_chain()

    def click_tap(self, strip_id: str, make_main: bool) -> None:
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
                self.log(f"{strip_id}.{name} = {value}: restarting.")
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
            taps=self.taps,
            advanced=self.advanced,
            selected=self.selected,
        )

    def change_channel_types(self, strip_id: str, types: list[str]) -> None:
        strip = self.doc.get(strip_id)
        if (strip.channel_types or ["eeg"]) == types:
            return
        strip.channel_types = [] if types == ["eeg"] else types
        self._structure_changed(f"{strip_id} processes: {', '.join(types)}.")

    def change_passthrough(self, strip_id: str, enabled: bool) -> None:
        self.doc.get(strip_id).pass_uncalibrated = enabled
        if self.running:
            self.call("set_pass_uncalibrated", node=strip_id, enabled=enabled)
        self.log(
            f"{strip_id}: pass through while uncalibrated {'on' if enabled else 'off'}."
        )

    def calibrate(self, strip_id: str, seconds: float) -> None:
        if not self.running:
            self.log("Start the pipeline before calibrating.")
            return
        if self.call("calibrate", node=strip_id, seconds=seconds) is not None:
            self.log(f"Calibrating {strip_id} on {seconds:g} s of input…")

    def clear_calibration(self, strip_id: str) -> None:
        if self.call("clear_calibration", node=strip_id) is not None:
            self.log(f"Cleared the calibration of {strip_id}.")

    # -- transport --------------------------------------------------------------------

    def start(self) -> None:
        problems = self.doc.problems()
        if problems:
            self.log("Cannot start: " + "; ".join(problems))
            return
        self._describe_input()
        if self.doc.mode == "offline":
            pipeline = self.doc.to_pipeline(tap_keep_seconds=0.0)
            result = self.call("run_offline", pipeline=pipeline)
            if result is not None:
                self.offline_state = "running"
                self.offline_probes.clear()
                self._offline_cache.clear()
                self.log("Offline run started.")
            self._transport()
            return
        if self.call("load_pipeline", pipeline=self.doc.to_pipeline()) is None:
            return
        self.buffers.clear()
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
            self, "Open pipeline or session", "",
            "Pipeline or session (*.json);;MeegXer session (*.meegxer.json)",
        )  # fmt: skip
        if not path:
            return
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

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    def save_session(self) -> None:
        """Save what is shown as a session, with the models fitted by the last
        offline run (offline mode) or calibrated live."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "session.meegxer.json",
            "MeegXer session (*.meegxer.json)",
        )  # fmt: skip
        if not path:
            return
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
        try:
            write_session(path, session)
        except OSError as exc:
            self.log(f"Could not save {path}: {exc}")
            return
        models = ", ".join(session.get("models", {})) or "none"
        outputs = len(session.get("outputs", []))
        self.log(f"Saved session {path} (fitted models: {models}; "
                 f"output files: {outputs}).")  # fmt: skip

    def save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pipeline", "pipeline.json", "Pipeline (*.json)"
        )
        if path:
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
            self.log(f"Daemon error: {err}")
        self._errors = list(status.get("errors", []))
        if offline:
            self._offline_status(offline)
        if not running and self.running:
            self.log("The pipeline stopped (end of input or an error).")
        changed = running != self.running or statuses != self.statuses
        self.running, self.statuses = running, statuses
        self._meters(status)
        if changed:
            self.refresh_chain()

    def _offline_status(self, offline: dict[str, Any]) -> None:
        """Probes become scrollable as soon as they have data (the run writes
        them as it goes); the scrollbar spans the whole recording."""
        self.offline_probes = offline.get("probes", {})
        total = int(offline.get("n_samples") or 0)
        sfreq = next((p["sfreq"] for p in self.offline_probes.values()), 0.0)
        span = int(self.view.seconds() * sfreq)
        self.timeline.setRange(0, max(0, total - span))
        self.timeline.setPageStep(max(1, span))
        self.timeline.setSingleStep(max(1, span // 10))
        state = offline.get("state", "")
        if state == self.offline_state:
            return
        self.offline_state = state
        if state == "done":
            self.log(f"Offline run done in {offline.get('elapsed_s', 0):.1f} s.")
        elif state == "error":
            self.log(f"Offline run failed: {offline.get('error')}")
        self._transport()

    def _meters(self, status: dict[str, Any]) -> None:
        m = self.meters
        if self.doc.mode == "offline":
            m["state"].setText(f"offline: {self.offline_state or 'ready'}")
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
            m["load"].setText(f"load {100 * busy / (1000 * uptime / blocks):.0f}%")
        p95 = (status.get("latency") or {}).get("p95_ms")
        m["latency"].setText(f"latency p95 {p95:.0f} ms" if p95 is not None else "")
        if "overruns" in status:
            m["overruns"].setText(f"overruns {status['overruns']}")
        shown = sum(b.dropped for b in self.buffers.values())
        m["display"].setText(f"frames dropped {shown}" if self.buffers else "")

    def on_frame(self, frame: DisplayFrame) -> None:
        buf = self.buffers.get(frame.probe)
        if buf is None:
            buf = self.buffers[frame.probe] = ProbeBuffer()
        buf.add(frame)

    # -- drawing ----------------------------------------------------------------------

    def _shown(self) -> tuple[str | None, list[str]]:
        main = next((tap_id(k) for k, v in self.taps.items() if v == "main"), None)
        comps = [tap_id(k) for k, v in self.taps.items() if v == "comparison"]
        return main, comps

    def _live_reference(self, buf: ProbeBuffer) -> int:
        """The newest bin: the Main probe's, so all probes line up by index."""
        main, _ = self._shown()
        ref = self.buffers.get(main) if main else None
        if ref is not None and ref.bin_size == buf.bin_size:
            return ref.end_bin
        return buf.end_bin

    def _live_window(
        self, probe: str, seconds: float, end: int | None = None
    ) -> Window | None:
        """The window ending at bin ``end`` (default: the newest); time in
        seconds before the newest data."""
        buf = self.buffers.get(probe)
        if buf is None or not buf.labels or not buf.bin_rate:
            return None
        newest = self._live_reference(buf)
        end = newest if end is None else min(end, newest)
        n = max(2, int(seconds * buf.bin_rate))
        mins, maxs, bypassed = buf.window(end, n)
        t = (np.arange(end - n, end) - newest) / buf.bin_rate
        return Window(probe, buf.labels, t, mins, maxs, bypassed, buf.bin_rate)

    def _offline_window(
        self, probe: str, seconds: float, offset: int | None = None
    ) -> Window | None:
        """The window starting ``offset`` samples into the recording (default:
        where the timeline is)."""
        info = self.offline_probes.get(probe)
        if info is None or not info.get("n_filled"):
            return None
        n = max(2, int(seconds * info["sfreq"]))
        offset = self.timeline.value() if offset is None else offset
        start = info["first_sample"] + offset
        key = (start, n, int(info["n_filled"]))
        cached = self._offline_cache.get(probe)
        if cached is not None and cached[0] == key:
            return cached[1]
        reply = self.call(
            "offline_window", probe=probe, start=start, n_samples=n, max_bins=2000
        )
        if reply is None:
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
        )
        self._offline_cache[probe] = (key, win)
        return win

    def redraw(self) -> None:
        main, comps = self._shown()
        if self.stacked.isVisible():  # the signal view rests meanwhile
            self.stacked.refresh(self.view, self._stacked_provider(main), main, comps)
            return
        provider = (
            self._live_window if self.doc.mode == "live" else self._offline_window
        )
        self.view.refresh(provider, main, comps)

    def _stacked_provider(self, main: str | None) -> Callable[..., Window | None]:
        """Windows at the stacked window's own scroll position, after setting
        its range: the whole recording offline, the kept history live."""
        seconds = self.stacked.seconds()
        if self.doc.mode == "live":
            buf = self.buffers.get(main) if main else None
            buf = buf or next(iter(self.buffers.values()), None)
            if buf is None or not buf.bin_rate:
                return self._live_window
            newest = self._live_reference(buf)
            n = int(seconds * buf.bin_rate)
            oldest = max(0, newest - buf.capacity)
            self.stacked.set_extent(min(oldest + n, newest), newest, n, live=True)
            end = self.stacked.position()
            return lambda p, s: self._live_window(p, s, end=end)
        info = self.offline_probes.get(main) if main else None
        info = info or next(iter(self.offline_probes.values()), None)
        if info is None:
            return self._offline_window
        span = int(seconds * info["sfreq"])
        total = int(info.get("n_samples") or 0)
        self.stacked.set_extent(0, max(0, total - span), span, live=False)
        offset = self.stacked.position()
        return lambda p, s: self._offline_window(p, s, offset=offset)

    def show_stacked(self) -> None:
        self.stacked.show()
        self.stacked.raise_()
        self.view.set_active(False)
        self.timeline.setEnabled(False)

    def _stacked_closed(self) -> None:
        self.view.set_active(True)
        self.timeline.setEnabled(True)

    def closeEvent(self, event: QCloseEvent) -> None:
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
