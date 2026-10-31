"""Real-time scheduler: block-major processing on a dedicated thread (rule 4).

Each loop iteration pulls whatever the source has ready and pushes that block
through the graph in topological order, using the same ``Stage.transform`` the
batch scheduler uses. All graph mutations (parameter changes, calibration
requests) are queued and applied by the scheduler thread between blocks, so node
code never runs concurrently with itself.

Timing: a block of ``n`` samples has a deadline of ``n / sfreq`` seconds; if
processing takes longer the scheduler is falling behind and the block counts as
an overrun. End-to-end latency is ``clock()`` at the sinks minus the LSL timestamp
of the block's last sample.

Calibration (one-shot, online): a stage that needs fitting passes nothing
downstream until it is calibrated. ``calibrate(node, seconds)`` buffers that many
seconds of the stage's input, fits, and from then on the stage runs. A node that
learns from labelled trials (``fit_events``) fits on the events of the Events rail
inside that stretch, after waiting ``event_wait`` seconds (the LSL input's
``marker_wait``) for markers still on their way; processing goes on meanwhile.

Events rail (docs/RAILS.md): the input's events (a played recording's, or LSL
markers placed on the signal's sample index) and the events strips emit.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pylsl

from meegxer.core.block import Block, FeatureBlock, SignalInfo, concat_blocks
from meegxer.core.display import (
    Decimator,
    DisplayFrame,
    DisplayPublisher,
    Envelope,
    FeatureFrame,
)
from meegxer.core.events import EventsRail
from meegxer.core.graph import Graph, GraphError, GraphNode, positions_message
from meegxer.core.node import (
    Probe,
    RailInput,
    RailOutput,
    Sink,
    Source,
    output_blocks,
)
from meegxer.core.ports import PortType
from meegxer.core.probes import ErpProbe
from meegxer.core.rails import (
    RailError,
    RailFeed,
    check_rates,
    join_inputs,
    rail_fingerprint,
    with_rails,
)
from meegxer.core.recording import recording_events
from meegxer.core.stage import Stage, StageStatus, StaleModelError, signal_fingerprint

Clock = Callable[[], float]
EVENTS_KEEP_SECONDS = 60.0  # live Events rail kept beyond open calibrations
RAIL_HOLD_SECONDS = 30.0  # how long a reader's input waits for a lagging rail
FEATURE_FRAMES_PER_FLUSH = 16  # newest feature windows sent per probe and frame


class _Readable(Protocol):
    @property
    def info(self) -> SignalInfo: ...

    def read(self, max_samples: int) -> Block | None: ...


class PacedSource:
    """Releases a recorded or synthetic source at its real sampling rate.

    Timestamps are rewritten onto the LSL clock, starting when the first block
    is read, so latency is measured the same way as for a live LSL stream.
    Samples are released in chunks of at least ``min_chunk_seconds``.
    """

    def __init__(
        self,
        source: Source,
        clock: Clock = pylsl.local_clock,
        min_chunk_seconds: float = 0.01,
    ) -> None:
        self.source = source
        self._clock = clock
        self._min_chunk = max(1, round(min_chunk_seconds * source.info.sfreq))
        self._t0: float | None = None
        self._pos = 0

    @property
    def info(self) -> SignalInfo:
        return self.source.info

    def read(self, max_samples: int) -> Block | None:
        now = self._clock()
        if self._t0 is None:
            self._t0 = now
        sfreq = self.info.sfreq
        due = int((now - self._t0) * sfreq) + 1 - self._pos
        n = min(due, max_samples)
        if n < self._min_chunk:  # release ~10 ms chunks, like an amplifier would
            return Block(np.empty((self.info.n_channels, 0)), self.info, self._pos,
                         self._t0 + self._pos / sfreq)  # fmt: skip
        block = self.source.read(n)
        if block is None:
            return None
        self._pos = block.stop_index
        return Block(block.data, block.info, block.sample_index,
                     self._t0 + block.sample_index / sfreq)  # fmt: skip


@dataclass
class StageMetrics:
    blocks: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, seconds: float) -> None:
        self.blocks += 1
        self.total_s += seconds
        self.max_s = max(self.max_s, seconds)

    def as_dict(self) -> dict[str, float]:
        mean = self.total_s / self.blocks if self.blocks else 0.0
        return {
            "blocks": self.blocks,
            "mean_ms": mean * 1e3,
            "max_ms": self.max_s * 1e3,
        }


@dataclass
class Metrics:
    blocks: int = 0
    samples: int = 0
    overruns: int = 0
    events: int = 0  # emitted by strips
    input_events: int = 0  # the input's (a played recording's, or markers)
    events_late: int = 0  # input events inside a stretch already fitted on
    errors: list[str] = field(default_factory=list)
    stages: dict[str, StageMetrics] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    started_at: float | None = None

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies:
            return {}
        arr = np.asarray(self.latencies[-5000:]) * 1e3
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }


class RealtimeScheduler:
    """Runs a graph live. Thread-safe API: ``start``, ``stop``, ``submit`` and the
    convenience wrappers ``set_params`` / ``calibrate`` / ``status``."""

    def __init__(
        self,
        graph: Graph,
        *,
        max_block: int | None = None,
        idle_sleep: float = 0.002,
        display: DisplayPublisher | None = None,
        display_fps: float = 30.0,
        display_bins_per_second: float = 250.0,
        clock: Clock = pylsl.local_clock,
    ) -> None:
        offline = [n.id for n in graph.nodes if not n.manifest.runs_online]
        if offline:
            raise GraphError(f"offline-only nodes cannot run in real time: {offline}")
        for node in graph.nodes:
            if isinstance(node.impl, Stage):
                try:
                    node.impl.processor.for_mode(node.impl.params, "online")
                except ValueError as exc:
                    raise GraphError(f"{node.id}: {exc}") from exc
        self.graph = graph
        self.max_block = max_block
        self.idle_sleep = idle_sleep
        self.clock = clock
        self.display = display
        self._frame_interval = 1.0 / display_fps
        self._bins_per_second = display_bins_per_second
        self.metrics = Metrics()
        self._commands: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = (
            queue.Queue()
        )
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._finished = threading.Event()
        self._calibration: dict[str, tuple[list[Block], float]] = {}
        # labelled fits waiting for markers: calibration, due time, rails over it
        self._fits: dict[str, tuple[Block, float, Mapping[str, RailInput] | None]] = {}
        self._calibration_rails: dict[str, list[Mapping[str, RailInput]]] = {}
        self._feeds: dict[str, RailFeed] = {}  # readers of rails (wait and align)
        self._written: dict[str, RailOutput] = {}  # the last stage's rails
        # (writer, output port) -> [(reader, its input port)]
        self._subscribers: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for node in graph.nodes:
            for port, (producer, output) in graph.rail_sources(node.id).items():
                if producer:
                    self._subscribers.setdefault((producer, output), []).append(
                        (node.id, port)
                    )
        self._rail = EventsRail()
        self._stop_index = 0  # end of the last block processed
        self._fitted: deque[tuple[int, int]] = deque(maxlen=32)  # labelled stretches
        self._input_fps: dict[str, str] = {}
        self._decimators: dict[str, Decimator] = {}
        self._pending: dict[str, list[Any]] = {}
        self._pending_features: dict[str, list[FeatureBlock]] = {}
        self._seq: dict[str, int] = {}
        self._last_flush = 0.0
        self._lock = threading.Lock()
        self._last_snapshot = 0.0
        self._snapshot: dict[str, Any] = {
            "running": False,
            "nodes": [
                {"id": n.id, "node": n.manifest.name, "role": n.role.value,
                 "status": "stopped", "params": dict(n.impl.params), "timing": {}}
                for n in graph.nodes
            ],
        }  # fmt: skip

    # -- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("scheduler already started")
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="meegxer-rt", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the stream ends or the scheduler stops."""
        return self._finished.wait(timeout)

    @property
    def running(self) -> bool:
        return self._running.is_set() and not self._finished.is_set()

    # -- commands (applied on the scheduler thread) -------------------------------

    def submit(self, fn: Callable[[], Any]) -> Future[Any]:
        future: Future[Any] = Future()
        if self._thread is None or self._finished.is_set():
            try:
                future.set_result(fn())
            except Exception as exc:
                future.set_exception(exc)
        else:
            self._commands.put((fn, future))
        return future

    def set_params(self, node_id: str, params: dict[str, object]) -> dict[str, Any]:
        def apply() -> dict[str, Any]:
            node = self.graph[node_id]
            if not isinstance(node.impl, Stage):
                raise GraphError(f"{node_id!r} is a {node.role.value}; only processor "
                                 "parameters can change while running")  # fmt: skip
            new = node.impl.manifest.resolve_params({**node.impl.params, **params})
            try:
                node.impl.processor.for_mode(new, "online")
            except ValueError as exc:
                raise GraphError(f"{node_id}: {exc}") from exc
            node.impl.set_params(**params)
            self._calibration.pop(node_id, None)
            self._calibration_rails.pop(node_id, None)
            self._fits.pop(node_id, None)
            self._refresh_fingerprints()
            return dict(node.impl.params)

        result: dict[str, Any] = self.submit(apply).result(timeout=10)
        return result

    def calibrate(self, node_id: str, seconds: float) -> None:
        def apply() -> None:
            node = self.graph[node_id]
            if not (isinstance(node.impl, Stage) and node.impl.manifest.requires_fit):
                raise GraphError(f"{node_id!r} does not need calibration")
            if seconds <= 0:
                raise ValueError("seconds must be positive")
            node.impl.begin_calibration()
            self._fits.pop(node_id, None)  # a new stretch replaces one still waiting
            self._calibration[node_id] = ([], seconds)
            self._calibration_rails[node_id] = []

        self.submit(apply).result(timeout=10)

    def set_pass_uncalibrated(self, node_id: str, enabled: bool) -> None:
        def apply() -> None:
            node = self.graph[node_id]
            if not isinstance(node.impl, Stage):
                raise GraphError(f"{node_id!r} is not a processor")
            node.impl.set_pass_uncalibrated(enabled)

        self.submit(apply).result(timeout=10)

    def reset_probe(self, node_id: str) -> None:
        """Start an ERP tap's running averages over (its view's Reset)."""

        def apply() -> None:
            node = self.graph[node_id]
            if not isinstance(node.impl, ErpProbe):
                raise GraphError(f"{node_id!r} is not an ERP tap")
            node.impl.reset()

        self.submit(apply).result(timeout=10)

    def clear_calibration(self, node_id: str | None = None) -> list[str]:
        """Drop fitted models explicitly (all nodes, or one). Returns the ids."""

        def apply() -> list[str]:
            nodes = [self.graph[node_id]] if node_id else self.graph.nodes
            cleared = []
            for node in nodes:
                if isinstance(node.impl, Stage) and node.impl.manifest.requires_fit:
                    node.impl.clear_model()
                    self._calibration.pop(node.id, None)
                    self._calibration_rails.pop(node.id, None)
                    self._fits.pop(node.id, None)
                    cleared.append(node.id)
            if node_id and not cleared:
                raise GraphError(f"{node_id!r} does not need calibration")
            self._refresh_fingerprints()
            return cleared

        result: list[str] = self.submit(apply).result(timeout=10)
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)

    # -- the loop -----------------------------------------------------------------

    def _run(self) -> None:
        source_node = self.graph.source
        assert isinstance(source_node.impl, Source)
        source: _Readable = (
            source_node.impl if getattr(source_node.impl, "live", False)
            else PacedSource(source_node.impl, self.clock)
        )  # fmt: skip
        self.metrics.started_at = self.clock()
        try:
            missing = self.graph.missing_positions(source.info)  # LSL: once open
            if missing:
                self.metrics.errors.append(positions_message(missing))
                return
            self._refresh_fingerprints()
            assert isinstance(source_node.impl, Source)
            self._rail.sfreq = source.info.sfreq  # the rail counts on its clock
            played = recording_events(source_node.impl)  # a played recording's
            self._rail.add(played)
            self.metrics.input_events += len(played)
            chunk = self.max_block or max(1, int(source.info.sfreq))
            while self._running.is_set():
                self._drain_commands()
                block = source.read(chunk)
                self._take_input_events(source_node.impl)
                self._finish_fits()
                if block is None:
                    self._finish_fits(now=True)
                    self._report_unfinished_calibration(source.info.sfreq)
                    break
                if block.n_samples == 0:
                    self._maybe_flush_display()
                    self._update_snapshot()
                    time.sleep(self.idle_sleep)
                    continue
                started = time.perf_counter()
                self._stop_index = block.stop_index
                self._process(block)
                elapsed = time.perf_counter() - started
                self.metrics.blocks += 1
                self.metrics.samples += block.n_samples
                if elapsed > block.n_samples / block.info.sfreq:
                    self.metrics.overruns += 1
                self._maybe_flush_display()
                self._update_snapshot()
        except Exception:
            self.metrics.errors.append(traceback.format_exc(limit=5))
        finally:
            self._drain_commands()
            for node in self.graph.nodes:
                if isinstance(node.impl, Sink):
                    try:
                        node.impl.close()
                        notes = node.impl.skipped
                        self.metrics.errors.extend(f"{node.id}: {n}" for n in notes)
                    except Exception:
                        self.metrics.errors.append(traceback.format_exc(limit=5))
            close = getattr(source_node.impl, "close", None)
            if callable(close):
                close()
            self._maybe_flush_display(force=True)
            self._running.clear()
            self._update_snapshot(force=True)
            self._finished.set()

    def _report_unfinished_calibration(self, sfreq: float) -> None:
        """The input ended while a calibration was still collecting: say so
        (its node stays uncalibrated) instead of stopping without a reason."""
        for node_id, (blocks, seconds) in self._calibration.items():
            got = sum(b.n_samples for b in blocks) / sfreq
            self.metrics.errors.append(
                f"calibration of {node_id!r} did not finish: the input ended after "
                f"{got:.1f} of {seconds:g} s; it stays uncalibrated (use a longer "
                "input or a shorter calibration)"
            )

    def _take_input_events(self, source: Source) -> None:
        """Live events from the input onto the rail; count the late ones (inside a
        stretch a labelled calibration already fitted on: a missed trial) and
        forget old ones."""
        sfreq = source.info.sfreq
        events = source.new_events()
        if events:
            self.metrics.events_late += sum(
                any(a <= e.sample_index < b for a, b in self._fitted) for e in events
            )
            self.metrics.input_events += len(events)
            self._rail.add(events)
        keep = self._stop_index - round(EVENTS_KEEP_SECONDS * sfreq)
        for blocks, _ in self._calibration.values():
            if blocks:
                first = blocks[0]
                keep = min(keep, self._rail.index(first.sample_index, first.info.sfreq))
        for calibration, _, _ in self._fits.values():
            keep = min(keep, self._rail.index(calibration.sample_index,
                                              calibration.info.sfreq))  # fmt: skip
        self._rail.trim(keep)

    def _finish_fits(self, now: bool = False) -> None:
        """Fit the labelled calibrations whose wait for late markers is over (all
        of them when the input has ended: no more can come)."""
        for node_id, (calibration, due, rails) in list(self._fits.items()):
            if now or time.monotonic() >= due:
                del self._fits[node_id]
                self._fit(self.graph[node_id], calibration, rails)

    def _fit(
        self,
        node: GraphNode,
        calibration: Block,
        rails: Mapping[str, RailInput] | None = None,
    ) -> None:
        stage = node.impl
        assert isinstance(stage, Stage)
        events = (
            self._rail.visible_to(
                self.graph, node.id, calibration.sample_index,
                calibration.stop_index, calibration.info.sfreq,
            )
            if stage.manifest.fit_events
            else ()
        )  # fmt: skip
        if stage.manifest.fit_events:  # on the rail's clock, as markers arrive
            rate = calibration.info.sfreq
            first = self._rail.index(calibration.sample_index, rate)
            self._fitted.append((first, self._rail.index(calibration.stop_index, rate)))
        try:
            stage.fit(calibration, self._input_fps[node.id], events, rails)
        except ValueError as exc:  # e.g. too few trials: say so, stay uncalibrated
            self.metrics.errors.append(f"calibration of {node.id!r} failed: {exc}")
        self._refresh_fingerprints()

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, future = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                result = fn()
                self._update_snapshot(force=True)  # status reflects the change
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)

    def _refresh_fingerprints(self) -> None:
        src = self.graph.source
        fp = signal_fingerprint(src.impl.info) if isinstance(src.impl, Source) else ""
        outputs = {src.id: fp}
        for port in src.manifest.rail_outputs:  # what the input carries
            outputs[f"{src.id}:{port.name}"] = rail_fingerprint(fp, port.name)
        for node in self.graph.nodes[1:]:
            assert node.input is not None
            self._input_fps[node.id] = with_rails(outputs.get(node.input, ""), {
                port: outputs.get(producer, "")
                for port, (producer, _) in self.graph.rail_sources(node.id).items()
                if producer
            })  # fmt: skip
            if isinstance(node.impl, ErpProbe):  # a change upstream: start over
                node.impl.follow(self._input_fps[node.id])
            if isinstance(node.impl, Stage):
                outputs[node.id] = node.impl.output_fingerprint(
                    self._input_fps[node.id]
                )
                for port in node.manifest.rail_outputs:  # read as a main input
                    outputs[f"{node.id}:{port.name}"] = rail_fingerprint(
                        outputs[node.id], port.name
                    )

    def _process(self, block: Block) -> None:
        outputs: dict[str, Block | tuple[FeatureBlock, ...] | None] = {
            self.graph.source.id: block
        }
        self._source_rails(block, outputs)
        sink_hit = False
        for node in self.graph.nodes[1:]:
            assert node.input is not None
            upstream = outputs.get(node.input)
            if upstream is None:
                outputs[node.id] = None
                continue
            started = time.perf_counter()
            if isinstance(node.impl, Stage):
                # processors take Signal input in v1 (port types keep it so)
                self._written = {}
                outputs[node.id] = (
                    self._run_stage(node, upstream)
                    if isinstance(upstream, Block)
                    else None
                )
                for port in node.manifest.rail_outputs:  # "id:port" main inputs
                    rail = self._written.get(port.name)
                    if isinstance(rail, Block) and not rail.n_samples:
                        rail = None
                    outputs[f"{node.id}:{port.name}"] = rail
            elif isinstance(node.impl, Probe):
                for item in output_blocks(upstream):
                    node.impl.tap(item)
                    if isinstance(item, Block):
                        self._feed_display(node, item)
                    elif (isinstance(item, FeatureBlock) and self.display is not None
                          and not isinstance(node.impl, ErpProbe)):  # fmt: skip
                        self._pending_features.setdefault(node.id, []).append(item)
            elif isinstance(node.impl, Sink):
                for item in output_blocks(upstream):  # feature windows one by one
                    node.impl.write(item)
                    sink_hit = True
                if node.impl.notes:
                    notes = node.impl.notes
                    self.metrics.errors.extend(f"{node.id}: {n}" for n in notes)
                    node.impl.notes.clear()
            self.metrics.stages.setdefault(node.id, StageMetrics()).add(
                time.perf_counter() - started
            )
        if sink_hit:
            last = block.timestamp + (block.n_samples - 1) / block.info.sfreq
            self.metrics.latencies.append(self.clock() - last)
            if len(self.metrics.latencies) > 20000:
                del self.metrics.latencies[:10000]

    def _run_stage(
        self, node: GraphNode, block: Block
    ) -> Block | tuple[FeatureBlock, ...] | None:
        """One block through a processor: a reader of rails first waits until
        they cover it (docs/RAILS.md); a writer hands its rails to their readers,
        along with how far its input has come."""
        stage = node.impl
        assert isinstance(stage, Stage)
        rails: dict[str, RailInput] | None = None
        if stage.manifest.rail_inputs:
            feed = self._feed(node.id, block.info.sfreq)
            feed.push_main(block)
            rate = block.info.sfreq
            try:
                got = feed.ready(
                    lambda a, b: self._rail.visible_to(self.graph, node.id, a, b, rate)
                )
                if got is not None:
                    check_rates(got[1], rate)
            except RailError as exc:
                self.metrics.errors.append(f"{node.id}: {exc}")
                got = None
            if got is None:
                return None  # a rail it reads lags: wait (rule 8)
            block, rails = got
        written: dict[str, RailOutput] = {}
        out = self._stage_step(node, block, rails, written)
        self._written = written  # the rails it wrote, for "id:port" readers
        for port in stage.manifest.rail_outputs:
            data = written.get(port.name)
            if data is None and port.type is PortType.FEATURES:
                data = ()  # no window, but the rail is known this far
            if data is None:
                continue
            for reader, reader_port in self._subscribers.get((node.id, port.name), []):
                feed = self._feed(reader, block.info.sfreq)
                feed.push_rail(reader_port, data, block.stop_index)
        return out

    def _source_rails(
        self, block: Block, outputs: dict[str, Block | tuple[FeatureBlock, ...] | None]
    ) -> None:
        """What the input's rail outputs carry over ``block`` (a file's ICs), to
        their readers and to strips taking one as their main input."""
        src = self.graph.source
        if not src.manifest.rail_outputs or not isinstance(src.impl, Source):
            return
        written = src.impl.rails(block)
        for port in src.manifest.rail_outputs:
            data = written.get(port.name)
            if data is None and port.type is PortType.FEATURES:
                data = ()
            if isinstance(data, Block) and not data.n_samples:
                data = None
            outputs[f"{src.id}:{port.name}"] = data
            if data is None:
                continue
            for reader, reader_port in self._subscribers.get((src.id, port.name), []):
                feed = self._feed(reader, block.info.sfreq)
                feed.push_rail(reader_port, data, block.stop_index)

    def _feed(self, node_id: str, sfreq: float) -> RailFeed:
        feed = self._feeds.get(node_id)
        if feed is None:
            ports = self.graph[node_id].manifest.rail_inputs
            hold = round(RAIL_HOLD_SECONDS * sfreq)
            feed = self._feeds[node_id] = RailFeed(ports, hold)
        return feed

    def _stage_step(
        self,
        node: GraphNode,
        block: Block,
        rails: dict[str, RailInput] | None,
        written: dict[str, RailOutput],
    ) -> Block | tuple[FeatureBlock, ...] | None:
        stage = node.impl
        assert isinstance(stage, Stage)
        input_fp = self._input_fps[node.id]
        pending = self._calibration.get(node.id)
        if pending is not None:
            blocks, seconds = pending
            blocks.append(block)
            if rails is not None:
                self._calibration_rails.setdefault(node.id, []).append(rails)
            if sum(b.n_samples for b in blocks) >= seconds * block.info.sfreq:
                del self._calibration[node.id]
                calibration = concat_blocks(blocks)
                pieces = self._calibration_rails.pop(node.id, [])
                joined = (
                    join_inputs(pieces, stage.manifest.rail_inputs) if pieces else None
                )
                if stage.manifest.reads_events:  # wait for markers on their way
                    source = self.graph.source.impl
                    assert isinstance(source, Source)
                    due = time.monotonic() + source.event_wait
                    self._fits[node.id] = (calibration, due, joined)
                else:
                    self._fit(node, calibration, joined)
            return self._without_model(node, block)
        if node.id in self._fits:
            return self._without_model(node, block)
        if stage.status(input_fp) is not StageStatus.READY:
            return self._without_model(node, block)
        try:
            result = stage.transform(block, input_fp, rails)
        except StaleModelError:
            return self._without_model(node, block)
        written.update(result.rails)
        self.metrics.events += len(result.events)
        self._rail.add(result.events, node.id, block.info.sfreq)
        if isinstance(result.block, FeatureBlock):
            return (result.block,)
        if isinstance(result.block, Block) and result.block.n_samples == 0:
            return None  # a delaying node holding back: nothing to hand on yet
        return result.block

    @staticmethod
    def _without_model(node: GraphNode, block: Block) -> Block | None:
        """No valid model: block (default, rule 7) or, if the node opted in, pass
        the data through unprocessed and mark it as bypassed by this node."""
        stage = node.impl
        assert isinstance(stage, Stage)
        if not stage.pass_uncalibrated:
            return None
        marked = tuple(dict.fromkeys((*block.bypassed, node.id)))
        return dataclasses.replace(block, bypassed=marked)

    # -- display --------------------------------------------------------------------

    def _feed_display(self, node: GraphNode, block: Block) -> None:
        if self.display is None:
            return
        decimator = self._decimators.get(node.id)
        if decimator is None:
            bin_size = max(1, round(block.info.sfreq / self._bins_per_second))
            decimator = self._decimators[node.id] = Decimator(bin_size)
        envelope = decimator.feed(block)
        if envelope is not None:
            self._pending.setdefault(node.id, []).append((block, envelope))

    def _maybe_flush_display(self, force: bool = False) -> None:
        if self.display is None:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self._frame_interval:
            return
        self._last_flush = now
        for probe_id, items in self._pending.items():
            if not items:
                continue
            info = items[-1][0].info
            bypassed = items[-1][0].bypassed
            groups: list[list[Envelope]] = []
            for _, env in items:  # merge only bins that follow each other
                last = groups[-1][-1] if groups else None
                if last is not None and last.start_bin + last.n_bins == env.start_bin:
                    groups[-1].append(env)
                else:
                    groups.append([env])
            for group in groups:
                merged = Envelope(
                    group[0].start_bin,
                    group[0].bin_size,
                    np.hstack([e.mins for e in group]),
                    np.hstack([e.maxs for e in group]),
                )
                seq = self._seq.get(probe_id, 0)
                self._seq[probe_id] = seq + 1
                self.display.publish(
                    DisplayFrame(probe_id, seq, info.ch_labels, info.sfreq, merged,
                                 bypassed, info.ch_classes, info.ch_scores,
                                 info.filters)
                )  # fmt: skip
            items.clear()
        sfreq = self.graph.source.impl.info.sfreq  # type: ignore[union-attr]
        for probe_id, windows in self._pending_features.items():
            for w in windows[-FEATURE_FRAMES_PER_FLUSH:]:  # lossy, like the rest
                seq = self._seq.get(probe_id, 0)
                self._seq[probe_id] = seq + 1
                self.display.publish(
                    FeatureFrame(probe_id, seq, w.window, sfreq, w.dims,
                                 dict(w.coords), w.data, w.unit, w.unit_dim,
                                 w.units)
                )  # fmt: skip
            windows.clear()
        for node in self.graph.nodes:  # an ERP tap: its running ERP, when new
            probe = node.impl
            if not isinstance(probe, ErpProbe) or not probe.changed:
                continue
            probe.changed = False
            erp = probe.average()
            if erp is None:
                continue
            seq = self._seq.get(node.id, 0)
            self._seq[node.id] = seq + 1
            counts = tuple(probe.counts[str(c)] for c in erp.coords["condition"])
            self.display.publish(
                FeatureFrame(node.id, seq, erp.window, sfreq, erp.dims,
                             dict(erp.coords), erp.data, erp.unit,
                             counts=counts)
            )  # fmt: skip

    # -- status ---------------------------------------------------------------------

    def node_status(self, node: GraphNode) -> str:
        if isinstance(node.impl, Stage):
            if node.id in self._calibration or node.id in self._fits:
                return StageStatus.CALIBRATING.value
            return node.impl.status(self._input_fps.get(node.id, "")).value
        return "ready"

    def _update_snapshot(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_snapshot < 0.1:
            return
        self._last_snapshot = now
        m = self.metrics
        uptime = self.clock() - m.started_at if m.started_at is not None else 0.0
        snapshot = {
            "running": self._running.is_set() and not self._finished.is_set(),
            "uptime_s": uptime,
            "blocks": m.blocks,
            "samples": m.samples,
            "overruns": m.overruns,
            "events": m.events,
            "input_events": m.input_events,
            "events_late": m.events_late,
            "errors": list(m.errors[-5:]),
            "latency": m.latency_summary(),
            "display": (
                {"sent": self.display.sent, "dropped": self.display.dropped}
                if self.display
                else {}
            ),
            "nodes": [
                {
                    "id": n.id,
                    "node": n.manifest.name,
                    "role": n.role.value,
                    "status": self.node_status(n),
                    "params": dict(n.impl.params),
                    "timing": m.stages[n.id].as_dict() if n.id in m.stages else {},
                    **(
                        {"pass_uncalibrated": n.impl.pass_uncalibrated}
                        if isinstance(n.impl, Stage) and n.impl.manifest.requires_fit
                        else {}
                    ),
                }
                for n in self.graph.nodes
            ],
        }
        source = self.graph.source.impl
        for attr in ("gaps", "duplicates", "first_index", "markers",
                     "markers_dropped"):  # fmt: skip
            if hasattr(source, attr):
                snapshot[f"source_{attr}"] = getattr(source, attr)
        with self._lock:
            self._snapshot = snapshot
