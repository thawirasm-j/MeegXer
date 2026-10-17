"""Batch scheduler: run a pipeline over a whole recording (CLAUDE.md rule 4).

The batch scheduler is node-major: each node processes its entire input before
the next node starts. That makes calibration natural (a stage that needs fitting
sees all of its input) and lets each node's output be cached. Within a node,
input is still fed block by block through the same ``transform`` the real-time
scheduler will use, so the processor code is identical in both.

Node availability (manifest ``availability``): offline-only processors (e.g.
zero-phase filters) receive the whole recording as a single block; online-only
nodes are refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from meegxer.core.block import Block, Event, EventBlock, FeatureBlock, concat_blocks
from meegxer.core.cache import Cache, CachedOutput
from meegxer.core.events import EventsRail
from meegxer.core.graph import Graph, GraphError, GraphNode
from meegxer.core.node import NodeRole, Output, Probe, Sink, Source, output_blocks
from meegxer.core.recording import recording_events
from meegxer.core.serialization import canonical_json, fingerprint
from meegxer.core.stage import Stage, StageStatus, labelled_input, signal_fingerprint

type Stream = list[Block | FeatureBlock | EventBlock]


def source_fingerprint(source: Source) -> str:
    """Identity of a source's output: node, data content (or, for sources without
    one, the parameters that generate it) and the channel description it delivers
    (which recording overrides change). A file is identified by its content, not
    its path, so a model fitted on it stays valid when the file moves (sessions)."""
    content = source.content_id()
    return fingerprint(
        "source",
        source.manifest.name,
        source.manifest.version,
        content or canonical_json(source.params),
        signal_fingerprint(source.info),
    )


def offline_input_fingerprints(graph: Graph) -> dict[str, str]:
    """The input fingerprint of every node as a batch run computes it (sources
    by content, offline variants of adaptable nodes), without running anything:
    tells whether a loaded model would be reused or refitted (sessions)."""
    outputs: dict[str, str] = {}
    inputs: dict[str, str] = {}
    events: tuple[Event, ...] | None = None
    for node in graph.nodes:
        if isinstance(node.impl, Source):
            outputs[node.id] = source_fingerprint(node.impl)
            continue
        assert node.input is not None
        inputs[node.id] = outputs.get(node.input, "")
        if isinstance(node.impl, Stage):
            try:
                stage = node.impl.for_mode("offline")
            except ValueError:
                stage = node.impl
            if stage.manifest.fit_events:
                if events is None:
                    source = graph.source.impl
                    assert isinstance(source, Source)
                    events = recording_events(source)
                inputs[node.id] = labelled_input(inputs[node.id], events)
            outputs[node.id] = stage.output_fingerprint(inputs[node.id])
    return inputs


def split_block(block: Block, size: int | None) -> list[Block]:
    """Cut a block into consecutive pieces of at most ``size`` samples."""
    if size is None or block.n_samples <= size:
        return [block]
    return [
        block.slice(start, min(start + size, block.stop_index))
        for start in range(block.sample_index, block.stop_index, size)
    ]


@dataclass
class BatchResult:
    """Outputs of one batch run.

    ``outputs`` holds the joined Signal output of every source/processor node that
    no other processor reads from (plus any requested with ``keep``); feature
    outputs are kept as block lists.
    """

    outputs: dict[str, Block | list[FeatureBlock]] = field(default_factory=dict)
    events: dict[str, tuple[Event, ...]] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    computed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)


class BatchScheduler:
    def __init__(
        self,
        *,
        block_size: int | None = None,
        cache: Cache | None = None,
        calibration_seconds: float | None = None,
    ) -> None:
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self.cache = cache
        self.calibration_seconds = calibration_seconds
        self._events: tuple[Event, ...] = ()  # the recording's, for labelled fits
        self._rail = EventsRail()  # those plus what strips emit (docs/RAILS.md)
        self._graph: Graph | None = None

    def run(self, graph: Graph, *, keep: Iterable[str] = ()) -> BatchResult:
        result = BatchResult()
        keep_ids = set(keep)
        streams: dict[str, Stream] = {}
        remaining = {n.id: len(graph.children(n.id)) for n in graph.nodes}

        online_only = [n.id for n in graph.nodes if not n.manifest.runs_offline]
        if online_only:
            raise GraphError(f"online-only nodes cannot run in batch: {online_only}")
        source = graph.source.impl
        assert isinstance(source, Source)
        graph.check_positions(source.info)  # before reading or writing anything
        labelled = any(isinstance(n.impl, Stage) and n.manifest.fit_events
                       for n in graph.nodes)  # fmt: skip
        self._events = recording_events(source) if labelled else ()
        self._rail = EventsRail()
        self._rail.add(self._events)
        self._graph = graph
        sinks = [n for n in graph.nodes if isinstance(n.impl, Sink)]
        for node in sinks:
            assert isinstance(node.impl, Sink)
            node.impl.bind(graph, node.id)

        for node in graph.nodes:
            if node.role is NodeRole.SOURCE:
                assert isinstance(node.impl, Source)
                streams[node.id] = list(self._read_source(node.impl))
                result.fingerprints[node.id] = source_fingerprint(node.impl)
                result.computed.append(node.id)
            else:
                assert node.input is not None
                upstream = streams[node.input]
                if isinstance(node.impl, Stage):
                    streams[node.id] = self._run_stage(node, upstream, result)
                elif isinstance(node.impl, Probe):
                    for block in upstream:
                        node.impl.tap(block)
                elif isinstance(node.impl, Sink):
                    for block in upstream:
                        node.impl.write(block)
                    node.impl.close()
                remaining[node.input] -= 1
                self._release(graph, node.input, remaining, streams, keep_ids, result)
            self._release(graph, node.id, remaining, streams, keep_ids, result)
        for node in sinks:
            assert isinstance(node.impl, Sink)
            node.impl.finish()
        return result

    def _read_source(self, source: Source) -> Iterable[Block]:
        total = source.n_samples
        if total is None:
            raise GraphError("batch runs need a finite source (set a duration)")
        chunk = self.block_size or max(total, 1)
        while (block := source.read(chunk)) is not None:
            if block.n_samples:
                yield block

    def _run_stage(
        self, node: GraphNode, upstream: Stream, result: BatchResult
    ) -> Stream:
        assert isinstance(node.impl, Stage) and node.input is not None
        try:
            stage = node.impl.for_mode("offline")  # e.g. zero-phase band-pass
        except ValueError as exc:
            raise GraphError(f"{node.id}: {exc}") from exc
        input_fp = result.fingerprints[node.input]
        if stage.manifest.fit_events:  # the trials' labels are part of its input
            input_fp = labelled_input(input_fp, self._events)
        signal_in = [b for b in upstream if isinstance(b, Block)]
        if len(signal_in) != len(upstream):
            raise GraphError(f"{node.id}: processors take Signal input in v1")

        if stage.status(input_fp) is not StageStatus.READY:
            # upstream strips' events follow from input_fp, so it names the
            # recording's only; the fit sees everything on the rail to its left
            assert self._graph is not None
            trials = self._rail.visible_to(self._graph, node.id)
            stage.fit(self._calibration(signal_in), input_fp, trials)
            result.fitted.append(node.id)
        output_fp = stage.output_fingerprint(input_fp)
        result.fingerprints[node.id] = output_fp
        cacheable = self.cache is not None and stage.manifest.kind == "stream"

        if cacheable and self.cache is not None:
            hit = self.cache.get(output_fp)
            if hit is not None:
                result.cached.append(node.id)
                result.events[node.id] = hit.events
                self._rail.add(hit.events, node.id)
                return list(split_block(hit.block, self.block_size))

        if stage.manifest.availability == "offline" and len(signal_in) > 1:
            signal_in = [concat_blocks(signal_in)]  # offline nodes see everything
        stage.state = None  # a batch run always starts the stream from scratch
        out: Stream = []
        events: list[Event] = []
        for block in signal_in:
            transformed = stage.transform(block, input_fp)
            out.extend(_non_empty(transformed.block))
            events.extend(transformed.events)
        tail = stage.flush(input_fp)  # what a delaying node still holds
        if tail is not None:
            out.extend(_non_empty(tail.block))
            events.extend(tail.events)
        result.events[node.id] = tuple(events)
        self._rail.add(events, node.id)
        result.computed.append(node.id)
        if cacheable and self.cache is not None and out:
            joined = concat_blocks([b for b in out if isinstance(b, Block)])
            self.cache.put(output_fp, CachedOutput(joined, tuple(events)))
        return out

    def _calibration(self, blocks: Sequence[Block]) -> Block:
        if not blocks:
            raise GraphError("no data to calibrate on")
        whole = concat_blocks(blocks)
        if self.calibration_seconds is None:
            return whole
        n = round(self.calibration_seconds * whole.info.sfreq)
        return whole.slice(
            whole.sample_index, min(whole.stop_index, whole.sample_index + n)
        )

    @staticmethod
    def _release(
        graph: Graph,
        node_id: str,
        remaining: dict[str, int],
        streams: dict[str, Stream],
        keep: set[str],
        result: BatchResult,
    ) -> None:
        """Drop a stream once all readers are done, keeping requested outputs."""
        if remaining[node_id] > 0 or node_id not in streams:
            return
        stream = streams.pop(node_id)
        processor_children = [
            c for c in graph.children(node_id) if c.role is NodeRole.PROCESSOR
        ]
        if node_id not in keep and processor_children:
            return
        signal_blocks = [b for b in stream if isinstance(b, Block)]
        if signal_blocks and len(signal_blocks) == len(stream):
            result.outputs[node_id] = concat_blocks(signal_blocks)
        else:
            result.outputs[node_id] = [b for b in stream if isinstance(b, FeatureBlock)]


def _non_empty(output: Output) -> tuple[Block | FeatureBlock, ...]:
    """A stage's output blocks without the empty ones a delaying node returns while
    it holds data back (spec 0.2.5): downstream nodes never see an empty block."""
    return tuple(
        b
        for b in output_blocks(output)
        if not (isinstance(b, Block) and not b.n_samples)
    )
