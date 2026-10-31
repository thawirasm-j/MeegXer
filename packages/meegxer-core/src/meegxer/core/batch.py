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

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meegxer.core.block import (
    Block,
    Event,
    EventBlock,
    FeatureBlock,
    Windows,
    contiguous_runs,
)
from meegxer.core.cache import Cache, CachedOutput
from meegxer.core.epoch_review import (
    EpochReview,
    EpochReviewRecord,
    EpochVerdict,
    epochs_to_review,
    load_epoch_review,
    save_epoch_review,
    verdicts_by_sample,
)
from meegxer.core.events import EventsRail
from meegxer.core.graph import Graph, GraphError, GraphNode
from meegxer.core.node import (
    NodeRole,
    Output,
    Probe,
    RailInput,
    Sink,
    Source,
    TransformResult,
    output_blocks,
)
from meegxer.core.ports import PortType
from meegxer.core.rails import (
    UNBOUNDED,
    RailBuffer,
    check_rates,
    events_input,
    rail_fingerprint,
    with_rails,
)
from meegxer.core.recording import recording_events
from meegxer.core.reject_segments import RejectSegments
from meegxer.core.review import (
    Mark,
    Review,
    ReviewAnswer,
    ReviewCallback,
    ReviewNeeded,
    ReviewRequest,
    ReviewStrip,
    VerdictsMissing,
    cut_blocks,
    display_presets,
    load_review,
    marks_to_events,
    merge_spans,
    now_iso,
    offered_labels,
    reject_words,
    rejected_spans,
    review_path,
    review_points,
    save_review,
    settle,
)
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
            for port in node.manifest.rail_outputs:  # what the file carries
                outputs[f"{node.id}:{port.name}"] = rail_fingerprint(
                    outputs[node.id], port.name
                )
            continue
        assert node.input is not None
        inputs[node.id] = outputs.get(node.input, "")
        if isinstance(node.impl, Stage):
            try:
                stage = node.impl.for_mode("offline")
            except ValueError:
                stage = node.impl
            if stage.manifest.reads_events:
                if events is None:
                    source = graph.source.impl
                    assert isinstance(source, Source)
                    events = recording_events(source)
                inputs[node.id] = labelled_input(inputs[node.id], events)
            inputs[node.id] = with_rails(inputs[node.id], {
                port: outputs.get(producer, "")
                for port, (producer, _) in graph.rail_sources(node.id).items()
                if producer
            })  # fmt: skip
            outputs[node.id] = stage.output_fingerprint(inputs[node.id])
            for port in stage.manifest.rail_outputs:  # read as a main input
                outputs[f"{node.id}:{port.name}"] = rail_fingerprint(
                    outputs[node.id], port.name
                )
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

    # kept streams: one Block, or its pieces when a Reject segments strip cut it
    outputs: dict[str, Block | list[Block] | list[FeatureBlock]] = field(
        default_factory=dict
    )
    events: dict[str, tuple[Event, ...]] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    computed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    fitted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # saves of existing files
    notes: list[str] = field(default_factory=list)  # what sinks changed mid-run


class RunStopped(RuntimeError):
    """The run was asked to stop (``BatchScheduler(stop=…)``)."""


class BatchScheduler:
    def __init__(
        self,
        *,
        block_size: int | None = None,
        cache: Cache | None = None,
        calibration_seconds: float | None = None,
        review: ReviewCallback | None = None,
        reviewer: str = "",
        stop: Callable[[], bool] | None = None,
    ) -> None:
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self.cache = cache
        self.calibration_seconds = calibration_seconds
        # asked for the marks of a review strip with no current review
        # (docs/REVIEW.md); without it such a recording stops (ReviewNeeded)
        self.review = review
        self.reviewer = reviewer  # saved with each review made in this run
        # asked before each node: true stops the run there (RunStopped). A node
        # runs over the whole recording at once, so a sink never stops halfway
        self.stop = stop
        self._events: tuple[Event, ...] = ()  # the recording's, for labelled fits
        self._rail = EventsRail()  # those plus what strips emit (docs/RAILS.md)
        self._graph: Graph | None = None
        # what extra output ports wrote: (node id, port) -> blocks or windows
        self._written: dict[tuple[str, str], list[Block | FeatureBlock]] = {}
        self._buffers: dict[tuple[str, str], RailBuffer] = {}  # joined, for readers
        # the input stretch of each node and its rate (a source: its output), for
        # the windows of a feature strip as a processor's input (spec 0.2.23)
        self._spans: dict[str, tuple[int, int, float]] = {}

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
        labelled = any(isinstance(n.impl, Stage) and n.manifest.reads_events
                       for n in graph.nodes)  # fmt: skip
        recorded = recording_events(source)  # on the rail for strips and writers
        self._events = recorded if labelled else ()
        self._rail = EventsRail(source.info.sfreq)  # counted on the input's clock
        self._rail.add(recorded)
        self._graph = graph
        self._written = {}
        self._buffers = {}
        self._spans = {}
        sinks = [n for n in graph.nodes if isinstance(n.impl, Sink)]
        for node in sinks:
            assert isinstance(node.impl, Sink)
            node.impl.bind(graph, node.id)

        for node in graph.nodes:
            if self.stop is not None and self.stop():
                raise RunStopped(f"stopped before {node.id}")
            if node.role is NodeRole.SOURCE:
                assert isinstance(node.impl, Source)
                streams[node.id] = list(self._read_source(node.impl))
                read = [b for b in streams[node.id] if isinstance(b, Block)]
                if read:
                    self._spans[node.id] = (read[0].sample_index, read[-1].stop_index,
                                            node.impl.info.sfreq)  # fmt: skip
                result.fingerprints[node.id] = source_fingerprint(node.impl)
                result.computed.append(node.id)
                if node.manifest.rail_outputs:  # what the file carries (ICs)
                    for block in streams[node.id]:
                        if isinstance(block, Block):
                            self._keep_rails(node.id, TransformResult(
                                block, {}, (), node.impl.rails(block)))  # fmt: skip
                    self._rail_streams(node, streams, result)
            else:
                assert node.input is not None
                upstream = streams[node.input]
                if isinstance(node.impl, Stage):
                    streams[node.id] = self._run_stage(node, upstream, result)
                    self._rail_streams(node, streams, result)
                elif isinstance(node.impl, Probe):
                    for block in upstream:
                        node.impl.tap(block)
                elif isinstance(node.impl, Sink):
                    for block in upstream:
                        node.impl.write(block)
                    rate = next((b.info.sfreq for b in upstream
                                 if isinstance(b, Block)), None)  # fmt: skip
                    node.impl.write_events(
                        self._rail.visible_to(graph, node.id, sfreq=rate)
                    )
                    node.impl.close()
                    result.skipped.extend(f"{node.id}: {note}"
                                          for note in node.impl.skipped)  # fmt: skip
                    result.notes.extend(f"{node.id}: {note}"
                                        for note in node.impl.notes)  # fmt: skip
                remaining[node.input_node] -= 1  # type: ignore[index]
                self._release(graph, node.input_node, remaining, streams,  # type: ignore[arg-type]
                              keep_ids, result)  # fmt: skip
            self._release(graph, node.id, remaining, streams, keep_ids, result)
        for node in sinks:
            assert isinstance(node.impl, Sink)
            node.impl.finish()
        return result

    def rail_events(self, node_id: str | None = None) -> tuple[Event, ...]:
        """The Events rail so far, on the input's clock: all of it (the input's
        events and every strip's that has run), or what ``node_id`` sees."""
        if node_id is None or self._graph is None:
            return self._rail.events()
        return self._rail.visible_to(self._graph, node_id)

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
        assert self._graph is not None
        input_fp = result.fingerprints[node.input]
        if stage.manifest.reads_events:  # the trials' labels are part of its input
            input_fp = labelled_input(input_fp, self._events)
        sources = self._graph.rail_sources(node.id)
        input_fp = with_rails(input_fp, {
            port: result.fingerprints[producer]
            for port, (producer, _) in sources.items() if producer
        })  # fmt: skip
        if stage.takes_features:  # a classifier reading a feature strip
            return self._run_on_features(node, stage, input_fp, upstream, result)
        signal_in = [b for b in upstream if isinstance(b, Block)]
        if len(signal_in) != len(upstream):
            raise GraphError(f"{node.id}: takes Signal input, not {node.input!r}'s")
        rate = signal_in[0].info.sfreq if signal_in else None  # after a resampler
        if signal_in and rate is not None:
            self._spans[node.id] = (signal_in[0].sample_index,
                                    signal_in[-1].stop_index, rate)  # fmt: skip
        if isinstance(stage.processor, ReviewStrip):  # its marks are its output
            marks = self._review_marks(node, stage, input_fp, signal_in, rate or 0.0)
            input_fp = labelled_input(input_fp, marks)
            if stage.processor.bads:  # and the channels it marked bad
                input_fp = fingerprint("bads", input_fp, ",".join(stage.processor.bads))
        if isinstance(stage.processor, EpochReview):  # its verdicts are its output
            verdicts = self._epoch_verdicts(node, stage, input_fp, signal_in,
                                            rate or 0.0)  # fmt: skip
            input_fp = fingerprint(
                "epoch verdicts",
                input_fp,
                canonical_json([v.to_wire() for v in verdicts]),
            )

        if stage.status(input_fp) is not StageStatus.READY:
            # upstream strips' events follow from input_fp, so it names the
            # recording's only; the fit sees everything on the rail to its left
            assert self._graph is not None
            trials = self._rail.visible_to(self._graph, node.id, sfreq=rate)
            calibration, trials = self._calibration(signal_in, trials)
            if len(contiguous_runs(signal_in)) > 1 and stage.manifest.rail_inputs:
                raise GraphError(f"{node.id}: a strip reading rails cannot fit on "
                                 "a stream cut by a Reject segments strip "
                                 "(yet)")  # fmt: skip
            rails = self._rails_for(node, calibration.sample_index,
                                    calibration.stop_index, rate)  # fmt: skip
            stage.fit(calibration, input_fp, trials, rails)
            result.fitted.append(node.id)
        output_fp = stage.output_fingerprint(input_fp)
        result.fingerprints[node.id] = output_fp
        cacheable = (  # a cached output keeps no rails: their writers always run
            self.cache is not None
            and stage.manifest.kind == "stream"
            and not stage.manifest.rail_outputs
        )

        if cacheable and self.cache is not None:
            hit = self.cache.get(output_fp)
            if hit is not None:
                result.cached.append(node.id)
                result.events[node.id] = hit.events
                self._rail.add(hit.events, node.id, rate)
                return list(split_block(hit.block, self.block_size))

        if stage.manifest.availability == "offline" and len(signal_in) > 1:
            # offline nodes see everything: every piece between cuts at once
            signal_in = contiguous_runs(signal_in)
        stage.state = None  # a batch run always starts the stream from scratch
        out: Stream = []
        events: list[Event] = []
        previous: Block | None = None
        for block in signal_in:
            if previous is not None and block.sample_index != previous.stop_index:
                cut = stage.restart(input_fp)  # a cut: nothing leaks across it
                if cut is not None:
                    out.extend(_non_empty(cut.block))
                    events.extend(cut.events)
                    self._keep_rails(node.id, cut)
            previous = block
            rails = self._rails_for(
                node, block.sample_index, block.stop_index, block.info.sfreq
            )
            transformed = stage.transform(block, input_fp, rails)
            out.extend(_non_empty(transformed.block))
            events.extend(transformed.events)
            self._keep_rails(node.id, transformed)
        tail = stage.flush(input_fp)  # what a delaying node still holds
        if tail is not None:
            out.extend(_non_empty(tail.block))
            events.extend(tail.events)
            self._keep_rails(node.id, tail)
        if isinstance(stage.processor, RejectSegments):  # rejected spans go
            assert self._graph is not None
            seen = self._rail.visible_to(self._graph, node.id, sfreq=rate)
            spans = rejected_spans(seen, stage.params)
            kept, boundaries = cut_blocks([b for b in out if isinstance(b, Block)],
                                          spans, node.id)  # fmt: skip
            out = list(kept)
            events.extend(boundaries)
        result.events[node.id] = tuple(events)
        self._rail.add(events, node.id, rate)  # counted on its input's clock
        result.computed.append(node.id)
        signal_out = [b for b in out if isinstance(b, Block)]
        if cacheable and self.cache is not None and signal_out:
            runs = contiguous_runs(signal_out)
            if len(runs) == 1:  # a cut stream is not cached (one block each)
                self.cache.put(output_fp, CachedOutput(runs[0], tuple(events)))
        return out

    def _run_on_features(
        self,
        node: GraphNode,
        stage: Stage,
        input_fp: str,
        upstream: Stream,
        result: BatchResult,
    ) -> Stream:
        """A processor taking Features (spec 0.2.23): fitted, when it needs it,
        on the windows of its input strip and the events it sees, then run on
        them all. Its windows count the samples of that strip's input."""
        assert self._graph is not None and node.input_node is not None
        span = self._spans.get(node.input_node)
        windows = tuple(b for b in upstream if isinstance(b, FeatureBlock))
        if len(windows) != len(upstream):
            raise GraphError(f"{node.id}: takes Features input, not {node.input!r}'s")
        if span is None:
            raise GraphError(f"{node.id}: {node.input!r} produced no windows")
        start, stop, rate = span
        self._spans[node.id] = span
        stretch = Windows(windows, rate, start, stop)
        if stage.status(input_fp) is not StageStatus.READY:
            calibration = stretch
            if self.calibration_seconds is not None:
                end = min(stop, start + round(self.calibration_seconds * rate))
                calibration = Windows(
                    tuple(w for w in windows if w.window[1] <= end), rate, start, end
                )
            trials = self._rail.visible_to(self._graph, node.id, sfreq=rate)
            stage.fit(calibration, input_fp, trials)
            result.fitted.append(node.id)
        result.fingerprints[node.id] = stage.output_fingerprint(input_fp)
        stage.state = None  # a batch run always starts the stream from scratch
        transformed = stage.transform(stretch, input_fp)
        self._keep_rails(node.id, transformed)
        result.events[node.id] = tuple(transformed.events)
        self._rail.add(transformed.events, node.id, rate)
        result.computed.append(node.id)
        return list(_non_empty(transformed.block))

    def _review_marks(
        self,
        node: GraphNode,
        stage: Stage,
        input_fp: str,
        signal_in: Sequence[Block],
        rate: float,
    ) -> tuple[Event, ...]:
        """The marks of a review strip: its sidecar's when made on this input,
        else a review asked for (docs/REVIEW.md), saved to the sidecar; with
        two reviewers, each one's blind review, then the resolve (step 6b)."""
        assert self._graph is not None and isinstance(stage.processor, ReviewStrip)
        try:
            path = review_path(self._graph, node.id, stage.params)
        except ValueError as exc:
            raise GraphError(str(exc)) from exc
        saved = load_review(path)
        labels = signal_in[0].info.ch_labels if signal_in else ()
        if saved is not None and saved.upstream == input_fp:
            review = saved
        elif int(stage.params.get("reviewers", 1)) == 2:
            review = self._two_reviews(node, stage, input_fp, signal_in, rate, path)
        else:
            review = self._one_review(node, stage, input_fp, signal_in, rate, path,
                                      saved, "not reviewed" if saved is None
                                      else "input changed")  # fmt: skip
            save_review(path, review, node.id)
        marks = review.marks
        events = marks_to_events(marks, rate)
        stage.processor.marks = events
        stage.processor.bads = tuple(b for b in review.bads if b in labels)
        stage.processor.record = {"reviewer": review.reviewer,
                                  "reviewed": review.reviewed,
                                  "upstream": review.upstream}  # fmt: skip
        return events

    def _ask(self, request: ReviewRequest, labels: Sequence[str]) -> ReviewAnswer:
        if self.review is None:
            raise ReviewNeeded(request)
        answer = self.review(request)
        if not isinstance(answer, ReviewAnswer):  # just the marks
            answer = ReviewAnswer(tuple(m for m in answer if isinstance(m, Mark)))
        unknown = sorted(set(answer.bads) - set(labels))
        if unknown:
            raise GraphError(f"{request.node_id}: no channels {unknown} to mark bad")
        return answer

    def _one_review(
        self,
        node: GraphNode,
        stage: Stage,
        input_fp: str,
        signal_in: Sequence[Block],
        rate: float,
        path: Path,
        saved: Review | None,
        reason: str,
    ) -> Review:
        """One person's review of the recording at the strip (``saved``: their
        earlier one, on another input, to confirm or change)."""
        assert self._graph is not None
        pairs = self._rail.visible_with_origins(self._graph, node.id,
                                                sfreq=rate or None)  # fmt: skip
        seen = [e for e, _ in pairs]
        suggestions = review_points(pairs, stage.params, rate) if rate else ()
        decision = str(stage.params["decision"])
        request = ReviewRequest(
            node.id, path, input_fp, reason,
            saved.marks if saved is not None else (),
            offered_labels(stage.params, [e.label for e in seen]),
            sum(b.n_samples for b in signal_in) / rate if rate else 0.0,
            reject_words(stage.params),
            saved.viewed if saved is not None else (),
            saved.bads if saved is not None else (),
            display_presets(stage.params),
            suggestions, decision=decision,
        )  # fmt: skip
        labels = signal_in[0].info.ch_labels if signal_in else ()
        answer = self._ask(request, labels)
        try:
            marks, bads = settle(suggestions, answer, decision)
        except VerdictsMissing as exc:
            raise GraphError(f"{node.id}: {exc}") from exc
        return Review(marks, input_fp, now_iso(), merge_spans(answer.viewed),
                      self.reviewer, bads)  # fmt: skip

    def _two_reviews(
        self,
        node: GraphNode,
        stage: Stage,
        input_fp: str,
        signal_in: Sequence[Block],
        rate: float,
        path: Path,
    ) -> Review:
        """Two reviewers, each blind to the other's marks, then the resolve
        (docs/REVIEW.md step 6b): the marks both made alike go on; the rest
        are suggestions to decide, in verify mode."""
        from meegxer.core.two_reviewers import (
            agreement,
            agreement_text,
            disagreements,
            rater_path,
            same,
        )

        reviews: list[Review] = []
        for k in (1, 2):
            rpath = rater_path(path, k)
            own = load_review(rpath)
            if own is not None and own.upstream == input_fp:
                reviews.append(own)
                continue
            if k == 2 and self.reviewer and reviews[0].reviewer == self.reviewer:
                raise GraphError(
                    f"{node.id}: needs a second reviewer: {self.reviewer} reviewed "
                    "as reviewer 1; run it again as another reviewer"
                )
            reason = f"reviewer {k} of 2" + (" (input changed)" if own else "")
            # blind: only this reviewer's own earlier marks are shown, if any
            review = self._one_review(node, stage, input_fp, signal_in, rate,
                                      rpath, own, reason)  # fmt: skip
            save_review(rpath, review, node.id)
            reviews.append(review)
        first, second = reviews
        duration = sum(b.n_samples for b in signal_in) / rate if rate else 0.0
        kappa = agreement(first, second, duration)
        agreed, points, both = disagreements(first, second)
        raters = (first.reviewer, second.reviewer)
        if points:
            note = ("Cohen's kappa per label (1 s windows): " + agreement_text(kappa)
                    if kappa else "")  # fmt: skip
            request = ReviewRequest(
                node.id, path, input_fp, "resolve", agreed,
                offered_labels(stage.params, [m.label for m in agreed]),
                duration, reject_words(stage.params), (), both,
                display_presets(stage.params), points, decision="verify", note=note,
            )  # fmt: skip
            labels = signal_in[0].info.ch_labels if signal_in else ()
            answer = self._ask(request, labels)
            try:
                marks, bads = settle(points, answer, "verify")
            except VerdictsMissing as exc:
                raise GraphError(f"{node.id}: {exc}") from exc
            # the agreed marks stay (verify mode: the answer may hold them too)
            marks = (*marks, *(a for a in agreed if not any(same(a, m) for m in marks)))
            viewed = merge_spans(answer.viewed)
        else:  # they agreed on everything: nothing to resolve
            marks, bads, viewed = agreed, both, ()
        review = Review(tuple(marks), input_fp, now_iso(), viewed, self.reviewer,
                        tuple(bads), raters=raters, agreement=kappa)  # fmt: skip
        save_review(path, review, node.id)
        return review

    def _epoch_verdicts(
        self,
        node: GraphNode,
        stage: Stage,
        input_fp: str,
        signal_in: Sequence[Block],
        rate: float,
    ) -> tuple[EpochVerdict, ...]:
        """The verdicts of an epoch review strip: its sidecar's when made on
        this input, else a review asked for (docs/REVIEW.md), saved."""
        assert self._graph is not None and isinstance(stage.processor, EpochReview)
        try:
            path = review_path(self._graph, node.id, stage.params)
        except ValueError as exc:
            raise GraphError(str(exc)) from exc
        windows: list[FeatureBlock] = []
        for piece in contiguous_runs(list(signal_in)):  # one unless cut
            rails = self._rails_for(node, piece.sample_index, piece.stop_index, rate)
            got = (rails or {}).get("epochs", ())
            if isinstance(got, tuple):
                windows.extend(w for w in got if isinstance(w, FeatureBlock))
        saved = load_epoch_review(path)
        if saved is not None and saved.upstream == input_fp:
            record = saved
        else:
            epochs = epochs_to_review(windows, rate, saved.epochs if saved else ())
            request = ReviewRequest(
                node.id, path, input_fp,
                "input changed" if saved is not None else "not reviewed",
                labels=offered_labels(stage.params, [v.event for v in epochs]),
                duration_s=sum(b.n_samples for b in signal_in) / rate if rate else 0.0,
                kind="epochs", epochs=epochs, windows=tuple(windows),
            )  # fmt: skip
            if self.review is None:
                raise ReviewNeeded(request)
            answer = self.review(request)
            given = (answer.epochs if isinstance(answer, ReviewAnswer)
                     else tuple(v for v in answer
                                if isinstance(v, EpochVerdict)))  # fmt: skip
            by_sample = verdicts_by_sample(given, rate)
            unknown = set(by_sample) - set(verdicts_by_sample(epochs, rate))
            if unknown:
                raise GraphError(f"{node.id}: no epochs start at samples "
                                 f"{sorted(unknown)}")  # fmt: skip
            chosen = tuple(
                dataclasses.replace(by_sample[k], onset=v.onset, duration=v.duration,
                                    event=v.event)
                if (k := round(v.onset * rate)) in by_sample else v
                for v in epochs
            )  # fmt: skip
            record = EpochReviewRecord(chosen, input_fp, now_iso(), self.reviewer)
            save_epoch_review(path, record, node.id)
        stage.processor.verdicts = verdicts_by_sample(record.epochs, rate)
        stage.processor.record = {"reviewer": record.reviewer,
                                  "reviewed": record.reviewed,
                                  "upstream": record.upstream}  # fmt: skip
        return record.epochs

    def _rail_streams(
        self, node: GraphNode, streams: dict[str, Stream], result: BatchResult
    ) -> None:
        """A node's rail outputs as streams, for nodes reading one as their main
        input (``"ica1:ics"``, spec 0.2.18)."""
        for port in node.manifest.rail_outputs:
            key = f"{node.id}:{port.name}"
            streams[key] = list(self._written.get((node.id, port.name), []))
            result.fingerprints[key] = rail_fingerprint(
                result.fingerprints[node.id], port.name
            )

    def _keep_rails(self, node_id: str, result: TransformResult) -> None:
        for port, data in result.rails.items():
            self._written.setdefault((node_id, port), []).extend(
                b for b in output_blocks(data)
                if not (isinstance(b, Block) and not b.n_samples)
            )  # fmt: skip

    def _rails_for(
        self, node: GraphNode, start: int, stop: int, sfreq: float | None
    ) -> dict[str, RailInput] | None:
        """The rails ``node`` reads, over ``[start, stop)`` counted at ``sfreq``,
        its input's rate (docs/RAILS.md)."""
        if not node.manifest.rail_inputs:
            return None
        assert self._graph is not None
        sources = self._graph.rail_sources(node.id)
        rails: dict[str, RailInput] = {}
        for port in node.manifest.rail_inputs:
            producer, output = sources[port.name]
            if not producer:  # the Events rail
                seen = self._rail.visible_to(self._graph, node.id, start, stop, sfreq)
                rails[port.name] = events_input(seen, start, stop)
                continue
            buffer = self._buffers.get((producer, output))
            if buffer is None:  # its writer has run to the end (node-major)
                buffer = self._buffers[producer, output] = RailBuffer(port.type)
                written = self._written.get((producer, output), [])
                if port.type is PortType.SIGNAL:
                    joined = [b for b in written if isinstance(b, Block)]
                    for piece in contiguous_runs(joined):  # one unless cut
                        buffer.add(piece, piece.stop_index)
                else:
                    windows = tuple(w for w in written if isinstance(w, FeatureBlock))
                    buffer.add(windows, UNBOUNDED)
            try:
                rails[port.name] = buffer.take(start, stop)
                if sfreq is not None:
                    check_rates({port.name: rails[port.name]}, sfreq)
            except ValueError as exc:
                raise GraphError(f"{node.id}: rail {port.name!r}: {exc}") from exc
        return rails

    def _calibration(
        self, blocks: Sequence[Block], events: Sequence[Event] = ()
    ) -> tuple[Block, tuple[Event, ...]]:
        """What a fit sees: the input, its pieces joined when a Reject segments
        strip cut it (the samples renumbered from the first, and the events in the
        pieces with them; those in a cut are left out)."""
        if not blocks:
            raise GraphError("no data to calibrate on")
        runs = contiguous_runs(blocks)
        whole = runs[0]
        kept = tuple(events)
        if len(runs) > 1:
            data = np.concatenate([r.data for r in runs], axis=1)
            whole = Block(data, whole.info, whole.sample_index, whole.timestamp)
            moved: list[Event] = []
            offset = 0  # samples cut so far
            for prev, run in zip([None, *runs[:-1]], runs, strict=True):
                if prev is not None:
                    offset += run.sample_index - prev.stop_index
                moved.extend(
                    dataclasses.replace(e, sample_index=e.sample_index - offset)
                    for e in events
                    if run.sample_index <= e.sample_index < run.stop_index
                )
            kept = tuple(moved)
        if self.calibration_seconds is not None:
            n = round(self.calibration_seconds * whole.info.sfreq)
            whole = whole.slice(
                whole.sample_index, min(whole.stop_index, whole.sample_index + n)
            )
        return whole, kept

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
            runs = contiguous_runs(signal_blocks)  # the pieces of a cut stream
            result.outputs[node_id] = runs[0] if len(runs) == 1 else runs
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
