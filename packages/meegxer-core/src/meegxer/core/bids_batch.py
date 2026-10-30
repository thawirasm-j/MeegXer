"""One pipeline over many recordings of a BIDS dataset (``meegxer run --bids`` and
the GUI's all-subjects offline run share this).

Each recording gets a fresh graph: the pipeline's input is replaced by that
recording (its Recording settings carried over), its outputs go to the dataset's
BIDS derivatives (a BIDS writer is added when the chain has none), and a recording
whose session file is already there counts as done (``spec/bids-derivatives.md``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meegxer.core.bids_out import (
    Target,
    bids_sinks,
    ensure_writer,
    find_recordings,
    target_for,
)
from meegxer.core.graph import Graph
from meegxer.core.node import Sink, Source
from meegxer.core.qc import recording_qc, write_summary
from meegxer.core.recording import RecordingOverrides, recording_of, with_recording
from meegxer.core.session import drop_shared_sinks
from meegxer.core.sources import BidsSource

_SHORT = {"subject": "sub", "session": "ses", "task": "task", "run": "run"}


def recording_label(params: dict[str, str]) -> str:
    """``sub-01_ses-a_task-rest``: a recording's BIDS entities."""
    return "_".join(f"{_SHORT[k]}-{v}" for k, v in params.items() if v and k in _SHORT)


def is_all_subjects(source: object) -> bool:
    """A BIDS input naming no subject: every subject of its dataset."""
    return isinstance(source, BidsSource) and not str(source.params["subject"])


def recordings_of(source: BidsSource) -> list[dict[str, str]]:
    """The recordings an all-subjects BIDS input stands for: every subject's,
    narrowed by its task, session and run when given."""
    p = source.params

    def only(name: str) -> list[str]:
        return [str(p[name])] if str(p[name]) else []

    return find_recordings(Path(str(p["root"])).resolve(), tasks=only("task"),
                           sessions=only("session"), runs=only("run"))  # fmt: skip


@dataclass
class PreparedRecording:
    label: str
    target: Target
    done: bool  # its session file exists (and overwrite is off): skip it
    notes: list[str] = field(default_factory=list)


def prepare_recording(
    graph: Graph, params: dict[str, str], root: str | Path | None, overwrite: bool
) -> PreparedRecording:
    """Point ``graph`` at one recording and its derivatives (in place)."""
    first = graph.source.impl
    assert isinstance(first, Source)
    settings = recording_of(first)
    source = (with_recording(BidsSource, params, RecordingOverrides.from_dict(settings))
              if settings else BidsSource(params))  # fmt: skip
    graph.replace(graph.source.id, source)
    drop_shared_sinks(graph)  # a shared session's file writers
    notes = []
    bids = set(map(id, bids_sinks(graph)))
    for node in list(graph.nodes):  # one fixed path cannot hold every recording
        if (isinstance(node.impl, Sink) and id(node.impl) not in bids
                and "path" in node.impl.params):  # fmt: skip
            graph.remove(node.id)
            notes.append(f"{node.id}: not written (one file cannot hold every "
                         "recording; the BIDS writer saves each)")  # fmt: skip
    writer = ensure_writer(graph, root, overwrite)
    target = target_for(graph, writer.params)
    # the session file is written last: without it the recording is unfinished
    # (e.g. interrupted while writing) and is done again
    done = target.path("meegxer.json").exists() and not overwrite
    if not done and target.path("eeg.set").exists():
        ensure_writer(graph, root, True)  # replace the partial files
    return PreparedRecording(recording_label(params), target, done, notes)


# -- a whole dataset ---------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    """How ``run_recording`` runs each recording (plain values: they go to
    worker processes)."""

    bids_out: str | None = None
    overwrite: bool = False
    allow_version_mismatch: bool = False
    block_size: int | None = None
    cache_dir: str | None = None
    report: bool = True  # a QC report per recording (meegxer.core.qc)


@dataclass
class Outcome:
    label: str
    status: str  # done, skipped, failed, unreviewed
    message: str = ""
    output: str = ""  # the processed recording
    root: str = ""  # its derivatives dataset
    fitted: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_recording(
    document: dict[str, Any],
    params: dict[str, str],
    options: RunOptions,
    stop: Callable[[], bool] | None = None,
) -> Outcome:
    """One recording of a dataset run, headless (a review strip without saved
    marks stops it: ``unreviewed``). A failure is an outcome; only ``stop``
    turning true raises (``RunStopped``)."""
    from meegxer.core.batch import (
        BatchScheduler,
        RunStopped,
        offline_input_fingerprints,
    )
    from meegxer.core.cache import DirectoryCache
    from meegxer.core.registry import Registry
    from meegxer.core.review import ReviewNeeded
    from meegxer.core.session import adopt_models, load_session

    label = recording_label(params)
    root = ""
    try:
        mismatch = options.allow_version_mismatch
        loaded = load_session(document, Registry.from_entry_points(),
                              allow_version_mismatch=mismatch)  # fmt: skip
        graph = loaded.graph
        # the input's Recording settings (renames, types, montage, bad
        # channels) apply to every recording
        prepared = prepare_recording(graph, params, options.bids_out, options.overwrite)
        target = prepared.target
        root = str(target.root)
        if prepared.done:
            return Outcome(label, "skipped", root=root)
        adopt_models(loaded, offline_input_fingerprints(graph))
        cache = DirectoryCache(options.cache_dir) if options.cache_dir else None
        scheduler = BatchScheduler(block_size=options.block_size, cache=cache,
                                   stop=stop)  # fmt: skip
        result = scheduler.run(graph)
    except RunStopped:
        raise
    except ReviewNeeded as exc:  # stopped at the strip; the others go on
        return Outcome(label, "unreviewed", str(exc), root=root)
    except Exception as exc:  # one bad recording does not stop the others
        return Outcome(label, "failed", str(exc), root=root)
    notes = list(prepared.notes)
    if options.report:
        notes += qc_or_note(graph, target, label)
    return Outcome(label, "done", output=str(target.path("eeg.set")),
                   root=root, fitted=list(result.fitted), notes=notes)  # fmt: skip


def qc_or_note(graph: Graph, target: Target, label: str) -> list[str]:
    """Write the recording's QC report; a failure is a note, not a failed
    recording (its files are written)."""
    try:
        recording_qc(graph, target, label)
    except Exception as exc:
        return [f"no QC report: {type(exc).__name__}: {exc}"]
    return []


def run_dataset(
    document: dict[str, Any],
    recordings: list[dict[str, str]],
    options: RunOptions,
    *,
    jobs: int = 1,
    on_note: Callable[[int, str], None] = lambda i, text: None,
    on_outcome: Callable[[int, Outcome], None] = lambda i, outcome: None,
    stop: Callable[[], bool] | None = None,
) -> list[Outcome]:
    """Every recording, in ``jobs`` worker processes (1: here, one after
    another; 0: one per CPU), each a fresh graph. A sample subject is
    downloaded here first, one at a time, while the workers run the others.
    ``on_note``/``on_outcome`` get the recording's number (from 1). With
    ``options.report``, the dataset's QC summary is written at the end.
    ``stop`` turning true ends the run early: no recording is started after
    it, the workers are stopped (their recordings are redone next time: they
    have no session file), and the outcomes so far are returned."""
    import multiprocessing
    import os
    from concurrent.futures import (
        FIRST_COMPLETED,
        Future,
        ProcessPoolExecutor,
        wait,
    )

    from meegxer.core.batch import RunStopped
    from meegxer.core.samples import ensure_recording, needs_fetch

    jobs = jobs or os.cpu_count() or 1
    outcomes: dict[int, Outcome] = {}

    def done(i: int, outcome: Outcome) -> None:
        outcomes[i] = outcome
        on_outcome(i, outcome)

    def fetch(i: int, params: dict[str, str]) -> bool:
        if not needs_fetch(params):
            return True
        on_note(i, "downloading this sample subject")
        try:
            ensure_recording(params)
        except Exception as exc:
            done(i, Outcome(recording_label(params), "failed", str(exc)))
            return False
        return True

    def stopping() -> bool:
        return stop is not None and stop()

    numbered = list(enumerate(recordings, 1))
    if jobs <= 1 or len(recordings) <= 1:
        for i, params in numbered:
            if stopping():
                break
            if fetch(i, params):
                try:
                    done(i, run_recording(document, params, options, stop))
                except RunStopped:
                    break
    else:
        # spawn, not fork: the same on every platform, and safe with threads
        context = multiprocessing.get_context("spawn")
        workers = min(jobs, len(recordings))
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            running: dict[Future[Outcome], tuple[int, str]] = {}

            def collect(future: Future[Outcome]) -> None:
                i, label = running.pop(future)
                try:
                    done(i, future.result())
                except Exception as exc:  # the worker died (out of memory, …)
                    done(i, Outcome(label, "failed", f"worker stopped: {exc}"))

            for i, params in numbered:
                if stopping():
                    break
                if fetch(i, params):
                    future = pool.submit(run_recording, document, params, options)
                    running[future] = (i, recording_label(params))
                for future in [f for f in running if f.done()]:
                    collect(future)
            while running and not stopping():
                finished, _ = wait(list(running), timeout=0.2,
                                   return_when=FIRST_COMPLETED)  # fmt: skip
                for future in finished:
                    collect(future)
            if running:  # stopped: end the recordings still running now
                for future in running:
                    future.cancel()
                # the executor has no public way to end a running task
                for proc in list(getattr(pool, "_processes", {}).values()):
                    proc.terminate()
    ordered = [outcomes[i] for i in sorted(outcomes)]
    if options.report:
        write_summaries(ordered)
    return ordered


def write_summaries(outcomes: list[Outcome]) -> list[Path]:
    """The QC summary of each derivatives dataset these outcomes went to."""
    rows: dict[str, list[dict[str, Any]]] = {o.root: [] for o in outcomes if o.root}
    # a recording that failed before its target was known (e.g. its download)
    # goes to the dataset the others went to
    fallback = next(iter(rows), "")
    for o in outcomes:
        if o.status in ("failed", "unreviewed"):
            rows.setdefault(o.root or fallback, []).append(
                {"recording": o.label, "status": o.status, "message": o.message}
            )
    written = []
    for root, extra in rows.items():
        if root and Path(root).is_dir():
            written += write_summary(Path(root), extra)
    return written
