"""Offline runs for clients: a batch run in the background, viewed while it runs.

The daemon runs a whole-recording pipeline with the batch scheduler (rule 4: the
same node code as live). Each probe writes what it sees to memory-mapped files in
a temporary folder as the run goes, so a client can scroll a probe as soon as it
has data instead of after the run, and the operating system keeps those files in
RAM when memory allows and on disk when it does not. Besides the samples (float32)
each probe keeps index-aligned min/max envelopes at a few zoom levels, so a
zoomed-out view of a long recording reads little data. Outputs to files come from
the pipeline's sinks. Processor outputs go to a size-capped disk cache keyed by
fingerprints, so re-running an unchanged chain skips the processing.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
import shutil
import tempfile
import threading
import time
import traceback
import warnings
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from meegxer.core import settings
from meegxer.core.batch import BatchResult, BatchScheduler, RunStopped
from meegxer.core.bids_batch import (
    Outcome,
    RunOptions,
    is_all_subjects,
    prepare_recording,
    qc_or_note,
    recordings_of,
    run_dataset,
    write_summaries,
)
from meegxer.core.block import Block, FeatureBlock, SignalInfo, units_wire
from meegxer.core.cache import DirectoryCache
from meegxer.core.display import Decimator
from meegxer.core.epoch_review import EpochVerdict
from meegxer.core.graph import Graph, GraphError
from meegxer.core.montage import (
    check_filter,
    display_filter,
    filter_pad,
    montage_matrix,
)
from meegxer.core.node import AnyBlock, Probe
from meegxer.core.passband import filters_wire
from meegxer.core.probes import FeatureProbe, MemoryProbe
from meegxer.core.review import (
    Mark,
    ReviewAnswer,
    ReviewCancelled,
    ReviewRequest,
    merge_spans,
)
from meegxer.core.sources import BidsSource
from meegxer.core.stage import Stage

MAX_WINDOW_BINS = 20000
BINS_PER_SECOND = 250.0
MAX_DERIVED_SECONDS = 120.0  # longest page drawn through a montage or filter
LEVEL_FACTOR = 8
_job_numbers = itertools.count(1)


def offline_dir() -> Path:
    """Where offline runs keep their files: ``MEEGXER_OFFLINE_DIR``, else the
    ``storage.offline_dir`` setting, else the system temp folder."""
    root = settings.resolve("storage.offline_dir", "MEEGXER_OFFLINE_DIR")
    return Path(root) if root else Path(tempfile.gettempdir()) / "meegxer-offline"


def batch_jobs() -> int:
    """Recordings an all-subjects run processes at once: ``MEEGXER_BATCH_JOBS``,
    else the ``general.batch_jobs`` setting (0: one per CPU)."""
    jobs = int(settings.resolve("general.batch_jobs", "MEEGXER_BATCH_JOBS"))
    return jobs or os.cpu_count() or 1


def _cache_cap_bytes() -> int:
    gb = settings.resolve("storage.cache_gb", "MEEGXER_OFFLINE_CACHE_GB")
    return int(float(gb) * 1e9)


class ProbeStore:
    """One probe's output on disk: samples plus min/max zoom levels."""

    def __init__(self, folder: Path, info: SignalInfo, n_total: int) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.info = info
        self.n_total = n_total
        self.filled = 0  # samples written, from the first sample of the stream
        self.first_index = 0
        self.bypassed = False
        n_ch = info.n_channels
        self.samples = np.lib.format.open_memmap(
            folder / "samples.npy", "w+", np.float32, (n_ch, max(n_total, 1))
        )
        base = max(1, round(info.sfreq / BINS_PER_SECOND))
        self.levels: list[tuple[int, np.memmap[Any, Any], Decimator]] = []
        size = base
        while True:
            n_bins = math.ceil(n_total / size) + 1
            env = np.lib.format.open_memmap(
                folder / f"env{size}.npy", "w+", np.float32, (2, n_ch, n_bins)
            )
            env[:] = np.nan
            self.levels.append((size, env, Decimator(size)))
            if n_bins <= 2000:
                break
            size *= LEVEL_FACTOR
        self.ends = [0] * len(self.levels)  # bins completed per level

    def write(self, block: Block) -> None:
        if self.filled == 0:
            self.first_index = block.sample_index
        lo = block.sample_index - self.first_index
        hi = min(lo + block.n_samples, self.n_total)
        if lo > self.filled:  # a cut (Reject segments): blank, not zeros
            self.samples[:, self.filled : min(lo, self.n_total)] = np.nan
        self.samples[:, lo:hi] = block.data[:, : hi - lo]
        for k, (_, env, decimator) in enumerate(self.levels):
            got = decimator.feed(block)
            if got is not None:
                stop = min(got.start_bin + got.n_bins, env.shape[2])
                env[0, :, got.start_bin : stop] = got.mins[:, : stop - got.start_bin]
                env[1, :, got.start_bin : stop] = got.maxs[:, : stop - got.start_bin]
                self.ends[k] = stop
        self.bypassed = self.bypassed or bool(block.bypassed)
        self.filled = hi

    def window(self, start: int, n: int, max_bins: int) -> dict[str, Any]:
        """Min/max of samples ``[start, start + n)`` in at most ``max_bins`` bins
        (single samples when zoomed in), clipped to what is written so far. Bins are
        aligned to the absolute sample index, as on the display stream."""
        if not 0 < max_bins <= MAX_WINDOW_BINS:
            raise ValueError(f"max_bins must be 1..{MAX_WINDOW_BINS}")
        want = max(1, math.ceil(n / max_bins))
        stop = min(start + n, self.first_index + self.filled)
        start = max(start, self.first_index)
        usable = [k for k, (size, _, _) in enumerate(self.levels) if size <= want]
        size = self.levels[usable[-1]][0] if usable else 1
        group = math.ceil(want / size)  # level bins per returned bin
        width = size * group
        first = start // width
        lo, hi = first * group, max(first * group, math.ceil(stop / width) * group)
        n_ch = self.info.n_channels
        mins = np.full((n_ch, hi - lo), np.nan)
        maxs = np.full((n_ch, hi - lo), np.nan)
        if usable:
            k = usable[-1]
            env = self.levels[k][1]
            top = min(hi, self.ends[k])
            if top > lo:
                mins[:, : top - lo] = env[0, :, lo:top]
                maxs[:, : top - lo] = env[1, :, lo:top]
        else:  # finer than the first zoom level: from the samples
            a = max(lo, self.first_index)
            z = min(hi, self.first_index + self.filled)
            if z > a:
                chunk = self.samples[:, a - self.first_index : z - self.first_index]
                mins[:, a - lo : z - lo] = chunk
                maxs[:, a - lo : z - lo] = chunk
        n_out = (hi - lo) // group
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins stay NaN
            mins = np.nanmin(mins.reshape(n_ch, n_out, group), axis=2)
            maxs = np.nanmax(maxs.reshape(n_ch, n_out, group), axis=2)
        return {"ch_labels": list(self.info.ch_labels), "sfreq": self.info.sfreq,
                "bin_size": width, "start_bin": first,
                "mins": _as_list(mins), "maxs": _as_list(maxs)}  # fmt: skip

    def derived_window(
        self,
        start: int,
        n: int,
        max_bins: int,
        montage: str = "as recorded",
        highpass: float = 0.0,
        lowpass: float = 0.0,
        notch: float = 0.0,
        bads: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """As :meth:`window`, through a display montage and filter
        (``meegxer.core.montage``), computed from the raw samples: those of
        the page, and a little either side for the filter to settle. A cut
        (a gap) is filtered on each side separately. ``bads`` (labels) replace
        the stream's bad channels, left out of the montage's references. A page
        longer than ``MAX_DERIVED_SECONDS`` is the plain envelope, with
        ``derived`` false."""
        info = self.info
        if bads is not None:
            info = dataclasses.replace(
                info, bads=tuple(b for b in bads if b in info.ch_labels)
            )
        if not 0 < max_bins <= MAX_WINDOW_BINS:
            raise ValueError(f"max_bins must be 1..{MAX_WINDOW_BINS}")
        sfreq = info.sfreq
        check_filter(sfreq, highpass, lowpass, notch)
        matrix, labels = montage_matrix(info, montage)
        if n > MAX_DERIVED_SECONDS * sfreq:
            return {**self.window(start, n, max_bins), "derived": False}
        width = max(1, math.ceil(n / max_bins))
        begin, end = self.first_index, self.first_index + self.filled
        stop = min(start + n, end)
        start = max(start, begin)
        first = start // width
        lo, hi = first * width, max(first * width, math.ceil(stop / width) * width)
        filtering = highpass > 0 or lowpass > 0 or notch > 0
        pad = filter_pad(sfreq, highpass) if filtering else 0
        a, z = max(begin, lo - pad), min(end, hi + pad)
        out = np.full((len(labels), hi - lo), np.nan)
        if z > a:
            raw = np.asarray(self.samples[:, a - begin : z - begin], dtype=float)
            shown = matrix @ np.nan_to_num(raw)
            finite = ~np.isnan(raw).any(axis=0)
            shown[:, ~finite] = np.nan
            if filtering:  # each stretch between gaps on its own
                edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
                for r0, r1 in zip(edges[::2], edges[1::2], strict=True):
                    shown[:, r0:r1] = display_filter(
                        shown[:, r0:r1], sfreq, highpass, lowpass, notch
                    )
            k0, k1 = max(lo, a), min(hi, z)
            out[:, k0 - lo : k1 - lo] = shown[:, k0 - a : k1 - a]
        n_out = (hi - lo) // width
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins stay NaN
            mins = np.nanmin(out.reshape(len(labels), n_out, width), axis=2)
            maxs = np.nanmax(out.reshape(len(labels), n_out, width), axis=2)
        return {"ch_labels": list(labels), "sfreq": sfreq, "bin_size": width,
                "start_bin": first, "mins": _as_list(mins), "maxs": _as_list(maxs),
                "derived": True}  # fmt: skip

    def close(self) -> None:
        for _, env, _ in self.levels:
            env.flush()
        self.samples.flush()


def _as_list(a: npt.NDArray[np.float64]) -> list[list[float | None]]:
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


class StoreProbe(Probe):
    """Stands in for a probe during an offline run and writes to a ProbeStore."""

    manifest = MemoryProbe.manifest

    def __init__(self, job: OfflineJob, probe_id: str) -> None:
        super().__init__({})
        self.job, self.probe_id = job, probe_id

    def tap(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            return
        store = self.job.probes.get(self.probe_id)
        if store is None:
            # a tap after a resampler holds the same span at its own rate
            scale = Fraction(str(block.info.sfreq)) / Fraction(str(self.job.sfreq))
            store = ProbeStore(self.job.store_folder / self.probe_id, block.info,
                               math.ceil(self.job.n_total * scale))  # fmt: skip
            self.job.probes[self.probe_id] = store
        store.write(block)


class FeatureStore:
    """One Features probe's windows of an offline run, kept in memory as
    float32 (feature windows are few compared with samples)."""

    def __init__(self, first: FeatureBlock) -> None:
        self.dims = first.dims
        self.coords = {k: list(v) for k, v in first.coords.items()}
        self.units = units_wire(first.unit, first.unit_dim, first.units)
        self.starts: list[int] = []
        self.stops: list[int] = []
        self.values: list[npt.NDArray[np.float32]] = []
        self.labels: list[str] = []  # an epoch's event label, per window

    def write(self, window: FeatureBlock) -> None:
        self.starts.append(window.window[0])
        self.stops.append(window.window[1])
        self.values.append(np.asarray(window.data, dtype=np.float32))
        if "epoch" in self.dims:  # epochs: each window is one, with its label
            self.labels.append(" ".join(map(str, window.coords.get("epoch", ()))))

    def windows(
        self, start: int, stop: int, max_windows: int, reduce: str
    ) -> dict[str, Any]:
        """The windows ending in ``(start, stop]``: at most ``max_windows`` of
        them, evenly picked, or their mean (``reduce="mean"``)."""
        ends = np.asarray(self.stops)
        idx = np.flatnonzero((ends > start) & (ends <= stop))
        out: dict[str, Any] = {"dims": list(self.dims), "coords": self.coords,
                               "n_windows": int(idx.size), **self.units}  # fmt: skip
        if not idx.size:
            return {**out, "windows": [], "values": []}
        if reduce == "mean":
            mean = np.nanmean(np.stack([self.values[i] for i in idx]), axis=0)
            return {**out, "windows": [[int(self.starts[idx[0]]),
                                        int(self.stops[idx[-1]])]],
                    "values": [_nested(mean)]}  # fmt: skip
        if idx.size > max_windows:
            idx = idx[np.linspace(0, idx.size - 1, max_windows).round().astype(int)]
        return {
            **out,
            "windows": [[int(self.starts[i]), int(self.stops[i])] for i in idx],
            "values": [_nested(self.values[i]) for i in idx],
            **({"labels": [self.labels[i] for i in idx]} if self.labels else {}),
        }


def _nested(a: npt.NDArray[Any]) -> Any:
    """An array as nested lists, NaN as None (JSON)."""
    if a.ndim == 0:
        v = float(a)
        return None if np.isnan(v) else v
    return [_nested(x) for x in a]


class FeatureStoreProbe(Probe):
    """Stands in for a Features probe during an offline run."""

    manifest = FeatureProbe.manifest

    def __init__(self, job: OfflineJob, probe_id: str) -> None:
        super().__init__({})
        self.job, self.probe_id = job, probe_id

    def tap(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            return
        store = self.job.features.get(self.probe_id)
        if store is None:
            store = self.job.features[self.probe_id] = FeatureStore(block)
        store.write(block)


class OfflineJob:
    """One background batch run of a graph; or, for a BIDS input naming no
    subject, one run per recording of its dataset (as ``meegxer run --bids``:
    fresh graphs from ``rebuild``, outputs into BIDS derivatives), the views
    showing the recording being run and, at the end, the last one."""

    def __init__(
        self,
        graph: Graph,
        *,
        folder: Path | None = None,
        rebuild: Callable[[], Graph] | None = None,
        document: dict[str, Any] | None = None,
        reviewer: str = "",
    ) -> None:
        online_only = [n.id for n in graph.nodes if not n.manifest.runs_offline]
        if online_only:
            raise GraphError(f"online-only nodes cannot run offline: {online_only}")
        self.graph = graph
        root = offline_dir()
        self.folder = folder or root / f"run-{os.getpid()}-{next(_job_numbers)}"
        self.store_folder = self.folder
        self.cache_dir = root / "cache"
        self.rebuild = rebuild
        # the pipeline as a session document, for worker processes
        self.document = document
        self.reviewer = reviewer  # saved with each review made in the run
        self.scheduler: BatchScheduler | None = None  # of the recording shown
        self.recordings: list[dict[str, str]] = []
        self.progress: dict[str, Any] = {}
        source = graph.source.impl
        if is_all_subjects(source):
            assert isinstance(source, BidsSource)
            if rebuild is None:
                raise GraphError("an all-subjects BIDS run needs its pipeline")
            self.recordings = recordings_of(source)
            if not self.recordings:
                root_dir = source.params["root"]
                raise GraphError(f"no matching EEG recordings in {root_dir}")
        self.state = "pending"
        self.error = ""
        self.skipped: list[str] = []  # saves skipped: the file existed
        self.started = 0.0
        self.finished = 0.0
        self.n_total = 0
        self.sfreq = 0.0  # the source's rate (feature windows are in its samples)
        self.probes: dict[str, ProbeStore] = {}
        self.features: dict[str, FeatureStore] = {}
        self._thread: threading.Thread | None = None
        # a review strip waiting for its marks (docs/REVIEW.md): the run thread
        # waits on the event; submit_review or cancel_review answers it
        self.review_request: ReviewRequest | None = None
        self._review_reply: ReviewAnswer | None = None
        self._review_answered = threading.Event()
        self._stop = threading.Event()  # stop(): the run ends at the next strip
        if not self.recordings:
            self._attach_stores(graph)

    def _attach_stores(self, graph: Graph) -> None:
        for node in graph.nodes:  # probes write to disk as the run goes
            if isinstance(node.impl, FeatureProbe):
                graph.replace(node.id, FeatureStoreProbe(self, node.id))
            elif isinstance(node.impl, Probe):
                graph.replace(node.id, StoreProbe(self, node.id))

    def start(self) -> None:
        self.state, self.started = "running", time.monotonic()
        self._thread = threading.Thread(target=self._run, name="meegxer-offline",
                                        daemon=True)  # fmt: skip
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run_graph(self, graph: Graph) -> BatchResult:
        source = graph.source.impl
        total = getattr(source, "n_samples", None)
        if total is None:
            raise GraphError("batch runs need a finite source (set a duration)")
        self.n_total = int(total)
        self.sfreq = float(source.info.sfreq)  # type: ignore[union-attr]
        self.store_folder.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = BatchScheduler(cache=DirectoryCache(self.cache_dir),
                                        review=self._await_review,
                                        reviewer=self.reviewer,
                                        stop=self._stop.is_set)  # fmt: skip
        result = self.scheduler.run(graph)
        for store in self.probes.values():
            store.close()
        return result

    def _await_review(self, request: ReviewRequest) -> ReviewAnswer:
        """Pause at a review strip until the GUI sends the marks (the views show
        what the run has computed so far: everything before the strip)."""
        self._review_reply = None
        self._review_answered.clear()
        self.review_request, self.state = request, "review"
        self._review_answered.wait()
        self.review_request = None
        if self._review_reply is None:
            raise ReviewCancelled("the review was cancelled")
        self.state = "running"
        return self._review_reply

    def submit_review(
        self,
        marks: list[dict[str, Any]],
        viewed: list[list[float]] | None = None,
        bads: list[str] | None = None,
        epochs: list[dict[str, Any]] | None = None,
    ) -> None:
        """The marks for the strip the run waits at (a suggestion answered no:
        a mark with verdict ``rejected``), the spans looked at (seconds) and
        the channels marked bad, or an epoch review's verdicts; the run goes
        on."""
        if self.review_request is None:
            raise RuntimeError("no review is waiting")
        self._review_reply = ReviewAnswer(
            tuple(Mark.from_wire(m) for m in marks), merge_spans(viewed or []),
            tuple(dict.fromkeys(str(b) for b in bads or ())),
            tuple(EpochVerdict.from_wire(v) for v in epochs or ()),
        )  # fmt: skip
        self._review_answered.set()

    def review_epochs(
        self, indices: list[int] | None, reduce: str, max_samples: int
    ) -> dict[str, Any]:
        """The data of the epochs the run's epoch review waits on (by their
        index in the request; None: all), each channel in at most
        ``max_samples`` points, or their mean with ``reduce="mean"``."""
        request = self.review_request
        if request is None or request.kind != "epochs":
            raise RuntimeError("no epoch review is waiting")
        windows = sorted(request.windows, key=lambda w: w.window[0])
        chosen = range(len(windows)) if indices is None else indices
        for i in chosen:
            if not 0 <= i < len(windows):
                raise ValueError(f"no epoch {i} (of {len(windows)})")
        if not windows:
            return {"times": [], "ch_labels": [], "unit": "V", "epochs": []}
        first = windows[0]
        times = list(first.coords.get("time", ()))
        step = max(1, -(-len(times) // max(1, max_samples)))  # ceil
        out: dict[str, Any] = {
            "times": times[::step],
            "ch_labels": [str(c) for c in first.coords.get("channel", ())],
            "unit": first.unit,
        }
        if reduce == "mean":
            stack = [windows[i].data[0] for i in chosen]
            mean = np.mean(stack, axis=0)[:, ::step] if stack else None
            out["mean"] = [] if mean is None else mean.astype(np.float32).tolist()
            out["n"] = len(stack)
            return out
        out["epochs"] = [{"index": i,
                          "data": windows[i].data[0][:, ::step]
                          .astype(np.float32).tolist()}
                         for i in chosen]  # fmt: skip
        return out

    def cancel_review(self) -> None:
        """Stop the run at the strip it waits at (nothing is saved)."""
        self._review_reply = None
        self._review_answered.set()

    def stop(self) -> None:
        """End the run: the recording being run stops before its next strip
        (its files are not written, so a later run does it again), workers
        are stopped, and no other recording starts. A waiting review is
        cancelled."""
        self._stop.set()
        if self.review_request is not None:
            self.cancel_review()

    def _run_recordings(self) -> None:
        """Every recording. A finished one (its session file exists) is skipped,
        except the last: it runs here for the views, its files are kept. With
        more than one worker (``general.batch_jobs``) the others run in worker
        processes first; one that stops at a review without saved marks runs
        again here, where the review window can answer it."""
        assert self.rebuild is not None
        n = len(self.recordings)
        done: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        outcomes: list[Outcome] = []
        self.progress = {"index": 0, "total": n, "current": "", "done": done,
                         "skipped": skipped, "failed": failed}  # fmt: skip
        here = list(enumerate(self.recordings, 1))
        jobs = batch_jobs()
        if jobs != 1 and n > 1 and self.document is not None:
            pooled, here = here[:-1], here[-1:]

            def finished(k: int, outcome: Outcome) -> None:
                i, params = pooled[k - 1]
                self.progress.update(index=len(outcomes) + 1, current=outcome.label)
                if outcome.status == "unreviewed":  # reviewed here, before the last
                    here.insert(len(here) - 1, (i, params))
                    return
                outcomes.append(outcome)
                if outcome.status == "failed":
                    failed.append(f"{outcome.label}: {outcome.message}")
                else:
                    (done if outcome.status == "done" else skipped).append(
                        outcome.label)  # fmt: skip

            options = RunOptions(cache_dir=str(self.cache_dir))
            run_dataset(self.document, [p for _, p in pooled], options, jobs=jobs,
                        on_outcome=finished, stop=self._stop.is_set)  # fmt: skip
        for i, params in here:
            if self._stop.is_set():
                break
            graph = self.rebuild()
            prepared = prepare_recording(graph, params, None, overwrite=False)
            self.progress.update(index=i, current=prepared.label)
            if prepared.done and i < n:
                skipped.append(prepared.label)
                outcomes.append(Outcome(prepared.label, "skipped",
                                        root=str(prepared.target.root)))  # fmt: skip
                continue
            self.probes, self.features = {}, {}  # the views follow this recording
            self.store_folder = self.folder / f"rec-{i}"
            self._attach_stores(graph)
            self.graph = graph
            root = str(prepared.target.root)
            try:
                result = self._run_graph(graph)
            except (ReviewCancelled, RunStopped):
                raise  # the person reviewing, or stop(), ended the whole run
            except Exception as exc:  # one bad recording does not stop the others
                traceback.print_exc()
                why = f"{type(exc).__name__}: {exc}"
                failed.append(f"{prepared.label}: {why}")
                outcomes.append(Outcome(prepared.label, "failed", why, root=root))
                continue
            (skipped if prepared.done else done).append(prepared.label)
            outcomes.append(Outcome(prepared.label, "done", root=root))
            notes = [*prepared.notes, *result.skipped]
            if not prepared.done:  # a rerun for the views keeps its report
                notes += qc_or_note(graph, prepared.target, prepared.label)
            self.skipped.extend(f"{prepared.label}: {s}" for s in notes)
        summary = [str(p) for p in write_summaries(outcomes) if p.suffix == ".html"]
        self.progress["summary"] = summary[0] if summary else ""
        if self._stop.is_set():
            raise RunStopped("stopped")
        self.progress.update(index=n)
        if len(failed) == n:
            raise RuntimeError(f"every recording failed; the first: {failed[0]}")

    def _run(self) -> None:
        try:
            if self.recordings:
                self._run_recordings()
            else:
                self.skipped = self._run_graph(self.graph).skipped
            self.state = "done"
        except (ReviewCancelled, RunStopped) as exc:
            if self._stop.is_set():  # stop(): what was finished is kept
                self.state = "stopped"
            else:
                self.state, self.error = "error", str(exc)
        except Exception as exc:
            traceback.print_exc()
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.monotonic()
            _prune_cache(self.cache_dir, _cache_cap_bytes())

    def status(self) -> dict[str, Any]:
        ended = self.state in ("done", "error", "stopped")
        end = self.finished if ended else time.monotonic()
        request = self.review_request
        return {
            "state": self.state,
            "stopping": self._stop.is_set() and self.busy,  # 0.2.19: stop_offline
            "error": self.error,
            # state "review": the strip waiting for marks (docs/REVIEW.md)
            "review": request.to_wire() if request is not None else None,
            "skipped": list(self.skipped),
            # an all-subjects BIDS run: the recording being run (counted from
            # 1), of how many, and how each went
            "recordings": dict(self.progress),
            "elapsed_s": round(end - self.started, 3) if self.started else 0.0,
            "n_samples": self.n_total,
            "sfreq": self.sfreq,
            # nodes with a fitted model after the run (fitted now or reused)
            "models": [
                n.id
                for n in self.graph.nodes
                if isinstance(n.impl, Stage) and n.impl.model is not None
            ]
            if self.state == "done"
            else [],
            "probes": {
                pid: {
                    "ch_labels": list(p.info.ch_labels),
                    "sfreq": p.info.sfreq,
                    "first_sample": p.first_index,
                    "n_filled": p.filled,
                    "n_samples": p.n_total,
                    "bypassed": p.bypassed,
                    "ch_classes": list(p.info.ch_classes),
                    "ch_scores": list(p.info.ch_scores),
                    "filters": filters_wire(p.info.filters),
                }
                for pid, p in list(self.probes.items())
            },
            "features": {
                pid: {
                    "dims": list(f.dims),
                    "coords": f.coords,
                    "n_windows": len(f.stops),
                    **f.units,
                }
                for pid, f in list(self.features.items())
            },
        }

    def window(
        self, probe: str, start: int, n: int, max_bins: int, **display: Any
    ) -> dict[str, Any]:
        """One probe's page; with ``display`` (montage, highpass, lowpass,
        notch, bads) through a display montage and filter."""
        store = self.probes.get(probe)
        if store is None:
            raise KeyError(f"no offline output for probe {probe!r} (yet)")
        if display:
            return {"probe": probe,
                    **store.derived_window(start, n, max_bins, **display)}  # fmt: skip
        return {"probe": probe, **store.window(start, n, max_bins)}

    def events(self, node: str | None = None) -> list[dict[str, Any]]:
        """The run's events so far (seconds from the recording's start): the
        input's and every strip's that has run, or those ``node`` sees."""
        if self.scheduler is None or not self.sfreq:
            return []
        if node is not None and node not in self.graph:
            raise KeyError(f"no node {node!r}")
        return [{"onset": e.sample_index / self.sfreq,
                 "duration": e.duration / self.sfreq, "label": e.label,
                 "origin": str(e.payload.get("origin", ""))}
                for e in self.scheduler.rail_events(node)]  # fmt: skip

    def feature_windows(
        self, probe: str, start: int, stop: int, max_windows: int, reduce: str
    ) -> dict[str, Any]:
        store = self.features.get(probe)
        if store is None:
            raise KeyError(f"no offline features for probe {probe!r} (yet)")
        return {"probe": probe, **store.windows(start, stop, max_windows, reduce)}

    @property
    def busy(self) -> bool:
        return self.state in ("running", "review")

    def discard(self) -> None:
        """Delete this run's files (after the run has stopped)."""
        self.cancel_review()
        self.join(5)
        self.probes.clear()
        shutil.rmtree(self.folder, ignore_errors=True)


def _prune_cache(folder: Path, cap: int) -> None:
    """Keep the cache under ``cap`` bytes, dropping the oldest entries first."""
    try:
        files = sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= cap:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
    except OSError:
        pass
