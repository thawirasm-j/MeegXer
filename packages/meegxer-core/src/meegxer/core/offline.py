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

import itertools
import math
import os
import shutil
import tempfile
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, SignalInfo
from meegxer.core.cache import DirectoryCache
from meegxer.core.display import Decimator
from meegxer.core.graph import Graph, GraphError
from meegxer.core.node import AnyBlock, Probe
from meegxer.core.probes import MemoryProbe

MAX_WINDOW_BINS = 20000
BINS_PER_SECOND = 250.0
LEVEL_FACTOR = 8
_job_numbers = itertools.count(1)


def offline_dir() -> Path:
    """Where offline runs keep their files (``MEEGXER_OFFLINE_DIR`` overrides)."""
    root = os.environ.get("MEEGXER_OFFLINE_DIR")
    return Path(root) if root else Path(tempfile.gettempdir()) / "meegxer-offline"


def _cache_cap_bytes() -> int:
    return int(float(os.environ.get("MEEGXER_OFFLINE_CACHE_GB", "5")) * 1e9)


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
            store = ProbeStore(self.job.folder / self.probe_id, block.info,
                               self.job.n_total)  # fmt: skip
            self.job.probes[self.probe_id] = store
        store.write(block)


class OfflineJob:
    """One background batch run of a graph."""

    def __init__(self, graph: Graph, *, folder: Path | None = None) -> None:
        online_only = [n.id for n in graph.nodes if not n.manifest.runs_offline]
        if online_only:
            raise GraphError(f"online-only nodes cannot run offline: {online_only}")
        self.graph = graph
        root = offline_dir()
        self.folder = folder or root / f"run-{os.getpid()}-{next(_job_numbers)}"
        self.cache_dir = root / "cache"
        self.state = "pending"
        self.error = ""
        self.started = 0.0
        self.finished = 0.0
        self.n_total = 0
        self.probes: dict[str, ProbeStore] = {}
        self._thread: threading.Thread | None = None
        for node in graph.nodes:  # probes write to disk as the run goes
            if isinstance(node.impl, Probe):
                graph.replace(node.id, StoreProbe(self, node.id))

    def start(self) -> None:
        self.state, self.started = "running", time.monotonic()
        self._thread = threading.Thread(target=self._run, name="meegxer-offline",
                                        daemon=True)  # fmt: skip
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            source = self.graph.source.impl
            total = getattr(source, "n_samples", None)
            if total is None:
                raise GraphError("batch runs need a finite source (set a duration)")
            self.n_total = int(total)
            self.folder.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            BatchScheduler(cache=DirectoryCache(self.cache_dir)).run(self.graph)
            for store in self.probes.values():
                store.close()
            self.state = "done"
        except Exception as exc:
            traceback.print_exc()
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.monotonic()
            _prune_cache(self.cache_dir, _cache_cap_bytes())

    def status(self) -> dict[str, Any]:
        end = self.finished if self.state in ("done", "error") else time.monotonic()
        return {
            "state": self.state,
            "error": self.error,
            "elapsed_s": round(end - self.started, 3) if self.started else 0.0,
            "n_samples": self.n_total,
            "probes": {
                pid: {
                    "ch_labels": list(p.info.ch_labels),
                    "sfreq": p.info.sfreq,
                    "first_sample": p.first_index,
                    "n_filled": p.filled,
                    "n_samples": p.n_total,
                    "bypassed": p.bypassed,
                }
                for pid, p in list(self.probes.items())
            },
        }

    def window(self, probe: str, start: int, n: int, max_bins: int) -> dict[str, Any]:
        store = self.probes.get(probe)
        if store is None:
            raise KeyError(f"no offline output for probe {probe!r} (yet)")
        return {"probe": probe, **store.window(start, n, max_bins)}

    def discard(self) -> None:
        """Delete this run's files (after the run has stopped)."""
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
