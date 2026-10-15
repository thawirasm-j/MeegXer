"""Offline runs for clients: a batch run in the background, viewed by probe windows.

The daemon runs a whole-recording pipeline with the batch scheduler (rule 4: the
same node code as live) and keeps each probe's output only as display envelopes
(index-aligned min/max bins, as on the display stream), so a client can scroll
through a long recording without the daemon holding the raw data twice. Outputs
to files come from the pipeline's sinks.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from meegxer.core.batch import BatchScheduler
from meegxer.core.display import Decimator, Envelope
from meegxer.core.graph import Graph, GraphError
from meegxer.core.probes import MemoryProbe

MAX_WINDOW_BINS = 20000


@dataclass(frozen=True, slots=True)
class ProbeEnvelope:
    ch_labels: tuple[str, ...]
    sfreq: float
    envelope: Envelope
    bypassed: tuple[str, ...]


class OfflineJob:
    """One background batch run of a graph."""

    def __init__(self, graph: Graph, *, bins_per_second: float = 250.0) -> None:
        online_only = [n.id for n in graph.nodes if not n.manifest.runs_offline]
        if online_only:
            raise GraphError(f"online-only nodes cannot run offline: {online_only}")
        self.graph = graph
        self.bins_per_second = bins_per_second
        self.state = "pending"
        self.error = ""
        self.started = 0.0
        self.finished = 0.0
        self.probes: dict[str, ProbeEnvelope] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.state, self.started = "running", time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="meegxer-offline", daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            BatchScheduler().run(self.graph)
            for node in self.graph.nodes:
                if isinstance(node.impl, MemoryProbe) and node.impl.blocks:
                    self.probes[node.id] = self._decimate(node.impl)
                    node.impl.clear()
            self.state = "done"
        except Exception as exc:
            traceback.print_exc()
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.finished = time.monotonic()

    def _decimate(self, probe: MemoryProbe) -> ProbeEnvelope:
        signal = probe.signal()
        bin_size = max(1, round(signal.info.sfreq / self.bins_per_second))
        decimator = Decimator(bin_size)
        envelope = decimator.feed(signal)
        if envelope is None:  # shorter than one bin
            envelope = Envelope(
                signal.sample_index // bin_size,
                bin_size,
                np.empty((signal.info.n_channels, 0)),
                np.empty((signal.info.n_channels, 0)),
            )
        return ProbeEnvelope(
            signal.info.ch_labels, signal.info.sfreq, envelope, signal.bypassed
        )

    def status(self) -> dict[str, Any]:
        end = self.finished if self.state in ("done", "error") else time.monotonic()
        return {
            "state": self.state,
            "error": self.error,
            "elapsed_s": round(end - self.started, 3) if self.started else 0.0,
            "probes": {
                pid: {
                    "ch_labels": list(p.ch_labels),
                    "sfreq": p.sfreq,
                    "bin_size": p.envelope.bin_size,
                    "first_bin": p.envelope.start_bin,
                    "n_bins": p.envelope.n_bins,
                    "bypassed": list(p.bypassed),
                }
                for pid, p in self.probes.items()
            },
        }

    def window(self, probe: str, start_bin: int, n_bins: int) -> dict[str, Any]:
        """Bins ``[start_bin, start_bin + n_bins)`` of one probe (clipped)."""
        if probe not in self.probes:
            raise KeyError(f"no offline output for probe {probe!r}")
        if not 0 < n_bins <= MAX_WINDOW_BINS:
            raise ValueError(f"n_bins must be 1..{MAX_WINDOW_BINS}")
        p = self.probes[probe]
        env = p.envelope
        lo = max(start_bin, env.start_bin)
        hi = min(start_bin + n_bins, env.start_bin + env.n_bins)
        cols = slice(lo - env.start_bin, max(lo, hi) - env.start_bin)

        def as_list(a: np.ndarray) -> list[list[float | None]]:
            return [[None if np.isnan(v) else float(v) for v in row] for row in a]

        return {
            "probe": probe,
            "ch_labels": list(p.ch_labels),
            "sfreq": p.sfreq,
            "bin_size": env.bin_size,
            "start_bin": lo,
            "mins": as_list(env.mins[:, cols]),
            "maxs": as_list(env.maxs[:, cols]),
        }
