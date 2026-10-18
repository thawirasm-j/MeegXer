"""Per-probe history of display frames: envelopes aligned by bin index (rule 8),
and the newest windows of a Features tap."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from meegxer.core.display import DisplayFrame, FeatureFrame

FloatArray = npt.NDArray[np.float64]


class ProbeBuffer:
    """Ring buffer of min/max bins for one probe.

    Bin ``k`` covers samples ``[k * bin_size, (k + 1) * bin_size)``, so buffers of
    different probes line up bin for bin. Missing bins (dropped frames, blocked
    data) read as NaN.
    """

    def __init__(self, capacity_bins: int = 250 * 30) -> None:
        self.capacity = capacity_bins
        self.labels: tuple[str, ...] = ()
        self.classes: tuple[str, ...] = ()  # e.g. ICLabel's class of each IC
        self.filters: tuple[tuple[str, float, float], ...] = ()  # gone through
        self.scores: tuple[float, ...] = ()
        self.sfreq = 0.0
        self.bin_size = 0
        self.end_bin = 0  # one past the newest bin received
        self.start_bin = 0  # the first bin received since the stream (re)started
        self.mins: FloatArray = np.empty((0, capacity_bins))
        self.maxs: FloatArray = np.empty((0, capacity_bins))
        self.bypassed = np.zeros(capacity_bins, dtype=bool)
        self.frames = 0
        self.last_seq = -1
        self.dropped = 0

    def _reset(self, frame: DisplayFrame) -> None:
        n = len(frame.ch_labels)
        self.labels, self.sfreq = frame.ch_labels, frame.sfreq
        self.bin_size = frame.envelope.bin_size
        self.mins = np.full((n, self.capacity), np.nan)
        self.maxs = np.full((n, self.capacity), np.nan)
        self.bypassed[:] = False
        self.end_bin = self.start_bin = frame.envelope.start_bin

    def add(self, frame: DisplayFrame) -> None:
        env = frame.envelope
        restarted = env.start_bin < self.end_bin  # went back: the stream restarted
        if (
            frame.ch_labels != self.labels
            or env.bin_size != self.bin_size
            or frame.sfreq != self.sfreq
            or restarted
        ):
            self._reset(frame)
        if self.last_seq >= 0 and frame.seq > self.last_seq + 1:
            self.dropped += frame.seq - self.last_seq - 1
        self.last_seq = frame.seq
        self.frames += 1
        self.classes, self.scores = frame.ch_classes, frame.ch_scores
        self.filters = frame.filters
        start, stop = env.start_bin, env.start_bin + env.n_bins
        if start > self.end_bin:  # a gap: forget what the ring holds there
            self._clear(self.end_bin, start)
        cols = np.arange(start, stop) % self.capacity
        self.mins[:, cols] = env.mins
        self.maxs[:, cols] = env.maxs
        self.bypassed[cols] = bool(frame.bypassed)
        self.end_bin = max(self.end_bin, stop)

    def _clear(self, start: int, stop: int) -> None:
        start = max(start, stop - self.capacity)
        cols = np.arange(start, stop) % self.capacity
        self.mins[:, cols] = np.nan
        self.maxs[:, cols] = np.nan
        self.bypassed[cols] = False

    def window(
        self, end_bin: int, n_bins: int
    ) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
        """``(mins, maxs, bypassed)`` for bins ``[end_bin - n_bins, end_bin)``."""
        n_ch = len(self.labels)
        mins = np.full((n_ch, n_bins), np.nan)
        maxs = np.full((n_ch, n_bins), np.nan)
        bypassed = np.zeros(n_bins, dtype=bool)
        lo = max(end_bin - n_bins, self.end_bin - self.capacity)
        hi = min(end_bin, self.end_bin)
        if hi > lo:
            cols = np.arange(lo, hi) % self.capacity
            dst = slice(lo - (end_bin - n_bins), hi - (end_bin - n_bins))
            mins[:, dst] = self.mins[:, cols]
            maxs[:, dst] = self.maxs[:, cols]
            bypassed[dst] = self.bypassed[cols]
        return mins, maxs, bypassed

    @property
    def oldest_bin(self) -> int:
        """The oldest bin still held (history shorter than the ring at first)."""
        return max(self.start_bin, self.end_bin - self.capacity)

    @property
    def bin_rate(self) -> float:
        return self.sfreq / self.bin_size if self.bin_size else 0.0


class FeatureBuffer:
    """The newest windows of one Features tap, in window order (live)."""

    def __init__(self, capacity: int = 600) -> None:
        self.capacity = capacity
        self.dims: tuple[str, ...] = ()
        self.coords: dict[str, tuple[str | float, ...]] = {}
        self.sfreq = 0.0
        self.units: tuple[str, str, tuple[str, ...]] = ("", "", ())
        self.windows: list[tuple[int, int]] = []
        self.values: list[FloatArray] = []
        self.last_seq = -1
        self.dropped = 0

    def add(self, frame: FeatureFrame) -> None:
        restarted = bool(self.windows) and frame.window[1] <= self.windows[-1][1]
        if frame.dims != self.dims or frame.coords != self.coords or restarted:
            self.dims, self.coords = frame.dims, dict(frame.coords)
            self.windows, self.values = [], []
        if self.last_seq >= 0 and frame.seq > self.last_seq + 1:
            self.dropped += frame.seq - self.last_seq - 1
        self.last_seq = frame.seq
        self.sfreq = frame.sfreq
        self.units = (frame.unit, frame.unit_dim, frame.units)
        self.windows.append(frame.window)
        self.values.append(np.asarray(frame.values, dtype=float))
        if len(self.windows) > self.capacity:
            del self.windows[0], self.values[0]
