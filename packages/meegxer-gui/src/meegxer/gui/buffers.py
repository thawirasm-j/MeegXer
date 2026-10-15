"""Per-probe history of display envelopes, aligned by bin index (rule 8)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from meegxer.core.display import DisplayFrame

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
        self.sfreq = 0.0
        self.bin_size = 0
        self.end_bin = 0  # one past the newest bin received
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
        self.end_bin = frame.envelope.start_bin

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
    def bin_rate(self) -> float:
        return self.sfreq / self.bin_size if self.bin_size else 0.0
