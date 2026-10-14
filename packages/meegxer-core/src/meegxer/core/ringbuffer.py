"""Preallocated multichannel ring buffer addressed by absolute sample index.

Writes and reads are vectorized slice copies (at most two per call); there are no
per-sample loops.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from meegxer.core.block import FloatArray, SampleGapError, SampleOverlapError


class RingBuffer:
    """Holds the most recent ``capacity`` samples of ``n_channels`` channels."""

    def __init__(self, n_channels: int, capacity: int, start_index: int = 0) -> None:
        if n_channels < 1 or capacity < 1:
            raise ValueError("n_channels and capacity must be >= 1")
        if start_index < 0:
            raise ValueError("start_index must be >= 0")
        self._buffer: FloatArray = np.zeros((n_channels, capacity), dtype=np.float64)
        self._capacity = capacity
        self._stop = start_index  # absolute index of the next sample to write
        self._count = 0  # valid samples currently held

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def n_channels(self) -> int:
        return int(self._buffer.shape[0])

    @property
    def start_index(self) -> int:
        """Absolute index of the oldest sample still held."""
        return self._stop - self._count

    @property
    def stop_index(self) -> int:
        """Absolute index one past the newest sample."""
        return self._stop

    def __len__(self) -> int:
        return self._count

    def write(self, data: npt.ArrayLike, sample_index: int | None = None) -> None:
        """Append ``data`` (n_channels, n). If ``sample_index`` is given it must
        continue the buffer exactly; gaps and overlaps raise."""
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != self.n_channels:
            raise ValueError(
                f"expected shape ({self.n_channels}, n), got {array.shape}"
            )
        if sample_index is not None:
            if sample_index > self._stop:
                raise SampleGapError(f"missing samples [{self._stop}, {sample_index})")
            if sample_index < self._stop:
                raise SampleOverlapError(
                    f"duplicated samples [{sample_index}, {self._stop})"
                )
        n = int(array.shape[1])
        if n == 0:
            return
        kept = array[:, -self._capacity :]
        k = int(kept.shape[1])
        start = (self._stop + n - k) % self._capacity
        first = min(k, self._capacity - start)
        self._buffer[:, start : start + first] = kept[:, :first]
        self._buffer[:, : k - first] = kept[:, first:]
        self._stop += n
        self._count = min(self._capacity, self._count + n)

    def read(self, start: int, stop: int) -> FloatArray:
        """Copy of samples ``[start, stop)`` by absolute index."""
        if not self.start_index <= start <= stop <= self.stop_index:
            raise IndexError(
                f"[{start}, {stop}) not held; buffer has "
                f"[{self.start_index}, {self.stop_index})"
            )
        n = stop - start
        out = np.empty((self.n_channels, n), dtype=np.float64)
        begin = start % self._capacity
        first = min(n, self._capacity - begin)
        out[:, :first] = self._buffer[:, begin : begin + first]
        out[:, first:] = self._buffer[:, : n - first]
        return out

    def latest(self, n: int) -> FloatArray:
        """Copy of the newest ``n`` samples."""
        return self.read(self.stop_index - n, self.stop_index)
