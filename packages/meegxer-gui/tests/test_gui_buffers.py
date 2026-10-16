import numpy as np

from meegxer.core.display import DisplayFrame, Envelope
from meegxer.gui.buffers import ProbeBuffer


def frame(start: int, n: int, seq: int, value: float = 1.0) -> DisplayFrame:
    mins = np.full((2, n), -value)
    maxs = np.full((2, n), value)
    return DisplayFrame("p", seq, ("Cz", "Pz"), 500.0, Envelope(start, 2, mins, maxs))


def test_bins_align_by_index_and_gaps_read_as_nan() -> None:
    buf = ProbeBuffer(capacity_bins=100)
    buf.add(frame(10, 5, 0))
    buf.add(frame(20, 5, 2, value=2.0))  # frames 15..19 missing, one dropped
    assert buf.dropped == 1 and buf.end_bin == 25
    _, maxs, _ = buf.window(25, 15)
    assert np.all(maxs[:, :5] == 1.0)
    assert np.all(np.isnan(maxs[:, 5:10]))
    assert np.all(maxs[:, 10:] == 2.0)
    assert buf.bin_rate == 250.0


def test_ring_wraps_and_a_restart_resets() -> None:
    buf = ProbeBuffer(capacity_bins=20)
    for k in range(10):
        buf.add(frame(k * 5, 5, k, value=float(k)))
    _, maxs, _ = buf.window(50, 20)
    assert maxs[0, 0] == 6.0 and maxs[0, -1] == 9.0
    _, old, _ = buf.window(50, 40)
    assert np.all(np.isnan(old[:, :20]))  # beyond the ring
    buf.add(frame(0, 5, 0, value=7.0))  # the stream restarted
    assert buf.end_bin == 5


def test_oldest_bin_is_the_first_received_until_the_ring_is_full() -> None:
    buf = ProbeBuffer(capacity_bins=20)
    buf.add(frame(100, 5, 0))
    assert buf.oldest_bin == 100  # a short history: nothing before it to show
    for k in range(1, 10):
        buf.add(frame(100 + k * 5, 5, k))
    assert buf.oldest_bin == buf.end_bin - 20
