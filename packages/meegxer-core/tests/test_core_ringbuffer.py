import numpy as np
import pytest

from meegxer.core.block import SampleGapError, SampleOverlapError
from meegxer.core.ringbuffer import RingBuffer


def test_matches_reference_under_random_block_sizes() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((4, 5000))
    buf = RingBuffer(n_channels=4, capacity=257, start_index=1000)
    pos = 0
    while pos < signal.shape[1]:
        n = int(rng.integers(0, 400))  # includes blocks larger than capacity
        chunk = signal[:, pos : pos + n]
        buf.write(chunk, sample_index=1000 + pos)
        pos += chunk.shape[1]
        held = min(pos, 257)
        assert len(buf) == held
        assert buf.stop_index == 1000 + pos
        np.testing.assert_array_equal(buf.latest(held), signal[:, pos - held : pos])
        if held > 10:
            a = buf.start_index + 3
            np.testing.assert_array_equal(
                buf.read(a, a + 7), signal[:, a - 1000 : a - 1000 + 7]
            )


def test_read_outside_held_range_raises() -> None:
    buf = RingBuffer(1, 10)
    buf.write(np.arange(25.0)[None, :])
    assert (buf.start_index, buf.stop_index) == (15, 25)
    with pytest.raises(IndexError):
        buf.read(14, 16)  # evicted
    with pytest.raises(IndexError):
        buf.read(24, 26)  # not written yet
    np.testing.assert_array_equal(buf.read(15, 25)[0], np.arange(15.0, 25.0))


def test_contiguity_checks() -> None:
    buf = RingBuffer(2, 8)
    buf.write(np.zeros((2, 3)), sample_index=0)
    with pytest.raises(SampleGapError):
        buf.write(np.zeros((2, 3)), sample_index=4)
    with pytest.raises(SampleOverlapError):
        buf.write(np.zeros((2, 3)), sample_index=2)
    with pytest.raises(ValueError):
        buf.write(np.zeros((3, 3)))


def test_read_returns_copy() -> None:
    buf = RingBuffer(1, 4)
    buf.write(np.ones((1, 4)))
    out = buf.read(0, 4)
    out[:] = 0
    np.testing.assert_array_equal(buf.read(0, 4), np.ones((1, 4)))
