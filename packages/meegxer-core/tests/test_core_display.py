import time

import numpy as np

from meegxer.core.block import Block, SignalInfo
from meegxer.core.display import (
    Decimator,
    DisplayFrame,
    DisplayPublisher,
    DisplaySubscriber,
    Envelope,
    decode_frame,
    encode_frame,
)

INFO = SignalInfo(("a", "b"), 100.0)


def ramp(start: int, n: int) -> Block:
    idx = np.arange(start, start + n, dtype=float)
    return Block(np.vstack([idx, -idx]), INFO, start, 0.0)


def test_bins_align_to_absolute_index_across_blocks() -> None:
    dec = Decimator(bin_size=10)
    envs = [
        e for e in (dec.feed(ramp(s, n)) for s, n in [(0, 7), (7, 18), (25, 30)]) if e
    ]
    starts = [e.start_bin for e in envs]
    mins = np.hstack([e.mins for e in envs])
    maxs = np.hstack([e.maxs for e in envs])
    assert starts == [0, 2] and mins.shape == (2, 5)  # bins 0..4 cover samples 0..49
    np.testing.assert_array_equal(mins[0], [0, 10, 20, 30, 40])
    np.testing.assert_array_equal(maxs[0], [9, 19, 29, 39, 49])
    np.testing.assert_array_equal(mins[1], [-9, -19, -29, -39, -49])


def test_stream_starting_mid_bin_and_gap_reset() -> None:
    dec = Decimator(bin_size=10)
    env = dec.feed(ramp(15, 20))  # first bin (10..19) only partly covered
    assert env is not None and env.start_bin == 1
    np.testing.assert_array_equal(env.mins[0], [15, 20])
    after_gap = dec.feed(ramp(100, 10))  # carry from 30..34 is discarded
    assert after_gap is not None and after_gap.start_bin == 10
    np.testing.assert_array_equal(after_gap.mins[0], [100])


def test_frame_round_trip() -> None:
    env = Envelope(
        5, 4, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]])
    )
    frame = DisplayFrame("main", 3, ("a", "b"), 250.0, env)
    back = decode_frame(encode_frame(frame))
    assert isinstance(back, DisplayFrame)
    assert (back.probe, back.seq, back.ch_labels, back.sfreq) == (
        "main",
        3,
        ("a", "b"),
        250.0,
    )
    assert back.envelope.start_bin == 5 and back.envelope.bin_size == 4
    np.testing.assert_array_equal(back.envelope.maxs, env.maxs)


def test_publisher_never_blocks_and_subscriber_receives() -> None:
    pub = DisplayPublisher("tcp://127.0.0.1:0")
    env = Envelope(0, 1, np.zeros((2, 100)), np.ones((2, 100)))
    started = time.perf_counter()
    for seq in range(5000):  # no subscriber: frames are dropped, never queued
        pub.publish(DisplayFrame("p", seq, ("a", "b"), 100.0, env))
    assert time.perf_counter() - started < 2.0

    sub = DisplaySubscriber(pub.endpoint, ["p"])
    frame = None
    for seq in range(200):  # PUB/SUB needs a moment to connect
        pub.publish(DisplayFrame("p", seq, ("a", "b"), 100.0, env))
        frame = sub.recv(timeout=0.02)
        if frame is not None:
            break
    assert frame is not None and frame.probe == "p"
    sub.close()
    pub.close()
