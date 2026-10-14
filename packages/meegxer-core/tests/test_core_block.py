import itertools

import numpy as np
import pytest

from meegxer.core.block import (
    Block,
    ChannelMismatchError,
    Event,
    EventBlock,
    FeatureBlock,
    SampleGapError,
    SampleOverlapError,
    SignalInfo,
    align_blocks,
    concat_blocks,
    match_channels,
)

INFO = SignalInfo(("Fz", "Cz", "Pz"), sfreq=500.0, stream_id="amp")


def make_block(start: int, n: int, info: SignalInfo = INFO) -> Block:
    rows = np.arange(info.n_channels)[:, None] * 1000
    data = (rows + np.arange(start, start + n)[None, :]).astype(np.float64)
    return Block(data, info, sample_index=start, timestamp=10.0 + start / info.sfreq)


def test_signal_info_validation() -> None:
    with pytest.raises(ChannelMismatchError, match="duplicate"):
        SignalInfo(("Fz", "Fz"), 500.0)
    with pytest.raises(ChannelMismatchError):
        SignalInfo((), 500.0)
    with pytest.raises(ValueError, match="sfreq"):
        SignalInfo(("Fz",), 0.0)
    with pytest.raises(ValueError, match="sfreq"):
        SignalInfo(("Fz",), float("nan"))


def test_block_is_float64_and_read_only() -> None:
    source = np.ones((3, 4), dtype=np.float32)
    block = Block(source, INFO, 0, 0.0)  # type: ignore[arg-type]
    assert block.data.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        block.data[0, 0] = 2.0
    owned = np.ones((3, 4))
    Block(owned, INFO, 0, 0.0)
    owned[0, 0] = 5.0  # the caller's own array stays writable


def test_block_shape_and_index_validation() -> None:
    with pytest.raises(ChannelMismatchError):
        Block(np.zeros((2, 4)), INFO, 0, 0.0)
    with pytest.raises(ValueError, match="2-D"):
        Block(np.zeros((3, 4, 1)), INFO, 0, 0.0)
    with pytest.raises(ValueError):
        Block(np.zeros((3, 4)), INFO, -1, 0.0)
    with pytest.raises(TypeError):
        Block(np.zeros((3, 4)), INFO, True, 0.0)


def test_slice_by_absolute_index_keeps_timestamps() -> None:
    block = make_block(100, 50)
    part = block.slice(110, 120)
    assert part.sample_index == 110 and part.n_samples == 10
    np.testing.assert_array_equal(part.sample_indices(), np.arange(110, 120))
    assert part.timestamps()[0] == pytest.approx(block.timestamps()[10])
    np.testing.assert_array_equal(part.data, block.data[:, 10:20])
    with pytest.raises(IndexError):
        block.slice(90, 110)
    assert block.slice(150, 150).n_samples == 0


def test_pick_reorders_by_label() -> None:
    block = make_block(0, 5)
    picked = block.pick(["Pz", "Fz"])
    assert picked.info.ch_labels == ("Pz", "Fz")
    np.testing.assert_array_equal(picked.data, block.data[[2, 0]])


def test_match_channels() -> None:
    np.testing.assert_array_equal(match_channels(["a", "b", "c"], ["c", "a"]), [2, 0])
    with pytest.raises(ChannelMismatchError, match="not found"):
        match_channels(["a", "b"], ["x"])
    with pytest.raises(ChannelMismatchError):
        match_channels(["a", "a"], ["a"])
    with pytest.raises(ChannelMismatchError):
        match_channels(["a", "b"], ["a", "a"])


def test_concat_contiguous_blocks_equals_whole() -> None:
    whole = make_block(0, 100)
    sizes = [7, 1, 30, 0, 62]
    edges = np.cumsum([0, *sizes])
    parts = [whole.slice(int(a), int(b)) for a, b in itertools.pairwise(edges)]
    joined = concat_blocks(parts)
    np.testing.assert_array_equal(joined.data, whole.data)
    assert joined.sample_index == 0 and joined.timestamp == whole.timestamp


def test_concat_detects_gap_and_overlap() -> None:
    with pytest.raises(SampleGapError):
        concat_blocks([make_block(0, 10), make_block(11, 5)])
    with pytest.raises(SampleOverlapError):
        concat_blocks([make_block(0, 10), make_block(9, 5)])
    other = SignalInfo(("Fz", "Cz", "Oz"), 500.0, "amp")
    with pytest.raises(ChannelMismatchError):
        concat_blocks([make_block(0, 10), make_block(10, 5, other)])


def test_align_overlays_by_index_and_label() -> None:
    ref = make_block(100, 50)
    other_info = SignalInfo(("Pz", "Oz", "Fz"), 500.0, "probe")
    other = Block(np.zeros((3, 40)), other_info, 120, 0.0)
    a, b = align_blocks(ref, other)
    assert a.info.ch_labels == b.info.ch_labels == ("Fz", "Pz")
    assert (
        (a.sample_index, a.stop_index) == (b.sample_index, b.stop_index) == (120, 150)
    )
    with pytest.raises(IndexError):
        align_blocks(ref, Block(np.zeros((3, 5)), other_info, 500, 0.0))


def test_feature_block_validation() -> None:
    fb = FeatureBlock(
        np.zeros((3, 4)),
        dims=("channel", "frequency"),
        window=(0, 500),
        timestamp=0.0,
        coords={"channel": INFO.ch_labels, "frequency": (1.0, 2.0, 3.0, 4.0)},
    )
    assert fb.coords["channel"] == INFO.ch_labels
    with pytest.raises(ValueError, match="dims"):
        FeatureBlock(np.zeros((3, 4)), ("channel",), (0, 1), 0.0)
    with pytest.raises(ValueError, match="length"):
        FeatureBlock(np.zeros(3), ("channel",), (0, 1), 0.0, {"channel": ("a", "b")})
    with pytest.raises(ValueError, match="window"):
        FeatureBlock(np.zeros(3), ("channel",), (5, 1), 0.0)


def test_event_block_sorts_and_checks_range() -> None:
    eb = EventBlock((Event(9, "b"), Event(2, "a", payload={"key": 1})), 0, 10)
    assert [e.label for e in eb.events] == ["a", "b"]
    assert eb.events[0].payload["key"] == 1
    with pytest.raises(IndexError):
        EventBlock((Event(10, "late"),), 0, 10)
    with pytest.raises(ValueError):
        Event(0, "")


def test_bad_channels() -> None:
    info = SignalInfo(("Fz", "Cz", "Pz"), 500.0, bads=("Cz",))
    block = Block(np.zeros((3, 4)), info, 0, 0.0)
    assert block.pick(["Pz", "Cz"]).info.bads == ("Cz",)
    assert block.pick(["Fz"]).info.bads == ()
    with pytest.raises(ChannelMismatchError):
        SignalInfo(("Fz",), 500.0, bads=("Oz",))
    with pytest.raises(ChannelMismatchError):
        SignalInfo(("Fz", "Cz"), 500.0, bads=("Fz", "Fz"))
