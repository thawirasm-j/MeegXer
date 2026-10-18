"""Filters a stream went through (``SignalInfo.filters``, spec 0.2.19) and the
frequencies they changed (``meegxer.core.passband``)."""

import json

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.cache import CachedOutput, DirectoryCache
from meegxer.core.display import (
    DisplayFrame,
    Envelope,
    decode_frame,
    encode_frame,
)
from meegxer.core.passband import (
    filtered_frequencies,
    filters_from_wire,
    filters_wire,
)

FILTERS = (("highpass", 1.0, 0.0), ("lowpass", 0.0, 40.0), ("notch", 50.0, 0.0))


def test_margins_around_each_filter() -> None:
    f = np.array([1.0, 1.4, 1.6, 10.0, 35.9, 36.1, 47.9, 48.1, 52.0, 52.1])
    hp = filtered_frequencies(f, [("highpass", 1.0, 0.0)])
    assert list(hp) == [True, True, False, False, False, False, False, False,
                        False, False]  # below 1.5 x the cutoff  # fmt: skip
    lp = filtered_frequencies(f, [("lowpass", 0.0, 40.0)])
    assert not lp[4] and lp[5]  # above 0.9 x the cutoff (36 Hz)
    notch = filtered_frequencies(f, [("notch", 50.0, 0.0)])
    assert list(notch[6:]) == [False, True, True, False]  # within 2 Hz
    stop = filtered_frequencies(f, [("bandstop", 8.0, 12.0)])
    assert stop[3] and not stop[2]
    assert not filtered_frequencies(f, []).any()


def test_filters_are_metadata_on_the_signal_info() -> None:
    info = SignalInfo(("Fz", "Cz"), 250.0)
    filtered = info.filtered(*FILTERS)
    assert filtered.filters == FILTERS and info.filters == ()
    assert filtered == info  # not part of equality: a cutoff change mid-stream
    assert filtered.subset(["Cz"]).filters == FILTERS
    assert filters_from_wire(json.loads(json.dumps(filters_wire(FILTERS)))) == FILTERS
    assert filters_from_wire(None) == ()
    with pytest.raises(ValueError, match="unknown filter kinds"):
        SignalInfo(("Fz",), 250.0, filters=(("wobble", 1.0, 0.0),))


def test_filters_reach_the_display_and_the_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    env = Envelope(0, 2, np.zeros((2, 3)), np.ones((2, 3)))
    frame = DisplayFrame("main", 0, ("Fz", "Cz"), 250.0, env, filters=FILTERS)
    back = decode_frame(encode_frame(frame))
    assert isinstance(back, DisplayFrame) and back.filters == FILTERS
    plain = decode_frame(encode_frame(DisplayFrame("main", 0, ("Fz", "Cz"), 250.0,
                                                   env)))  # fmt: skip
    assert isinstance(plain, DisplayFrame) and plain.filters == ()

    cache = DirectoryCache(tmp_path)
    info = SignalInfo(("Fz", "Cz"), 250.0).filtered(*FILTERS)
    cache.put("k", CachedOutput(Block(np.zeros((2, 10)), info, 0, 0.0)))
    got = cache.get("k")
    assert got is not None and got.block.info.filters == FILTERS
    # an entry from before filters were recorded is a miss, not "no filters"
    with np.load(tmp_path / "k.npz") as npz:
        old = {name: npz[name] for name in npz.files if name != "filters"}
    np.savez(tmp_path / "old.npz", allow_pickle=False, **old)
    assert cache.get("old") is None
