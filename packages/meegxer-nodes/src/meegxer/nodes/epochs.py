"""Epochs: the signal cut around events, on a side rail (docs/RAILS.md).

The strip passes the signal on unchanged and writes its epochs to the
``epochs`` rail: one feature window per event, ``dims`` ``(epoch, channel,
time)`` with the event's label as the one ``epoch`` coordinate, the times in
seconds from the event, its sample span as the window, in volts. It reads the
Events rail: the input's events, a review's marks, and those of the strips
before it.

- ``events``: the labels to cut around (empty: every event but the boundaries
  of cuts and the marks to reject).
- ``tmin``/``tmax``: the epoch, in seconds from the event (``tmin`` may be
  negative: before it).
- ``baseline`` ``pre-stimulus``: each channel's mean over ``tmin``..0 is
  subtracted (MNE's ``baseline=(None, 0)``); ``none`` keeps the signal as is.
- ``reject``: an epoch overlapping a mark whose label starts with one of these
  words (``BAD``) is left out (MNE's ``reject_by_annotation``). An epoch
  across a cut a Reject segments strip made is left out too: its samples are not all
  there (the strip gets each piece between cuts on its own).

Live and offline: between blocks it keeps the newest samples an epoch still
needs and the events whose epochs are not complete yet, so an epoch is written
in the block that brings its last sample (the same epochs as one pass over the
recording). A gap in the sample index (a cut) starts it over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from meegxer.core.block import CHANNEL_TYPES, Block, Event, FeatureBlock, concat_blocks
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, RailInput, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes.references import credits

BOUNDARY = "boundary"  # the event a Reject segments strip puts at a cut


def _words(text: object) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(text).split(",") if s.strip())


class Epochs(Processor):
    """Cut the signal around events onto the ``epochs`` rail."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.20",
            "name": "meegxer.nodes.epochs",
            "version": "0.2.0",
            "type": "stream",
            "description": "Cut epochs around events onto a side rail (the "
                           "signal goes on unchanged).",
            "availability": "both",
            "inputs": [{"name": "in", "type": "signal"},
                       {"name": "events", "type": "events", "rail": "events"}],
            "outputs": [{"name": "out", "type": "signal"},
                        {"name": "epochs", "type": "features", "rail": "epochs"}],
            "channel_types": list(CHANNEL_TYPES),  # every channel, as given
            "params": [
                {"name": "events", "type": "string", "default": "",
                 "primary": True,
                 "description": "Comma-separated event labels to cut around "
                                "(empty: every event)"},
                {"name": "tmin", "type": "float", "default": -0.2, "min": -60.0,
                 "max": 60.0, "unit": "s", "primary": True,
                 "description": "Start of the epoch, from the event"},
                {"name": "tmax", "type": "float", "default": 0.8, "min": -60.0,
                 "max": 60.0, "unit": "s", "primary": True,
                 "description": "End of the epoch, from the event"},
                {"name": "baseline", "type": "enum", "default": "pre-stimulus",
                 "choices": ["pre-stimulus", "none"],
                 "description": "Subtract each channel's mean over tmin..0"},
                {"name": "reject", "type": "string", "default": "BAD",
                 "advanced": True,
                 "description": "Leave out epochs overlapping a mark whose "
                                "label starts with one of these words "
                                "(empty: keep them)"},
            ],
            **credits("meegxer.nodes.epochs"),
        }
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return self.transform_rails(block, model, state, params, {"events": ()})

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        events = rails.get("events", ())
        assert isinstance(events, tuple)
        marks = [e for e in events if isinstance(e, Event)]
        seen, held = _held(state, block)
        joined = concat_blocks([seen, block]) if seen is not None else block
        marks = held + marks
        windows = cut_epochs(joined, marks, params)
        return TransformResult(block, _keep(joined, marks, params), (),
                               {"epochs": tuple(windows)})  # fmt: skip


def _span(params: Params, sfreq: float) -> tuple[int, int]:
    """The epoch's first and last sample, counted from its event."""
    tmin, tmax = float(params["tmin"]), float(params["tmax"])
    if tmax <= tmin:
        raise ValueError(f"tmax ({tmax:g} s) must be after tmin ({tmin:g} s)")
    return round(tmin * sfreq), round(tmax * sfreq)


def _keep(joined: Block, marks: list[Event], params: Params) -> State:
    """What the next block needs: the newest samples an epoch can still start
    in, the events whose epochs end after this block, and the marks to reject
    that reach into the samples kept."""
    first, last = _span(params, joined.info.sfreq)
    start = max(joined.sample_index,
                joined.stop_index - max(last - first, -first))  # fmt: skip
    reject = _words(params["reject"])
    held = [e for e in marks
            if e.sample_index + last + 1 > joined.stop_index
            or (reject and e.duration and e.label.startswith(reject)
                and e.sample_index + e.duration > start)]  # fmt: skip
    return {
        "start": start,
        "timestamp": joined.timestamp
        + (start - joined.sample_index) / joined.info.sfreq,
        "data": np.array(joined.data[:, start - joined.sample_index :]),
        "events": [
            {"sample": e.sample_index, "label": e.label, "duration": e.duration}
            for e in held
        ],
    }


def _held(state: State | None, block: Block) -> tuple[Block | None, list[Event]]:
    """The samples and events kept from the blocks before, if ``block``
    follows on from them (else nothing: a cut, or a new stream)."""
    if not state or state.get("start") is None:
        return None, []
    data = state["data"]
    assert isinstance(data, np.ndarray)
    start, stamp, kept = state["start"], state["timestamp"], state["events"]
    assert isinstance(start, int) and isinstance(stamp, float)
    assert isinstance(kept, list)
    if start + data.shape[1] != block.sample_index or (
        data.shape[0] != block.info.n_channels
    ):
        return None, []
    seen = Block(data, block.info, start, stamp)
    events = []
    for e in kept:
        assert isinstance(e, dict)
        sample, label, duration = e["sample"], e["label"], e["duration"]
        assert isinstance(sample, int) and isinstance(label, str)
        assert isinstance(duration, int)
        events.append(Event(sample, label, duration))
    return seen, events


def cut_epochs(block: Block, events: list[Event], params: Params) -> list[FeatureBlock]:
    """The epochs of ``block`` around the chosen events, in event order."""
    sfreq = block.info.sfreq
    first, last = _span(params, sfreq)  # samples from the event
    if params["baseline"] == "pre-stimulus" and first >= 0:
        raise ValueError("a pre-stimulus baseline needs tmin < 0 (or baseline none)")
    wanted = set(_words(params["events"]))
    reject = _words(params["reject"])
    bad = [(e.sample_index, e.sample_index + e.duration) for e in events
           if reject and e.duration and e.label.startswith(reject)]  # fmt: skip
    times = tuple(round(k / sfreq, 9) for k in range(first, last + 1))
    out: list[FeatureBlock] = []
    for e in sorted(events, key=lambda e: e.sample_index):
        if e.label == BOUNDARY or (reject and e.label.startswith(reject)):
            continue
        if wanted and e.label not in wanted:
            continue
        lo, hi = e.sample_index + first, e.sample_index + last + 1
        if lo < block.sample_index or hi > block.stop_index:
            continue  # not all its samples are here (an edge, or a cut)
        if any(a < hi and lo < b for a, b in bad):
            continue
        data = np.array(
            block.data[:, lo - block.sample_index : hi - block.sample_index]
        )
        if params["baseline"] == "pre-stimulus":
            data -= data[:, : 1 - first].mean(axis=1, keepdims=True)  # tmin..0
        out.append(FeatureBlock(
            data[np.newaxis], ("epoch", "channel", "time"), (lo, hi),
            block.timestamp + (lo - block.sample_index) / sfreq,
            {"epoch": (e.label,), "channel": block.info.ch_labels, "time": times},
            unit="V",
        ))  # fmt: skip
    return out
