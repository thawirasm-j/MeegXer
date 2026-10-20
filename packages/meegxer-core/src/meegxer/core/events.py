"""The Events rail of a run (docs/RAILS.md): one per chain, many contributors.

The input starts it (a recording's events, or LSL markers placed on the signal's
sample index at the inlet, rule 8), and any strip adds the events its transform
emits. Events are kept in sample-index order and tagged with their origin; a
strip sees the input's events and those of the strips upstream of it (a rail
exists only to the right of where it is written), never its own or later ones.

The rail counts samples on the input's clock. A strip after a rate change (a
resampler) works on another clock, so it passes its rate and the rail maps
indices by the exact ratio of the rates (rule 8: index / rate is the same time
on both sides), to the nearest sample: what it sees, and what it adds.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from collections.abc import Iterable
from fractions import Fraction

from meegxer.core.block import Event
from meegxer.core.graph import Graph

INPUT = "input"  # origin of the events the input carries


def upstream_ids(graph: Graph, node_id: str) -> set[str]:
    """The nodes whose output reaches ``node_id`` (its ancestors), source included."""
    found: set[str] = set()
    current = graph[node_id].input_node
    while current is not None and current not in found:
        found.add(current)
        current = graph[current].input_node
    return found


def _nearest(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def rescale(events: Iterable[Event], scale: Fraction) -> tuple[Event, ...]:
    """``events`` on a clock ``scale`` times as fast, to the nearest sample."""
    if scale == 1:
        return tuple(events)
    return tuple(
        dataclasses.replace(e, sample_index=_nearest(e.sample_index * scale),
                            duration=_nearest(e.duration * scale))
        for e in events
    )  # fmt: skip


class EventsRail:
    """Origin-tagged events in sample-index order, on the input's clock
    (``sfreq``, its rate; None: every caller is on that clock)."""

    def __init__(self, sfreq: float | None = None) -> None:
        self.sfreq = sfreq
        self._keys: list[tuple[int, int]] = []  # (sample index, arrival) for bisect
        self._items: list[tuple[Event, str]] = []
        self._added = 0

    def __len__(self) -> int:
        return len(self._items)

    def _scale(self, sfreq: float | None) -> Fraction:
        """The caller's clock over the rail's."""
        if sfreq is None or self.sfreq is None or sfreq == self.sfreq:
            return Fraction(1)
        return Fraction(str(sfreq)) / Fraction(str(self.sfreq))

    def add(
        self, events: Iterable[Event], origin: str = INPUT, sfreq: float | None = None
    ) -> None:
        """Add ``events`` counted at ``sfreq`` (None: the input's rate)."""
        for event in rescale(events, 1 / self._scale(sfreq)):
            key = (event.sample_index, self._added)
            self._added += 1
            at = bisect.bisect(self._keys, key)
            self._keys.insert(at, key)
            self._items.insert(at, (event, origin))

    def events(
        self,
        start: int = 0,
        stop: int | None = None,
        origins: set[str] | None = None,
    ) -> tuple[Event, ...]:
        """Events with ``start <= sample_index < stop`` from the given origins (all
        when None)."""
        lo = bisect.bisect_left(self._keys, (start, -1))
        hi = (
            len(self._keys)
            if stop is None
            else bisect.bisect_left(self._keys, (stop, -1))
        )
        return tuple(
            e for e, origin in self._items[lo:hi]
            if origins is None or origin in origins
        )  # fmt: skip

    def visible_to(
        self,
        graph: Graph,
        node_id: str,
        start: int = 0,
        stop: int | None = None,
        sfreq: float | None = None,
    ) -> tuple[Event, ...]:
        """What ``node_id`` sees: the input's events and its upstream strips',
        over ``[start, stop)`` counted at ``sfreq`` (its input's rate; None: the
        input's), and counted at that rate."""
        scale = self._scale(sfreq)
        if scale != 1:  # the samples that land in [start, stop) once mapped
            start = math.ceil((start - Fraction(1, 2)) / scale)
            stop = None if stop is None else math.ceil((stop - Fraction(1, 2)) / scale)
        seen = self.events(start, stop, {INPUT, *upstream_ids(graph, node_id)})
        return rescale(seen, scale)

    def index(self, sample_index: int, sfreq: float | None = None) -> int:
        """``sample_index`` counted at ``sfreq`` (None: the input's rate) on the
        rail's clock (the sample at or before it)."""
        return math.floor(sample_index / self._scale(sfreq))

    def trim(self, before: int) -> None:
        """Forget events before sample index ``before`` (bounded live memory)."""
        cut = bisect.bisect_left(self._keys, (before, -1))
        del self._keys[:cut]
        del self._items[:cut]
