"""The Events rail of a run (docs/RAILS.md): one per chain, many contributors.

The input starts it (a recording's events, or LSL markers placed on the signal's
sample index at the inlet, rule 8), and any strip adds the events its transform
emits. Events are kept in sample-index order and tagged with their origin; a
strip sees the input's events and those of the strips upstream of it (a rail
exists only to the right of where it is written), never its own or later ones.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable

from meegxer.core.block import Event
from meegxer.core.graph import Graph

INPUT = "input"  # origin of the events the input carries


def upstream_ids(graph: Graph, node_id: str) -> set[str]:
    """The nodes whose output reaches ``node_id`` (its ancestors), source included."""
    found: set[str] = set()
    current = graph[node_id].input
    while current is not None and current not in found:
        found.add(current)
        current = graph[current].input
    return found


class EventsRail:
    """Origin-tagged events in sample-index order."""

    def __init__(self) -> None:
        self._keys: list[tuple[int, int]] = []  # (sample index, arrival) for bisect
        self._items: list[tuple[Event, str]] = []
        self._added = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, events: Iterable[Event], origin: str = INPUT) -> None:
        for event in events:
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
        self, graph: Graph, node_id: str, start: int = 0, stop: int | None = None
    ) -> tuple[Event, ...]:
        """What ``node_id`` sees: the input's events and its upstream strips'."""
        return self.events(start, stop, {INPUT, *upstream_ids(graph, node_id)})

    def trim(self, before: int) -> None:
        """Forget events before sample index ``before`` (bounded live memory)."""
        cut = bisect.bisect_left(self._keys, (before, -1))
        del self._keys[:cut]
        del self._items[:cut]
