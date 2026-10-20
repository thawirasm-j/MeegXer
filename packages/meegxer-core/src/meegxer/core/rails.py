"""Rails a node reads and writes through its extra ports (docs/RAILS.md, 0.2.17).

A reader gets each rail over exactly the samples of its main block (rule 8): a
Signal rail's samples ``[start, stop)``, the feature windows that ended inside
the block (``start < window end <= stop``, so every window is delivered once
whatever the block sizes), or the Events rail's events inside it. When a rail
lags (its producer delays its output), the reader waits until every rail it
reads covers the samples: both schedulers and the conformance check cut the
rails the same way, so live and batch results agree (rule 4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from meegxer.core.block import (
    Block,
    Event,
    FeatureBlock,
    concat_blocks,
    contiguous_runs,
)
from meegxer.core.node import RailInput, RailOutput
from meegxer.core.ports import Port, PortType
from meegxer.core.serialization import fingerprint

UNBOUNDED = 2**62  # coverage of a rail that is complete (batch, conformance)


class RailError(ValueError):
    """A rail does not cover the samples a reader needs."""


def check_rates(rails: Mapping[str, RailInput], sfreq: float) -> None:
    """A Signal rail counts samples on its writer's clock: a reader at another
    rate (a resampler between them) would cut it at the wrong samples."""
    for port, data in rails.items():
        if isinstance(data, Block) and data.info.sfreq != sfreq:
            raise RailError(
                f"rail {port!r} is at {data.info.sfreq:g} Hz but the strip reading "
                f"it is at {sfreq:g} Hz: read it before the rate changes"
            )


class RailBuffer:
    """What a producer's extra output has written so far, for one reader port.

    ``covered`` is how far the rail is known: a Signal rail up to the end of its
    data, a Features rail up to the end of the producer's input (every window
    ending there has been emitted)."""

    def __init__(self, port_type: PortType) -> None:
        if port_type is PortType.EVENTS:
            raise ValueError("the Events rail is read from the run's EventsRail")
        self.type = port_type
        self._blocks: list[Block] = []
        self._windows: list[FeatureBlock] = []
        self._through = 0
        self.first: int | None = None  # first sample the rail has data for

    def add(self, output: RailOutput, through: int) -> None:
        if self.type is PortType.SIGNAL:
            if not isinstance(output, Block):
                raise RailError("a Signal rail takes Signal blocks")
            if output.n_samples:
                if self.first is None:
                    self.first = output.sample_index
                self._blocks.append(output)
                self._through = max(self._through, output.stop_index)
            return
        if isinstance(output, Block):
            raise RailError("a Features rail takes feature windows")
        if self.first is None:
            self.first = 0
        self._windows.extend(output)
        self._through = max(self._through, through)

    @property
    def covered(self) -> int:
        return self._through

    @property
    def start(self) -> int | None:
        """First sample a Signal rail still holds (None: no samples yet)."""
        if self.type is not PortType.SIGNAL or not self._blocks:
            return None
        return self._blocks[0].sample_index

    def take(self, start: int, stop: int) -> RailInput:
        """The rail over ``[start, stop)``; a Signal rail must cover all of it."""
        if self.type is PortType.FEATURES:
            return tuple(w for w in self._windows if start < w.window[1] <= stop)
        if not self._blocks:
            raise RailError(f"the rail has no samples for [{start}, {stop})")
        self._blocks = contiguous_runs(self._blocks)  # pieces when cut
        for piece in self._blocks:
            if piece.sample_index <= start <= stop <= piece.stop_index:
                return piece.slice(start, stop)
        covers = ", ".join(f"[{b.sample_index}, {b.stop_index})" for b in self._blocks)
        raise RailError(f"the rail covers {covers}, not [{start}, {stop})")

    def drop_before(self, index: int) -> None:
        """Forget what lies before ``index`` (already handed over)."""
        if self.type is PortType.FEATURES:
            self._windows = [w for w in self._windows if w.window[1] > index]
            return
        self._blocks = [
            b.slice(max(index, b.sample_index), b.stop_index)
            for b in contiguous_runs(self._blocks)
            if b.stop_index > index
        ]


def events_input(events: tuple[Event, ...], start: int, stop: int) -> tuple[Event, ...]:
    return tuple(e for e in events if start <= e.sample_index < stop)


def cut(
    full: Mapping[str, RailInput], ports: tuple[Port, ...], start: int, stop: int
) -> dict[str, RailInput]:
    """Rails given whole (conformance, offline-only nodes) cut to one block."""
    out: dict[str, RailInput] = {}
    for port in ports:
        data = full[port.name]
        if port.type is PortType.EVENTS:
            out[port.name] = events_input(_events(data), start, stop)
            continue
        buffer = RailBuffer(port.type)
        if port.type is PortType.SIGNAL:
            assert isinstance(data, Block)
            buffer.add(data, data.stop_index)
        else:
            buffer.add(_windows(data), UNBOUNDED)
        out[port.name] = buffer.take(start, stop)
    return out


def _events(data: RailInput) -> tuple[Event, ...]:
    if isinstance(data, Block) or not all(isinstance(e, Event) for e in data):
        raise RailError("an Events rail takes events")
    return tuple(e for e in data if isinstance(e, Event))


def _windows(data: RailInput) -> tuple[FeatureBlock, ...]:
    if isinstance(data, Block) or not all(isinstance(w, FeatureBlock) for w in data):
        raise RailError("a Features rail takes feature windows")
    return tuple(w for w in data if isinstance(w, FeatureBlock))


def rail_fingerprint(output_fp: str, port: str) -> str:
    """What a node's rail output carries: its output fingerprint plus the port
    (a node reading ``"ica1:ics"`` as its main input depends on this)."""
    return fingerprint("rail", output_fp, port)


def with_rails(input_fp: str, producers: Mapping[str, str]) -> str:
    """A reader's input fingerprint: its main input plus each rail's producer
    output, so a change upstream of any rail it reads asks for a refit."""
    if not producers:
        return input_fp
    named = [f"{port}={fp}" for port, fp in sorted(producers.items())]
    return fingerprint("rails", input_fp, *named)


def join_inputs(
    pieces: list[Mapping[str, RailInput]], ports: tuple[Port, ...]
) -> dict[str, RailInput]:
    """Rails handed over block by block, joined over the whole stretch (a live
    calibration fits on rails collected with its blocks)."""
    out: dict[str, RailInput] = {}
    for port in ports:
        parts = [p[port.name] for p in pieces]
        if port.type is PortType.SIGNAL:
            blocks = [b for b in parts if isinstance(b, Block) and b.n_samples]
            if not blocks:
                raise RailError(f"rail {port.name!r} has no samples to fit on")
            out[port.name] = concat_blocks(blocks)
        elif port.type is PortType.EVENTS:
            out[port.name] = tuple(e for p in parts for e in _events(p))
        else:
            out[port.name] = tuple(w for p in parts for w in _windows(p))
    return out


class RailFeed:
    """Live (rule 8, wait and align): a reader's main input is held until every
    rail it reads covers the same samples, then handed over with the rails cut
    to them. A rail that starts late (its producer was calibrated later) drops
    the main samples before it; at most ``hold`` samples wait, the oldest go."""

    def __init__(self, ports: tuple[Port, ...], hold: int) -> None:
        self.ports = ports
        self.hold = max(1, hold)
        self._main: list[Block] = []
        self.buffers = {
            p.name: RailBuffer(p.type) for p in ports if p.type is not PortType.EVENTS
        }

    def push_rail(self, port: str, output: RailOutput, through: int) -> None:
        buffer = self.buffers[port]
        buffer.add(output, through)
        buffer.drop_before(buffer.covered - self.hold)

    def push_main(self, block: Block) -> None:
        self._main.append(block)

    def ready(
        self, events: Callable[[int, int], tuple[Event, ...]]
    ) -> tuple[Block, dict[str, RailInput]] | None:
        """The main samples every rail covers, with the rails over them (None
        while a rail lags). ``events(start, stop)`` reads the Events rail."""
        if not self._main:
            return None
        joined = concat_blocks(self._main) if len(self._main) > 1 else self._main[0]
        start, stop = joined.sample_index, joined.stop_index
        for buffer in self.buffers.values():
            stop = min(stop, buffer.covered)
            if buffer.type is PortType.SIGNAL:
                first = buffer.start
                start = max(start, first if first is not None else stop)
        if stop <= start:  # a rail lags: wait, keeping at most `hold` samples
            keep = max(start if start < joined.stop_index else joined.sample_index,
                       joined.stop_index - self.hold)  # fmt: skip
            self._main = [joined.slice(min(keep, joined.stop_index),
                                       joined.stop_index)]  # fmt: skip
            return None
        block = joined.slice(start, stop)
        self._main = (
            [joined.slice(stop, joined.stop_index)] if stop < joined.stop_index else []
        )
        rails: dict[str, RailInput] = {}
        for port in self.ports:
            if port.type is PortType.EVENTS:
                rails[port.name] = events(start, stop)
            else:
                rails[port.name] = self.buffers[port.name].take(start, stop)
                self.buffers[port.name].drop_before(stop)
        return block, rails
