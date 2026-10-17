"""Typed ports (CLAUDE.md rule 6): Signal, Features and Events never mix.

A node's first input and first output are its main ports (the chain). Further
ports name a rail (docs/RAILS.md, spec 0.2.17): an extra input reads that rail, an
extra output writes it. ``events`` is the run's Events rail; a node adds to it
through the events its transform returns, so no output port names it. A Signal
port may say which space its channels are in: sensor (electrodes), component
(e.g. ICs) or source (regions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from meegxer.core.block import Block, EventBlock, FeatureBlock


class PortType(StrEnum):
    SIGNAL = "signal"
    FEATURES = "features"
    EVENTS = "events"


BLOCK_TYPES: dict[PortType, type[Block | FeatureBlock | EventBlock]] = {
    PortType.SIGNAL: Block,
    PortType.FEATURES: FeatureBlock,
    PortType.EVENTS: EventBlock,
}


class PortTypeError(TypeError):
    """A connection or block does not match the declared port type."""


EVENTS_RAIL = "events"
SPACES = ("sensor", "component", "source")


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    type: PortType
    rail: str = ""  # the rail an extra port reads or writes (spec 0.2.17)
    space: str = ""  # Signal ports: sensor, component or source


def check_connection(output: Port, input: Port) -> None:
    """Raise unless an output port may feed the given input port."""
    if output.type is not input.type:
        raise PortTypeError(
            f"cannot connect {output.type} output {output.name!r} to "
            f"{input.type} input {input.name!r}"
        )


def check_block(port: Port, block: Block | FeatureBlock | EventBlock) -> None:
    """Raise unless ``block`` is the block type that ``port`` carries."""
    expected = BLOCK_TYPES[port.type]
    if not isinstance(block, expected):
        raise PortTypeError(
            f"port {port.name!r} carries {expected.__name__}, "
            f"got {type(block).__name__}"
        )
