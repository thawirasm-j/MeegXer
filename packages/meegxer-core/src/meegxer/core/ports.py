"""Typed ports (CLAUDE.md rule 6): Signal, Features and Events never mix."""

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


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    type: PortType


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
