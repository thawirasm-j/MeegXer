import numpy as np
import pytest

from meegxer.core.block import Block, EventBlock, SignalInfo
from meegxer.core.ports import (
    Port,
    PortType,
    PortTypeError,
    check_block,
    check_connection,
)


def test_matching_port_types_connect() -> None:
    check_connection(Port("out", PortType.SIGNAL), Port("in", PortType.SIGNAL))


@pytest.mark.parametrize(
    ("out_type", "in_type"),
    [
        (PortType.SIGNAL, PortType.FEATURES),
        (PortType.FEATURES, PortType.EVENTS),
        (PortType.EVENTS, PortType.SIGNAL),
    ],
)
def test_mismatched_port_types_rejected(out_type: PortType, in_type: PortType) -> None:
    with pytest.raises(PortTypeError):
        check_connection(Port("out", out_type), Port("in", in_type))


def test_check_block() -> None:
    block = Block(np.zeros((1, 2)), SignalInfo(("Cz",), 100.0), 0, 0.0)
    check_block(Port("x", PortType.SIGNAL), block)
    with pytest.raises(PortTypeError):
        check_block(Port("x", PortType.EVENTS), block)
    check_block(Port("x", PortType.EVENTS), EventBlock((), 0, 2))


def test_spaces_connect_only_when_they_match() -> None:
    """Spec 0.2.22: a component rail (the ICs) cannot feed an input that takes
    sensor space; a port naming no space connects to any."""
    from meegxer.core.ports import Port, PortType, PortTypeError, check_connection

    ics = Port("ics", PortType.SIGNAL, "ics", "component")
    sensor_in = Port("in", PortType.SIGNAL, space="sensor")
    any_in = Port("in", PortType.SIGNAL)
    check_connection(ics, any_in)  # e.g. a spectrum of the components
    check_connection(Port("out", PortType.SIGNAL), sensor_in)  # a main signal
    check_connection(Port("out", PortType.SIGNAL, space="sensor"), sensor_in)
    with pytest.raises(PortTypeError, match="component-space output 'ics'"):
        check_connection(ics, sensor_in)
