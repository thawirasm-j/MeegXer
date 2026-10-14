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
