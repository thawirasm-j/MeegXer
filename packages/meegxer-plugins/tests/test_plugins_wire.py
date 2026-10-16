import importlib.util
import io
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from meegxer.plugins.wire import (
    WireError,
    encode,
    read_message,
    tree_from_wire,
    tree_to_wire,
)

SDK = Path(__file__).resolve().parents[1] / "sdk" / "python" / "meegxer_plugin.py"


def sdk() -> ModuleType:
    spec = importlib.util.spec_from_file_location("meegxer_plugin", SDK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARRAYS: dict[str, np.ndarray] = {
    "data": np.arange(12, dtype=np.float32).reshape(3, 4),
    "zi": np.linspace(0, 1, 8).reshape(2, 2, 2),
    "count": np.array([1, 2, 3], dtype=np.int64),
    "mask": np.array([True, False]),
}


def test_host_and_sdk_frames_interoperate() -> None:
    frame = encode({"type": "transform", "id": 4}, ARRAYS)
    header, arrays = sdk().read_frame(io.BytesIO(frame))
    assert header == {"type": "transform", "id": 4}
    for name, arr in ARRAYS.items():
        np.testing.assert_array_equal(arrays[name], arr)
        assert arrays[name].dtype == arr.dtype

    buffer = io.BytesIO()
    sdk().write_frame(buffer, {"type": "result", "id": 4}, ARRAYS)
    msg = read_message(io.BytesIO(buffer.getvalue()))
    assert msg.header == {"type": "result", "id": 4}
    for name, arr in ARRAYS.items():
        np.testing.assert_array_equal(msg.arrays[name], arr)
        assert msg.arrays[name].dtype == arr.dtype


def test_trees_round_trip_through_both_implementations() -> None:
    tree = {"zi": np.ones((2, 3)), "n": 5, "nan": float("nan"), "inf": float("-inf"),
            "nested": [np.zeros(2), {"label": "x"}]}  # fmt: skip
    arrays: dict[str, np.ndarray] = {}
    wire = tree_to_wire(tree, arrays, "s")
    assert wire["nan"] == {"mxfloat": "nan"} and wire["zi"] == {"mxarray": "s0"}
    via_sdk = sdk().from_wire(wire, arrays)  # what a Python plugin sees
    np.testing.assert_array_equal(via_sdk["zi"], tree["zi"])
    assert np.isnan(via_sdk["nan"]) and via_sdk["nested"][1] == {"label": "x"}
    restored = tree_from_wire(wire, arrays)
    np.testing.assert_array_equal(restored["zi"], tree["zi"])
    assert np.isnan(restored["nan"]) and restored["inf"] == float("-inf")
    assert restored["nested"][1] == {"label": "x"}


def test_malformed_frames_are_rejected() -> None:
    with pytest.raises(WireError, match="magic"):
        read_message(io.BytesIO(b"oops" + bytes(20)))
    with pytest.raises(EOFError):
        read_message(io.BytesIO(encode({"type": "x"}, ARRAYS)[:30]))
    with pytest.raises(EOFError):
        read_message(io.BytesIO(b""))
    with pytest.raises(WireError, match="dtype"):
        encode({"type": "x"}, {"c": np.array([1 + 2j])})
    with pytest.raises(WireError, match="missing array"):
        tree_from_wire({"mxarray": "nope"}, {})
