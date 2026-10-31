"""The plugin wire reader on untrusted bytes (rule 9; M11.2): every frame a
plugin may send either decodes or is refused with WireError (or EOFError at
the end of the stream), without reading what it declares beyond the limits."""

from __future__ import annotations

import contextlib
import io
import json
import struct
from typing import Any

import numpy as np
import pytest

from meegxer.plugins.wire import (
    MAGIC,
    MAX_PAYLOAD,
    WireError,
    encode,
    read_message,
    tree_from_wire,
)

ARRAYS: dict[str, np.ndarray] = {
    "block": np.arange(12, dtype=np.float32).reshape(3, 4),
    "index": np.array([1, 2, 3], dtype=np.int64),
    "mask": np.array([True, False]),
    "empty": np.zeros((0, 5)),
}


def frame(header: dict[str, Any] | None = None) -> bytes:
    return encode(header or {"type": "result", "id": 7, "state": {"k": [1, 2]}},
                  ARRAYS)  # fmt: skip


def raw(header: object, payload: bytes = b"", length: int | None = None) -> bytes:
    head = json.dumps(header).encode()
    size = len(payload) if length is None else length
    sizes = struct.pack("<I", len(head)), struct.pack("<Q", size)
    return MAGIC + sizes[0] + head + sizes[1] + payload


def decodes_or_refuses(data: bytes) -> None:
    with contextlib.suppress(WireError, EOFError):
        read_message(io.BytesIO(data))


def test_a_good_frame_round_trips() -> None:
    message = read_message(io.BytesIO(frame()))
    assert message.type == "result" and message.header["state"] == {"k": [1, 2]}
    for name, array in ARRAYS.items():
        np.testing.assert_array_equal(message.arrays[name], array)
        assert message.arrays[name].dtype == array.dtype


def test_declared_sizes_are_checked_before_reading() -> None:
    with pytest.raises(WireError, match="too large"):  # nothing is allocated
        read_message(io.BytesIO(raw({"type": "x"}, length=2**62)))
    assert MAX_PAYLOAD < 2**31
    with pytest.raises(WireError, match="too large"):
        read_message(io.BytesIO(MAGIC + struct.pack("<I", 2**31)))


@pytest.mark.parametrize(
    ("arrays", "why"),
    [
        ({"name": "a"}, "must be a list"),
        (["a"], "JSON object"),
        ([{"name": 1, "dtype": "float32", "shape": [1], "offset": 0, "nbytes": 4}],
         "name must be a string"),
        ([{"name": "a", "dtype": "object", "shape": [1], "offset": 0, "nbytes": 4}],
         "unknown dtype"),
        ([{"name": "a", "dtype": "float32", "shape": [-1], "offset": 0, "nbytes": 4}],
         "dimension"),
        ([{"name": "a", "dtype": "float32", "shape": [1], "offset": -4, "nbytes": 4}],
         "offset"),
        ([{"name": "a", "dtype": "float32", "shape": [1], "offset": 0, "nbytes": True}],
         "nbytes"),
        ([{"name": "a", "dtype": "float32", "shape": [3], "offset": 0, "nbytes": 4}],
         "do not hold"),
        ([{"name": "a", "dtype": "float32", "shape": [2], "offset": 4, "nbytes": 8}],
         "runs past"),
        ([{"name": "a", "dtype": "float32", "shape": [1], "offset": 0,
           "nbytes": 4}] * 2, "twice"),
        ([{"name": "a", "dtype": "float32", "shape": "1", "offset": 0, "nbytes": 4}],
         "shape must be a list"),
    ],
)  # fmt: skip
def test_bad_descriptors_are_refused(arrays: object, why: str) -> None:
    with pytest.raises(WireError, match=why):
        read_message(io.BytesIO(raw({"type": "x", "arrays": arrays}, b"\0" * 8)))


def test_bad_headers_are_refused() -> None:
    for head in (b"\xff\xfe", b"{", b"[" * 100000 + b"]" * 100000, b"[1, 2]"):
        data = MAGIC + struct.pack("<I", len(head)) + head + struct.pack("<Q", 0)
        with pytest.raises(WireError):
            read_message(io.BytesIO(data))
    with pytest.raises(WireError, match="bad magic"):
        read_message(io.BytesIO(b"hello world, a print to stdout"))


def test_model_trees_refer_only_to_their_arrays() -> None:
    arrays = {"m0": np.ones(2)}
    assert tree_from_wire({"w": {"mxarray": "m0"}}, arrays)["w"].shape == (2,)
    for bad in ({"mxarray": "m9"}, {"mxarray": ["m0"]}, {"mxfloat": "1e400x"},
                {"mxfloat": 3}):  # fmt: skip
        with pytest.raises(WireError):
            tree_from_wire({"w": bad}, arrays)


def test_random_mutations_never_escape_as_other_errors() -> None:
    """3000 seeded mutations of good frames: bit flips, cuts, insertions and
    spliced headers; each decodes or is refused, quickly."""
    rng = np.random.default_rng(0)
    good = [frame(), frame({"type": "manifest", "manifest": {"name": "x"}}),
            encode({"type": "ready"})]  # fmt: skip
    for k in range(3000):
        data = bytearray(good[k % len(good)])
        for _ in range(int(rng.integers(1, 6))):
            op, at = int(rng.integers(4)), int(rng.integers(len(data)))
            if op == 0:
                data[at] ^= 1 << int(rng.integers(8))
            elif op == 1:
                del data[at:]
            elif op == 2:
                data[at:at] = rng.bytes(int(rng.integers(1, 9)))
            else:
                data[at] = int(rng.integers(256))
            if not data:
                break
        decodes_or_refuses(bytes(data))


def test_random_headers_never_escape_as_other_errors() -> None:
    rng = np.random.default_rng(1)
    values: list[Any] = [None, True, -1, 0, 2**40, 1.5, "x", "float32", [], [-1],
                         [2**40], {}, [{}], {"mxarray": 1}]  # fmt: skip
    keys = ("name", "dtype", "shape", "offset", "nbytes")
    for _ in range(2000):
        desc = {k: values[int(rng.integers(len(values)))] for k in keys
                if rng.random() < 0.8}  # fmt: skip
        decodes_or_refuses(raw({"type": "x", "arrays": [desc]}, rng.bytes(16)))
