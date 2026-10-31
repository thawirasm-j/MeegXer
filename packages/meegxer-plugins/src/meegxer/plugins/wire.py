"""Host-side implementation of the plugin wire format (spec/plugin-protocol.md).

The SDKs carry their own implementations of the same spec; tests cross-check them.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, Any

import numpy as np
import numpy.typing as npt

MAGIC = b"MXP1"
PROTOCOL = "meegxer.plugin/0.2"
DTYPES = {"float32", "float64", "int32", "int64", "uint8", "bool"}
_ARRAY = "mxarray"
_FLOAT = "mxfloat"
MAX_HEADER = 16 * 1024 * 1024
# a plugin is untrusted (rule 9): what it declares is checked before it is
# read, and a frame is read in pieces, so memory grows only with what arrives
MAX_PAYLOAD = 256 * 1024 * 1024
READ_CHUNK = 1024 * 1024


class WireError(ValueError):
    """A frame is malformed or violates the protocol."""


@dataclass
class Message:
    header: dict[str, Any]
    arrays: dict[str, npt.NDArray[Any]] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return str(self.header.get("type", ""))


def _to_wire_dtype(array: npt.NDArray[Any]) -> str:
    if array.dtype == np.bool_:
        return "bool"
    name = array.dtype.newbyteorder("=").name
    if name not in DTYPES:
        raise WireError(f"dtype {array.dtype} is not allowed on the wire")
    return name


def encode(
    header: Mapping[str, Any], arrays: Mapping[str, npt.NDArray[Any]] = {}
) -> bytes:
    """Build one frame."""
    descriptors = []
    chunks = []
    offset = 0
    for name, array in arrays.items():
        dtype = _to_wire_dtype(np.asarray(array))
        data = np.ascontiguousarray(
            array, dtype=np.dtype(dtype).newbyteorder("<")
        ).tobytes()
        descriptors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(np.shape(array)),
                "offset": offset,
                "nbytes": len(data),
            }
        )
        chunks.append(data)
        offset += len(data)
    head = json.dumps({**header, "arrays": descriptors}, allow_nan=False).encode()
    payload = b"".join(chunks)
    return (
        MAGIC
        + struct.pack("<I", len(head))
        + head
        + struct.pack("<Q", len(payload))
        + payload
    )


def _read_exact(stream: IO[bytes], n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(min(remaining, READ_CHUNK))
        if not chunk:
            raise EOFError("stream closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream: IO[bytes]) -> Message:
    """Read one frame; raises EOFError on a closed stream."""
    magic = stream.read(4)
    if not magic:
        raise EOFError("stream closed")
    if len(magic) < 4:
        magic += _read_exact(stream, 4 - len(magic))
    if magic != MAGIC:
        raise WireError(f"bad magic {magic!r} (is the plugin printing to stdout?)")
    (head_len,) = struct.unpack("<I", _read_exact(stream, 4))
    if head_len > MAX_HEADER:
        raise WireError(f"header of {head_len} bytes is too large")
    try:
        header = json.loads(_read_exact(stream, head_len))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise WireError(f"header is not JSON: {exc}") from None
    if not isinstance(header, dict):
        raise WireError("header must be a JSON object")
    (payload_len,) = struct.unpack("<Q", _read_exact(stream, 8))
    if payload_len > MAX_PAYLOAD:
        raise WireError(f"payload of {payload_len} bytes is too large")
    payload = _read_exact(stream, payload_len)
    descriptors = header.pop("arrays", [])
    if not isinstance(descriptors, list):
        raise WireError("'arrays' must be a list of array descriptors")
    arrays: dict[str, npt.NDArray[Any]] = {}
    for desc in descriptors:
        name, arr = _array(desc, payload)
        if name in arrays:
            raise WireError(f"array {name!r} is described twice")
        arrays[name] = arr
    return Message(header, arrays)


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WireError(f"{what} must be a whole number >= 0, got {value!r}")
    return value


def _array(desc: object, payload: bytes) -> tuple[str, npt.NDArray[Any]]:
    """One array of the payload, after checking its descriptor whole."""
    if not isinstance(desc, dict):
        raise WireError("an array descriptor must be a JSON object")
    name, kind, shape = desc.get("name"), desc.get("dtype"), desc.get("shape")
    if not isinstance(name, str):
        raise WireError(f"array name must be a string, got {name!r}")
    if not isinstance(kind, str) or kind not in DTYPES:
        raise WireError(f"unknown dtype {kind!r}")
    if not isinstance(shape, list):
        raise WireError(f"array {name!r}: shape must be a list")
    dims = [_count(n, f"array {name!r}: a dimension") for n in shape]
    start = _count(desc.get("offset"), f"array {name!r}: offset")
    nbytes = _count(desc.get("nbytes"), f"array {name!r}: nbytes")
    dtype = np.dtype(kind).newbyteorder("<")
    if math.prod(dims) * dtype.itemsize != nbytes:
        raise WireError(f"array {name!r}: {nbytes} bytes do not hold shape {dims}")
    if start + nbytes > len(payload):
        raise WireError(f"array {name!r} runs past the payload")
    arr = np.frombuffer(payload[start : start + nbytes], dtype=dtype).reshape(dims)
    return name, arr.astype(arr.dtype.newbyteorder("="))


def tree_to_wire(tree: object, arrays: dict[str, npt.NDArray[Any]], prefix: str) -> Any:
    """Replace arrays (and non-finite floats) in a model/state tree by references."""
    if isinstance(tree, np.ndarray):
        name = f"{prefix}{len(arrays)}"
        arrays[name] = tree
        return {_ARRAY: name}
    if isinstance(tree, np.generic):
        return tree_to_wire(tree.item(), arrays, prefix)
    if isinstance(tree, float) and not math.isfinite(tree):
        return {_FLOAT: "nan" if math.isnan(tree) else ("inf" if tree > 0 else "-inf")}
    if isinstance(tree, list):
        return [tree_to_wire(v, arrays, prefix) for v in tree]
    if isinstance(tree, dict):
        return {str(k): tree_to_wire(v, arrays, prefix) for k, v in tree.items()}
    return tree


def tree_from_wire(tree: object, arrays: Mapping[str, npt.NDArray[Any]]) -> Any:
    if isinstance(tree, dict):
        if set(tree) == {_ARRAY}:
            name = tree[_ARRAY]
            if not isinstance(name, str) or name not in arrays:
                raise WireError(f"tree refers to missing array {name!r}")
            return np.array(arrays[name])
        if set(tree) == {_FLOAT}:
            if tree[_FLOAT] not in ("nan", "inf", "-inf"):
                raise WireError(f"bad non-finite float {tree[_FLOAT]!r}")
            return float(tree[_FLOAT])
        return {k: tree_from_wire(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [tree_from_wire(v, arrays) for v in tree]
    return tree
