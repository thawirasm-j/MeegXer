"""Serializable model/state values and content fingerprints.

Node models and states must be serializable (docs/WORKPLAN.md decision): sessions
are saved and resumed, and a future Rust core must be able to hold them. Allowed
values are numeric/bool numpy arrays, Python scalars, strings, None, and lists or
str-keyed dicts of those.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

type StateValue = (
    npt.NDArray[Any]
    | int
    | float
    | bool
    | str
    | list[StateValue]
    | dict[str, StateValue]
    | None
)
type State = dict[str, StateValue]

_ARRAY_KEY = "__ndarray__"
_TREE_ENTRY = "__tree__"


class StateSerializationError(TypeError):
    """A model or state contains a value that cannot be serialized."""


def _check_array(path: str, array: npt.NDArray[Any]) -> None:
    if array.dtype.kind not in "biufc":
        raise StateSerializationError(
            f"{path}: arrays must be numeric or bool, got dtype {array.dtype}"
        )


def _encode(value: object, path: str, arrays: dict[str, npt.NDArray[Any]]) -> Any:
    if isinstance(value, np.ndarray):
        _check_array(path, value)
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {_ARRAY_KEY: key}
    if isinstance(value, np.generic):
        return _encode(value.item(), path, arrays)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return {"__float__": repr(value)}  # keeps nan/inf and exact bits
    if isinstance(value, list):
        return [_encode(v, f"{path}[{i}]", arrays) for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in sorted(value.items(), key=lambda item: str(item[0])):
            if not isinstance(k, str) or k.startswith("__"):
                raise StateSerializationError(
                    f"{path}: keys must be str and not start with '__', got {k!r}"
                )
            out[k] = _encode(v, f"{path}.{k}", arrays)
        return out
    raise StateSerializationError(
        f"{path}: unsupported type {type(value).__name__} "
        "(use arrays, scalars, str, None, list or dict)"
    )


def _decode(node: Any, arrays: Mapping[str, npt.NDArray[Any]]) -> StateValue:
    if isinstance(node, list):
        return [_decode(v, arrays) for v in node]
    if isinstance(node, dict):
        if _ARRAY_KEY in node:
            return arrays[node[_ARRAY_KEY]]
        if "__float__" in node:
            return float(node["__float__"])
        return {k: _decode(v, arrays) for k, v in node.items()}
    value: StateValue = node
    return value


def check_state(state: Mapping[str, object]) -> None:
    """Raise :class:`StateSerializationError` if ``state`` is not serializable."""
    _encode(dict(state), "state", {})


def state_to_bytes(state: Mapping[str, object]) -> bytes:
    """Serialize a model or state (npz container, no pickle)."""
    arrays: dict[str, npt.NDArray[Any]] = {}
    tree = _encode(dict(state), "state", arrays)
    tree_bytes = np.frombuffer(json.dumps(tree).encode(), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **{_TREE_ENTRY: tree_bytes}, **arrays)
    return buffer.getvalue()


def state_from_bytes(data: bytes) -> State:
    """Inverse of :func:`state_to_bytes`."""
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files if name != _TREE_ENTRY}
        tree = json.loads(npz[_TREE_ENTRY].tobytes().decode())
    decoded = _decode(tree, arrays)
    assert isinstance(decoded, dict)
    return decoded


def hash_state(state: Mapping[str, object]) -> str:
    """Deterministic SHA-256 of a model or state's content (dtype, shape, bytes)."""
    arrays: dict[str, npt.NDArray[Any]] = {}
    tree = _encode(dict(state), "state", arrays)
    digest = hashlib.sha256(json.dumps(tree, sort_keys=True).encode())
    for key in sorted(arrays, key=lambda k: int(k[1:])):
        array = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}:{array.dtype.str}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    """Stable JSON used for hashing parameters (sorted keys, no NaN)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(*parts: str) -> str:
    """SHA-256 over an ordered sequence of strings."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()
