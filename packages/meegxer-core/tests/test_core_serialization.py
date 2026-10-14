import numpy as np
import pytest

from meegxer.core.serialization import (
    StateSerializationError,
    canonical_json,
    check_state,
    fingerprint,
    hash_state,
    state_from_bytes,
    state_to_bytes,
)


def sample_state() -> dict[str, object]:
    return {
        "zi": np.arange(12, dtype=np.float64).reshape(2, 3, 2),
        "phase": np.array([1 + 2j, 3 - 1j]),
        "counts": np.array([1, 2, 3], dtype=np.int32),
        "mask": np.array([True, False]),
        "n_seen": 42,
        "gain": 0.1,
        "nan": float("nan"),
        "mode": "causal",
        "done": False,
        "nothing": None,
        "history": [np.zeros(2), {"inner": 1.5}],
    }


def test_round_trip_preserves_values_and_dtypes() -> None:
    state = sample_state()
    restored = state_from_bytes(state_to_bytes(state))
    for key in ("zi", "phase", "counts", "mask"):
        original = state[key]
        assert isinstance(original, np.ndarray)
        np.testing.assert_array_equal(restored[key], original)
        assert restored[key].dtype == original.dtype  # type: ignore[union-attr]
    assert restored["n_seen"] == 42 and restored["gain"] == 0.1
    assert np.isnan(restored["nan"])  # type: ignore[arg-type]
    assert restored["mode"] == "causal" and restored["done"] is False
    assert restored["nothing"] is None
    history = restored["history"]
    assert isinstance(history, list) and history[1] == {"inner": 1.5}


@pytest.mark.parametrize(
    "bad",
    [
        {"obj": np.array(["a", "b"])},
        {"obj": np.array([object()], dtype=object)},
        {"tuple": (1, 2)},
        {"set": {1}},
        {"nested": {1: "int key"}},
        {"__reserved": 1},
    ],
)
def test_unserializable_values_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(StateSerializationError):
        check_state(bad)


def test_numpy_scalars_become_python_scalars() -> None:
    restored = state_from_bytes(
        state_to_bytes({"x": np.float64(2.5), "n": np.int64(3)})
    )
    assert restored == {"x": 2.5, "n": 3}


def test_hash_is_content_based_and_order_independent() -> None:
    a = {"x": np.ones(3), "y": np.zeros(2), "k": 1}
    b = {"k": 1, "y": np.zeros(2), "x": np.ones(3)}
    assert hash_state(a) == hash_state(b)
    assert hash_state(a) == hash_state(state_from_bytes(state_to_bytes(a)))
    assert hash_state(a) != hash_state({**a, "x": np.ones(3) * 2})
    assert hash_state({"x": np.ones(3)}) != hash_state({"x": np.ones(3, np.float32)})
    assert hash_state({"x": np.ones(4)}) != hash_state({"x": np.ones((2, 2))})


def test_canonical_json_and_fingerprint() -> None:
    assert canonical_json({"b": 1, "a": 2.0}) == '{"a":2.0,"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    assert fingerprint("ab", "c") != fingerprint("a", "bc")
