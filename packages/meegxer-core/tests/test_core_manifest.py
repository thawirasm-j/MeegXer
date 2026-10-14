import copy
import json
from pathlib import Path
from typing import Any

import pytest

from meegxer.core.manifest import Manifest, ManifestError, ParamError, manifest_schema
from meegxer.core.ports import PortType

REPO = Path(__file__).resolve().parents[3]

VALID: dict[str, Any] = {
    "spec_version": "0.2.0",
    "name": "meegxer.sterile.iir_bandpass",
    "version": "1.0.0",
    "type": "stream",
    "inputs": [{"name": "in", "type": "signal"}],
    "outputs": [{"name": "out", "type": "signal"}],
    "params": [
        {"name": "low", "type": "float", "default": 1.0, "min": 0.0, "unit": "Hz"},
        {"name": "order", "type": "int", "default": 4, "min": 1, "max": 8},
        {"name": "zero_phase", "type": "bool", "default": False},
        {
            "name": "design",
            "type": "enum",
            "default": "butter",
            "choices": ["butter", "cheby1"],
        },
        {"name": "label", "type": "string", "default": "bp"},
    ],
    "latency_samples": 12,
}


def with_changes(**changes: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    data.update(changes)
    return data


def test_package_schema_matches_spec() -> None:
    spec = json.loads((REPO / "spec" / "manifest.schema.json").read_text())
    assert manifest_schema() == spec


def test_valid_manifest_loads() -> None:
    m = Manifest.from_dict(VALID)
    assert m.kind == "stream" and m.latency_samples == 12 and not m.requires_fit
    assert m.inputs[0].type is PortType.SIGNAL
    assert [p.name for p in m.params] == [
        "low",
        "order",
        "zero_phase",
        "design",
        "label",
    ]
    assert Manifest.from_json(json.dumps(VALID)) == m


@pytest.mark.parametrize(
    "data",
    [
        {k: v for k, v in VALID.items() if k != "name"},
        with_changes(name="Bad Name"),
        with_changes(version="1.0"),
        with_changes(spec_version="1.0.0"),
        with_changes(type="realtime"),
        with_changes(inputs=[]),
        with_changes(outputs=[{"name": "out", "type": "audio"}]),
        with_changes(params=[{"name": "x", "type": "float", "default": "high"}]),
        with_changes(params=[{"name": "x", "type": "int", "default": 1.5}]),
        with_changes(params=[{"name": "x", "type": "bool", "default": 1}]),
        with_changes(params=[{"name": "x", "type": "enum", "default": "a"}]),
        with_changes(params=[{"name": "x", "type": "float", "default": 1, "foo": 2}]),
        with_changes(latency_samples=-1),
        with_changes(unexpected=True),
    ],
)
def test_schema_violations_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ManifestError):
        Manifest.from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        with_changes(params=[{"name": "x", "type": "float", "default": 5, "max": 1}]),
        with_changes(
            params=[{"name": "x", "type": "float", "default": 0, "min": 2, "max": 1}]
        ),
        with_changes(
            params=[
                {"name": "x", "type": "enum", "default": "c", "choices": ["a", "b"]}
            ]
        ),
        with_changes(
            params=[
                {"name": "x", "type": "bool", "default": True},
                {"name": "x", "type": "bool", "default": False},
            ]
        ),
        with_changes(outputs=[{"name": "out", "type": "features"}]),
        with_changes(
            outputs=[{"name": "a", "type": "signal"}, {"name": "a", "type": "signal"}]
        ),
    ],
)
def test_semantic_violations_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ManifestError):
        Manifest.from_dict(data)


def test_analysis_node_may_output_features() -> None:
    m = Manifest.from_dict(
        with_changes(type="analysis", outputs=[{"name": "psd", "type": "features"}])
    )
    assert m.outputs[0].type is PortType.FEATURES


def test_resolve_params_defaults_and_overrides() -> None:
    m = Manifest.from_dict(VALID)
    assert m.resolve_params() == {
        "low": 1.0,
        "order": 4,
        "zero_phase": False,
        "design": "butter",
        "label": "bp",
    }
    resolved = m.resolve_params({"low": 2, "order": 6.0, "design": "cheby1"})
    assert resolved["low"] == 2.0 and isinstance(resolved["low"], float)
    assert resolved["order"] == 6 and isinstance(resolved["order"], int)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": 1},
        {"low": -1.0},
        {"order": 9},
        {"order": 2.5},
        {"order": True},
        {"zero_phase": 1},
        {"design": "bessel"},
        {"label": 3},
        {"low": float("inf")},
    ],
)
def test_resolve_params_rejects_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ParamError):
        Manifest.from_dict(VALID).resolve_params(overrides)


def role_manifest(role: str, inputs: int, outputs: int, **extra: Any) -> dict[str, Any]:
    return with_changes(
        role=role,
        inputs=[{"name": f"i{k}", "type": "signal"} for k in range(inputs)],
        outputs=[{"name": f"o{k}", "type": "signal"} for k in range(outputs)],
        **extra,
    )


@pytest.mark.parametrize(
    ("role", "inputs", "outputs"),
    [("source", 0, 1), ("processor", 1, 1), ("probe", 1, 0), ("sink", 1, 0)],
)
def test_roles_with_valid_ports(role: str, inputs: int, outputs: int) -> None:
    assert Manifest.from_dict(role_manifest(role, inputs, outputs)).role == role


@pytest.mark.parametrize(
    ("role", "inputs", "outputs"),
    [("source", 1, 1), ("source", 0, 0), ("processor", 0, 1), ("probe", 1, 1),
     ("sink", 0, 0), ("probe", 2, 0)],
)  # fmt: skip
def test_roles_with_invalid_ports(role: str, inputs: int, outputs: int) -> None:
    with pytest.raises(ManifestError):
        Manifest.from_dict(role_manifest(role, inputs, outputs))


def test_live_params_must_exist_and_fit_only_for_processors() -> None:
    assert Manifest.from_dict(with_changes(live_params=["low"])).live_params == {"low"}
    with pytest.raises(ManifestError, match="live_params"):
        Manifest.from_dict(with_changes(live_params=["nope"]))
    with pytest.raises(ManifestError, match="fitting"):
        Manifest.from_dict(role_manifest("sink", 1, 0, requires_fit=True))


def test_availability_and_advanced_flags() -> None:
    m = Manifest.from_dict(VALID)
    assert m.availability == "both" and m.runs_online and m.runs_offline
    assert not any(p.advanced for p in m.params)
    offline = Manifest.from_dict(with_changes(availability="offline"))
    assert offline.runs_offline and not offline.runs_online
    params = [{"name": "x", "type": "int", "default": 1, "advanced": True}]
    assert Manifest.from_dict(with_changes(params=params)).params[0].advanced
    with pytest.raises(ManifestError):
        Manifest.from_dict(with_changes(availability="sometimes"))
