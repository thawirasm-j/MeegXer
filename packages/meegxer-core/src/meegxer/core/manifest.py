"""Node manifests: parse, validate against the spec schema, resolve parameters.

The canonical schema is ``spec/manifest.schema.json`` (CLAUDE.md rule 11). The core
ships an identical copy so it can validate at runtime; a test keeps them in sync.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Literal

import jsonschema

from meegxer.core.ports import EVENTS_RAIL, Port, PortType

SPEC_VERSION = "0.2.19"

type ParamValue = float | int | bool | str
type ParamType = Literal["float", "int", "bool", "enum", "string"]
type NodeKind = Literal["stream", "analysis"]
type NodeRoleName = Literal["source", "processor", "probe", "sink"]
type Availability = Literal["online", "offline", "both"]


class ManifestError(ValueError):
    """A manifest is invalid (schema or semantic check)."""


class ParamError(ValueError):
    """Parameter values do not satisfy the manifest."""


@cache
def manifest_schema() -> dict[str, Any]:
    text = resources.files("meegxer.core").joinpath("schema/manifest.schema.json")
    schema: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    return schema


def _port(data: Mapping[str, Any]) -> Port:
    return Port(data["name"], PortType(data["type"]), data.get("rail", ""),
                data.get("space", ""))  # fmt: skip


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    type: ParamType
    default: ParamValue
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    description: str = ""
    advanced: bool = False
    primary: bool = False
    step: float | None = None

    def coerce(self, value: object) -> ParamValue:
        """Validate one value against this spec and return it in canonical form."""
        name = self.name
        if self.type == "bool":
            if not isinstance(value, bool):
                raise ParamError(f"{name}: expected bool, got {value!r}")
            return value
        if self.type in ("enum", "string"):
            if not isinstance(value, str):
                raise ParamError(f"{name}: expected str, got {value!r}")
            if self.type == "enum" and value not in self.choices:
                raise ParamError(f"{name}: {value!r} not in {list(self.choices)}")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ParamError(f"{name}: expected a number, got {value!r}")
        number: int | float
        if self.type == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ParamError(f"{name}: expected an integer, got {value!r}")
            number = int(value)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ParamError(f"{name}: must be finite, got {value!r}")
        if self.min is not None and number < self.min:
            raise ParamError(f"{name}: {number} is below the minimum {self.min}")
        if self.max is not None and number > self.max:
            raise ParamError(f"{name}: {number} is above the maximum {self.max}")
        return number


@dataclass(frozen=True, slots=True)
class Reference:
    """Something to cite when using a node (spec 0.2.12)."""

    citation: str
    doi: str = ""
    url: str = ""

    def as_dict(self) -> dict[str, str]:
        out = {"citation": self.citation}
        if self.doi:
            out["doi"] = self.doi
        if self.url:
            out["url"] = self.url
        return out


@dataclass(frozen=True, slots=True)
class Manifest:
    spec_version: str
    name: str
    version: str
    kind: NodeKind
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    params: tuple[ParamSpec, ...] = ()
    latency_samples: int = 0
    requires_fit: bool = False
    requires_positions: bool = False
    fit_events: bool = False  # fit also gets the calibration's events (0.2.16)
    image: str | None = None
    description: str = ""
    role: NodeRoleName = "processor"
    live_params: frozenset[str] = frozenset()
    availability: Availability = "both"
    authors: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    license: str = ""
    category: str = ""
    channel_types: tuple[str, ...] = ()  # processed unless the pipeline says otherwise

    @property
    def runs_online(self) -> bool:
        return self.availability in ("online", "both")

    @property
    def runs_offline(self) -> bool:
        return self.availability in ("offline", "both")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Validate ``data`` against the schema and semantic rules."""
        try:
            jsonschema.validate(dict(data), manifest_schema())
        except jsonschema.ValidationError as exc:
            location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            raise ManifestError(f"{location}: {exc.message}") from exc

        params = tuple(
            ParamSpec(
                name=p["name"],
                type=p["type"],
                default=p["default"],
                min=p.get("min"),
                max=p.get("max"),
                choices=tuple(p.get("choices", ())),
                unit=p.get("unit", ""),
                description=p.get("description", ""),
                advanced=p.get("advanced", False),
                primary=p.get("primary", False),
                step=p.get("step"),
            )
            for p in data.get("params", [])
        )
        manifest = cls(
            spec_version=data["spec_version"],
            name=data["name"],
            version=data["version"],
            kind=data["type"],
            inputs=tuple(_port(p) for p in data["inputs"]),
            outputs=tuple(_port(p) for p in data["outputs"]),
            params=params,
            latency_samples=data.get("latency_samples", 0),
            requires_fit=data.get("requires_fit", False),
            requires_positions=data.get("requires_positions", False),
            fit_events=data.get("fit_events", False),
            image=data.get("image"),
            description=data.get("description", ""),
            role=data.get("role", "processor"),
            live_params=frozenset(data.get("live_params", ())),
            availability=data.get("availability", "both"),
            authors=tuple(data.get("authors", ())),
            references=tuple(
                Reference(r["citation"], r.get("doi", ""), r.get("url", ""))
                for r in data.get("references", ())
            ),
            license=data.get("license", ""),
            category=data.get("category", ""),
            channel_types=tuple(data.get("channel_types", ())),
        )
        manifest._check_semantics()
        return manifest

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        return cls.from_dict(json.loads(text))

    def _check_semantics(self) -> None:
        for ports, kind in ((self.inputs, "input"), (self.outputs, "output")):
            names = [p.name for p in ports]
            if len(set(names)) != len(names):
                raise ManifestError(f"duplicate {kind} port names: {names}")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ManifestError(f"duplicate parameter names: {names}")
        n_in, n_out = len(self.inputs), len(self.outputs)
        expected = {
            "source": n_in == 0 and n_out >= 1,
            "processor": n_in >= 1 and n_out >= 1,
            "probe": n_in == 1 and n_out == 0,
            "sink": n_in >= 1 and n_out == 0,
        }
        if not expected[self.role]:
            raise ManifestError(
                f"a {self.role} cannot have {n_in} input(s) and {n_out} output(s)"
            )
        self._check_rail_ports()
        if (
            self.role == "processor"
            and self.kind == "stream"
            and any(p.type is not PortType.SIGNAL
                    for p in (*self.inputs[:1], *self.outputs[:1]))
        ):  # fmt: skip
            raise ManifestError("stream processors take and return Signal ports only")
        if self.requires_fit and self.role != "processor":
            raise ManifestError("only processors can require fitting")
        if self.fit_events and not self.requires_fit:
            raise ManifestError("fit_events needs requires_fit")
        unknown_live = sorted(self.live_params - set(names))
        if unknown_live:
            raise ManifestError(f"live_params names unknown parameters {unknown_live}")
        for spec in self.params:
            if spec.min is not None and spec.max is not None and spec.min > spec.max:
                raise ManifestError(f"{spec.name}: min is greater than max")
            try:
                spec.coerce(spec.default)
            except ParamError as exc:
                raise ManifestError(f"invalid default: {exc}") from exc

    def _check_rail_ports(self) -> None:
        """Main ports (the first of each) name no rail; every further port does.
        Only a processor has rail ports; no output writes the Events rail (events
        travel with the transform's result); the Events rail is read by an Events
        port; a space is for Signal ports."""
        for kind, ports in (("input", self.inputs), ("output", self.outputs)):
            for i, port in enumerate(ports):
                if port.space and port.type is not PortType.SIGNAL:
                    raise ManifestError(f"{kind} {port.name!r}: only Signal ports "
                                        "have a space")  # fmt: skip
                if i == 0 and port.rail:
                    raise ManifestError(f"{kind} {port.name!r} is the main port; "
                                        "only further ports name a rail")  # fmt: skip
                if i == 0:
                    continue
                if self.role != "processor":
                    raise ManifestError(f"a {self.role} has one {kind} port")
                if not port.rail:
                    raise ManifestError(f"extra {kind} {port.name!r} must name the "
                                        "rail it reads or writes")  # fmt: skip
                if port.rail == EVENTS_RAIL and (
                    kind == "output" or port.type is not PortType.EVENTS
                ):
                    raise ManifestError(
                        f"{kind} {port.name!r}: the Events rail is read by an Events "
                        "input; events are written through the transform's events"
                    )
                if kind == "output" and port.type is PortType.EVENTS:
                    raise ManifestError(f"output {port.name!r}: events are written "
                                        "through the transform's events")  # fmt: skip

    @property
    def rail_inputs(self) -> tuple[Port, ...]:
        """The extra input ports: the rails this node reads."""
        return self.inputs[1:]

    @property
    def rail_outputs(self) -> tuple[Port, ...]:
        """The extra output ports: the rails this node writes."""
        return self.outputs[1:]

    @property
    def reads_events(self) -> bool:
        """Fit or transform sees the Events rail (fit_events, or an Events port)."""
        return self.fit_events or any(p.rail == EVENTS_RAIL for p in self.rail_inputs)

    def resolve_params(
        self, overrides: Mapping[str, object] | None = None
    ) -> dict[str, ParamValue]:
        """Defaults merged with ``overrides``; unknown names and bad values raise."""
        overrides = dict(overrides or {})
        known = {spec.name: spec for spec in self.params}
        unknown = sorted(set(overrides) - set(known))
        if unknown:
            raise ParamError(f"{self.name}: unknown parameter(s) {unknown}")
        return {
            name: spec.coerce(overrides.get(name, spec.default))
            for name, spec in known.items()
        }
