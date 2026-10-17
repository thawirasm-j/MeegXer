"""The pipeline graph (CLAUDE.md rule 5): a DAG, built from an ordered node list.

Nodes are kept in insertion order and may only take input from an earlier node,
so insertion order is a topological order and cycles cannot be expressed. The v1
UI's linear chain with probe taps is the special case where each node's input
defaults to the nearest earlier source or processor whose output type matches the
node's input (so a Features strip branches off the Signal chain, spec 0.2.7).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

from meegxer.core.block import SignalInfo
from meegxer.core.manifest import SPEC_VERSION, Manifest, ParamValue
from meegxer.core.node import NodeRole, Probe, Processor, Sink, Source
from meegxer.core.ports import EVENTS_RAIL, Port, check_connection
from meegxer.core.recording import (
    RecordingError,
    RecordingOverrides,
    montage_positions,
    recording_of,
    with_recording,
)
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage

type NodeImpl = Source | Stage | Probe | Sink


class GraphError(ValueError):
    """The pipeline is not a valid graph (ids, inputs, roles or port types)."""


def positions_message(missing: Mapping[str, list[str]]) -> str:
    """Why a pipeline cannot run: which nodes lack positions for which channels."""
    parts = []
    for node_id, labels in missing.items():
        shown = ", ".join(labels[:8]) + (f" and {len(labels) - 8} more"
                                         if len(labels) > 8 else "")  # fmt: skip
        parts.append(f"{node_id} needs electrode positions for {shown}")
    return ("; ".join(parts) + ". Set a montage or positions (Recording window, "
            "Channels tab; recording.montage in a pipeline file)")  # fmt: skip


@cache
def pipeline_schema() -> dict[str, Any]:
    text = resources.files("meegxer.core").joinpath("schema/pipeline.schema.json")
    schema: dict[str, Any] = json.loads(text.read_text(encoding="utf-8"))
    return schema


def _role_of(impl: NodeImpl) -> NodeRole:
    if isinstance(impl, Stage):
        return NodeRole.PROCESSOR
    return impl.role


def _params_of(impl: NodeImpl) -> dict[str, ParamValue]:
    return dict(impl.params)


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    impl: NodeImpl
    input: str | None
    strip_params: tuple[str, ...] = ()  # UI only: shown on the strip (advanced mode)

    @property
    def role(self) -> NodeRole:
        return _role_of(self.impl)

    @property
    def manifest(self) -> Manifest:
        return self.impl.manifest


class Graph:
    """Ordered DAG of sources, processor stages, probes and sinks."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._counter: dict[str, int] = {}

    # -- construction -----------------------------------------------------------

    def add(
        self,
        impl: NodeImpl,
        *,
        id: str | None = None,
        input: str | None = None,
        strip_params: Iterable[str] = (),
    ) -> GraphNode:
        """Append a node. ``input`` defaults to the nearest earlier source or
        processor with a matching output port type; sources take no input."""
        role = _role_of(impl)
        if impl.manifest.role != role.value:
            raise GraphError(
                f"{impl.manifest.name} declares role {impl.manifest.role!r} "
                f"but is a {role.value}"
            )
        node_id = id or self._next_id(impl.manifest.name)
        if node_id in self._nodes:
            raise GraphError(f"duplicate node id {node_id!r}")

        if role is NodeRole.SOURCE:
            if input is not None:
                raise GraphError(f"source {node_id!r} cannot take an input")
            if any(n.role is NodeRole.SOURCE for n in self._nodes.values()):
                raise GraphError("v1 pipelines have exactly one source")
        else:
            input = input or self._default_input(impl.manifest.inputs[:1])
            if input is None:
                wanted = ", ".join(p.type.value for p in impl.manifest.inputs[:1])
                raise GraphError(
                    f"{node_id!r} has no earlier node with {wanted or 'an'} output "
                    "to read from"
                )
            parent = self._nodes.get(input)
            if parent is None:
                raise GraphError(
                    f"{node_id!r} reads from {input!r}, which is not an earlier node"
                )
            if not parent.manifest.outputs:
                raise GraphError(f"{input!r} ({parent.role.value}) has no output")
            check_connection(parent.manifest.outputs[0], impl.manifest.inputs[0])
            for port in impl.manifest.rail_inputs:  # extra inputs read rails
                self._rail_source(node_id, port)

        node = GraphNode(node_id, impl, input, self._check_strip(impl, strip_params))
        self._nodes[node_id] = node
        return node

    @staticmethod
    def _check_strip(impl: NodeImpl, names: Iterable[str]) -> tuple[str, ...]:
        chosen = tuple(names)
        known = {p.name for p in impl.manifest.params}
        unknown = [n for n in chosen if n not in known]
        if unknown:
            raise GraphError(
                f"{impl.manifest.name}: unknown strip parameter(s) {unknown}"
            )
        if len(chosen) > 3 or len(set(chosen)) != len(chosen):
            raise GraphError("strip_params: up to 3 distinct parameters")
        return chosen

    def set_strip_params(self, node_id: str, names: Iterable[str]) -> None:
        """Choose the parameters a node's strip shows (display only)."""
        node = self._nodes[node_id]
        self._nodes[node_id] = dataclasses.replace(
            node, strip_params=self._check_strip(node.impl, names)
        )

    def _default_input(self, inputs: Iterable[Port]) -> str | None:
        wanted = next((port.type for port in inputs), None)
        for node in reversed(self._nodes.values()):
            if node.role not in (NodeRole.SOURCE, NodeRole.PROCESSOR):
                continue
            outputs = node.manifest.outputs
            if wanted is None or (outputs and outputs[0].type is wanted):
                return node.id
        return None

    def _next_id(self, name: str) -> str:
        base = name.rsplit(".", 1)[-1]
        while True:
            self._counter[base] = self._counter.get(base, 0) + 1
            candidate = f"{base}{self._counter[base]}"
            if candidate not in self._nodes:
                return candidate

    @classmethod
    def chain(cls, impls: Iterable[NodeImpl]) -> Graph:
        """Linear chain; probes and sinks tap the nearest earlier processor."""
        graph = cls()
        for impl in impls:
            graph.add(impl)
        return graph

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], registry: Registry) -> Graph:
        try:
            jsonschema.validate(dict(data), pipeline_schema())
        except jsonschema.ValidationError as exc:
            location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            raise GraphError(f"{location}: {exc.message}") from exc
        graph = cls()
        for entry in data["nodes"]:
            node_cls = registry.get(entry["node"])
            params = entry.get("params", {})
            impl: NodeImpl
            passthrough = bool(entry.get("pass_uncalibrated", False))
            if issubclass(node_cls, Processor):
                try:
                    impl = Stage(
                        node_cls(),
                        params,
                        pass_uncalibrated=passthrough,
                        channel_types=entry.get("channel_types"),
                    )
                except ValueError as exc:
                    raise GraphError(
                        f"{entry.get('id', entry['node'])}: {exc}"
                    ) from exc
            else:
                if passthrough:
                    raise GraphError(
                        f"{entry['node']}: pass_uncalibrated applies to processors only"
                    )
                if "channel_types" in entry:
                    raise GraphError(
                        f"{entry['node']}: channel_types applies to processors only"
                    )
                impl = cls._configured(node_cls, params, entry.get("recording"))
            graph.add(
                impl,
                id=entry.get("id"),
                input=entry.get("input"),
                strip_params=entry.get("strip_params", ()),
            )
        return graph

    @staticmethod
    def _configured(
        node_cls: type[Any], params: Mapping[str, Any], recording: Any
    ) -> NodeImpl:
        if recording is None:
            impl: NodeImpl = node_cls(params)
            return impl
        if not issubclass(node_cls, Source):
            raise GraphError(f"{node_cls.manifest.name}: recording applies to sources")
        try:
            overrides = RecordingOverrides.from_dict(recording)
            if overrides.montage:
                montage_positions(overrides.montage)  # fail early on a bad montage
        except (RecordingError, ValueError, OSError) as exc:
            raise GraphError(f"recording: {exc}") from exc
        return with_recording(node_cls, params, overrides)

    @classmethod
    def load(cls, path: str | Path, registry: Registry | None = None) -> Graph:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, registry or Registry.from_entry_points())

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "nodes": [
                {
                    "id": n.id,
                    "node": n.manifest.name,
                    "params": _params_of(n.impl),
                    **({"input": n.input} if n.input is not None else {}),
                    **(
                        {"recording": rec}
                        if isinstance(n.impl, Source) and (rec := recording_of(n.impl))
                        else {}
                    ),
                    **(
                        {"strip_params": list(n.strip_params)} if n.strip_params else {}
                    ),
                    **(
                        {"pass_uncalibrated": True}
                        if isinstance(n.impl, Stage) and n.impl.pass_uncalibrated
                        else {}
                    ),
                    **(
                        {"channel_types": list(n.impl.channel_types)}
                        if isinstance(n.impl, Stage)
                        and n.impl.channel_types != n.impl.default_channel_types
                        else {}
                    ),
                }
                for n in self._nodes.values()
            ],
        }

    # -- queries ----------------------------------------------------------------

    @property
    def nodes(self) -> list[GraphNode]:
        """All nodes in topological (insertion) order."""
        return list(self._nodes.values())

    def __getitem__(self, node_id: str) -> GraphNode:
        return self._nodes[node_id]

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def children(self, node_id: str) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.input == node_id]

    def _rail_source(
        self, node_id: str, port: Port, before: str | None = None
    ) -> tuple[str, str]:
        """Who writes the rail an extra input reads: ``("", "")`` for the Events
        rail, else the nearest earlier node with an extra output writing it (a
        rail exists only to the right of its producer, docs/RAILS.md)."""
        if port.rail == EVENTS_RAIL:
            return "", ""
        earlier = list(self._nodes.values())
        if before is not None:
            earlier = earlier[: [n.id for n in earlier].index(before)]
        for node in reversed(earlier):
            for out in node.manifest.rail_outputs:
                if out.rail == port.rail:
                    check_connection(out, port)
                    return node.id, out.name
        raise GraphError(
            f"{node_id!r} reads the {port.rail!r} rail, which no earlier node writes"
        )

    def rail_sources(self, node_id: str) -> dict[str, tuple[str, str]]:
        """For each extra input of ``node_id``: (producer id, its output port),
        or ("", "") for the Events rail."""
        node = self._nodes[node_id]
        return {
            port.name: self._rail_source(node_id, port, before=node_id)
            for port in node.manifest.rail_inputs
        }

    def rail_readers(self, node_id: str) -> list[GraphNode]:
        """The nodes that read a rail ``node_id`` writes."""
        return [
            n for n in self._nodes.values()
            if n.manifest.rail_inputs and n.id != node_id
            and any(p == node_id for p, _ in self.rail_sources(n.id).values())
        ]  # fmt: skip

    @property
    def source(self) -> GraphNode:
        for node in self._nodes.values():
            if node.role is NodeRole.SOURCE:
                return node
        raise GraphError("pipeline has no source")

    def missing_positions(self, info: SignalInfo) -> dict[str, list[str]]:
        """Nodes that need electrode positions (manifest ``requires_positions``)
        and the good channels of ``info`` they process that have none."""
        positions = dict(zip(info.ch_labels, info.positions, strict=True)) \
            if info.positions else {}  # fmt: skip
        found: dict[str, list[str]] = {}
        for node in self.nodes:
            if not (node.manifest.requires_positions and isinstance(node.impl, Stage)):
                continue
            missing = [
                label
                for label in info.labels_of_types(node.impl.channel_types)
                if label not in info.bads and positions.get(label) is None
            ]
            if missing:
                found[node.id] = missing
        return found

    def check_positions(self, info: SignalInfo) -> None:
        """Refuse to run nodes that need electrode positions the input lacks."""
        missing = self.missing_positions(info)
        if missing:
            raise GraphError(positions_message(missing))

    def remove(self, node_id: str) -> None:
        """Remove a node that nothing reads from."""
        if self.children(node_id) or self.rail_readers(node_id):
            raise GraphError(f"cannot remove {node_id!r}: other nodes read from it")
        del self._nodes[node_id]

    def replace(self, node_id: str, impl: NodeImpl) -> None:
        """Swap the implementation of a node, keeping its id and wiring."""
        old = self._nodes[node_id]
        if _role_of(impl) is not old.role:
            raise GraphError("replacement must have the same role")
        if old.input is not None:
            parent = self._nodes[old.input]
            check_connection(parent.manifest.outputs[0], impl.manifest.inputs[0])
        for child in self.children(node_id):
            check_connection(impl.manifest.outputs[0], child.manifest.inputs[0])
        self._nodes[node_id] = GraphNode(node_id, impl, old.input, old.strip_params)
