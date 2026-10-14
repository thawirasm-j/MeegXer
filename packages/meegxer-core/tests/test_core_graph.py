import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from meegxer.core.block import Block
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest
from meegxer.core.node import NodeRole, Params, Processor, TransformResult
from meegxer.core.ports import PortTypeError
from meegxer.core.probes import MemoryProbe
from meegxer.core.registry import Registry, UnknownNodeError
from meegxer.core.serialization import State
from meegxer.core.sinks import FeatureTsvSink, SetSink
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage


class Passthrough(Processor):
    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.pass", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": [{"name": "k", "type": "float", "default": 1.0}],
    })  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block, {})


REGISTRY = Registry(
    [SyntheticSource, Passthrough, MemoryProbe, SetSink, FeatureTsvSink]
)


def pipeline(*nodes: dict[str, Any]) -> dict[str, Any]:
    return {"spec_version": "0.2.0", "nodes": list(nodes)}


SRC = {"node": "meegxer.core.synthetic", "params": {"duration": 1}}


def test_plain_list_is_a_chain_with_probe_taps() -> None:
    graph = Graph.from_dict(
        pipeline(
            SRC,
            {"id": "a", "node": "test.pass"},
            {"id": "r1", "node": "meegxer.core.probe"},
            {"id": "b", "node": "test.pass", "params": {"k": 2}},
            {"id": "main", "node": "meegxer.core.probe"},
        ),
        REGISTRY,
    )
    inputs = {n.id: n.input for n in graph.nodes}
    assert inputs == {"synthetic1": None, "a": "synthetic1", "r1": "a", "b": "a",
                      "main": "b"}  # fmt: skip
    assert graph.source.id == "synthetic1"
    assert [n.id for n in graph.children("a")] == ["r1", "b"]
    assert graph["b"].role is NodeRole.PROCESSOR
    assert isinstance(graph["b"].impl, Stage) and graph["b"].impl.params["k"] == 2.0


def test_explicit_fan_out_and_round_trip() -> None:
    data = pipeline(
        SRC,
        {"id": "a", "node": "test.pass"},
        {"id": "b", "node": "test.pass", "input": "synthetic1"},
    )
    graph = Graph.from_dict(data, REGISTRY)
    assert graph["b"].input == "synthetic1"
    again = Graph.from_dict(graph.to_dict(), REGISTRY)
    assert again.to_dict() == graph.to_dict()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (pipeline({"node": "test.pass"}), "no upstream"),
        (pipeline(SRC, SRC), "exactly one source"),
        (pipeline(SRC, {"id": "x", "node": "test.pass", "input": "later"},
                  {"id": "later", "node": "test.pass"}), "not an earlier node"),
        (pipeline(SRC, {"id": "a", "node": "test.pass"},
                  {"id": "a", "node": "test.pass"}), "duplicate"),
        (pipeline(SRC, {"id": "p", "node": "meegxer.core.probe"},
                  {"node": "test.pass", "input": "p"}), "no output"),
        (pipeline(dict(SRC, input="x")), "<root>|input|source"),
        ({"spec_version": "0.2.0", "nodes": []}, "nodes"),
        ({"spec_version": "9.0.0", "nodes": [SRC]}, "spec_version"),
    ],
)  # fmt: skip
def test_invalid_pipelines_rejected(data: dict[str, Any], match: str) -> None:
    with pytest.raises(GraphError, match=match):
        Graph.from_dict(data, REGISTRY)


def test_port_type_mismatch_rejected() -> None:
    with pytest.raises(PortTypeError):
        Graph.from_dict(
            pipeline(
                SRC, {"node": "meegxer.core.feature_tsv", "params": {"path": "x"}}
            ),
            REGISTRY,
        )


def test_unknown_node() -> None:
    with pytest.raises(UnknownNodeError, match="available"):
        Graph.from_dict(pipeline(SRC, {"node": "test.missing"}), REGISTRY)


def test_replace_keeps_wiring(tmp_path: Path) -> None:
    graph = Graph.chain([SyntheticSource({"duration": 1}), Stage(Passthrough())])
    graph.replace("synthetic1", SyntheticSource({"duration": 2}))
    assert graph.source.impl.params["duration"] == 2.0
    with pytest.raises(GraphError):
        graph.replace("synthetic1", Stage(Passthrough()))
    path = tmp_path / "p.json"
    path.write_text(json.dumps(graph.to_dict()))
    assert Graph.load(path, REGISTRY).to_dict() == graph.to_dict()


def test_installed_entry_points_resolve() -> None:
    registry = Registry.from_entry_points()
    for name in ("meegxer.core.synthetic", "meegxer.core.file", "meegxer.core.probe",
                 "meegxer.core.set_writer", "meegxer.sterile.iir_filter",
                 "meegxer.sterile.notch", "meegxer.sterile.rereference"):  # fmt: skip
        assert registry.get(name).manifest.name == name


def test_registry_rejects_name_clash() -> None:
    class Other(Passthrough):
        pass

    with pytest.raises(ValueError, match="two node classes"):
        Registry([Passthrough, Other])
