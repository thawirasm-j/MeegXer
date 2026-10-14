"""Spec 0.2.2: ``primary`` parameters and per-node ``strip_params`` (UI only)."""

import pytest

from meegxer.core.control import manifest_to_dict
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.sources import SyntheticSource


def pipeline(**node: object) -> dict[str, object]:
    return {
        "spec_version": SPEC_VERSION,
        "nodes": [{"id": "src", "node": "meegxer.core.synthetic", **node}],
    }


def test_primary_flag_reaches_the_manifest_listing() -> None:
    listed = manifest_to_dict(SyntheticSource.manifest)
    primary = {p["name"] for p in listed["params"] if p["primary"]}
    assert primary == {"n_channels", "sfreq"}


def test_strip_params_round_trip() -> None:
    registry = Registry.from_entry_points()
    graph = Graph.from_dict(pipeline(strip_params=["sfreq", "seed"]), registry)
    assert graph["src"].strip_params == ("sfreq", "seed")
    assert graph.to_dict()["nodes"][0]["strip_params"] == ["sfreq", "seed"]
    graph.set_strip_params("src", ["duration"])
    assert graph.to_dict()["nodes"][0]["strip_params"] == ["duration"]
    graph.set_strip_params("src", [])
    assert "strip_params" not in graph.to_dict()["nodes"][0]


@pytest.mark.parametrize(
    "names", [["nope"], ["sfreq", "seed", "duration", "n_channels"], ["sfreq", "sfreq"]]
)
def test_strip_params_are_checked(names: list[str]) -> None:
    with pytest.raises(GraphError):
        Graph.from_dict(pipeline(strip_params=names), Registry.from_entry_points())


def test_memory_probe_keeps_a_bounded_history() -> None:
    from meegxer.core.probes import MemoryProbe

    src = SyntheticSource({"duration": 10.0, "n_channels": 2, "sfreq": 100.0})
    probe = MemoryProbe({"keep_seconds": 2.0})
    while (block := src.read(30)) is not None:
        probe.tap(block)
    kept = probe.signal()
    assert 200 <= kept.n_samples < 230
    assert kept.stop_index == 1000  # the newest data is kept
    everything = MemoryProbe({"keep_seconds": 0.0})
    src = SyntheticSource({"duration": 10.0, "n_channels": 2, "sfreq": 100.0})
    while (block := src.read(30)) is not None:
        everything.tap(block)
    assert everything.signal().n_samples == 1000
