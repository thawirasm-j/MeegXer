"""Node names from before the 0.2.1 package rename keep working."""

from importlib.metadata import EntryPoint

from meegxer.core.graph import Graph
from meegxer.core.registry import Registry, _legacy, canonical_name


def test_old_builtin_names_resolve_to_the_new_ones() -> None:
    assert canonical_name("meegxer.sterile.notch") == "meegxer.nodes.notch"
    assert canonical_name("meegxer.nodes.notch") == "meegxer.nodes.notch"
    assert canonical_name("eeglab.asr") == "eeglab.asr"
    registry = Registry.from_entry_points()
    assert registry.get("meegxer.sterile.notch") is registry.get("meegxer.nodes.notch")
    assert not [n for n in registry.names() if ".sterile." in n]


def test_a_pipeline_saved_with_old_names_still_builds() -> None:
    saved = {"spec_version": "0.2.0", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic", "params": {"duration": 1.0}},
        {"id": "hp", "node": "meegxer.sterile.iir_filter",
         "params": {"kind": "highpass", "low": 1.0}},
    ]}  # fmt: skip
    graph = Graph.from_dict(saved, Registry.from_entry_points())
    assert graph["hp"].manifest.name == "meegxer.nodes.iir_filter"


def test_entry_points_of_stale_old_distributions_are_ignored() -> None:
    group = "meegxer.nodes"
    old = EntryPoint("meegxer.sterile.notch", "meegxer.sterile.filters:Notch", group)
    plugins = EntryPoint("plugins", "meegxer.runtime.index:node_classes", group)
    new = EntryPoint("meegxer.nodes.notch", "meegxer.nodes.filters:Notch", group)
    assert _legacy(old) and _legacy(plugins) and not _legacy(new)
