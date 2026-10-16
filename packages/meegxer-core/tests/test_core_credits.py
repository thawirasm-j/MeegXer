"""Credit for nodes: authors, references and licence (spec 0.2.12)."""

import re
from typing import Any, ClassVar

import pytest

from meegxer.core.control import manifest_to_dict
from meegxer.core.graph import Graph
from meegxer.core.manifest import Manifest, ManifestError
from meegxer.core.references import as_text, collect
from meegxer.core.registry import Registry, node_origin
from meegxer.core.session import build_session

DOI = re.compile(r"^10\.[0-9]{4,9}/\S+$")
BASE: dict[str, Any] = {
    "spec_version": "0.2.12", "name": "lab.x", "version": "0.1.0", "type": "stream",
    "inputs": [{"name": "in", "type": "signal"}],
    "outputs": [{"name": "out", "type": "signal"}],
}  # fmt: skip


def test_credits_are_optional_and_round_trip() -> None:
    plain = Manifest.from_dict(BASE)
    assert plain.authors == () and plain.references == () and plain.license == ""
    refs = [
        {"citation": "Doe J (2025). X.", "doi": "10.1234/abc.5"},
        {"citation": "Tool", "url": "https://x.org"},
    ]
    m = Manifest.from_dict(
        {**BASE, "authors": ["Jane Doe"], "license": "MIT", "references": refs}
    )
    out = manifest_to_dict(m)
    assert out["authors"] == ["Jane Doe"] and out["license"] == "MIT"
    assert out["references"] == refs


@pytest.mark.parametrize(
    "refs",
    [[{"doi": "10.1/x"}], [{"citation": "x", "doi": "https://doi.org/10.1234/x"}],
     [{"citation": "x", "year": 2020}]],
)  # fmt: skip
def test_malformed_references_are_explained(refs: list[dict[str, Any]]) -> None:
    with pytest.raises(ManifestError, match="references"):
        Manifest.from_dict({**BASE, "references": refs})


def test_built_in_nodes_cite_their_methods_and_software() -> None:
    registry = Registry.from_entry_points()
    builtin = [registry.get(n) for n in registry.names() if n.startswith("meegxer.")]
    assert all(node_origin(cls) == "builtin" for cls in builtin)
    for cls in builtin:
        m = cls.manifest
        assert m.authors and m.license == "GPL-3.0-or-later", m.name
        assert all(DOI.match(r.doi) for r in m.references if r.doi), m.name
    cites = {n: {r.doi for r in registry.get(n).manifest.references}
             for n in registry.names()}  # fmt: skip
    assert "10.1016/j.neuroimage.2019.05.026" in cites["meegxer.nodes.ica"]  # ICLabel
    assert "10.1038/s41592-019-0686-2" in cites["meegxer.nodes.iir_filter"]  # SciPy
    assert "10.1016/j.clinph.2004.04.029" in cites["meegxer.nodes.connectivity"]
    assert "10.21105/joss.01896" in cites["meegxer.core.bids"]  # MNE-BIDS


def test_references_are_collected_once_and_written_as_text() -> None:
    a = {"citation": "A", "doi": "10.1/a"}
    again = {"citation": "A again", "doi": "10.1/a"}
    assert collect([a, again, {"citation": "B"}]) == [a, {"citation": "B"}]
    assert as_text(a) == "A https://doi.org/10.1/a"
    assert as_text({"citation": "T", "url": "https://t.org"}) == "T https://t.org"


class Custom:
    manifest: ClassVar[Manifest] = Manifest.from_dict(BASE)

    @classmethod
    def provenance(cls) -> dict[str, Any]:
        return {"command": ["python", "x.py"], "unsandboxed": True}


def test_origin_tells_custom_and_plugin_code_from_built_ins() -> None:
    assert node_origin(Custom) == "custom"
    Custom.provenance = classmethod(lambda cls: {"image": "lab/x:1"})  # type: ignore[method-assign,assignment]
    assert node_origin(Custom) == "plugin"
    assert node_origin(object) == "builtin"


def test_sessions_record_what_each_node_cites() -> None:
    graph = Graph.from_dict(
        {"spec_version": "0.2.12", "nodes": [
            {"id": "input", "node": "meegxer.core.synthetic",
             "params": {"duration": 1.0}},
            {"id": "hp", "node": "meegxer.nodes.iir_filter"},
        ]},
        Registry.from_entry_points(),
    )  # fmt: skip
    nodes = build_session(graph)["nodes"]
    assert "references" not in nodes["input"]  # nothing to cite
    assert nodes["hp"]["references"][0]["doi"] == "10.1038/s41592-019-0686-2"
