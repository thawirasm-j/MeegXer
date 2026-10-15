import pytest

from meegxer.core.control import manifest_to_dict
from meegxer.core.registry import Registry
from meegxer.gui.pipeline_doc import PipelineDoc, tap_id


@pytest.fixture
def doc() -> PipelineDoc:
    registry = Registry.from_entry_points()
    d = PipelineDoc(
        [manifest_to_dict(registry.get(n).manifest) for n in registry.names()]
    )
    d.set_source("meegxer.core.synthetic")
    d.insert(1, "meegxer.sterile.iir_filter")
    d.insert(2, "meegxer.sterile.notch")
    return d


def test_chain_edits_keep_the_input_first(doc: PipelineDoc) -> None:
    assert [s.id for s in doc.strips] == ["input", "iirfilter1", "notch1"]
    doc.move("notch1", 1)
    assert [s.id for s in doc.strips] == ["input", "notch1", "iirfilter1"]
    doc.move("notch1", 3)
    assert [s.id for s in doc.strips] == ["input", "iirfilter1", "notch1"]
    doc.insert(0, "meegxer.sterile.notch")  # cannot go before the input
    assert doc.strips[1].id == "notch2"
    with pytest.raises(ValueError):
        doc.move("input", 2)
    with pytest.raises(ValueError):
        doc.remove("input")
    with pytest.raises(ValueError):
        doc.insert(1, "meegxer.core.synthetic")


def test_mode_comes_from_the_input(doc: PipelineDoc) -> None:
    modes = [doc.mode]
    assert not doc.mode_is_fixed()
    doc.set_source("meegxer.core.file")
    modes.append(doc.mode)
    doc.set_source("meegxer.core.lsl_in")
    modes.append(doc.mode)
    assert modes == ["live", "offline", "live"] and doc.mode_is_fixed()
    doc.input_mode = "offline"
    assert str(doc.mode) == "live"  # LSL is always live


def test_strict_compatibility_with_the_adaptable_band_pass(doc: PipelineDoc) -> None:
    zero = doc.insert(3, "meegxer.sterile.iir_filter_zerophase")
    out = doc.insert(4, "meegxer.core.lsl_out")
    assert doc.incompatibility(zero) == "Offline only"
    assert doc.incompatibility(doc.get("iirfilter1")) is None
    doc.get("iirfilter1").params["phase"] = "zero"
    assert doc.incompatibility(doc.get("iirfilter1")) == "Zero-phase needs offline"
    doc.input_mode = "offline"
    assert doc.incompatibility(zero) is None
    assert doc.incompatibility(doc.get("iirfilter1")) is None
    assert doc.incompatibility(out) == "Live only"
    assert any("lsl" in p for p in doc.problems())


def test_taps_follow_input_and_processors(doc: PipelineDoc) -> None:
    doc.insert(3, "meegxer.core.lsl_out")
    pipeline = doc.to_pipeline()
    ids = [n["id"] for n in pipeline["nodes"]]
    assert ids == [
        "input",
        tap_id("input"),
        "iirfilter1",
        tap_id("iirfilter1"),
        "notch1",
        tap_id("notch1"),
        "lslout1",
    ]
    saved = doc.to_pipeline(taps=False)
    assert all(n["node"] != "meegxer.core.probe" for n in saved["nodes"])
    again = PipelineDoc(list(doc.manifests.values()))
    again.load_pipeline(pipeline)
    assert [s.id for s in again.strips] == [s.id for s in doc.strips]


def test_strip_shows_crucial_or_picked_params(doc: PipelineDoc) -> None:
    hp = doc.get("iirfilter1")
    hp.params["kind"] = "highpass"
    assert doc.shown_params(hp, advanced=False) == ["kind", "low"]
    hp.strip_params = ["order", "phase"]
    assert doc.shown_params(hp, advanced=True) == ["order", "phase"]
    assert doc.shown_params(hp, advanced=False) == ["kind", "low"]
    assert doc.format_param(hp, "low") == "1 Hz"


def test_feature_strips_branch_off_untapped_and_build_a_graph(doc: PipelineDoc) -> None:
    from meegxer.core.graph import Graph

    psd = doc.insert(3, "meegxer.sterile.band_power")
    tsv = doc.insert(4, "meegxer.core.feature_tsv")
    tsv.params["path"] = "bands.tsv"
    reref = doc.insert(5, "meegxer.sterile.rereference")
    assert psd.id not in {s.id for s in doc.tappable()}
    graph = Graph.from_dict(doc.to_pipeline(), Registry.from_entry_points())
    inputs = {node.id: node.input for node in graph.nodes}
    assert inputs[tsv.id] == psd.id
    assert inputs[reref.id] == "notch1"  # the signal chain goes on
