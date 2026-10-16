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
    d.insert(1, "meegxer.nodes.iir_filter")
    d.insert(2, "meegxer.nodes.notch")
    return d


def test_chain_edits_keep_the_input_first(doc: PipelineDoc) -> None:
    assert [s.id for s in doc.strips] == ["input", "iirfilter1", "notch1"]
    doc.move("notch1", 1)
    assert [s.id for s in doc.strips] == ["input", "notch1", "iirfilter1"]
    doc.move("notch1", 3)
    assert [s.id for s in doc.strips] == ["input", "iirfilter1", "notch1"]
    doc.insert(0, "meegxer.nodes.notch")  # cannot go before the input
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
    zero = doc.insert(3, "meegxer.nodes.iir_filter_zerophase")
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


@pytest.mark.parametrize("data", [[1, 2], "x", {"nodes": "abc"}, {"nodes": [1]}])
def test_opening_a_json_file_that_is_not_a_pipeline_is_refused(
    doc: PipelineDoc, data: object
) -> None:
    before = [s.id for s in doc.strips]
    with pytest.raises(ValueError, match="not a pipeline"):
        doc.load_pipeline(data)  # type: ignore[arg-type]
    assert [s.id for s in doc.strips] == before


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

    psd = doc.insert(3, "meegxer.nodes.band_power")
    tsv = doc.insert(4, "meegxer.core.feature_tsv")
    tsv.params["path"] = "bands.tsv"
    reref = doc.insert(5, "meegxer.nodes.rereference")
    assert psd.id not in {s.id for s in doc.tappable()}
    graph = Graph.from_dict(doc.to_pipeline(), Registry.from_entry_points())
    inputs = {node.id: node.input for node in graph.nodes}
    assert inputs[tsv.id] == psd.id
    assert inputs[reref.id] == "notch1"  # the signal chain goes on


def test_synthetic_input_runs_forever_live_and_needs_a_duration_offline(
    doc: PipelineDoc,
) -> None:
    assert doc.source is not None
    assert doc.source.params["duration"] == 0.0  # the GUI's default: endless
    assert doc.problems() == []
    doc.input_mode = "offline"
    assert any("runs forever" in p for p in doc.problems())
    doc.source.params["duration"] = 30.0
    assert doc.problems() == []
    # an opened pipeline keeps the node's own default (60 s) when it omits it
    doc.load_pipeline({"nodes": [{"id": "input", "node": "meegxer.core.synthetic"}]})
    assert doc.source.params["duration"] == 60.0


def test_strips_that_need_positions_list_the_channels_without_them(
    doc: PipelineDoc,
) -> None:
    channels = [
        {"label": "Fp1", "type": "eeg", "bad": False, "position": None},
        {"label": "Cz", "type": "eeg", "bad": False, "position": [0, 0, 0.1]},
        {"label": "O1", "type": "eeg", "bad": True, "position": None},
        {"label": "VEOG", "type": "eog", "bad": False, "position": None},
    ]
    assert doc.missing_positions(channels) == {}  # no strip needs them yet
    doc.insert(3, "meegxer.nodes.ica")
    assert doc.missing_positions(channels) == {"ica1": ["Fp1"]}
    doc.get("ica1").channel_types = ["eeg", "eog"]
    assert doc.missing_positions(channels) == {"ica1": ["Fp1", "VEOG"]}


def test_groups_and_what_each_strip_reads_from(doc: PipelineDoc) -> None:
    """Feature extraction inserted between re-reference and ICA forks off after
    re-reference; ICA still reads from re-reference (one lane, forks drawn)."""
    for node in ["meegxer.nodes.rereference", "meegxer.nodes.band_power",
                 "meegxer.core.lsl_features_out", "meegxer.nodes.ica",
                 "meegxer.core.lsl_out"]:  # fmt: skip
        doc.insert(len(doc.strips), node)
    groups = {s.id: doc.group(s) for s in doc.strips}
    assert groups == {
        "input": "input", "iirfilter1": "preprocess", "notch1": "preprocess",
        "rereferenc1": "preprocess", "bandpower1": "features",
        "lslfeature1": "output", "ica1": "preprocess", "lslout1": "output",
    }  # fmt: skip
    main = [s.id for s in doc.strips if doc.on_main_line(s)]
    assert main == ["input", "iirfilter1", "notch1", "rereferenc1", "ica1"]
    reads = {s.id: doc.input_of(s) for s in doc.strips}
    assert reads["bandpower1"] == "rereferenc1"
    assert reads["lslfeature1"] == "bandpower1"
    assert reads["ica1"] == "rereferenc1"  # the main line continues
    assert reads["lslout1"] == "ica1"
    assert reads["input"] is None
    # and the pipeline the core builds agrees
    from meegxer.core.graph import Graph

    graph = Graph.from_dict(doc.to_pipeline(taps=False), Registry.from_entry_points())
    for strip in doc.strips[1:]:
        assert graph[strip.id].input == reads[strip.id], strip.id


def test_tap_delays_add_up_along_each_tap_path(doc: PipelineDoc) -> None:
    """Live legend estimate: computing time of the strips before a tap plus any
    declared built-in delay, relative to the input tap."""
    timing = {"input": 0.3, "iirfilter1": 0.5, "notch1": 0.25}
    delays = doc.tap_delays(timing, 500.0)
    assert delays["input"] == 0.0
    assert delays["iirfilter1"] == pytest.approx(0.5)
    assert delays["notch1"] == pytest.approx(0.75)
    doc.manifest(doc.get("iirfilter1"))["latency_samples"] = 50  # look-ahead
    assert doc.tap_delays(timing, 500.0)["notch1"] == pytest.approx(100.75)


def test_a_pipeline_saved_with_old_node_names_loads(doc: PipelineDoc) -> None:
    saved = doc.to_pipeline(taps=False)
    for entry in saved["nodes"]:
        entry["node"] = entry["node"].replace("meegxer.nodes.", "meegxer.sterile.")
    doc.load_pipeline(saved)
    assert [s.node for s in doc.strips][1:] == [
        "meegxer.nodes.iir_filter",
        "meegxer.nodes.notch",
    ]
