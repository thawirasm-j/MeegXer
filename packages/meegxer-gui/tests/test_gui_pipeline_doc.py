from typing import Any

import pytest

from meegxer.core.control import manifest_to_dict
from meegxer.core.graph import Graph
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


def test_scripts_and_plugins_are_grouped_as_custom(doc: PipelineDoc) -> None:
    script = {**doc.manifest("meegxer.nodes.notch"), "name": "script.notch",
              "origin": "custom"}  # fmt: skip
    feature = {**doc.manifest("meegxer.nodes.band_power"), "name": "lab.bp",
               "origin": "plugin"}  # fmt: skip
    doc.set_manifests([*doc.manifests.values(), script, feature])
    assert doc.group_label("meegxer.nodes.notch") == "Preprocessing"
    assert doc.group_label("script.notch") == "Custom preprocessing"
    assert doc.group_label("lab.bp") == "Custom feature extraction"
    assert not doc.is_custom("meegxer.nodes.notch")


def test_the_add_strip_menu_is_grouped_by_category(doc: PipelineDoc) -> None:
    script = {**doc.manifest("meegxer.nodes.notch"), "name": "script.notch",
              "origin": "custom", "category": "filters"}  # fmt: skip
    plugin = {**doc.manifest("meegxer.nodes.band_power"), "name": "lab.bp",
              "origin": "plugin", "category": "my-measures"}  # fmt: skip
    loose = {**doc.manifest("meegxer.nodes.notch"), "name": "script.loose",
             "origin": "custom", "category": ""}  # fmt: skip
    doc.set_manifests([*doc.manifests.values(), script, plugin, loose])
    sections = dict(doc.palette_sections())
    assert list(sections) == ["Preprocessing", "Feature extraction", "Output"]
    pre = dict(sections["Preprocessing"])
    assert list(pre)[:3] == ["Filters", "Reference & channels", "Artifacts"]
    assert list(pre)[-1] == "Other"  # no category: last
    filters = pre["Filters"]
    assert filters[-1] == "script.notch"  # built-ins first, then the user's
    assert "meegxer.nodes.fir_filter" in filters
    assert pre["Other"] == ["script.loose"]
    features = dict(sections["Feature extraction"])
    assert list(features)[-1] == "My measures"  # unknown categories after known
    assert dict(sections["Output"])["Files"][0].startswith("meegxer.core.")
    assert doc.mark("script.notch") == "✎" and doc.mark("lab.bp") == "⬡"
    assert doc.mark("meegxer.nodes.notch") == ""


def test_a_node_can_process_more_than_eeg_by_default(doc: PipelineDoc) -> None:
    strip = doc.insert(len(doc.strips), "meegxer.nodes.eog_regression")
    assert doc.default_types(strip) == ["eeg", "eog"]
    assert doc.processed_types(strip) == ["eeg", "eog"]
    assert "channel_types" not in doc.to_pipeline(taps=False)["nodes"][-1]  # default
    strip.channel_types = ["eeg"]  # narrower than the node's default: saved
    assert doc.to_pipeline(taps=False)["nodes"][-1]["channel_types"] == ["eeg"]


def test_rails_start_at_feature_strips_one_lane_each(doc: PipelineDoc) -> None:
    """docs/RAILS.md: each feature strip starts a named rail on its own lane, the
    earliest highest, and feature and output strips sit below the signal."""
    psd = doc.insert(len(doc.strips), "meegxer.nodes.psd")
    tsv = doc.insert(len(doc.strips), "meegxer.core.feature_tsv")
    bp = doc.insert(len(doc.strips), "meegxer.nodes.band_power")
    bp2 = doc.insert(len(doc.strips), "meegxer.nodes.band_power")
    lsl = doc.insert(len(doc.strips), "meegxer.core.lsl_features_out")
    doc.set_input(lsl.id, bp.id)  # reads the first band power, past the second
    rails, needs = doc.rails()
    assert needs == {}
    by = {r.producer: r for r in rails}
    assert (by[psd.id].label, by[psd.id].readers, by[psd.id].lane) == (
        "PSD", [tsv.id], 2)  # fmt: skip
    assert by[bp.id].label == "Band power" and by[bp.id].readers == [lsl.id]
    assert by[bp.id].lane == 1
    assert by[bp2.id].label == "Band power·2" and by[bp2.id].readers == []
    assert by[bp2.id].lane == 0
    assert doc.rail_lanes() == 3
    assert [doc.lowered(s) for s in (doc.strips[1], psd, tsv, lsl)] == [
        False, True, True, True]  # fmt: skip
    assert doc.reads_signal(psd) and not doc.reads_signal(tsv)


def test_events_rail_comes_from_the_input_and_csp_needs_it(doc: PipelineDoc) -> None:
    csp = doc.insert(len(doc.strips), "meegxer.nodes.csp")
    rails, needs = doc.rails()
    assert needs == {csp.id: "Events"}
    assert doc.rail_lanes() == 2  # CSP's own features rail, a row for Events
    doc.input_events = 12
    rails, needs = doc.rails()
    events = next(r for r in rails if r.kind == "events")
    assert (events.producer, events.readers, needs) == ("input", [csp.id], {})


def test_a_live_lsl_input_with_markers_starts_the_events_rail(
    doc: PipelineDoc,
) -> None:
    doc.set_source("meegxer.core.lsl_in")
    csp = doc.insert(len(doc.strips), "meegxer.nodes.csp")
    assert doc.rails()[1] == {csp.id: "Events"}
    assert doc.source is not None
    doc.source.params["marker_stream"] = "PsychoPy markers"
    rails, needs = doc.rails()
    events = next(r for r in rails if r.kind == "events")
    assert (events.readers, needs) == ([csp.id], {})


def test_rail_ports_draw_their_rails_and_readers() -> None:
    """Spec 0.2.17: an extra output starts a rail named by its port; later
    strips reading that rail draw from it; one with no earlier writer needs it."""
    registry = Registry.from_entry_points()
    base = [manifest_to_dict(registry.get(n).manifest) for n in registry.names()]

    def node(name: str, inputs: list[Any], outputs: list[Any]) -> dict[str, Any]:
        return {"spec_version": "0.2.17", "name": name, "version": "0.1.0",
                "type": "stream", "role": "processor", "params": [],
                "inputs": [{"name": "in", "type": "signal"}, *inputs],
                "outputs": [{"name": "out", "type": "signal"}, *outputs]}  # fmt: skip

    ics = {"name": "ics", "type": "signal", "rail": "ics", "space": "component"}
    doc = PipelineDoc([*base, node("lab.ica", [], [ics]), node("lab.rej", [ics], [])])
    doc.set_source("meegxer.core.synthetic")
    early = doc.insert(1, "lab.rej")
    ica = doc.insert(2, "lab.ica")
    rej = doc.insert(3, "lab.rej")
    rails, needs = doc.rails()
    rail = next(r for r in rails if r.kind == "signal")
    assert (rail.id, rail.label, rail.producer, rail.readers) == (
        f"{ica.id}:ics", "ICs", ica.id, [rej.id])  # fmt: skip
    assert needs == {early.id: "ICs"}


def test_a_spectrum_can_read_the_ics_rail(doc: PipelineDoc) -> None:
    """Spec 0.2.18: a signal strip may take a rail output (the ICs) as its input;
    it is saved as ``input: "ica1:ics"``, drawn as a reader of the ICs rail, and
    no longer pulls from the signal line."""
    ica = doc.insert(len(doc.strips), "meegxer.nodes.ica")
    psd = doc.insert(len(doc.strips), "meegxer.nodes.psd")
    assert doc.rail_choices(psd) == [f"{ica.id}:ics"]
    assert doc.choice_label(f"{ica.id}:ics") == "ICs (ICA + ICLabel)"
    assert doc.input_of(psd) == ica.id and doc.reads_signal(psd)  # the default
    doc.set_input(psd.id, f"{ica.id}:ics")
    assert not doc.reads_signal(psd)
    rails, _ = doc.rails()
    ics = next(r for r in rails if r.id == f"{ica.id}:ics")
    assert (ics.label, ics.kind, ics.readers) == ("ICs", "signal", [psd.id])
    entry = next(e for e in doc.to_pipeline()["nodes"] if e["id"] == psd.id)
    assert entry["input"] == f"{ica.id}:ics"
    graph = Graph.from_dict(doc.to_pipeline(), Registry.from_entry_points())
    assert graph[psd.id].input_port == "ics"
    doc.remove(ica.id)
    assert doc.get(psd.id).input == ""


def test_a_chosen_input_is_saved_loaded_and_dropped_with_its_strip(
    doc: PipelineDoc,
) -> None:
    bp = doc.insert(len(doc.strips), "meegxer.nodes.band_power")
    psd = doc.insert(len(doc.strips), "meegxer.nodes.psd")
    tsv = doc.insert(len(doc.strips), "meegxer.core.feature_tsv")
    tsv.params["path"] = "features.tsv"
    assert doc.input_choices(tsv) == [psd.id, bp.id]
    assert doc.input_of(tsv) == psd.id  # nearest by default
    doc.set_input(tsv.id, bp.id)
    entry = next(e for e in doc.to_pipeline()["nodes"] if e["id"] == tsv.id)
    assert entry["input"] == bp.id
    graph = Graph.from_dict(doc.to_pipeline(), Registry.from_entry_points())
    assert {n.id: n.input for n in graph.nodes}[tsv.id] == bp.id
    other = PipelineDoc(list(doc.manifests.values()))
    other.load_pipeline(doc.to_pipeline())
    assert other.get(tsv.id).input == bp.id
    with pytest.raises(ValueError, match="cannot read"):
        doc.set_input(tsv.id, doc.strips[1].id)  # a signal strip
    doc.remove(bp.id)
    assert doc.get(tsv.id).input == "" and doc.input_of(tsv) == psd.id
