from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.conformance import ConformanceError, check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.node import RailInput
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, StageStatus, signal_fingerprint
from meegxer.plugins.adapter import PluginProcessor, plugin_class, probe_manifest
from meegxer.plugins.worker import LocalLauncher, PluginError

INFO = SignalInfo(("Fz", "Cz", "Pz"), 100.0)
BLOCK = Block(np.random.default_rng(1).standard_normal((3, 300)) * 2e-5, INFO, 50, 0.5)
FP = signal_fingerprint(INFO)


def make(plugin: Callable[..., LocalLauncher], name: str) -> type[PluginProcessor]:
    launcher = plugin(name)
    return plugin_class(probe_manifest(launcher), launcher)


def test_adapter_is_an_ordinary_processor(plugin: Callable[..., LocalLauncher]) -> None:
    cls = make(plugin, "ema")
    assert cls.manifest.name == "test.ema" and cls.manifest.params[0].name == "alpha"
    result = Stage(cls(), {"alpha": 0.5}).transform(BLOCK, FP)
    assert isinstance(result.block, Block)
    assert result.block.sample_index == 50 and result.block.info == INFO
    assert result.state["n"] == 350  # state comes back to the host
    assert all(isinstance(e, Event) for e in result.events)


def test_plugin_reads_and_writes_rails(plugin: Callable[..., LocalLauncher]) -> None:
    """Spec 0.2.17: rails travel in both directions, by port name."""
    processor = make(plugin, "railed")()
    try:
        assert [p.rail for p in processor.manifest.rail_outputs] == ["twice", "marks"]
        half = Block(BLOCK.data / 2, INFO, BLOCK.sample_index, BLOCK.timestamp)
        rails: dict[str, RailInput] = {
            "mix": half, "cues": (Event(60, "a"), Event(200, "b"))}  # fmt: skip
        result = Stage(processor).transform(BLOCK, FP, rails)
        assert isinstance(result.block, Block)
        np.testing.assert_allclose(result.block.data, BLOCK.data / 2, rtol=1e-6)
        twice = result.rails["twice"]
        assert isinstance(twice, Block) and twice.sample_index == 50
        np.testing.assert_allclose(twice.data, BLOCK.data * 2, rtol=1e-6)
        marks = result.rails["marks"]
        assert isinstance(marks, tuple)
        assert [w.window for w in marks] == [(60, 61), (200, 201)]
        check_blockwise_equivalence(processor, BLOCK, rails=rails, trials=3,
                                    rtol=1e-6, atol=1e-12)  # fmt: skip
    finally:
        processor.close()


def test_stateful_plugin_passes_acceptance_test_1(
    plugin: Callable[..., LocalLauncher],
) -> None:
    processor = make(plugin, "ema")()
    try:
        assert check_blockwise_equivalence(processor, BLOCK, trials=4) == 0.0
    finally:
        processor.close()


def test_forgetful_plugin_fails_acceptance_test_1(
    plugin: Callable[..., LocalLauncher],
) -> None:
    processor = make(plugin, "forgetful")()
    with pytest.raises(ConformanceError):
        check_blockwise_equivalence(processor, BLOCK)
    processor.close()


def test_fitted_plugin_models_are_loaded_once_and_after_restart(
    plugin: Callable[..., LocalLauncher],
) -> None:
    stage = Stage(make(plugin, "center")())
    assert stage.status(FP) is StageStatus.UNCALIBRATED
    stage.fit(BLOCK, FP)
    assert stage.status(FP) is StageStatus.READY
    out = stage.transform(BLOCK, FP).block
    assert isinstance(out, Block)
    np.testing.assert_allclose(out.data.mean(axis=1), 0, atol=1e-10)
    stage.processor.worker._kill()  # type: ignore[attr-defined]
    again = stage.transform(BLOCK, FP).block  # model re-sent to the new worker
    assert isinstance(again, Block)
    np.testing.assert_allclose(again.data, out.data)
    stage.processor.close()


def test_reserved_names_and_bad_manifests_rejected(
    plugin: Callable[..., LocalLauncher],
) -> None:
    launcher = plugin("reserved")
    with pytest.raises(PluginError, match="reserved"):
        plugin_class(probe_manifest(launcher), launcher)
    with pytest.raises(PluginError, match="processors"):
        sink = {
            "spec_version": "0.2.1", "name": "lab.sink", "version": "0.1.0",
            "type": "stream", "role": "sink", "outputs": [],
            "inputs": [{"name": "in", "type": "signal"}],
        }  # fmt: skip
        plugin_class(sink, launcher)
    with pytest.raises(PluginError, match="Signal input"):  # spec 0.2.23
        classifier = {
            "spec_version": "0.2.23", "name": "lab.classify", "version": "0.1.0",
            "type": "analysis", "inputs": [{"name": "in", "type": "features"}],
            "outputs": [{"name": "out", "type": "features"}],
        }  # fmt: skip
        plugin_class(classifier, launcher)


def test_plugins_run_in_both_schedulers(plugin: Callable[..., LocalLauncher]) -> None:
    cls = make(plugin, "ema")

    def graph(probe: MemoryProbe) -> Graph:
        src = SyntheticSource({"duration": 1.0, "n_channels": 3, "sfreq": 200.0})
        return Graph.chain([src, Stage(cls(), {"alpha": 0.2}), probe])

    batch_probe, live_probe = MemoryProbe(), MemoryProbe()
    batch = graph(batch_probe)
    BatchScheduler(block_size=37).run(batch)
    live = graph(live_probe)
    scheduler = RealtimeScheduler(live)
    scheduler.start()
    assert scheduler.wait(20)
    assert scheduler.status()["errors"] == []
    np.testing.assert_allclose(live_probe.signal().data, batch_probe.signal().data,
                               rtol=1e-6, atol=1e-12)  # fmt: skip
    for g in (batch, live):
        for node in g.nodes:
            if isinstance(node.impl, Stage):
                node.impl.processor.close()


def test_fitted_plugin_round_trips_through_a_session(
    plugin: Callable[..., LocalLauncher],
) -> None:
    """A plugin's model is host-owned state like any other: a session stores it
    as readable JSON, records where the plugin came from, and a reload uses the
    model without refitting."""
    from meegxer.core.registry import Registry
    from meegxer.core.session import (
        build_session,
        load_session,
        model_status,
    )
    from meegxer.plugins.index import origin_of

    launcher = plugin("center")
    entry = {"command": list(launcher.argv), "unsandboxed": True}
    cls = plugin_class(probe_manifest(launcher), launcher, origin=origin_of(entry))
    registry = Registry.from_entry_points()
    registry.add(cls)
    source = {"id": "input", "node": "meegxer.core.synthetic",
              "params": {"duration": 2.0, "n_channels": 3, "sfreq": 100.0}}  # fmt: skip
    pipeline = {
        "spec_version": "0.2.9",
        "nodes": [source, {"id": "c", "node": "test.center"}],
    }
    graph = Graph.from_dict(pipeline, registry)
    first = BatchScheduler().run(graph)
    session = build_session(graph)
    assert session["nodes"]["c"]["provenance"] == {
        "command": list(launcher.argv),
        "unsandboxed": True,
        "sha256": origin_of(entry)["sha256"],  # of the plugin's code file
    }
    assert session["models"]["c"]["data"]  # readable JSON of the plugin's model

    from meegxer.core.batch import offline_input_fingerprints

    loaded = load_session(session, registry)
    assert model_status(loaded.graph, offline_input_fingerprints(loaded.graph)) == {
        "c": "reused"
    }
    again = BatchScheduler().run(loaded.graph)
    assert again.fitted == [] and first.fitted == ["c"]

    # the code changed since the session was saved: refused, like another image
    from meegxer.core.session import SessionError

    script = Path(launcher.argv[-1])
    script.write_text(script.read_text() + "\n# edited\n")
    changed = plugin_class(probe_manifest(launcher), launcher, origin=origin_of(entry))
    other = Registry.from_entry_points()
    other.add(changed)
    with pytest.raises(SessionError, match="plugin code changed"):
        load_session(session, other)
    for g in (graph, loaded.graph):
        impl = g["c"].impl
        assert isinstance(impl, Stage)
        impl.processor.close()
    a, b = first.outputs["c"], again.outputs["c"]
    assert isinstance(a, Block) and isinstance(b, Block)
    np.testing.assert_array_equal(a.data, b.data)
