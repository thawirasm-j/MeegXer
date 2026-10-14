from collections.abc import Callable

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.conformance import ConformanceError, check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, StageStatus, signal_fingerprint
from meegxer.runtime.adapter import PluginProcessor, plugin_class, probe_manifest
from meegxer.runtime.worker import LocalLauncher, PluginError

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
