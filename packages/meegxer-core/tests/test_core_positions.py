"""Nodes that need electrode positions (manifest requires_positions, spec 0.2.11)
refuse to run before reading or writing anything when the input lacks them."""

import time
from typing import Any

import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import SignalInfo
from meegxer.core.control import manifest_to_dict
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.node import Source
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry


def pipeline(recording: dict[str, Any] | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {
        "id": "input", "node": "meegxer.core.synthetic",
        "params": {"duration": 10.0, "n_channels": 8, "sfreq": 250.0},
    }  # fmt: skip
    if recording:
        source["recording"] = recording
    return {"spec_version": SPEC_VERSION, "nodes": [
        source,
        {"id": "ica", "node": "meegxer.nodes.ica", "params": {"max_iter": 50}},
    ]}  # fmt: skip


def graph(recording: dict[str, Any] | None = None) -> Graph:
    return Graph.from_dict(pipeline(recording), Registry.from_entry_points())


def info_of(g: Graph) -> SignalInfo:
    source = g.source.impl
    assert isinstance(source, Source)
    return source.info


def test_ica_declares_it_needs_positions() -> None:
    manifest = Registry.from_entry_points().get("meegxer.nodes.ica").manifest
    assert manifest.requires_positions
    assert manifest_to_dict(manifest)["requires_positions"] is True
    assert manifest_to_dict(manifest)["latency_samples"] == manifest.latency_samples
    other = Registry.from_entry_points().get("meegxer.nodes.notch").manifest
    assert not other.requires_positions


def test_missing_positions_counts_good_processed_channels_only() -> None:
    g = graph()
    info = info_of(g)
    missing = g.missing_positions(info)
    assert list(missing) == ["ica"] and len(missing["ica"]) == 8
    with pytest.raises(GraphError, match="ica needs electrode positions for Fp1"):
        g.check_positions(info)
    # a montage gives every channel a position
    placed = graph({"montage": "colin27_1020"})
    assert placed.missing_positions(info_of(placed)) == {}
    # bad channels and channel types the strip does not process do not count
    bad = graph({"channels": {"Fp1": {"bad": True}, "Fp2": {"type": "eog"}}})
    assert "Fp1" not in bad.missing_positions(info_of(bad))["ica"]
    assert "Fp2" not in bad.missing_positions(info_of(bad))["ica"]


def test_batch_refuses_before_running() -> None:
    with pytest.raises(GraphError, match="needs electrode positions"):
        BatchScheduler().run(graph())


def test_live_run_refuses_once_the_input_is_open() -> None:
    scheduler = RealtimeScheduler(graph())
    scheduler.start()
    assert scheduler.wait(10)
    status = scheduler.status()
    assert not status["running"]
    assert len(status["errors"]) == 1
    assert status["errors"][0].startswith("ica needs electrode positions for Fp1")
    assert scheduler.metrics.blocks == 0

    ok = RealtimeScheduler(graph({"montage": "colin27_1020"}))
    ok.start()
    time.sleep(0.5)
    assert ok.status()["running"] and not ok.status()["errors"]
    ok.stop()
