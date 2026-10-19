"""The Events rail (docs/RAILS.md): the input's events and those strips emit, seen
by labelled fits in both schedulers, live with a wait for late markers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event
from meegxer.core.events import INPUT, EventsRail
from meegxer.core.graph import Graph
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage


def manifest(name: str, **extra: object) -> Manifest:
    data: dict[str, object] = {
        "spec_version": "0.2.16", "name": f"test.{name}", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}], "params": [],
    }  # fmt: skip
    data.update(extra)
    return Manifest.from_dict(data)


class Labelled(Processor):
    """Remembers the events its fit was given; passes the signal on."""

    manifest: ClassVar[Manifest] = manifest(
        "labelled", requires_fit=True, fit_events=True
    )

    def fit(self, calibration: Block, params: Params) -> State:
        raise AssertionError("fits with events")

    def fit_labelled(self, calibration: Block, events: object, params: Params) -> State:
        assert isinstance(events, tuple)
        return {
            "onsets": np.array([e.sample_index for e in events], dtype=float),
            "labels": [e.label for e in events],
        }

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block, {})


class Detector(Processor):
    """Emits a 'beat' at every 100th sample."""

    manifest: ClassVar[Manifest] = manifest("detector")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        first = -(-block.sample_index // 100) * 100
        beats = tuple(Event(i, "beat") for i in range(first, block.stop_index, 100))
        return TransformResult(block, {}, beats)


def fitted_events(graph: Graph, node_id: str) -> list[tuple[int, str]]:
    stage = graph[node_id].impl
    assert isinstance(stage, Stage) and stage.model is not None
    model = stage.model.data
    onsets = np.asarray(model["onsets"]).astype(int).tolist()
    labels = model["labels"]
    assert isinstance(labels, list)
    return list(zip(onsets, map(str, labels), strict=True))


def test_rail_keeps_index_order_origins_and_trims() -> None:
    graph = Graph.chain([SyntheticSource({}), Stage(Detector()), Stage(Labelled())])
    rail = EventsRail()
    rail.add([Event(30, "b"), Event(10, "a")])
    rail.add([Event(20, "beat"), Event(10, "beat")], "detector1")
    rail.add([Event(15, "late")], "labelled1")
    assert [e.label for e in rail.events()] == ["a", "beat", "late", "beat", "b"]
    assert [e.label for e in rail.events(15, 30)] == ["late", "beat"]
    assert [e.label for e in rail.events(origins={INPUT})] == ["a", "b"]
    # a strip sees the input's events and its upstream strips', not its own
    seen = rail.visible_to(graph, "labelled1")
    assert [e.label for e in seen] == ["a", "beat", "beat", "b"]
    assert [e.label for e in rail.visible_to(graph, "detector1")] == ["a", "b"]
    rail.trim(20)
    assert [e.sample_index for e in rail.events()] == [20, 30]


def events_file(tmp_path: Path, onsets: list[float]) -> Path:
    path = tmp_path / "events.tsv"
    rows = "".join(f"{t}\t0\tcue\n" for t in onsets)
    path.write_text("onset\tduration\ttrial_type\n" + rows)
    return path


def synthetic(tmp_path: Path, seconds: float, onsets: list[float]) -> dict[str, Any]:
    return {
        "id": "src", "node": "meegxer.core.synthetic",
        "params": {"duration": seconds, "n_channels": 2, "sfreq": 100.0},
        "recording": {"events": {"file": str(events_file(tmp_path, onsets))}},
    }  # fmt: skip


def graph_of(nodes: list[dict[str, Any]], *impls: tuple[str, Processor]) -> Graph:
    """A chain from a pipeline's source dict plus test processors by id."""
    graph = Graph.from_dict({"spec_version": "0.2.16", "nodes": nodes},
                            Registry.from_entry_points())  # fmt: skip
    for node_id, impl in impls:
        graph.add(Stage(impl), id=node_id)
    return graph


def test_batch_fit_sees_the_recordings_and_upstream_strips_events(
    tmp_path: Path,
) -> None:
    graph = graph_of([synthetic(tmp_path, 3.0, [0.5, 2.5])],
                     ("before", Labelled()), ("detector", Detector()),
                     ("after", Labelled()))  # fmt: skip
    BatchScheduler().run(graph)
    assert fitted_events(graph, "before") == [(50, "cue"), (250, "cue")]
    beats = [(0, "beat"), (100, "beat"), (200, "beat")]
    assert fitted_events(graph, "after") == sorted([*beats, (50, "cue"), (250, "cue")])


def test_played_recording_events_calibrate_a_labelled_node_live(
    tmp_path: Path,
) -> None:
    """Live playback of a recording carries its events onto the rail."""
    graph = graph_of([synthetic(tmp_path, 1.5, [0.2, 0.7, 1.3])],
                     ("labelled", Labelled()))  # fmt: skip
    graph.add(MemoryProbe())
    scheduler = RealtimeScheduler(graph)
    scheduler.calibrate("labelled", 1.0)  # from the first sample
    scheduler.start()
    assert scheduler.wait(20)
    assert fitted_events(graph, "labelled") == [(20, "cue"), (70, "cue")]
    assert scheduler.status()["input_events"] == 3


class Markers(SyntheticSource):
    """A live source whose markers arrive late: each (index, label) is handed
    over only once the signal has reached ``arrives``."""

    live: ClassVar[bool] = True

    def __init__(self, markers: list[tuple[int, int, str]], wait: float) -> None:
        super().__init__({"duration": 40.0, "n_channels": 2, "sfreq": 200.0})
        self._markers = sorted(markers)
        self._wait = wait
        self._pos = 0

    @property
    def event_wait(self) -> float:
        return self._wait

    def read(self, max_samples: int) -> Block | None:
        time.sleep(0.02)  # 200 samples per 20 ms
        block = super().read(max_samples)
        if block is not None:
            self._pos = block.stop_index
        return block

    def new_events(self) -> tuple[Event, ...]:
        due = [m for m in self._markers if m[0] <= self._pos]
        self._markers = [m for m in self._markers if m[0] > self._pos]
        return tuple(Event(index, label) for _, index, label in due)


def test_live_labelled_fit_waits_for_late_markers() -> None:
    """The fit waits event_wait after its last sample: a marker arriving within
    that wait is used, one arriving after it is counted as late (missed)."""
    source = Markers(
        [(200, 50, "on time"), (400, 150, "in the wait"), (7000, 180, "too late")],
        wait=0.3,
    )
    graph = Graph.chain([source, Stage(Labelled()), MemoryProbe()])
    scheduler = RealtimeScheduler(graph)
    scheduler.calibrate("labelled1", 1.0)  # samples 0-200
    scheduler.start()
    assert scheduler.wait(20)
    assert fitted_events(graph, "labelled1") == [(50, "on time"), (150, "in the wait")]
    status = scheduler.status()
    assert (status["input_events"], status["events_late"]) == (3, 1)


def test_a_failed_labelled_fit_is_reported_and_the_run_goes_on() -> None:
    source = Markers([], wait=0.0)
    probe = MemoryProbe()
    graph = Graph.chain([source, Stage(Labelled()), probe])
    scheduler = RealtimeScheduler(graph)
    scheduler.calibrate("labelled1", 1.0)
    scheduler.start()
    assert scheduler.wait(20)
    errors = scheduler.status()["errors"]
    assert any("calibration of 'labelled1' failed" in e and "no events" in e
               for e in errors), errors  # fmt: skip
    stage = graph["labelled1"].impl
    assert isinstance(stage, Stage) and stage.model is None
    assert scheduler.node_status(graph["labelled1"]) == "uncalibrated"
    assert scheduler.metrics.samples == 8000  # the run went on to the end


def test_rail_maps_indices_for_a_strip_at_another_rate() -> None:
    graph = Graph.chain([SyntheticSource({}), Stage(Detector()), Stage(Labelled())])
    rail = EventsRail(100.0)  # counted on the input's clock
    rail.add([Event(50, "cue", duration=10), Event(51, "near")])
    rail.add([Event(10, "beat")], "detector1", 50.0)  # emitted at 50 Hz
    assert [(e.sample_index, e.label) for e in rail.events()] == [
        (20, "beat"), (50, "cue"), (51, "near")]  # fmt: skip
    seen = rail.visible_to(graph, "labelled1", sfreq=50.0)
    assert [(e.sample_index, e.duration) for e in seen] == [(10, 0), (25, 5), (26, 0)]
    # a window at 50 Hz holds exactly the events that map into it
    assert [e.label for e in rail.visible_to(graph, "labelled1", 25, 26, 50.0)] == [
        "cue"]  # fmt: skip
    assert rail.index(25, 50.0) == 50 and rail.index(25) == 25
