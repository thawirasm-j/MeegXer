"""The keyboard sink (rule 9: only host sinks act): armed by hand for one live
run, a fixed map of event labels to keys, rate-limited, stopped by Esc."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Event
from meegxer.core.graph import Graph, GraphError
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.nodes.actions import Keyboard, KeyboardSink, parse_keys


class FakeKeyboard:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.listening = False
        self.escape: Callable[[], None] | None = None

    def __call__(self, on_escape: Callable[[], None]) -> Keyboard:
        self.listening, self.escape = True, on_escape

        def stop() -> None:
            self.listening = False

        return self.keys.append, stop


@pytest.fixture
def keyboard(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeKeyboard]:
    fake = FakeKeyboard()
    monkeypatch.setattr(KeyboardSink, "keyboard", staticmethod(fake))
    yield fake


def test_the_key_map() -> None:
    assert parse_keys("pred_left=left, pred_right=Right, go=a") == {
        "pred_left": "left", "pred_right": "right", "go": "a"}  # fmt: skip
    for bad, why in (("pred_left", "label=key"), ("x=ctrl", "not a key"),
                     ("x=alt_l", "not a key"), ("", "at least one")):  # fmt: skip
        with pytest.raises(ValueError, match=why):
            parse_keys(bad)


def test_it_presses_only_when_armed_and_mapped(keyboard: FakeKeyboard) -> None:
    sink = KeyboardSink({"keys": "pred_left=left, pred_right=right"})
    events = [Event(10, "pred_left"), Event(20, "left"), Event(30, "pred_right")]
    sink.act(events)
    assert keyboard.keys == [] and not keyboard.listening  # off until armed
    sink.arm(True)
    assert keyboard.listening  # Esc is watched while armed
    for event in events:  # half a second apart: within max_rate (2/s)
        sink.act([event])
        sink._last -= 0.5
    assert keyboard.keys == ["left", "right"]  # the cue "left" is not mapped
    assert sink.action_status() == {"armed": True, "pressed": 2, "dropped": 0}
    assert keyboard.escape is not None
    keyboard.escape()  # Esc: disarmed, and the run is told
    assert not sink.armed and not keyboard.listening
    assert sink.notes == ["disarmed by Esc"]
    sink.act(events)
    assert keyboard.keys == ["left", "right"]
    sink.arm(True)
    sink.close()  # the run's end disarms it
    assert not sink.armed and not keyboard.listening


def test_at_most_max_rate_keys_per_second(keyboard: FakeKeyboard) -> None:
    sink = KeyboardSink({"keys": "a=a", "max_rate": 2.0})
    sink.arm(True)
    sink.act([Event(i, "a") for i in range(10)])  # all at once
    assert keyboard.keys == ["a"] and sink.dropped == 9
    time.sleep(0.55)  # half a second later: the next one goes
    sink.act([Event(20, "a")])
    assert keyboard.keys == ["a", "a"]


def test_no_desktop_no_arming(monkeypatch: pytest.MonkeyPatch) -> None:
    def headless(on_escape: Callable[[], None]) -> Keyboard:
        raise RuntimeError("no keyboard to press here (no display)")

    monkeypatch.setattr(KeyboardSink, "keyboard", staticmethod(headless))
    sink = KeyboardSink({})
    with pytest.raises(RuntimeError, match="no keyboard"):
        sink.arm(True)
    assert not sink.armed


class Marker(Processor):
    """Emits 'pred_left' at every 25th sample: a classifier's stand-in."""

    manifest = Manifest.from_dict({
        "spec_version": "0.2.23", "name": "test.marker", "version": "0.1.0",
        "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}], "params": [],
    })  # fmt: skip

    def transform(
        self, block: object, model: State, state: State | None, params: Params
    ) -> TransformResult:
        from meegxer.core.block import Block

        assert isinstance(block, Block)
        first = -(-block.sample_index // 25) * 25
        events = tuple(Event(i, "pred_left")
                       for i in range(first, block.stop_index, 25))  # fmt: skip
        return TransformResult(block, state or {}, events)


def chain(duration: float) -> tuple[Graph, KeyboardSink]:
    graph = Graph()
    graph.add(SyntheticSource({"duration": duration, "n_channels": 2,
                               "sfreq": 100.0}))  # fmt: skip
    graph.add(Stage(Marker()), id="marker")
    sink = KeyboardSink({"keys": "pred_left=space", "max_rate": 20.0})
    graph.add(sink, id="keys")
    return graph, sink


def test_live_it_acts_on_the_rail_once_armed(keyboard: FakeKeyboard) -> None:
    graph, _ = chain(3.0)
    scheduler = RealtimeScheduler(graph, max_block=10)
    with pytest.raises(GraphError, match="start the live run first"):
        scheduler.arm("keys", True)
    scheduler.start()
    time.sleep(0.5)
    assert keyboard.keys == []  # events flow, but nothing is pressed yet
    status = scheduler.arm("keys", True)
    assert status["armed"]
    assert scheduler.wait(10)
    pressed = len(keyboard.keys)
    assert pressed > 0 and set(keyboard.keys) == {"space"}
    node = next(n for n in scheduler.status()["nodes"] if n["id"] == "keys")
    assert node["pressed"] == pressed and not node["armed"]  # the end disarms


def test_a_batch_run_never_presses(keyboard: FakeKeyboard, tmp_path: Path) -> None:
    graph, sink = chain(1.0)
    sink.arm(True)  # even armed by mistake: a batch run gives it no events
    BatchScheduler().run(graph)
    assert keyboard.keys == []


def test_it_loads_from_a_pipeline_unarmed() -> None:
    spec = {"spec_version": "0.2.23", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic"},
        {"id": "keys", "node": "meegxer.nodes.keyboard",
         "params": {"keys": "pred_left=a"}},
    ]}  # fmt: skip
    graph = Graph.from_dict(spec, Registry.from_entry_points())
    sink = graph["keys"].impl
    assert isinstance(sink, KeyboardSink) and not sink.armed
    assert "armed" not in graph.to_dict()["nodes"][1]["params"]  # never saved
