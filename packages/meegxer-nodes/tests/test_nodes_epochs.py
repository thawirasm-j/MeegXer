"""Epochs on a side rail: cut around events, baseline before the event, left out
over a BAD mark or a cut; the same epochs live, block by block."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import FeatureProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.registry import Registry
from meegxer.core.review import Mark, ReviewRequest
from meegxer.nodes.epochs import Epochs, cut_epochs

SFREQ = 100.0


def params(**kw: float | str) -> dict[str, float | str]:
    return {"events": "", "tmin": -0.2, "tmax": 0.5, "baseline": "pre-stimulus",
            "reject": "BAD", **kw}  # fmt: skip


def block(offset: float = 5.0) -> Block:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, 600)) * 1e-6 + offset * 1e-6
    return Block(data, SignalInfo(("Fz", "Cz"), SFREQ), 0, 0.0)


def test_epochs_are_cut_around_the_chosen_events() -> None:
    b = block()
    events = [Event(100, "go"), Event(200, "stop"), Event(300, "go"),
              Event(10, "go"),   # its start is before the recording
              Event(590, "go")]  # its end is after it  # fmt: skip
    got = cut_epochs(b, events, params(events="go"))
    assert [(w.window, w.coords["epoch"]) for w in got] == [
        ((80, 151), ("go",)), ((280, 351), ("go",))]  # fmt: skip
    first = got[0]
    assert first.dims == ("epoch", "channel", "time") and first.unit == "V"
    assert first.data.shape == (1, 2, 71)
    assert first.coords["time"][0] == pytest.approx(-0.2)
    assert first.coords["time"][20] == pytest.approx(0.0)
    # baseline: each channel's mean over tmin..0 (21 samples) is zero
    np.testing.assert_allclose(first.data[0, :, :21].mean(axis=1), 0.0, atol=1e-18)
    raw = cut_epochs(b, events, params(events="go", baseline="none"))[0]
    np.testing.assert_array_equal(raw.data[0], b.data[:, 80:151])
    every = cut_epochs(b, events, params())  # empty: every event
    assert [w.coords["epoch"][0] for w in every] == ["go", "stop", "go"]


def test_epochs_over_a_bad_mark_or_a_boundary_label_are_left_out() -> None:
    b = block()
    events = [Event(100, "go"), Event(300, "go"), Event(320, "BAD_muscle", 50),
              Event(400, "boundary", 20)]  # fmt: skip
    kept = cut_epochs(b, events, params())
    assert [w.window[0] for w in kept] == [80]  # 300's epoch overlaps the mark
    loose = cut_epochs(b, events, params(reject=""))  # no reject words
    # every epoch kept, the mark itself is an event too; never a boundary
    assert [w.coords["epoch"][0] for w in loose] == ["go", "go", "BAD_muscle"]


def test_wrong_spans_are_refused() -> None:
    with pytest.raises(ValueError, match="after tmin"):
        cut_epochs(block(), [Event(100, "go")], params(tmin=0.5, tmax=0.1))
    with pytest.raises(ValueError, match="tmin < 0"):
        cut_epochs(block(), [Event(100, "go")], params(tmin=0.0))


def test_the_same_epochs_whatever_the_block_sizes() -> None:
    """Acceptance test 1: an epoch across blocks is written once its last
    sample comes, and a BAD mark from an earlier block still leaves it out."""
    events = (Event(10, "go"), Event(100, "go"), Event(160, "stop"),
              Event(230, "BAD_eyes", 40), Event(300, "go"),  # under the mark
              Event(420, "go"), Event(590, "go"))  # fmt: skip
    for kw in (
        {},
        {"tmin": -0.5, "tmax": -0.1},  # an epoch before its event
        {"tmin": 0.1, "tmax": 1.2, "baseline": "none"},
        {"reject": ""},
    ):
        check_blockwise_equivalence(Epochs(), block(), params(**kw),
                                    rails={"events": events})  # fmt: skip


def graph_from(nodes: list[dict[str, Any]]) -> Graph:
    return Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes},
                           Registry.from_entry_points())  # fmt: skip


def test_live_epochs_match_the_batch_run(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n1.0\t0\tgo\n"
                      "2.5\t0\tstop\n4.0\t0\tgo\n")  # fmt: skip

    def nodes() -> list[dict[str, Any]]:
        return [
            {"id": "src", "node": "meegxer.core.synthetic",
             "params": {"duration": 5.0, "n_channels": 2, "sfreq": SFREQ},
             "recording": {"events": {"file": str(events)}}},
            {"id": "ep", "node": "meegxer.nodes.epochs",
             "params": {"tmin": -0.2, "tmax": 0.5}},
            {"id": "tap", "node": "meegxer.core.feature_probe",
             "input": "ep:epochs", "params": {"keep_windows": 0}},
        ]  # fmt: skip

    def windows(graph: Graph) -> list[FeatureBlock]:
        tap = graph["tap"].impl
        assert isinstance(tap, FeatureProbe)
        return [w for w in tap.windows if isinstance(w, FeatureBlock)]

    live = graph_from(nodes())
    scheduler = RealtimeScheduler(live, max_block=13)
    scheduler.start()
    assert scheduler.wait(20)
    assert scheduler.status()["errors"] == []
    batch = graph_from(nodes())
    BatchScheduler(block_size=64).run(batch)
    got, want = windows(live), windows(batch)
    assert [(w.window, w.coords["epoch"]) for w in got] == [
        ((80, 151), ("go",)), ((230, 301), ("stop",)),
        ((380, 451), ("go",))]  # fmt: skip
    assert [w.window for w in want] == [w.window for w in got]
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(a.data, b.data)


def test_a_batch_run_writes_the_epochs_rail_and_skips_epochs_over_a_cut(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.tsv"
    events.write_text("onset\tduration\ttrial_type\n"
                      "1.0\t0\tgo\n2.5\t0\tgo\n4.0\t0\tgo\n")  # fmt: skip
    nodes: list[dict[str, Any]] = [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 6.0, "n_channels": 2, "sfreq": SFREQ},
         "recording": {"events": {"file": str(events)}}},
        {"id": "rev", "node": "meegxer.core.review",
         "params": {"file": str(tmp_path / "marks.tsv")}},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
        {"id": "ep", "node": "meegxer.nodes.epochs",
         "params": {"events": "go", "tmin": -0.2, "tmax": 0.5}},
        {"id": "tap", "node": "meegxer.core.feature_probe", "input": "ep:epochs",
         "params": {"keep_windows": 0}},
    ]  # fmt: skip
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes},
                            Registry.from_entry_points())  # fmt: skip

    def review(req: ReviewRequest) -> list[Mark]:
        return [Mark(2.6, 0.3, "BAD_eyes")]  # cut out inside the second epoch

    BatchScheduler(block_size=37, review=review).run(graph)
    tap = graph["tap"].impl
    assert isinstance(tap, FeatureProbe)
    windows = [w for w in tap.windows if isinstance(w, FeatureBlock)]
    assert [w.window for w in windows] == [(80, 151), (380, 451)]


def test_the_offline_store_keeps_each_epochs_label() -> None:
    from meegxer.core.offline import FeatureStore

    windows = cut_epochs(block(), [Event(100, "go"), Event(300, "stop")], params())
    store = FeatureStore(windows[0])
    for w in windows:
        store.write(w)
    reply = store.windows(0, 600, 10, "none")
    assert reply["labels"] == ["go", "stop"]
    assert reply["dims"] == ["epoch", "channel", "time"]
