"""CSP, fitted on labelled trials (manifest fit_events, spec 0.2.16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.graph import Graph
from meegxer.core.registry import Registry
from meegxer.core.stage import Stage, labelled_input
from meegxer.nodes.decoding import CSP

FS = 100.0
LABELS = tuple(f"E{i}" for i in range(6))


def imagery(n_trials: int = 20, seed: int = 0) -> tuple[Block, list[Event]]:
    """Alternating 'left' and 'right' trials 4 s apart: a 10 Hz rhythm that is
    strong on E1 and weak on E4 during 'left', the other way round during
    'right', over background noise on every channel."""
    rng = np.random.default_rng(seed)
    n = int(n_trials * 4 * FS)
    x = rng.standard_normal((len(LABELS), n))
    t = np.arange(n) / FS
    events = []
    for k in range(n_trials):
        label = "left" if k % 2 == 0 else "right"
        onset = int(k * 4 * FS)
        events.append(Event(onset, label))
        span = slice(onset + int(0.5 * FS), onset + int(3.0 * FS))
        rhythm = np.sin(2 * np.pi * 10 * t[span] + rng.uniform(0, 2 * np.pi))
        strong, weak = (1, 4) if label == "left" else (4, 1)
        x[strong, span] += 3 * rhythm
        x[weak, span] += 0.3 * rhythm
    return Block(1e-6 * x, SignalInfo(LABELS, FS, "mi"), 0, 0.0), events


def fitted(block: Block, events: list[Event], **params: Any) -> Stage:
    stage = Stage(CSP(), params)
    stage.fit(block, labelled_input("fp", events), events)
    return stage


def test_csp_matches_mne_on_the_same_trials() -> None:
    from mne.decoding import CSP as MNECSP

    block, events = imagery()
    stage = fitted(block, events, n_components=2)
    starts = [e.sample_index + 50 for e in events]
    epochs = np.stack([block.data[:, s : s + 200] for s in starts])
    y = np.array([0 if e.label == "left" else 1 for e in events])
    ref = MNECSP(n_components=2, log=True).fit(epochs, y)
    model = stage.model.data  # type: ignore[union-attr]
    assert model["classes"] == ["left", "right"] and model["trials"] == [10, 10]
    filters = np.asarray(model["filters"])
    np.testing.assert_allclose(np.abs(filters), np.abs(ref.filters_[:2]), rtol=1e-8)
    # windows of 2 s every 0.5 s: those on a trial give MNE's features for it
    out = stage.transform(block, labelled_input("fp", events)).block
    assert isinstance(out, tuple)
    by_start = {w.window[0]: w for w in out}
    for s, feats in zip(starts, ref.transform(epochs), strict=True):
        np.testing.assert_allclose(by_start[s].data, feats, rtol=1e-8)
    assert out[0].coords["component"] == ("CSP1", "CSP2")


def test_csp_features_tell_the_classes_apart() -> None:
    block, events = imagery(seed=1)
    stage = fitted(block, events, window=2.5, step=4.0, tmin=0.5, tmax=3.0)
    test_block, test_events = imagery(seed=2)  # new trials, same model
    shifted = Block(test_block.data, test_block.info, 0, 0.0)
    stage.state = None
    out = stage.transform(shifted, labelled_input("fp", events)).block
    assert isinstance(out, tuple)
    # windows start at 0, 4, 8 s ... (step 4 s): one per trial, offset 0.5 s
    first = np.array([w.data[0] for w in out])
    labels = [test_events[i].label for i in range(len(first))]
    left = first[[lb == "left" for lb in labels]]
    right = first[[lb == "right" for lb in labels]]
    assert abs(left.mean() - right.mean()) > 3 * max(left.std(), right.std())


def test_csp_explains_missing_trials() -> None:
    block, events = imagery(n_trials=6)
    with pytest.raises(ValueError, match="no events"):
        Stage(CSP(), {}).fit(block, "fp")
    with pytest.raises(ValueError, match=r"too few of \['up'\].*'left', 'right'"):
        fitted(block, events, classes="left, up")
    with pytest.raises(ValueError, match="two different classes"):
        fitted(block, events, classes="left")
    with pytest.raises(ValueError, match="only 6 channels"):
        fitted(block, events, n_components=8)


def test_csp_runs_block_by_block() -> None:
    """Acceptance test 1, fitted on the labelled calibration."""
    block, events = imagery(n_trials=8)
    worst = check_blockwise_equivalence(
        CSP(), block.slice(0, 1600), {}, calibration=block, events=events, trials=2
    )
    assert worst < 1e-12


def test_a_batch_run_fits_on_the_recordings_events(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    rows = [f"{4.0 * k + 1.0}\t0\t{'left' if k % 2 == 0 else 'right'}"
            for k in range(6)]  # fmt: skip
    events.write_text("onset\tduration\ttrial_type\n" + "\n".join(rows) + "\n")
    spec = {"spec_version": "0.2.16", "nodes": [
        {"id": "src", "node": "meegxer.core.synthetic",
         "params": {"duration": 30.0, "n_channels": 6, "sfreq": 100.0},
         "recording": {"events": {"file": str(events)}}},
        {"id": "csp", "node": "meegxer.nodes.csp", "params": {"n_components": 2}},
    ]}  # fmt: skip
    graph = Graph.from_dict(spec, Registry.from_entry_points())
    result = BatchScheduler().run(graph)
    assert result.fitted == ["csp"]
    stage = graph["csp"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    assert stage.model.data["trials"] == [3, 3]
    before = offline_input_fingerprints(graph)["csp"]
    # other labels: a different input, so the model no longer fits it
    events.write_text("onset\tduration\ttrial_type\n" + "\n".join(
        r.replace("left", "rest") for r in rows) + "\n")  # fmt: skip
    after = offline_input_fingerprints(graph)["csp"]
    assert after != before
    assert stage.status(after).value == "needs-refit"


def test_live_calibration_from_a_played_recording_matches_batch(
    tmp_path: Path,
) -> None:
    """Rule 4 for labelled fits: CSP calibrated live on the first 5 s of a played
    recording, from the events on the Events rail, is the model a batch run fits
    on the same 5 s."""
    from meegxer.core.probes import MemoryProbe
    from meegxer.core.realtime import RealtimeScheduler

    events = tmp_path / "events.tsv"
    rows = [f"{0.5 * k}\t0\t{'left' if k % 2 else 'right'}" for k in range(1, 12)]
    events.write_text("onset\tduration\ttrial_type\n" + "\n".join(rows) + "\n")

    def graph() -> Graph:
        spec = {"spec_version": "0.2.16", "nodes": [
            {"id": "src", "node": "meegxer.core.synthetic",
             "params": {"duration": 6.0, "n_channels": 6, "sfreq": 100.0},
             "recording": {"events": {"file": str(events)}}},
            {"id": "csp", "node": "meegxer.nodes.csp",
             "params": {"n_components": 2, "tmin": 0.0, "tmax": 0.4}},
        ]}  # fmt: skip
        return Graph.from_dict(spec, Registry.from_entry_points())

    offline = graph()
    BatchScheduler(calibration_seconds=5.0).run(offline)
    live = graph()
    live.add(MemoryProbe())
    scheduler = RealtimeScheduler(live)
    scheduler.calibrate("csp", 5.0)  # from the first sample
    scheduler.start()
    assert scheduler.wait(30)
    assert scheduler.status()["errors"] == []
    models = []
    for g in (offline, live):
        stage = g["csp"].impl
        assert isinstance(stage, Stage) and stage.model is not None
        models.append(stage.model.data)
    assert models[0]["trials"] == models[1]["trials"] == [5, 4]
    np.testing.assert_allclose(
        np.asarray(models[1]["filters"]), np.asarray(models[0]["filters"])
    )
