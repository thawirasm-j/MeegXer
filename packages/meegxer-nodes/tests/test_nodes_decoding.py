"""CSP, fitted on labelled trials (manifest fit_events, spec 0.2.16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo
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


# -- LDA (spec 0.2.23: a processor reading a feature strip) ---------------------


def windows_of(
    n: int = 40, seed: int = 0, sfreq: float = FS
) -> tuple[Any, list[Event]]:
    """Feature windows of 2 values, 1 s long, one per trial of 'left' or
    'right' (4 s apart, starting 0.5 s after the cue), the classes apart in
    their mean; and the cues."""
    from meegxer.core.block import FeatureBlock, Windows

    rng = np.random.default_rng(seed)
    out, events = [], []
    for k in range(n):
        label = "left" if k % 2 == 0 else "right"
        onset = int(k * 4 * sfreq)
        events.append(Event(onset, label))
        start = onset + int(0.5 * sfreq)
        mean = (1.0, -1.0) if label == "left" else (-1.0, 1.0)
        values = np.array(mean) + rng.standard_normal(2)
        span = (start, start + int(sfreq))
        names = {"component": ("CSP1", "CSP2")}
        out.append(FeatureBlock(values, ("component",), span, start / sfreq, names))
    return Windows(tuple(out), sfreq, 0, int(n * 4 * sfreq)), events


@pytest.mark.parametrize("shrinkage", [True, False])
def test_lda_matches_scikit_learn(shrinkage: bool) -> None:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from meegxer.nodes.decoding import LDA

    windows, events = windows_of()
    stage = Stage(LDA(), {"shrinkage": shrinkage, "tmin": 0.0, "tmax": 2.0})
    fp = labelled_input("fp", events)
    stage.fit(windows, fp, events)
    model = stage.model.data  # type: ignore[union-attr]
    assert model["classes"] == ["left", "right"] and model["windows"] == [20, 20]
    x = np.stack([w.data for w in windows.windows])
    y = np.array([k % 2 for k in range(len(x))])
    ref = LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto" if shrinkage else None
    ).fit(x, y)
    result = stage.transform(windows, fp)
    out = result.block
    assert isinstance(out, tuple) and len(out) == len(x)
    np.testing.assert_allclose(np.stack([w.data for w in out]), ref.predict_proba(x),
                               rtol=1e-10)  # fmt: skip
    assert out[0].coords["class"] == ("left", "right") and out[0].unit == "1"
    assert [w.window for w in out] == [w.window for w in windows.windows]
    # a class event at each window's last sample, named apart from the cues
    predicted = [e.label for e in result.events]
    assert predicted == [f"pred_{('left', 'right')[k]}" for k in ref.predict(x)]
    assert result.events[0].sample_index == windows.windows[0].window[1] - 1
    assert sum(lb == f"pred_{e.label}" for lb, e in zip(predicted, events,
                                                       strict=True)) >= 36  # fmt: skip


def test_lda_least_probability_and_prefix_need_no_refit() -> None:
    from meegxer.nodes.decoding import LDA

    windows, events = windows_of()
    stage = Stage(LDA(), {"tmin": 0.0, "tmax": 2.0})
    fp = labelled_input("fp", events)
    stage.fit(windows, fp, events)
    stage.set_params(min_probability=0.999, event_prefix="key_")
    assert stage.status(fp).value == "ready"
    result = stage.transform(windows, fp)
    assert 0 < len(result.events) < len(windows.windows)
    assert all(e.label.startswith("key_") for e in result.events)
    assert all(float(e.payload["probability"]) >= 0.999  # type: ignore[arg-type]
               for e in result.events)  # fmt: skip


def test_lda_explains_what_it_cannot_fit() -> None:
    from meegxer.core.block import Windows
    from meegxer.core.stage import StaleModelError
    from meegxer.nodes.decoding import LDA

    windows, events = windows_of(n=6)
    fp = labelled_input("fp", events)
    with pytest.raises(ValueError, match="no events"):
        Stage(LDA(), {}).fit(windows, "fp")
    # default trial 0.5-4 s: the 1 s windows from 0.5 s lie in it
    Stage(LDA(), {}).fit(windows, fp, events)
    with pytest.raises(ValueError, match=r"too few of \['left', 'right'\].*shorter"):
        Stage(LDA(), {"tmin": 1.0}).fit(windows, fp, events)
    with pytest.raises(ValueError, match="no feature windows"):
        Stage(LDA(), {}).fit(Windows((), FS, 0, 100), fp, events)
    stage = Stage(LDA(), {})
    stage.fit(windows, fp, events)
    other = Windows(windows.windows, 2 * FS, 0, windows.stop_index)
    with pytest.raises(StaleModelError, match="differ from the calibration"):
        stage.transform(other, fp)


def mi_raw(n_trials: int = 30, seed: int = 0) -> Any:
    import mne

    block, events = imagery(n_trials, seed)
    raw = mne.io.RawArray(block.data, mne.create_info(list(LABELS), FS, "eeg"),
                          verbose="error")  # fmt: skip
    raw.set_annotations(mne.Annotations([e.sample_index / FS for e in events],
                                        [0.0] * len(events),
                                        [e.label for e in events]))  # fmt: skip
    return raw


def csp_lda(raw: Any) -> Graph:
    from meegxer.core.memory import RawInput
    from meegxer.nodes.decoding import LDA

    graph = Graph()
    graph.add(RawInput(raw), id="src")
    graph.add(Stage(CSP(), {"n_components": 2, "window": 1.0, "step": 0.5,
                            "tmin": 0.5, "tmax": 3.0}), id="csp")  # fmt: skip
    graph.add(Stage(LDA(), {"tmin": 0.5, "tmax": 3.0}), id="lda")
    return graph


def test_csp_then_lda_in_a_batch_run() -> None:
    """The LDA strip reads the CSP strip (its default input: the nearest
    feature strip) and puts its class events on the Events rail."""
    from meegxer.core.memory import RawOutput

    graph = csp_lda(mi_raw())
    assert graph["lda"].input == "csp"
    out_sink = RawOutput()
    graph.add(out_sink, id="out", input="src")  # an output reading the signal
    scheduler = BatchScheduler()
    result = scheduler.run(graph)
    assert result.fitted == ["csp", "lda"]
    out = result.outputs["lda"]
    assert isinstance(out, list) and out
    assert isinstance(out[0], FeatureBlock) and out[0].dims == ("class",)
    stage = graph["lda"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    assert stage.model.input_info.ch_labels == ("CSP1", "CSP2")
    # windows wholly in a trial: classified as its cue, nearly always
    predicted = result.events["lda"]
    assert len(predicted) == len(out)
    hits = total = 0
    for e in predicted:
        onset = (e.sample_index // 400) * 400  # trials every 4 s
        if 50 + 100 <= e.sample_index - onset + 1 <= 300:  # its window: 0.5-3 s
            total += 1
            hits += e.label == "pred_" + ("left" if onset // 400 % 2 == 0 else "right")
    assert total >= 30 and hits / total > 0.9, (hits, total)
    # an output to its right gets the class events (the Events rail as drawn);
    # a processor sees its ancestors' only (its fingerprint names them, rule 7)
    labels = set(out_sink.raw.annotations.description)
    assert {"pred_left", "pred_right", "left", "right"} <= labels
    assert not any(e.label.startswith("pred_")
                   for e in scheduler.rail_events("csp"))  # fmt: skip
    # a change upstream of CSP marks both for a refit (rule 7)
    before = offline_input_fingerprints(graph)["lda"]
    graph["csp"].impl.set_params(n_components=3)  # type: ignore[union-attr]
    after = offline_input_fingerprints(graph)["lda"]
    assert after != before and stage.status(after).value == "needs-refit"


def test_lda_calibrated_live_after_csp() -> None:
    """Live: CSP calibrates on the first stretch, then LDA on the windows CSP
    gives after it; each window's class comes as it ends, and is what the
    same model gives on the same windows offline."""
    from meegxer.core.block import Windows
    from meegxer.core.probes import FeatureProbe
    from meegxer.core.realtime import RealtimeScheduler

    graph = csp_lda(mi_raw(n_trials=16))
    taps = {name: FeatureProbe({"keep_windows": 0}) for name in ("csp", "lda")}
    for name, tap in taps.items():
        graph.add(tap, input=name)
    import time

    # the recording plays ten times faster than real time
    scheduler = RealtimeScheduler(graph, max_block=37,
                                  clock=lambda: 10 * time.perf_counter())  # fmt: skip
    scheduler.calibrate("csp", 20.0)
    scheduler.calibrate("lda", 20.0)
    scheduler.start()
    assert scheduler.wait(60)
    status = scheduler.status()
    assert status["errors"] == []
    stage = graph["lda"].impl
    assert isinstance(stage, Stage) and stage.model is not None
    got = taps["lda"].windows
    # LDA's calibration starts with CSP's first windows, once CSP is fitted
    assert got and got[0].window[1] > 40 * FS
    seen = {w.window: w for w in taps["csp"].windows}
    again = stage.processor.transform_features(
        Windows(tuple(seen[w.window] for w in got), FS, 0, 64 * 100),
        stage.model.data,
        None,
        stage.params,
    )
    assert isinstance(again.block, tuple)
    np.testing.assert_allclose([w.data for w in again.block],
                               [w.data for w in got])  # fmt: skip
    assert len(again.events) == len(got)
