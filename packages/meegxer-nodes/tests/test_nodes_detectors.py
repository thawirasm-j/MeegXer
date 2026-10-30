"""Detector strips (docs/REVIEW.md step 4): amplitude and flat signal, muscle,
blinks; each wraps an MNE function and emits events."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.registry import Registry
from meegxer.core.review import BAD_CHANNEL
from meegxer.core.stage import Stage, signal_fingerprint
from meegxer.nodes.detectors import (
    AmplitudeDetector,
    BlinkDetector,
    MuscleDetector,
)

EEG = ("Fz", "Cz", "Pz", "Oz")
RNG = np.random.default_rng(7)


def noise(n_ch: int, seconds: float, sfreq: float) -> np.ndarray:
    return RNG.normal(0.0, 5e-6, (n_ch, round(seconds * sfreq)))


def detect(proc: object, b: Block, **params: Any) -> tuple[Block, tuple[Event, ...]]:
    stage = Stage(proc, params).for_mode("offline")  # type: ignore[arg-type]
    result = stage.transform(b, signal_fingerprint(b.info))
    assert isinstance(result.block, Block)
    return result.block, tuple(result.events)


def spans(events: tuple[Event, ...], label: str) -> list[tuple[int, int]]:
    return [(e.sample_index, e.sample_index + e.duration) for e in events
            if e.label == label]  # fmt: skip


def test_detectors_are_registered_under_event_detection() -> None:
    registry = Registry.from_entry_points()
    for name in ("meegxer.nodes.detect_amplitude", "meegxer.nodes.detect_muscle",
                 "meegxer.nodes.detect_blinks"):  # fmt: skip
        manifest = registry.get(name).manifest
        assert (manifest.category, manifest.availability) == ("events", "offline")


# -- amplitude and flat ---------------------------------------------------------------


def test_amplitude_marks_jumps_and_flat_spans_and_proposes_bad_channels() -> None:
    sfreq = 250.0
    data = noise(4, 10.0, sfreq)
    data[0, 550] += 300e-6  # a 300 µV spike on Fz at 2.2 s
    data[:3, 1500:1550] = 0.0  # flat, 6-6.2 s (2%: spans, not a bad channel)
    data[3] = 0.0  # Oz flat throughout: proposed as bad, no spans
    info = SignalInfo(EEG, sfreq)
    b = Block(data, info, 1000, 4.0)  # offset: events land on the block's samples
    out, events = detect(AmplitudeDetector(), b)
    assert out is b  # the signal passes on unchanged
    (peak,) = spans(events, "BAD_peak")
    assert peak[0] <= 1000 + 550 < peak[1] <= 1000 + 553
    (flat,) = spans(events, "BAD_flat")
    assert abs(flat[0] - 2500) <= 2 and abs(flat[1] - 2550) <= 2
    proposed = [e for e in events if e.label == BAD_CHANNEL]
    assert [(e.sample_index, e.duration, dict(e.payload)) for e in proposed] == [
        (1000, 0, {"channel": "Oz", "reason": "flat"})]  # fmt: skip
    assert events[0].label == BAD_CHANNEL  # proposals first, then spans by onset


def test_amplitude_leaves_out_bad_channels_and_needs_a_limit() -> None:
    data = noise(4, 4.0, 250.0)
    data[3, 200] += 500e-6
    b = Block(data, SignalInfo(EEG, 250.0, bads=("Oz",)), 0, 0.0)
    assert detect(AmplitudeDetector(), b)[1] == ()
    with pytest.raises(ValueError, match="peak or a flat limit"):
        detect(AmplitudeDetector(), b, peak=0.0, flat=0.0)


# -- muscle ---------------------------------------------------------------------------


def test_muscle_marks_a_high_frequency_burst() -> None:
    sfreq = 500.0
    data = noise(4, 20.0, sfreq)
    t = np.arange(500) / sfreq
    data[:, 5000:5500] += 40e-6 * np.sin(2 * np.pi * 125 * t)  # 10-11 s
    _, events = detect(MuscleDetector(), Block(data, SignalInfo(EEG, sfreq), 0, 0.0))
    found = spans(events, "BAD_muscle")
    assert found and all(lo >= 4900 and hi <= 5600 for lo, hi in found)
    assert sum(hi - lo for lo, hi in found) > 300


def test_muscle_band_must_fit_below_nyquist() -> None:
    b = Block(noise(4, 4.0, 250.0), SignalInfo(EEG, 250.0), 0, 0.0)
    with pytest.raises(ValueError, match="below half the sampling rate"):
        detect(MuscleDetector(), b)
    detect(MuscleDetector(), b, low=70.0, high=110.0)  # fits at 250 Hz


# -- blinks ---------------------------------------------------------------------------


def blink_recording(sfreq: float, peaks_s: list[float]) -> Block:
    data = noise(5, 30.0, sfreq)
    half = round(0.15 * sfreq)
    bump = 200e-6 * np.hanning(2 * half + 1)
    for p in peaks_s:
        i = round(p * sfreq)
        data[4, i - half : i + half + 1] += bump
    info = SignalInfo((*EEG, "VEOG"), sfreq, ch_types=("eeg",) * 4 + ("eog",))
    return Block(data, info, 0, 0.0)


def test_blinks_become_short_spans_around_each_peak() -> None:
    sfreq = 250.0
    peaks = [3.0, 8.0, 14.0, 21.0, 26.0]
    b = blink_recording(sfreq, peaks)
    # MNE's threshold is a peak's prominence; its default (a quarter of the
    # filtered range) also takes the filter's ringing around a lone synthetic bump
    out, events = detect(BlinkDetector(), b, blink_threshold=80.0)
    assert out is b
    found = [e for e in events if e.label == "blink"]
    assert len(found) == len(peaks)
    for e, p in zip(found, peaks, strict=True):
        peak = int(e.payload["peak"])  # type: ignore[arg-type]
        assert abs(peak - p * sfreq) <= 0.05 * sfreq
        assert e.sample_index == peak - round(0.25 * sfreq)
        assert e.duration == 2 * round(0.25 * sfreq) + 1


def test_blink_label_and_eog_channel_settings() -> None:
    b = blink_recording(250.0, [5.0, 15.0])
    _, events = detect(BlinkDetector(), b, mark_label="BAD_blink", eog="VEOG",
                       half_width=0.1, blink_threshold=80.0)  # fmt: skip
    assert [e.label for e in events] == ["BAD_blink", "BAD_blink"]
    assert {e.duration for e in events} == {2 * 25 + 1}
    with pytest.raises(ValueError, match="no channel named EOG9"):
        detect(BlinkDetector(), b, eog="EOG9")
    no_eog = Block(b.data[:4], SignalInfo(EEG, 250.0), 0, 0.0)
    bad_eog = Block(b.data, dataclasses.replace(b.info, bads=("VEOG",)), 0, 0.0)
    for block in (no_eog, bad_eog):  # auto leaves out a bad EOG channel
        with pytest.raises(ValueError, match="no EOG channel found"):
            detect(BlinkDetector(), block)
    named = detect(BlinkDetector(), bad_eog, eog="VEOG", blink_threshold=80.0)[1]
    assert len(named) == 2
    short = Block(b.data[:, :2000], b.info, 0, 0.0)  # 8 s: under MNE's 10 s filter
    assert detect(BlinkDetector(), short, blink_threshold=80.0)[1] == ()
    one = Block(noise(4, 1 / 250, 250.0), SignalInfo(EEG, 250.0), 0, 0.0)
    assert detect(AmplitudeDetector(), one)[1] == ()  # a sliver left between cuts


# -- in a pipeline --------------------------------------------------------------------


def test_reject_segments_cuts_what_a_detector_marks(tmp_path: Path) -> None:
    sfreq = 250.0
    data = noise(3, 8.0, sfreq)
    data[1, 1000] += 400e-6  # a spike at 4 s
    raw = mne.io.RawArray(data, mne.create_info(["Fz", "Cz", "Pz"], sfreq, "eeg"),
                          verbose="error")  # fmt: skip
    path = tmp_path / "sub-01_raw.fif"
    raw.save(path, verbose="error")
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": [
        {"id": "src", "node": "meegxer.core.file", "params": {"path": str(path)}},
        {"id": "amp", "node": "meegxer.nodes.detect_amplitude"},
        {"id": "rej", "node": "meegxer.core.reject_segments"},
    ]}, Registry.from_entry_points())  # fmt: skip
    result = BatchScheduler().run(graph, keep=["rej"])
    (peak,) = spans(result.events["amp"], "BAD_peak")
    assert peak[0] <= 1000 < peak[1]
    (boundary,) = result.events["rej"]
    assert boundary.label == "boundary" and boundary.sample_index == peak[1]
    pieces = result.outputs["rej"]
    assert isinstance(pieces, list)
    cut = [(p.sample_index, p.stop_index) for p in pieces if isinstance(p, Block)]
    assert cut == [(0, peak[0]), (peak[1], 2000)]
