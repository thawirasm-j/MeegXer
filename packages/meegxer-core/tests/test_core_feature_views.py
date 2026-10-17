"""Feature views in the core: Features probes, feature frames on the display
stream (protocol 0.2), channel classes on signal frames, offline feature stores."""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.display import (
    Decimator,
    DisplayFrame,
    DisplayPublisher,
    DisplaySubscriber,
    FeatureFrame,
    decode_frame,
    encode_features,
    encode_frame,
)
from meegxer.core.graph import Graph
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.offline import OfflineJob
from meegxer.core.probes import FeatureProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.serialization import State
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage


class WindowMean(Processor):
    """Mean of each channel over non-overlapping windows of n samples."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.0", "name": "test.window_mean", "version": "0.1.0",
        "type": "analysis", "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "features"}],
        "params": [{"name": "n", "type": "int", "default": 50}],
    })  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        n = int(params["n"])
        held = block.data if state is None else np.hstack(
            [np.asarray(state["held"]), block.data])  # fmt: skip
        start = block.sample_index if state is None else int(state["start"])  # type: ignore[arg-type]
        out = []
        while held.shape[1] >= n:
            out.append(FeatureBlock(held[:, :n].mean(axis=1), ("channel",),
                                    (start, start + n), 0.0,
                                    {"channel": block.info.ch_labels}))  # fmt: skip
            held, start = held[:, n:], start + n
        return TransformResult(tuple(out), {"held": held, "start": start})


def source(seconds: float = 2.0) -> SyntheticSource:
    return SyntheticSource({"duration": seconds, "n_channels": 3, "sfreq": 100.0})


def test_feature_frames_and_channel_classes_round_trip() -> None:
    values = np.arange(6.0).reshape(2, 3)
    frame = FeatureFrame("tap", 4, (100, 200), 100.0, ("channel", "frequency"),
                         {"channel": ("Fz", "Cz"), "frequency": (1.0, 2.0, 3.0)},
                         values)  # fmt: skip
    back = decode_frame(encode_features(frame))
    assert isinstance(back, FeatureFrame)
    assert (back.window, back.dims, back.coords["frequency"]) == (
        (100, 200), ("channel", "frequency"), (1.0, 2.0, 3.0))  # fmt: skip
    np.testing.assert_array_equal(back.values, values)
    env = Decimator(2).feed(Block(np.ones((2, 4)), SignalInfo(("IC1", "IC2"), 100.0),
                                  0, 0.0))  # fmt: skip
    assert env is not None
    ics = DisplayFrame("ics", 0, ("IC1", "IC2"), 100.0, env,
                       ch_classes=("eye", "brain"), ch_scores=(0.97, 0.8))  # fmt: skip
    back_ics = decode_frame(encode_frame(ics))
    assert isinstance(back_ics, DisplayFrame)
    assert back_ics.ch_classes == ("eye", "brain")
    assert back_ics.ch_scores == (0.97, 0.8)


def test_feature_probe_keeps_its_newest_windows() -> None:
    probe = FeatureProbe({"keep_windows": 3})
    for k in range(5):
        probe.tap(FeatureBlock(np.array([k]), ("feature",), (k, k + 1), 0.0))
    assert [w.window[0] for w in probe.windows] == [2, 3, 4]


def test_live_features_reach_the_display_stream() -> None:
    graph = Graph.chain([source(), Stage(WindowMean(), {"n": 20})])
    graph.add(FeatureProbe(), id="tap_windowmean1")
    publisher = DisplayPublisher("tcp://127.0.0.1:*")
    subscriber = DisplaySubscriber(publisher.endpoint, ["tap_windowmean1"])
    time.sleep(0.2)
    scheduler = RealtimeScheduler(graph, display=publisher)
    scheduler.start()
    frames = []
    while (frame := subscriber.recv(timeout=1.0)) is not None:
        frames.append(frame)
    assert scheduler.wait(10)
    assert frames and all(isinstance(f, FeatureFrame) for f in frames)
    feature_frames = [f for f in frames if isinstance(f, FeatureFrame)]
    windows = [f.window for f in feature_frames]
    assert windows == sorted(windows) and windows[0] == (0, 20)
    assert feature_frames[0].coords["channel"] == ("Fp1", "Fp2", "F3")
    subscriber.close()
    publisher.close()


def test_offline_runs_keep_feature_windows_for_the_view() -> None:
    graph = Graph.chain([source(), Stage(WindowMean(), {"n": 20})])
    graph.add(FeatureProbe(), id="tap_windowmean1")
    job = OfflineJob(graph)
    job.start()
    job.join(30)
    assert job.state == "done", job.error
    info = job.status()["features"]["tap_windowmean1"]
    assert (info["dims"], info["n_windows"]) == (["channel"], 10)
    some = job.feature_windows("tap_windowmean1", 0, 100, 3, "none")
    assert some["windows"] == [[0, 20], [40, 60], [80, 100]]
    assert len(some["values"]) == 3 and len(some["values"][0]) == 3
    mean = job.feature_windows("tap_windowmean1", 0, 200, 2000, "mean")
    assert mean["windows"] == [[0, 200]] and mean["n_windows"] == 10
    job.discard()
