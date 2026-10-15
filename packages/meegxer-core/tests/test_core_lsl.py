"""LSL round trips and a short acceptance test 2 (no dropped/duplicated samples).

The full 30-minute run is ``tools/stability_test.py``.
"""

import time
import uuid

import numpy as np
import pylsl
import pytest

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.graph import Graph
from meegxer.core.lsl import INDEX_CHANNEL, LslFeatureSink, LslSink, LslSource
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource


def unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


def test_sink_to_source_round_trip_with_units_and_index() -> None:
    name = unique("mx-rt")
    info = SignalInfo(("Fz", "Cz"), 250.0)
    sink = LslSink({"stream_name": name, "include_index": True})
    data = np.random.default_rng(0).standard_normal((2, 500)) * 1e-5
    t0 = pylsl.local_clock()
    sink.write(Block(data[:, :250], info, 0, t0))
    source = LslSource({"stream_name": name, "timeout": 5.0})
    assert source.info.ch_labels == ("Fz", "Cz")  # index channel is hidden
    assert source.info.sfreq == 250.0
    sink.write(Block(data[:, 250:], info, 250, t0 + 1.0))
    received: list[Block] = []
    deadline = time.monotonic() + 5
    while sum(b.n_samples for b in received) < 250 and time.monotonic() < deadline:
        block = source.read(1000)
        assert block is not None
        if block.n_samples:
            received.append(block)
        time.sleep(0.01)
    got = np.hstack([b.data for b in received])
    np.testing.assert_allclose(
        got[:, -250:], data[:, 250:], rtol=1e-6
    )  # uV on the wire
    assert source.gaps == 0 and source.duplicates == 0
    source.close()
    sink.close()


def test_feature_outlet_sends_one_labelled_sample_per_window() -> None:
    name = unique("mx-feat")
    sink = LslFeatureSink({"stream_name": name})
    coords = {"channel": ("Fz", "Cz"), "band": ("alpha", "beta")}
    t0 = pylsl.local_clock()

    def window(k: int) -> FeatureBlock:
        values = np.arange(4.0).reshape(2, 2) + 10 * k
        return FeatureBlock(values, ("channel", "band"), (k * 50, k * 50 + 100),
                            t0 + k * 0.2, coords)  # fmt: skip

    sink.write(window(0))  # opens the outlet
    streams = pylsl.resolve_byprop("name", name, timeout=5.0)
    assert streams
    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    assert info.nominal_srate() == pylsl.IRREGULAR_RATE
    labels, ch = [], info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    assert labels == ["Fz|alpha", "Fz|beta", "Cz|alpha", "Cz|beta"]
    inlet.open_stream(timeout=5.0)
    time.sleep(0.2)
    sink.write(window(1))
    sample, stamp = inlet.pull_sample(timeout=5.0)
    assert sample == [10.0, 11.0, 12.0, 13.0]
    assert stamp == pytest.approx(t0 + 0.2, abs=1e-6)
    sink.close()


def test_source_detects_gaps_and_duplicates() -> None:
    name = unique("mx-gap")
    stream = pylsl.StreamInfo(name, "EEG", 2, 100.0, "double64", name)
    chans = stream.desc().append_child("channels")
    for label in ("Cz", INDEX_CHANNEL):
        chans.append_child("channel").append_child_value("label", label)
    outlet = pylsl.StreamOutlet(stream)
    source = LslSource({"stream_name": name, "timeout": 5.0})
    _ = source.info
    time.sleep(0.5)
    for idx in [*range(0, 10), *range(15, 20), *range(18, 25)]:
        outlet.push_sample([0.0, float(idx)])
    deadline = time.monotonic() + 5
    total = 0
    while total < 22 and time.monotonic() < deadline:
        block = source.read(100)
        assert block is not None
        total += block.n_samples
        time.sleep(0.02)
    assert total == 22
    assert source.gaps == 5 and source.duplicates == 2
    source.close()


@pytest.mark.timeout(60)
def test_short_stability_lsl_in_to_lsl_out() -> None:
    """Acceptance test 2, short: 64 ch / 1 kHz through LSL in -> filter -> LSL out."""
    sterile = pytest.importorskip("meegxer.sterile.filters")
    from meegxer.core.stage import Stage

    gen_name, out_name = unique("mx-gen"), unique("mx-out")
    seconds = 4.0
    generator = RealtimeScheduler(Graph.chain([
        SyntheticSource({"duration": seconds, "n_channels": 64, "sfreq": 1000.0}),
        LslSink({"stream_name": gen_name, "include_index": True}),
    ]))  # fmt: skip
    generator.start()
    pipeline = RealtimeScheduler(Graph.chain([
        LslSource({"stream_name": gen_name, "timeout": 10.0}),
        Stage(sterile.IIRFilter(), {"kind": "highpass", "low": 1.0}),
        MemoryProbe(),
        LslSink({"stream_name": out_name, "include_index": True}),
    ]))  # fmt: skip
    pipeline.start()
    found = pylsl.resolve_byprop("name", out_name, timeout=10)
    assert found
    inlet = pylsl.StreamInlet(found[0], max_buflen=30)
    inlet.open_stream(timeout=10)
    indices: list[float] = []
    assert generator.wait(seconds + 10)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        samples, _ = inlet.pull_chunk(timeout=0.2, max_samples=100_000)
        indices.extend(row[-1] for row in samples)
        if (
            pipeline.status()["samples"]
            and indices
            and indices[-1] >= pipeline.status()["samples"] - 1
        ):
            break
    pipeline.stop()
    status = pipeline.status()
    idx = np.asarray(indices)
    assert status["errors"] == []
    assert status.get("source_gaps") == 0 and status.get("source_duplicates") == 0
    assert idx.size > 0 and np.all(np.diff(idx) == 1)  # no gaps, no duplicates out
    assert status["latency"]["p95_ms"] < 150
