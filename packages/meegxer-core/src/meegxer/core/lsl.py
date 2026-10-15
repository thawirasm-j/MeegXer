"""Lab Streaming Layer source and sink (online only).

LSL streams usually carry microvolts; the core works in volts, so both nodes scale
by ``unit``. A stream may carry a ``SAMPLE_INDEX`` channel (the sink can add one):
the source then uses it as the sample index and counts gaps and duplicates, which
is how acceptance test 2 verifies that no sample is dropped or repeated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pylsl

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink, Source
from meegxer.core.ports import PortTypeError
from meegxer.core.sinks import feature_columns

INDEX_CHANNEL = "SAMPLE_INDEX"
_UNIT_SCALE = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}


def _manifest(name: str, role: str, params: list[dict[str, object]]) -> Manifest:
    ports = [{"name": "out" if role == "source" else "in", "type": "signal"}]
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": "0.1.0",
            "type": "stream",
            "role": role,
            "availability": "online",
            "inputs": [] if role == "source" else ports,
            "outputs": ports if role == "source" else [],
            "params": params,
        }
    )


_UNIT_PARAM = {
    "name": "unit",
    "type": "enum",
    "default": "uV",
    "choices": list(_UNIT_SCALE),
    "advanced": True,
    "description": "Unit of the values on the LSL stream",
}


def channel_labels(info: pylsl.StreamInfo) -> list[str]:
    labels: list[str] = []
    channel = info.desc().child("channels").child("channel")
    while not channel.empty():
        labels.append(channel.child_value("label"))
        channel = channel.next_sibling()
    if len(labels) != info.channel_count() or not all(labels):
        labels = [f"Ch{i + 1}" for i in range(info.channel_count())]
    return labels


class LslSource(Source):
    """Pulls an LSL stream, resolved by name (and optionally type)."""

    manifest: ClassVar[Manifest] = _manifest(
        "meegxer.core.lsl_in",
        "source",
        [
            {"name": "stream_name", "type": "string", "default": "", "primary": True},
            {"name": "stream_type", "type": "string", "default": "", "advanced": True},
            {"name": "timeout", "type": "float", "default": 10.0, "min": 0.1,
             "unit": "s", "advanced": True, "description": "How long to search"},
            _UNIT_PARAM,
        ],
    )  # fmt: skip
    live: ClassVar[bool] = True

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._inlet: pylsl.StreamInlet | None = None
        self._info: SignalInfo | None = None
        self._index_col: int | None = None
        self._keep: list[int] = []
        self._next = 0
        self._last_index: int | None = None
        self.first_index: int | None = None  # first counter value seen
        self.gaps = 0  # samples missing according to the index channel
        self.duplicates = 0

    def open(self) -> None:
        if self._inlet is not None:
            return
        name = str(self.params["stream_name"])
        if not name:
            raise ValueError("lsl_in: 'stream_name' is required")
        found = pylsl.resolve_byprop(
            "name", name, timeout=float(self.params["timeout"])
        )
        kind = str(self.params["stream_type"])
        found = [s for s in found if not kind or s.type() == kind]
        if not found:
            raise TimeoutError(f"no LSL stream named {name!r} found")
        inlet = pylsl.StreamInlet(
            found[0],
            max_buflen=30,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        full = inlet.info(timeout=float(self.params["timeout"]))
        labels = channel_labels(full)
        if INDEX_CHANNEL in labels:
            self._index_col = labels.index(INDEX_CHANNEL)
        self._keep = [i for i, lab in enumerate(labels) if lab != INDEX_CHANNEL]
        sfreq = full.nominal_srate()
        if sfreq <= 0:
            raise ValueError(f"LSL stream {name!r} has no regular sampling rate")
        self._info = SignalInfo(
            tuple(labels[i] for i in self._keep), sfreq, full.source_id() or name
        )
        inlet.open_stream(timeout=float(self.params["timeout"]))
        self._inlet = inlet

    @property
    def info(self) -> SignalInfo:
        self.open()
        assert self._info is not None
        return self._info

    def read(self, max_samples: int) -> Block | None:
        self.open()
        assert self._inlet is not None and self._info is not None
        samples, stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
        if not stamps:
            empty = np.empty((self._info.n_channels, 0))
            return Block(empty, self._info, self._next, pylsl.local_clock())
        chunk = np.asarray(samples, dtype=np.float64)
        scale = _UNIT_SCALE[str(self.params["unit"])]
        if self._index_col is not None:
            self._check_index(chunk[:, self._index_col])
        data = chunk[:, self._keep].T * scale
        block = Block(data, self._info, self._next, float(stamps[0]))
        self._next += data.shape[1]
        return block

    def _check_index(self, index: npt.NDArray[np.float64]) -> None:
        """Count missing and repeated samples from consecutive index steps."""
        idx = index.astype(np.int64)
        if self.first_index is None:
            self.first_index = int(idx[0])
        previous = (
            idx[:1] - 1 if self._last_index is None else np.array([self._last_index])
        )
        steps = np.diff(np.concatenate([previous, idx]))
        self.gaps += int(np.sum(steps[steps > 1] - 1))
        self.duplicates += int(np.sum(1 - steps[steps < 1]))
        self._last_index = int(idx[-1])

    def close(self) -> None:
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None


class LslSink(Sink):
    """Publishes Signal blocks as an LSL outlet, with the original timestamps."""

    manifest: ClassVar[Manifest] = _manifest(
        "meegxer.core.lsl_out",
        "sink",
        [
            {"name": "stream_name", "type": "string", "default": "MeegXer",
             "primary": True},
            {"name": "stream_type", "type": "string", "default": "EEG",
             "advanced": True},
            {"name": "include_index", "type": "bool", "default": False,
             "advanced": True,
             "description": f"Append a {INDEX_CHANNEL} channel (sample counter)"},
            _UNIT_PARAM,
        ],
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._outlet: pylsl.StreamOutlet | None = None
        self._labels: tuple[str, ...] = ()

    def _open(self, info: SignalInfo) -> pylsl.StreamOutlet:
        with_index = bool(self.params["include_index"])
        labels = [*info.ch_labels, *([INDEX_CHANNEL] if with_index else [])]
        stream = pylsl.StreamInfo(
            str(self.params["stream_name"]),
            str(self.params["stream_type"]),
            len(labels),
            info.sfreq,
            "double64" if with_index else "float32",
            f"meegxer-{self.params['stream_name']}",
        )
        channels = stream.desc().append_child("channels")
        for label in labels:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            unit = "" if label == INDEX_CHANNEL else str(self.params["unit"])
            ch.append_child_value("unit", unit)
        self._labels = info.ch_labels
        return pylsl.StreamOutlet(stream, chunk_size=0, max_buffered=60)

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError("lsl_out accepts Signal blocks only")
        if block.n_samples == 0:
            return
        if self._outlet is None:
            self._outlet = self._open(block.info)
        elif block.info.ch_labels != self._labels:
            raise ValueError("lsl_out: channel layout changed while streaming")
        data = block.data / _UNIT_SCALE[str(self.params["unit"])]
        if self.params["include_index"]:
            data = np.vstack([data, block.sample_indices()[None, :]])
        last = block.timestamp + (block.n_samples - 1) / block.info.sfreq
        self._outlet.push_chunk(np.ascontiguousarray(data.T), timestamp=last)

    def close(self) -> None:
        self._outlet = None


class LslFeatureSink(Sink):
    """Publishes feature windows as an irregular-rate LSL outlet: one sample per
    window, one channel per feature value (labelled like the feature TSV columns,
    e.g. ``Fz|alpha``), stamped with the window's start time."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.core.lsl_features_out",
            "version": "0.1.0",
            "type": "stream",
            "role": "sink",
            "description": "Feature windows as an LSL stream (one sample per window).",
            "inputs": [{"name": "in", "type": "features"}],
            "outputs": [],
            "params": [
                {"name": "stream_name", "type": "string",
                 "default": "MeegXer features", "primary": True},
                {"name": "stream_type", "type": "string", "default": "Features",
                 "advanced": True},
            ],
            "availability": "both",
        }
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._outlet: pylsl.StreamOutlet | None = None
        self._columns: list[str] = []

    def _open(self, columns: list[str]) -> pylsl.StreamOutlet:
        stream = pylsl.StreamInfo(
            str(self.params["stream_name"]),
            str(self.params["stream_type"]),
            len(columns),
            pylsl.IRREGULAR_RATE,
            "float32",
            f"meegxer-{self.params['stream_name']}",
        )
        channels = stream.desc().append_child("channels")
        for label in columns:
            channels.append_child("channel").append_child_value("label", label)
        self._columns = columns
        return pylsl.StreamOutlet(stream, chunk_size=0, max_buffered=60)

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            raise PortTypeError("lsl_features_out accepts Features blocks only")
        columns = feature_columns(block)
        if self._outlet is None:
            self._outlet = self._open(columns)
        elif columns != self._columns:
            raise ValueError("lsl_features_out: feature layout changed while streaming")
        sample = [float(v) for v in block.data.ravel()]
        self._outlet.push_sample(sample, timestamp=block.timestamp)

    def close(self) -> None:
        self._outlet = None
