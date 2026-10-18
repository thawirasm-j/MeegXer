"""Display stream: decimated min/max envelopes over a lossy ZeroMQ PUB socket.

Rules 2 and 8: the core decimates before sending, never waits for a viewer, and
aligns everything by sample index. Bins cover fixed absolute sample ranges
(bin ``k`` is samples ``[k * bin_size, (k + 1) * bin_size)``), so envelopes from
different probes of the same stream line up bin for bin.

Wire format (one multipart message per probe per frame; see
``spec/display-protocol.md``): ``[topic, header, payload]`` where topic is the probe
id (UTF-8), header is JSON and payload is float32 ``(2, n_channels, n_bins)``
little-endian (row 0 = min, row 1 = max). A Features probe sends its feature
windows instead (header ``kind: "features"``, payload the window's values).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import zmq

from meegxer.core.auth import Guard, login
from meegxer.core.block import Block, FloatArray

DISPLAY_PROTOCOL = "meegxer.display/0.2"
_ACCEPTED = ("meegxer.display/0.1", DISPLAY_PROTOCOL)

type ZmqContext = zmq.Context[zmq.Socket[bytes]]


@dataclass(frozen=True, slots=True)
class Envelope:
    """Completed bins of one probe: ``mins``/``maxs`` are (n_channels, n_bins)."""

    start_bin: int
    bin_size: int
    mins: FloatArray
    maxs: FloatArray

    @property
    def n_bins(self) -> int:
        return int(self.mins.shape[1])


class Decimator:
    """Turns a contiguous Signal stream into index-aligned min/max bins."""

    def __init__(self, bin_size: int) -> None:
        if bin_size < 1:
            raise ValueError("bin_size must be >= 1")
        self.bin_size = bin_size
        self._carry: FloatArray | None = None
        self._carry_start = 0

    def feed(self, block: Block) -> Envelope | None:
        data, start = block.data, block.sample_index
        if (
            self._carry is not None
            and self._carry_start + self._carry.shape[1] == start
        ):
            data = np.hstack([self._carry, data])
            start = self._carry_start
        bs = self.bin_size
        first_bin = start // bs
        complete_end = ((start + data.shape[1]) // bs) * bs
        m = complete_end - start
        if m <= 0:
            self._carry, self._carry_start = np.array(data), start
            return None
        head = np.asarray(data[:, :m])
        self._carry, self._carry_start = np.array(data[:, m:]), complete_end
        offset = start - first_bin * bs  # first bin may start before the stream
        if offset:
            pad = np.full((head.shape[0], offset), np.nan)
            head = np.hstack([pad, head])
        bins = head.reshape(head.shape[0], -1, bs)
        return Envelope(first_bin, bs, np.nanmin(bins, axis=2), np.nanmax(bins, axis=2))


@dataclass(frozen=True, slots=True)
class DisplayFrame:
    probe: str
    seq: int
    ch_labels: tuple[str, ...]
    sfreq: float
    envelope: Envelope
    bypassed: tuple[str, ...] = ()  # nodes this data passed through unprocessed
    ch_classes: tuple[str, ...] = ()  # e.g. ICLabel's class of each component
    ch_scores: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """One feature window of a Features probe (display protocol 0.2)."""

    probe: str
    seq: int
    window: tuple[int, int]  # source samples [start, stop) it was computed from
    sfreq: float
    dims: tuple[str, ...]
    coords: dict[str, tuple[str | float, ...]]
    values: FloatArray


def encode_frame(frame: DisplayFrame) -> list[bytes]:
    env = frame.envelope
    header = {
        "protocol": DISPLAY_PROTOCOL,
        "probe": frame.probe,
        "seq": frame.seq,
        "ch_labels": list(frame.ch_labels),
        "sfreq": frame.sfreq,
        "start_bin": env.start_bin,
        "bin_size": env.bin_size,
        "n_bins": env.n_bins,
        "bypassed": list(frame.bypassed),
    }
    if frame.ch_classes:
        header["ch_classes"] = list(frame.ch_classes)
        header["ch_scores"] = list(frame.ch_scores)
    payload = np.stack([env.mins, env.maxs]).astype("<f4").tobytes()
    return [frame.probe.encode(), json.dumps(header).encode(), payload]


def encode_features(frame: FeatureFrame) -> list[bytes]:
    header = {
        "protocol": DISPLAY_PROTOCOL,
        "kind": "features",
        "probe": frame.probe,
        "seq": frame.seq,
        "window": list(frame.window),
        "sfreq": frame.sfreq,
        "dims": list(frame.dims),
        "coords": {k: list(v) for k, v in frame.coords.items()},
        "shape": list(frame.values.shape),
    }
    payload = np.asarray(frame.values).astype("<f4").tobytes()
    return [frame.probe.encode(), json.dumps(header).encode(), payload]


def decode_frame(parts: Sequence[bytes]) -> DisplayFrame | FeatureFrame:
    _, header_bytes, payload = parts
    header = json.loads(header_bytes)
    if header.get("protocol") not in _ACCEPTED:
        raise ValueError(f"unsupported display protocol {header.get('protocol')!r}")
    if header.get("kind") == "features":
        values = np.frombuffer(payload, dtype="<f4").reshape(header["shape"])
        return FeatureFrame(
            header["probe"],
            header["seq"],
            (int(header["window"][0]), int(header["window"][1])),
            float(header["sfreq"]),
            tuple(header["dims"]),
            {k: tuple(v) for k, v in header.get("coords", {}).items()},
            values.astype(np.float64),
        )
    n_ch, n_bins = len(header["ch_labels"]), header["n_bins"]
    arr: npt.NDArray[np.float32] = np.frombuffer(payload, dtype="<f4").reshape(
        2, n_ch, n_bins
    )
    env = Envelope(
        header["start_bin"],
        header["bin_size"],
        arr[0].astype(np.float64),
        arr[1].astype(np.float64),
    )
    return DisplayFrame(
        header["probe"],
        header["seq"],
        tuple(header["ch_labels"]),
        header["sfreq"],
        env,
        tuple(header.get("bypassed", ())),
        tuple(header.get("ch_classes", ())),
        tuple(float(s) for s in header.get("ch_scores", ())),
    )


class DisplayPublisher:
    """Publishes frames without ever blocking; slow subscribers lose frames.

    Owned by the scheduler thread (ZeroMQ sockets are not thread-safe).
    """

    def __init__(
        self,
        endpoint: str,
        *,
        context: ZmqContext | None = None,
        high_water_mark: int = 8,
        secure: bool = False,
    ) -> None:
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, high_water_mark)
        self._socket.setsockopt(zmq.LINGER, 0)
        if secure:  # only subscribers with the daemon's token
            Guard.protect(self._socket)
        self._socket.bind(endpoint)
        self.endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.sent = 0
        self.dropped = 0

    def publish(self, frame: DisplayFrame | FeatureFrame) -> None:
        parts = (
            encode_features(frame)
            if isinstance(frame, FeatureFrame)
            else encode_frame(frame)
        )
        try:
            self._socket.send_multipart(parts, flags=zmq.NOBLOCK)
            self.sent += 1
        except zmq.Again:
            self.dropped += 1

    def close(self) -> None:
        self._socket.close(linger=0)


class DisplaySubscriber:
    """Client side: receive frames for some or all probes."""

    def __init__(
        self,
        endpoint: str,
        probes: Sequence[str] = (),
        *,
        context: ZmqContext | None = None,
        token: str | None = None,
    ) -> None:
        self._context = context or zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        login(self._socket, token)
        self._socket.setsockopt(zmq.RCVHWM, 32)
        for probe in probes or [""]:
            self._socket.setsockopt(zmq.SUBSCRIBE, probe.encode())
        self._socket.connect(endpoint)

    def recv(self, timeout: float = 1.0) -> DisplayFrame | FeatureFrame | None:
        if not self._socket.poll(int(timeout * 1000)):
            return None
        return decode_frame(self._socket.recv_multipart())

    def close(self) -> None:
        self._socket.close(linger=0)
