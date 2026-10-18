"""Data blocks that flow between nodes, plus label and sample-index helpers.

Every block is aligned by absolute sample index and carries the LSL timestamp of
its first sample (CLAUDE.md rule 8). Channels are matched by label, never by
position. Block arrays are read-only views: a processor must return new data
rather than modify its input, because probes may hold the same block.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]
Position = tuple[float, float, float]
# a filter a stream has been through: (kind, low, high) in Hz, kind highpass
# (low), lowpass (high), bandstop (low-high) or notch (low); unused edges 0
Filter = tuple[str, float, float]
FILTER_KINDS = ("highpass", "lowpass", "bandstop", "notch")

CHANNEL_TYPES = ("eeg", "eog", "emg", "ecg", "resp", "stim", "misc")
DEFAULT_PROCESSED_TYPES = ("eeg",)
type JSONValue = (
    str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
)


class ChannelMismatchError(ValueError):
    """Channel labels are missing, duplicated, or differ between blocks."""


class SampleGapError(ValueError):
    """Blocks are not contiguous: samples are missing between them."""


class SampleOverlapError(ValueError):
    """Blocks are not contiguous: samples are duplicated between them."""


def _readonly(array: npt.ArrayLike, ndim_min: int) -> FloatArray:
    data = np.asarray(array, dtype=np.float64)
    if data.ndim < ndim_min:
        raise ValueError(f"expected at least {ndim_min} dimension(s), got {data.ndim}")
    view = data.view()
    view.flags.writeable = False
    return view


def _check_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


@dataclass(frozen=True, slots=True)
class SignalInfo:
    """Static description of a Signal stream.

    ``bads`` marks channels known to be bad. They stay in the data; nodes that
    pool channels (e.g. average reference) can leave them out. ``ch_types``
    (e.g. ``eeg``, ``eog``) and ``positions`` (x, y, z in metres, head coordinates:
    x right, y front, z up; None where unknown) are per channel, or empty when not
    known at all. They describe the channels but do not change the data.
    ``ch_classes`` and ``ch_scores`` are attributes a classifier gave each channel
    (ICLabel for components: its most likely class and that probability), or empty.
    ``filters`` lists the filters the stream has been through, in order (filter
    nodes add theirs; spec 0.2.19): metadata, so it takes no part in equality and
    a live change of a cutoff does not split the stream.
    """

    ch_labels: tuple[str, ...]
    sfreq: float
    stream_id: str = ""
    bads: tuple[str, ...] = ()
    ch_types: tuple[str, ...] = ()
    positions: tuple[Position | None, ...] = ()
    ch_classes: tuple[str, ...] = ()
    ch_scores: tuple[float, ...] = ()
    filters: tuple[Filter, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        labels = tuple(self.ch_labels)
        object.__setattr__(self, "ch_labels", labels)
        if not labels:
            raise ChannelMismatchError("a signal needs at least one channel")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ChannelMismatchError("channel labels must be non-empty strings")
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ChannelMismatchError(f"duplicate channel labels: {duplicates}")
        if not math.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive and finite, got {self.sfreq}")
        bads = tuple(self.bads)
        object.__setattr__(self, "bads", bads)
        unknown = sorted(set(bads) - set(labels))
        if unknown or len(set(bads)) != len(bads):
            raise ChannelMismatchError(f"bad channels must be unique labels: {bads}")
        types = tuple(self.ch_types)
        object.__setattr__(self, "ch_types", types)
        positions = tuple(
            None if p is None else (float(p[0]), float(p[1]), float(p[2]))
            for p in self.positions
        )
        object.__setattr__(self, "positions", positions)
        classes = tuple(str(c) for c in self.ch_classes)
        object.__setattr__(self, "ch_classes", classes)
        scores = tuple(float(s) for s in self.ch_scores)
        object.__setattr__(self, "ch_scores", scores)
        filters = tuple((str(k), float(lo), float(hi)) for k, lo, hi in self.filters)
        unknown_kinds = sorted({k for k, _, _ in filters} - set(FILTER_KINDS))
        if unknown_kinds:
            raise ValueError(f"unknown filter kinds: {unknown_kinds}")
        object.__setattr__(self, "filters", filters)
        per_channel = (("ch_types", types), ("positions", positions),
                       ("ch_classes", classes), ("ch_scores", scores))  # fmt: skip
        for name, values in per_channel:
            if values and len(values) != len(labels):
                raise ChannelMismatchError(
                    f"{name} has {len(values)} entries for {len(labels)} channels"
                )

    def subset(self, labels: Sequence[str]) -> SignalInfo:
        """The description of the given channels, in the given order."""
        idx = [int(i) for i in match_channels(self.ch_labels, labels)]
        return SignalInfo(
            tuple(labels),
            self.sfreq,
            self.stream_id,
            tuple(label for label in labels if label in self.bads),
            tuple(self.ch_types[i] for i in idx) if self.ch_types else (),
            tuple(self.positions[i] for i in idx) if self.positions else (),
            tuple(self.ch_classes[i] for i in idx) if self.ch_classes else (),
            tuple(self.ch_scores[i] for i in idx) if self.ch_scores else (),
            self.filters,
        )

    def filtered(self, *filters: Filter) -> SignalInfo:
        """This description with ``filters`` added after the earlier ones."""
        return dataclasses.replace(self, filters=self.filters + tuple(filters))

    @property
    def n_channels(self) -> int:
        return len(self.ch_labels)

    def labels_of_types(self, types: Sequence[str]) -> tuple[str, ...]:
        """Channels whose type is in ``types``. Channels of unknown type count as
        EEG, so streams that carry no types behave as all-EEG."""
        if not self.ch_types:
            return self.ch_labels if "eeg" in types else ()
        return tuple(
            label
            for label, kind in zip(self.ch_labels, self.ch_types, strict=True)
            if (kind or "eeg") in types
        )


def match_channels(available: Sequence[str], wanted: Sequence[str]) -> IndexArray:
    """Return indices into ``available`` for each label in ``wanted``, in order."""
    position = {label: i for i, label in enumerate(available)}
    if len(position) != len(available):
        raise ChannelMismatchError("available channel labels are not unique")
    if len(set(wanted)) != len(wanted):
        raise ChannelMismatchError("requested channel labels are not unique")
    missing = [label for label in wanted if label not in position]
    if missing:
        raise ChannelMismatchError(f"channels not found: {missing}")
    return np.array([position[label] for label in wanted], dtype=np.intp)


@dataclass(frozen=True, eq=False, slots=True)
class Block:
    """A chunk of a Signal stream: ``data`` is (n_channels, n_samples), float64.

    ``sample_index`` is the absolute index of the first sample; ``timestamp`` is the
    LSL time of that sample. Per-sample times are ``timestamp + k / sfreq``.
    ``bypassed`` lists the nodes that passed this data through unprocessed because
    they had no valid model yet (only when that node was set to pass through).
    """

    data: FloatArray
    info: SignalInfo
    sample_index: int
    timestamp: float
    bypassed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        data = _readonly(self.data, ndim_min=2)
        if data.ndim != 2:
            raise ValueError(f"signal data must be 2-D, got shape {data.shape}")
        if data.shape[0] != self.info.n_channels:
            raise ChannelMismatchError(
                f"data has {data.shape[0]} rows but info has "
                f"{self.info.n_channels} channel labels"
            )
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "data", data)
        object.__setattr__(
            self, "sample_index", _check_index("sample_index", self.sample_index)
        )

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def stop_index(self) -> int:
        """Absolute index one past the last sample."""
        return self.sample_index + self.n_samples

    def sample_indices(self) -> npt.NDArray[np.int64]:
        return np.arange(self.sample_index, self.stop_index, dtype=np.int64)

    def timestamps(self) -> FloatArray:
        return self.timestamp + np.arange(self.n_samples) / self.info.sfreq

    def slice(self, start: int, stop: int) -> Block:
        """Return samples ``[start, stop)`` by absolute sample index."""
        if not self.sample_index <= start <= stop <= self.stop_index:
            raise IndexError(
                f"[{start}, {stop}) is outside block [{self.sample_index}, "
                f"{self.stop_index})"
            )
        offset = start - self.sample_index
        return Block(
            data=self.data[:, offset : offset + stop - start],
            info=self.info,
            sample_index=start,
            timestamp=self.timestamp + offset / self.info.sfreq,
            bypassed=self.bypassed,
        )

    def pick(self, labels: Sequence[str]) -> Block:
        """Return the given channels, in the given order."""
        idx = match_channels(self.info.ch_labels, labels)
        info = self.info.subset(labels)
        return Block(
            self.data[idx], info, self.sample_index, self.timestamp, self.bypassed
        )


def concat_blocks(blocks: Sequence[Block]) -> Block:
    """Join contiguous blocks of one stream; gaps and overlaps raise."""
    if not blocks:
        raise ValueError("no blocks to concatenate")
    first = blocks[0]
    for prev, nxt in itertools.pairwise(blocks):
        if nxt.info != first.info:
            raise ChannelMismatchError("blocks come from streams with different info")
        if nxt.sample_index > prev.stop_index:
            raise SampleGapError(
                f"missing samples [{prev.stop_index}, {nxt.sample_index})"
            )
        if nxt.sample_index < prev.stop_index:
            raise SampleOverlapError(
                f"duplicated samples [{nxt.sample_index}, {prev.stop_index})"
            )
    data = np.concatenate([b.data for b in blocks], axis=1)
    bypassed = tuple(dict.fromkeys(n for b in blocks for n in b.bypassed))
    return Block(data, first.info, first.sample_index, first.timestamp, bypassed)


def align_blocks(reference: Block, other: Block) -> tuple[Block, Block]:
    """Overlay two blocks by sample index and channel label.

    Returns both blocks cut to their common sample range and to the channels they
    share, in ``reference`` order. Used by probes to compare taps.
    """
    if reference.info.sfreq != other.info.sfreq:
        raise ValueError("cannot align blocks with different sampling rates")
    start = max(reference.sample_index, other.sample_index)
    stop = min(reference.stop_index, other.stop_index)
    if start >= stop:
        raise IndexError("blocks share no samples")
    other_labels = set(other.info.ch_labels)
    common = [label for label in reference.info.ch_labels if label in other_labels]
    if not common:
        raise ChannelMismatchError("blocks share no channel labels")
    return (
        reference.slice(start, stop).pick(common),
        other.slice(start, stop).pick(common),
    )


type Coordinate = tuple[str, ...] | tuple[float, ...]


def unit_at(
    unit: str, unit_dim: str, units: tuple[str, ...], index: Mapping[str, int]
) -> str:
    """The unit at ``index`` given a feature block's ``unit``/``unit_dim``/``units``
    (``unit`` where the dimension is not chosen or there are no per-entry units)."""
    if unit_dim and units and unit_dim in index:
        return units[index[unit_dim]]
    return unit


def units_wire(unit: str, unit_dim: str, units: tuple[str, ...]) -> dict[str, Any]:
    """A feature block's units as protocol fields (only those that are set)."""
    out: dict[str, Any] = {"unit": unit} if unit else {}
    if unit_dim:
        out |= {"unit_dim": unit_dim, "units": list(units)}
    return out


def units_from_wire(wire: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """``(unit, unit_dim, units)`` from protocol fields (``units_wire``)."""
    return (str(wire.get("unit", "")), str(wire.get("unit_dim", "")),
            tuple(str(u) for u in wire.get("units", ())))  # fmt: skip


@dataclass(frozen=True, eq=False, slots=True)
class FeatureBlock:
    """Windowed features, e.g. PSD (channel x frequency) or band power.

    ``dims`` names each axis of ``data``; ``coords`` optionally labels the entries
    of a dimension (``coords["channel"]`` must hold channel labels). ``window`` is
    the source sample range ``[start, stop)`` the features were computed from.

    ``unit`` is the unit of every value (``"V^2/Hz"``, ``"dB"``; ``"1"``:
    dimensionless; ``""``: not stated). When the values of one dimension differ
    in unit (specparam's measures), ``unit_dim`` names it and ``units`` gives the
    unit of each of its entries, in order (spec 0.2.19).
    """

    data: FloatArray
    dims: tuple[str, ...]
    window: tuple[int, int]
    timestamp: float
    coords: Mapping[str, Coordinate] = field(default_factory=dict)
    stream_id: str = ""
    unit: str = ""
    unit_dim: str = ""
    units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        data = _readonly(self.data, ndim_min=1)
        dims = tuple(self.dims)
        if len(dims) != data.ndim or len(set(dims)) != len(dims):
            raise ValueError(f"dims {dims} must name each of {data.ndim} axes once")
        for dim, values in self.coords.items():
            if dim not in dims:
                raise ValueError(f"coordinate for unknown dimension {dim!r}")
            if len(values) != data.shape[dims.index(dim)]:
                raise ValueError(f"coordinate {dim!r} length does not match data")
        if self.unit_dim or self.units:
            if self.unit_dim not in dims:
                raise ValueError(f"unit_dim {self.unit_dim!r} is not one of {dims}")
            if len(self.units) != data.shape[dims.index(self.unit_dim)]:
                raise ValueError(f"units must give one unit per {self.unit_dim!r}")
        start, stop = self.window
        _check_index("window start", start)
        if stop < start:
            raise ValueError("window stop must be >= start")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "dims", dims)
        coords = {k: tuple(v) for k, v in self.coords.items()}
        object.__setattr__(self, "coords", MappingProxyType(coords))
        object.__setattr__(self, "units", tuple(self.units))

    def unit_of(self, index: Mapping[str, int] | None = None) -> str:
        """The unit of the values at ``index`` (a position along ``unit_dim``)."""
        return unit_at(self.unit, self.unit_dim, self.units, index or {})


@dataclass(frozen=True, slots=True)
class Event:
    """A marker or command at an absolute sample index."""

    sample_index: int
    label: str
    duration: int = 0
    payload: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_index("sample_index", self.sample_index)
        _check_index("duration", self.duration)
        if not self.label:
            raise ValueError("event label must be non-empty")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class EventBlock:
    """Events of one stream covering samples ``[start_index, stop_index)``.

    The covered range is explicit so a consumer can tell "no events here" apart
    from "not yet seen".
    """

    events: tuple[Event, ...]
    start_index: int
    stop_index: int
    stream_id: str = ""

    def __post_init__(self) -> None:
        events = tuple(sorted(self.events, key=lambda e: e.sample_index))
        _check_index("start_index", self.start_index)
        if self.stop_index < self.start_index:
            raise ValueError("stop_index must be >= start_index")
        start, stop = self.start_index, self.stop_index
        outside = [e for e in events if not start <= e.sample_index < stop]
        if outside:
            raise IndexError(f"{len(outside)} event(s) fall outside the covered range")
        object.__setattr__(self, "events", events)
