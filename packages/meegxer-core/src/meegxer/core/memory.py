"""In-memory input and outputs for the Python API (``meegxer.api``, M10.7).

An MNE Raw in memory as the File input, and the signal or the epochs a strip
gives back as MNE objects. The batch scheduler runs them as it runs any
source and writer (rule 4: no second code path): the output nodes are sinks,
so they see what a file writer at the same place would (the events on the
Events rail, the cuts of a Reject segments strip), and turn it into MNE
objects with the writers' own code (``make_raw``, ``epochs_array``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, Event, FeatureBlock
from meegxer.core.epochs_out import (
    MNE_TYPES,
    channel_positions,
    channel_types,
    epochs_array,
)
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink
from meegxer.core.ports import PortTypeError
from meegxer.core.sinks import BOUNDARY, make_raw
from meegxer.core.sources import FileSource

RAW_OUTPUT = "meegxer.api.raw"
EPOCHS_OUTPUT = "meegxer.api.epochs"


class RawInput(FileSource):
    """An MNE Raw in memory as the File input (a copy: the caller's Raw is
    left as it is). Its annotations are the input's events; it has no review
    at import (that belongs to a file)."""

    def __init__(self, raw: Any) -> None:
        super().__init__({"import_review": False})
        self._mne_raw = raw

    def _files(self) -> list[Path]:
        return []

    def _load(self) -> Any:
        return self._mne_raw.copy().load_data(verbose="error")

    def _stream_id(self) -> str:
        names = [Path(str(f)).name for f in self._mne_raw.filenames if f]
        return names[0] if names else "raw"

    def content_id(self) -> str:
        """The recording itself (its samples, channels, rate and events), as
        a file's content identifies it: a model fitted on it stays valid for
        the same data (sessions)."""
        data, info = self._raw
        digest = hashlib.sha256(np.ascontiguousarray(data).tobytes())
        digest.update(repr((info.ch_labels, info.sfreq, info.bads,
                            self._annotations)).encode())  # fmt: skip
        return digest.hexdigest()

    def input_files(self) -> list[Path]:
        return []


def _output_manifest(name: str, input_type: str, what: str) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.0", "name": name, "version": "0.1.0",
        "type": "stream", "role": "sink", "availability": "offline",
        "description": f"{what} as an MNE object (meegxer.api.run)",
        "inputs": [{"name": "in", "type": input_type}], "outputs": [],
    })  # fmt: skip


class RawOutput(Sink):
    """The signal it reads as an ``mne.io.RawArray`` (``raw`` after the run):
    the same events and cuts (``boundary``) as a ``.set`` written there, with
    the channels' types and positions. ``components``: it reads a component
    rail (the ICs), whose channels are ``misc``."""

    manifest: ClassVar[Manifest] = _output_manifest(RAW_OUTPUT, "signal", "The signal")

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self.components = False
        self.raw: Any = None
        self._blocks: list[Block] = []
        self._boundaries: list[list[int]] = []  # [position in data, samples cut]
        self._events: list[list[Any]] = []
        self._written = 0

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError(f"{self.manifest.name} accepts Signal blocks only")
        if not block.n_samples:
            return
        if self._blocks:
            last = self._blocks[-1]
            if block.info != last.info:
                raise ValueError(f"the channels changed at sample {last.stop_index}")
            if block.sample_index > last.stop_index:  # a cut (Reject segments)
                cut = block.sample_index - last.stop_index
                self._boundaries.append([self._written, cut])
        self._blocks.append(block)
        self._written += block.n_samples

    def write_events(self, events: Sequence[Event]) -> None:
        self._events = [[e.sample_index, e.duration, e.label]
                        for e in events if e.label != BOUNDARY]  # fmt: skip

    def close(self) -> None:
        if not self._blocks:
            return
        first = self._blocks[0]
        info = first.info
        header = {"ch_labels": list(info.ch_labels), "sfreq": info.sfreq,
                  "bads": list(info.bads), "sample_index": first.sample_index,
                  "events": self._events, "boundaries": self._boundaries}  # fmt: skip
        if self.components:
            types: list[str] = ["misc"] * info.n_channels
        else:  # a channel with no type is EEG, as everywhere in MeegXer
            given = info.ch_types or ("eeg",) * info.n_channels
            types = [t if (t := str(g).lower()) in MNE_TYPES else "misc"
                     for g in given]  # fmt: skip
        positions = {lb: tuple(p) for lb, p in zip(info.ch_labels,
                     info.positions or (), strict=False) if p}  # fmt: skip
        data = np.concatenate([b.data for b in self._blocks], axis=1)
        self.raw = make_raw(data, header, types,
                            None if self.components else positions)  # fmt: skip
        self._blocks = []


class EpochsOutput(Sink):
    """The epochs rail it reads as ``mne.EpochsArray`` (``epochs`` after the
    run), as the epochs writer makes them; None when no event was cut."""

    manifest: ClassVar[Manifest] = _output_manifest(
        EPOCHS_OUTPUT, "features", "The epochs"
    )

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self.epochs: Any = None
        self._windows: list[FeatureBlock] = []
        self._graph: Any = None

    def bind(self, graph: Any, node_id: str) -> None:
        self._graph = graph

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock) or block.dims[:1] != ("epoch",):
            raise PortTypeError(f"{self.manifest.name} reads an epochs rail")
        self._windows.append(block)

    def close(self) -> None:
        if self._windows:
            windows, self._windows = self._windows, []
            self.epochs = epochs_array(windows, channel_types(self._graph),
                                       channel_positions(self._graph))  # fmt: skip
