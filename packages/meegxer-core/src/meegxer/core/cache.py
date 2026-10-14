"""Per-node output cache for batch runs, keyed by stage output fingerprints.

A fingerprint covers the source (parameters and file content), every upstream
node's identity, version and parameters, and every fitted model, so changing
stage N invalidates stage N and everything downstream while earlier stages are
served from the cache. Block size is not part of the key: by the processor
contract (acceptance test 1) outputs do not depend on it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from meegxer.core.block import Block, Event, SignalInfo


@dataclass(frozen=True, slots=True)
class CachedOutput:
    block: Block
    events: tuple[Event, ...] = ()


class Cache(Protocol):
    def get(self, key: str) -> CachedOutput | None: ...

    def put(self, key: str, value: CachedOutput) -> None: ...


class MemoryCache:
    """Keeps outputs for the lifetime of the object (GUI sessions, tests)."""

    def __init__(self) -> None:
        self._items: dict[str, CachedOutput] = {}

    def get(self, key: str) -> CachedOutput | None:
        return self._items.get(key)

    def put(self, key: str, value: CachedOutput) -> None:
        self._items[key] = value

    def __len__(self) -> int:
        return len(self._items)


def _events_to_json(events: tuple[Event, ...]) -> str:
    return json.dumps(
        [
            {
                "sample_index": e.sample_index,
                "label": e.label,
                "duration": e.duration,
                "payload": dict(e.payload),
            }
            for e in events
        ]
    )


def _events_from_json(text: str) -> tuple[Event, ...]:
    return tuple(Event(**item) for item in json.loads(text))


class DirectoryCache:
    """One ``<fingerprint>.npz`` per node output under ``root`` (no pickle)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key.isalnum():
            raise ValueError("cache keys must be hex fingerprints")
        return self.root / f"{key}.npz"

    def get(self, key: str) -> CachedOutput | None:
        path = self._path(key)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as npz:
            info = SignalInfo(
                tuple(str(x) for x in npz["ch_labels"]),
                float(npz["sfreq"]),
                str(npz["stream_id"]),
            )
            block = Block(
                npz["data"], info, int(npz["sample_index"]), float(npz["timestamp"])
            )
            events = _events_from_json(str(npz["events"]))
        return CachedOutput(block, events)

    def put(self, key: str, value: CachedOutput) -> None:
        block = value.block
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                np.savez(
                    fh,
                    allow_pickle=False,
                    data=block.data,
                    ch_labels=np.array(block.info.ch_labels),
                    sfreq=np.float64(block.info.sfreq),
                    stream_id=np.array(block.info.stream_id),
                    sample_index=np.int64(block.sample_index),
                    timestamp=np.float64(block.timestamp),
                    events=np.array(_events_to_json(value.events)),
                )
            os.replace(tmp, self._path(key))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
