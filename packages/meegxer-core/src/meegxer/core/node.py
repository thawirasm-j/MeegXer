"""Node roles and the processor contract (CLAUDE.md rules 6 and 7).

A processor object is stateless: everything that changes while streaming lives in
``state`` and everything learned from calibration lives in ``model``. That is what
lets the real-time and batch schedulers share one implementation (rule 4) and lets
sessions be saved and resumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, NamedTuple

from meegxer.core.block import Block, Event, EventBlock, FeatureBlock, SignalInfo
from meegxer.core.manifest import Manifest, ParamValue
from meegxer.core.serialization import State

type Params = Mapping[str, ParamValue]
type AnyBlock = Block | FeatureBlock | EventBlock


class NodeRole(StrEnum):
    SOURCE = "source"
    PROCESSOR = "processor"
    PROBE = "probe"
    SINK = "sink"


class TransformResult(NamedTuple):
    block: Block | FeatureBlock
    state: State
    events: tuple[Event, ...] = ()


class Processor(ABC):
    """Base class for every processing node, built-in or plugin adapter.

    Contract:

    - ``fit(calibration, params) -> model``: learn from calibration data. Only
      called when ``manifest.requires_fit`` is true.
    - ``transform(block, model, state, params) -> (block, state, events)``: process
      one block. ``state`` is ``None`` on the first block of a stream (and after a
      parameter change or refit); the processor initialises it then. Model and state
      must be serializable (see :mod:`meegxer.core.serialization`).
    - Never modify the input block (its data is read-only) and keep no per-stream
      data on ``self``.
    """

    role: ClassVar[NodeRole] = NodeRole.PROCESSOR
    manifest: ClassVar[Manifest]

    def fit(self, calibration: Block, params: Params) -> State:
        raise NotImplementedError(f"{self.manifest.name} does not implement fit")

    @abstractmethod
    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult: ...


class Source(ABC):
    """Produces Signal blocks. Shape is provisional until the M2 schedulers."""

    role: ClassVar[NodeRole] = NodeRole.SOURCE

    @property
    @abstractmethod
    def info(self) -> SignalInfo: ...

    @abstractmethod
    def read(self, max_samples: int) -> Block | None:
        """Return the next block, an empty block if none is ready, None at end."""


class Probe(ABC):
    """Non-modifying tap. Shape is provisional until the M2 schedulers."""

    role: ClassVar[NodeRole] = NodeRole.PROBE

    @abstractmethod
    def tap(self, block: AnyBlock) -> None: ...


class Sink(ABC):
    """Consumes blocks (files, LSL outlets, action sinks). Provisional until M2."""

    role: ClassVar[NodeRole] = NodeRole.SINK

    @abstractmethod
    def write(self, block: AnyBlock) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Flush and release resources."""
