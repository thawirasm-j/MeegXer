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
from typing import ClassVar, Literal, NamedTuple

from meegxer.core.block import Block, Event, EventBlock, FeatureBlock, SignalInfo
from meegxer.core.manifest import Manifest, ParamValue
from meegxer.core.serialization import State

Mode = Literal["online", "offline"]

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

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        """The implementation to run in ``mode``. Adaptable nodes return a variant
        (e.g. the zero-phase filter offline) and raise ``ValueError`` when ``params``
        cannot run in that mode. Default: this processor in every mode."""
        return self

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release external resources (e.g. a plugin worker). May be called more
        than once; a processor must work again after ``close`` if used again."""


class _ConfiguredNode(ABC):
    """Sources, probes and sinks are configured once from resolved parameters."""

    manifest: ClassVar[Manifest]

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        self.params: dict[str, ParamValue] = self.manifest.resolve_params(params)


class Source(_ConfiguredNode):
    """Produces Signal blocks from the start of a stream."""

    role: ClassVar[NodeRole] = NodeRole.SOURCE

    @property
    @abstractmethod
    def info(self) -> SignalInfo: ...

    @property
    def n_samples(self) -> int | None:
        """Total length if known (files, finite synthetic runs), else None."""
        return None

    @abstractmethod
    def read(self, max_samples: int) -> Block | None:
        """Return up to ``max_samples`` new samples, an empty block if none are
        ready yet, or None at the end of the stream."""

    def content_id(self) -> str:
        """Identity of the data beyond the parameters (e.g. a file content hash).

        Part of the source fingerprint, so caches notice a changed file.
        """
        return ""


class Probe(_ConfiguredNode):
    """Non-modifying tap: sees every block of its input, changes nothing."""

    role: ClassVar[NodeRole] = NodeRole.PROBE

    @abstractmethod
    def tap(self, block: AnyBlock) -> None: ...


class Sink(_ConfiguredNode):
    """Consumes blocks (files, LSL outlets, action sinks)."""

    role: ClassVar[NodeRole] = NodeRole.SINK

    @abstractmethod
    def write(self, block: AnyBlock) -> None: ...

    def close(self) -> None:
        """Flush and release resources."""
