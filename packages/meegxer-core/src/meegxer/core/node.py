"""Node roles and the processor contract (CLAUDE.md rules 6 and 7).

A processor object is stateless: everything that changes while streaming lives in
``state`` and everything learned from calibration lives in ``model``. That is what
lets the real-time and batch schedulers share one implementation (rule 4) and lets
sessions be saved and resumed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NamedTuple

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


type Output = Block | FeatureBlock | tuple[FeatureBlock, ...]
# what an extra input port receives (docs/RAILS.md, spec 0.2.17): a Signal rail's
# block over the same samples as the main block, the feature windows that ended
# inside it, or the Events rail's events inside it
type RailInput = Block | tuple[FeatureBlock, ...] | tuple[Event, ...]
# what an extra output port writes: a Signal block, or completed feature windows
type RailOutput = Block | tuple[FeatureBlock, ...]


class TransformResult(NamedTuple):
    """``block`` is the output: a Signal block, or for an analysis processor the
    feature windows completed by this call (a tuple, possibly empty). ``rails``
    holds what the extra output ports write, by port name."""

    block: Output
    state: State
    events: tuple[Event, ...] = ()
    rails: Mapping[str, RailOutput] = MappingProxyType({})


def output_blocks(output: Output) -> tuple[Block | FeatureBlock, ...]:
    """A result's output as a tuple of blocks (feature windows one by one)."""
    return output if isinstance(output, tuple) else (output,)


class Processor(ABC):
    """Base class for every processing node, built-in or plugin adapter.

    Contract:

    - ``fit(calibration, params) -> model``: learn from calibration data. Only
      called when ``manifest.requires_fit`` is true. A node that learns from
      labelled trials (manifest ``fit_events``) implements
      ``fit_labelled(calibration, events, params)`` instead: ``events`` are the
      events inside the calibration stretch, at absolute sample indices.
    - Rails (spec 0.2.17): a node with extra input ports (the rails it reads)
      implements ``transform_rails(block, model, state, params, rails)`` and, if it
      fits, ``fit_rails(calibration, params, rails)``; ``rails`` maps each extra
      input's name to its data over the same samples (``RailInput``). Extra output
      ports are written through ``TransformResult.rails``.
    - ``transform(block, model, state, params) -> (block, state, events)``: process
      one block. ``state`` is ``None`` on the first block of a stream (and after a
      parameter change or refit); the processor initialises it then. Model and state
      must be serializable (see :mod:`meegxer.core.serialization`).
    - Never modify the input block (its data is read-only) and keep no per-stream
      data on ``self``.
    - An analysis processor buffers its input in ``state`` and returns every
      feature window the block completes, as a tuple of FeatureBlocks (none when
      no window is complete yet), so its windows never depend on block sizes.
    - Output samples are aligned by sample index (rule 8). A processor that needs
      look-ahead may return output that lags its input (an earlier
      ``sample_index``, possibly fewer samples, even none) and hand over the rest
      in ``flush`` at the end of the stream. The schedulers drop empty results,
      so a processor is never handed a block with zero samples (spec 0.2.8).
    """

    role: ClassVar[NodeRole] = NodeRole.PROCESSOR
    manifest: ClassVar[Manifest]

    def fit(self, calibration: Block, params: Params) -> State:
        raise NotImplementedError(f"{self.manifest.name} does not implement fit")

    def fit_labelled(
        self, calibration: Block, events: Sequence[Event], params: Params
    ) -> State:
        """Fit with the calibration stretch's events (manifest ``fit_events``).
        Default: the events are not needed, ``fit``."""
        return self.fit(calibration, params)

    def fit_rails(
        self, calibration: Block, params: Params, rails: Mapping[str, RailInput]
    ) -> State:
        """Fit with the rails the extra input ports read, over the calibration
        stretch. Default: the rails are not needed, ``fit``."""
        return self.fit(calibration, params)

    @abstractmethod
    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult: ...

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        """Transform with the rails the extra input ports read, over the block's
        samples. Default: the rails are not needed, ``transform``."""
        return self.transform(block, model, state, params)

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        """End of stream (batch runs). A processor that delays its output (its
        result starts before its input, by ``sample_index``) returns here what it
        still holds for the last samples; ``info`` describes the stream as its last
        block did. Default: nothing held, None."""
        return None

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        """The implementation to run in ``mode``. Adaptable nodes return a variant
        (e.g. the zero-phase filter offline) and raise ``ValueError`` when ``params``
        cannot run in that mode. Default: this processor in every mode."""
        return self

    @classmethod  # noqa: B027 - optional hook
    def warm_up(cls) -> None:
        """Load slow dependencies ahead of use (the daemon calls this in the
        background at startup), so the first block does not stall the real-time
        thread on an import. Default: nothing to load."""

    @classmethod
    def provenance(cls) -> dict[str, Any]:
        """Where this implementation comes from, recorded in sessions: for an
        external plugin its image (``image``, ``image_id``, ``repo_digests``) or
        local command. Built-in nodes: empty (the package version says it all)."""
        return {}

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

        Part of the source fingerprint, so caches notice a changed file. A source
        with a content id is identified by its content, not by its parameters (a
        moved file is the same data).
        """
        return ""

    def input_files(self) -> list[Path]:
        """The files this source reads (recorded with their hashes in sessions)."""
        return []

    def new_events(self) -> tuple[Event, ...]:
        """Live events the input carries that arrived since the last call (LSL
        markers), already placed on this source's sample index (rule 8). A
        recording's events come from ``recording_events`` instead."""
        return ()

    @property
    def event_wait(self) -> float:
        """Seconds a labelled calibration waits after its last sample for live
        events still on their way (0: none can arrive late)."""
        return 0.0


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

    def bind(self, graph: Any, node_id: str) -> None:
        """Batch runs call this before running, with the graph the sink is in and
        its id: for writers that need the input's identity (BIDS derivatives)."""

    def finish(self) -> None:
        """Batch runs call this once every node has run (after ``close``): for
        writers that record the whole run (e.g. its session file)."""

    def close(self) -> None:
        """Flush and release resources."""
