"""A configured processor in a pipeline, with calibration status (rule 7).

Staleness is derived, not tracked by hand. Each stage has a fingerprint over its
node identity, its parameters and the fingerprint of its input; a fitted model
records the fingerprint it was fitted under. If anything upstream changes (a
parameter, a refit, the input stream) the fingerprints differ and the stage reports
``NEEDS_REFIT``, so a stale model can never be used silently.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from meegxer.core.block import (
    CHANNEL_TYPES,
    DEFAULT_PROCESSED_TYPES,
    Block,
    SignalInfo,
    concat_blocks,
)
from meegxer.core.manifest import ParamValue
from meegxer.core.node import Mode, Processor, TransformResult, output_blocks
from meegxer.core.ports import check_block
from meegxer.core.serialization import (
    State,
    canonical_json,
    check_state,
    fingerprint,
    hash_state,
)


class StageStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATING = "calibrating"
    READY = "ready"
    NEEDS_REFIT = "needs-refit"


class StaleModelError(RuntimeError):
    """Transform was attempted without a current model for this input."""


def signal_fingerprint(info: SignalInfo) -> str:
    """Fingerprint of a source stream: what a pipeline's first stage depends on."""
    return fingerprint(
        "signal",
        info.stream_id,
        repr(float(info.sfreq)),
        ",".join(info.ch_labels),
        ",".join(info.bads),
    )


@dataclass(frozen=True, slots=True)
class FittedModel:
    data: State
    fingerprint: str
    content_hash: str
    input_info: SignalInfo


class Stage:
    """One processor plus its parameters, model and streaming state."""

    def __init__(
        self,
        processor: Processor,
        params: Mapping[str, object] | None = None,
        *,
        pass_uncalibrated: bool = False,
        channel_types: Sequence[str] = DEFAULT_PROCESSED_TYPES,
    ) -> None:
        self.processor = processor
        self.manifest = processor.manifest
        self.params: dict[str, ParamValue] = self.manifest.resolve_params(params)
        unknown = sorted(set(channel_types) - set(CHANNEL_TYPES))
        if unknown or not channel_types:
            raise ValueError(f"channel_types must name some of {CHANNEL_TYPES}")
        # which channels this stage processes; the others pass through untouched
        self.channel_types = tuple(dict.fromkeys(channel_types))
        self._rest: list[Block] = []  # untouched channels not yet output (delays)
        self._full_info: SignalInfo | None = None
        self._last_info: SignalInfo | None = None  # what the processor last saw
        self.model: FittedModel | None = None
        self.state: State | None = None
        self._calibrating = False
        self.pass_uncalibrated = False
        self.set_pass_uncalibrated(pass_uncalibrated)

    def for_mode(self, mode: Mode) -> Stage:
        """This stage, or a stage running the processor's variant for ``mode``
        (parameters the variant does not know, such as ``phase``, are dropped)."""
        variant = self.processor.for_mode(self.params, mode)
        if variant is self.processor:
            return self
        if self.manifest.requires_fit or variant.manifest.requires_fit:
            raise ValueError(
                f"{self.manifest.name}: adaptable nodes cannot need fitting"
            )
        known = {p.name for p in variant.manifest.params}
        return Stage(
            variant,
            {k: v for k, v in self.params.items() if k in known},
            channel_types=self.channel_types,
        )

    def set_pass_uncalibrated(self, enabled: bool) -> None:
        """Opt in to passing data through unprocessed while there is no valid
        model (uncalibrated, calibrating or needs-refit). The default blocks."""
        if enabled and not self.manifest.requires_fit:
            raise ValueError(
                f"{self.manifest.name} does not need calibration, so it has no "
                "uncalibrated state to pass through"
            )
        self.pass_uncalibrated = bool(enabled)

    def clear_model(self) -> None:
        """Explicitly drop the fitted model (models are otherwise kept)."""
        self.model = None
        self.state = None
        self._calibrating = False

    def set_params(self, **updates: object) -> None:
        """Change parameters.

        Streaming state restarts on the next block unless every changed parameter
        is listed in the manifest's ``live_params``.
        """
        new = self.manifest.resolve_params({**self.params, **updates})
        changed = {name for name, value in new.items() if self.params[name] != value}
        self.params = new
        if not changed <= self.manifest.live_params:
            self.state = None

    def fingerprint(self, input_fp: str) -> str:
        """What a model fitted for this stage depends on.

        A fitted node's live parameters act in ``transform`` only (manifest
        ``live_params``), so they are left out: changing one needs no refit.
        """
        params = self.params
        if self.manifest.requires_fit:
            params = {
                k: v for k, v in params.items() if k not in self.manifest.live_params
            }
        return fingerprint(
            "stage",
            self.manifest.name,
            self.manifest.version,
            canonical_json(params),
            ",".join(self.channel_types),
            input_fp,
        )

    def output_fingerprint(self, input_fp: str) -> str:
        """What the next stage's input depends on (includes this stage's model)."""
        own = self.fingerprint(input_fp)
        if not self.manifest.requires_fit:
            return own
        model_hash = self.model.content_hash if self.model else "unfitted"
        live = {k: v for k, v in self.params.items() if k in self.manifest.live_params}
        return fingerprint("output", own, model_hash, canonical_json(live))

    def status(self, input_fp: str) -> StageStatus:
        if not self.manifest.requires_fit:
            return StageStatus.READY
        if self._calibrating:
            return StageStatus.CALIBRATING
        if self.model is None:
            return StageStatus.UNCALIBRATED
        if self.model.fingerprint != self.fingerprint(input_fp):
            return StageStatus.NEEDS_REFIT
        return StageStatus.READY

    def begin_calibration(self) -> None:
        """Mark the stage as collecting calibration data (online flow)."""
        if not self.manifest.requires_fit:
            raise RuntimeError(f"{self.manifest.name} does not require fitting")
        self._calibrating = True

    def fit(self, calibration: Block, input_fp: str) -> FittedModel:
        """Fit on ``calibration`` for the input identified by ``input_fp``."""
        if not self.manifest.requires_fit:
            raise RuntimeError(f"{self.manifest.name} does not require fitting")
        selected = self._selected(calibration)
        if selected is None:
            raise ValueError(
                f"{self.manifest.name}: no channels of types {self.channel_types} "
                "to fit on"
            )
        calibration = selected
        try:
            data = self.processor.fit(calibration, self.params)
            check_state(data)
        finally:
            self._calibrating = False
        self.model = FittedModel(
            data=data,
            fingerprint=self.fingerprint(input_fp),
            content_hash=hash_state(data),
            input_info=calibration.info,
        )
        self.state = None
        return self.model

    def transform(self, block: Block, input_fp: str) -> TransformResult:
        """Process one block; refuses to run on a missing or stale model."""
        status = self.status(input_fp)
        if status is not StageStatus.READY:
            raise StaleModelError(f"{self.manifest.name} is {status}, not ready")
        full = block
        if self.state is None:  # a new stream: forget what an earlier one left
            self._rest = []
        selected = self._selected(block)
        if selected is None:  # no channels of the processed types: pass through
            if self.manifest.kind == "analysis":
                return TransformResult((), self.state or {})  # nothing to analyse
            return TransformResult(full, self.state or {})
        block = selected
        self._last_info = block.info
        model: State = {}
        if self.model is not None:
            if block.info.ch_labels != self.model.input_info.ch_labels or (
                block.info.sfreq != self.model.input_info.sfreq
            ):
                raise StaleModelError(
                    f"{self.manifest.name}: block channels or rate differ from the "
                    "calibration data"
                )
            model = self.model.data
        result = self.processor.transform(block, model, self.state, self.params)
        for port in self.manifest.outputs[:1]:
            for out_block in output_blocks(result.block):
                check_block(port, out_block)
        out = result.block
        if block.bypassed and isinstance(out, Block) and out.bypassed != block.bypassed:
            merged = tuple(dict.fromkeys((*block.bypassed, *out.bypassed)))
            result = result._replace(block=dataclasses.replace(out, bypassed=merged))
        check_state(result.state)
        self.state = result.state
        if block is not full and isinstance(result.block, Block):
            rest = [lb for lb in full.info.ch_labels if lb not in block.info.ch_labels]
            self._rest.append(full.pick(rest))
            self._full_info = full.info
            joined = self._join(block.info.ch_labels, result.block)
            result = result._replace(block=joined)
        return result

    def flush(self, input_fp: str) -> TransformResult | None:
        """End of stream: what a delaying processor still holds (its output for the
        last samples), merged with the channels it did not process; None when it
        holds nothing."""
        if self._last_info is None:
            return None
        model: State = self.model.data if self.model is not None else {}
        result = self.processor.flush(self._last_info, model, self.state, self.params)
        if result is None:
            return None
        check_state(result.state)
        self.state = result.state
        if self._rest and self._full_info and isinstance(result.block, Block):
            labels = self._full_info.labels_of_types(self.channel_types)
            result = result._replace(block=self._join(labels, result.block))
        return result

    def _join(self, selected: tuple[str, ...], processed: Block) -> Block:
        """Merge processed channels with the untouched ones over the samples the
        processor returned (which may lag its input: a delaying node)."""
        rest = concat_blocks(self._rest)
        lo, hi = processed.sample_index, processed.stop_index
        if not rest.sample_index <= lo <= hi <= rest.stop_index:
            raise ValueError(
                f"{self.manifest.name}: output [{lo}, {hi}) is outside the input seen "
                f"[{rest.sample_index}, {rest.stop_index})"
            )
        self._rest = [rest.slice(hi, rest.stop_index)]
        assert self._full_info is not None
        return _merge(self._full_info, selected, processed, rest.slice(lo, hi))

    def _selected(self, block: Block) -> Block | None:
        """The channels this stage processes: the block itself when that is all of
        it, None when there are none."""
        labels = block.info.labels_of_types(self.channel_types)
        if labels == block.info.ch_labels:
            return block
        return block.pick(labels) if labels else None


def _merge(
    full: SignalInfo, selected: tuple[str, ...], processed: Block, others: Block
) -> Block:
    """Put a stage's output back among the channels it did not process (``others``,
    covering the same samples). Same channels out as in: each goes back to its
    place. Otherwise (a node that adds or drops channels) the output comes first,
    then the untouched channels."""
    picked = processed.info.ch_labels
    if picked == selected:
        index = {label: i for i, label in enumerate(full.ch_labels)}
        data = np.empty((full.n_channels, processed.n_samples))
        data[[index[label] for label in picked]] = processed.data
        data[[index[label] for label in others.info.ch_labels]] = others.data
        bads = tuple(
            label
            for label in full.ch_labels
            if label in (processed.info.bads if label in picked else others.info.bads)
        )
        info = dataclasses.replace(full, bads=bads)
        return Block(
            data, info, processed.sample_index, processed.timestamp, processed.bypassed
        )
    a, b = processed.info, others.info
    info = SignalInfo(
        a.ch_labels + b.ch_labels,
        a.sfreq,
        a.stream_id,
        a.bads + b.bads,
        (a.ch_types or ("",) * a.n_channels) + (b.ch_types or ("",) * b.n_channels)
        if a.ch_types or b.ch_types
        else (),
        (a.positions or (None,) * a.n_channels)
        + (b.positions or (None,) * b.n_channels)
        if a.positions or b.positions
        else (),
    )
    return Block(
        np.vstack([processed.data, others.data]),
        info,
        processed.sample_index,
        processed.timestamp,
        processed.bypassed,
    )


def chain_input_fingerprints(stages: Sequence[Stage], source_fp: str) -> list[str]:
    """Input fingerprint of each stage in a linear chain fed by ``source_fp``."""
    fps: list[str] = []
    current = source_fp
    for stage in stages:
        fps.append(current)
        current = stage.output_fingerprint(current)
    return fps
