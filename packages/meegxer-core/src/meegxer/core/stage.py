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
)
from meegxer.core.manifest import ParamValue
from meegxer.core.node import Mode, Processor, TransformResult
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
        """What a model fitted for this stage depends on."""
        return fingerprint(
            "stage",
            self.manifest.name,
            self.manifest.version,
            canonical_json(self.params),
            ",".join(self.channel_types),
            input_fp,
        )

    def output_fingerprint(self, input_fp: str) -> str:
        """What the next stage's input depends on (includes this stage's model)."""
        own = self.fingerprint(input_fp)
        if not self.manifest.requires_fit:
            return own
        model_hash = self.model.content_hash if self.model else "unfitted"
        return fingerprint("output", own, model_hash)

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
        selected = self._selected(block)
        if selected is None:  # no channels of the processed types: pass through
            return TransformResult(full, self.state or {})
        block = selected
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
            check_block(port, result.block)
        out = result.block
        if block.bypassed and isinstance(out, Block) and out.bypassed != block.bypassed:
            merged = tuple(dict.fromkeys((*block.bypassed, *out.bypassed)))
            result = result._replace(block=dataclasses.replace(out, bypassed=merged))
        check_state(result.state)
        self.state = result.state
        if block is not full and isinstance(result.block, Block):
            joined = _merge(full, block.info.ch_labels, result.block)
            result = result._replace(block=joined)
        return result

    def _selected(self, block: Block) -> Block | None:
        """The channels this stage processes: the block itself when that is all of
        it, None when there are none."""
        labels = block.info.labels_of_types(self.channel_types)
        if labels == block.info.ch_labels:
            return block
        return block.pick(labels) if labels else None


def _merge(full: Block, selected: tuple[str, ...], processed: Block) -> Block:
    """Put a stage's output back among the channels it did not process. Same
    channels out as in: each goes back to its place. Otherwise (a node that adds
    or drops channels) the output comes first, then the untouched channels."""
    if processed.n_samples != full.n_samples:
        raise ValueError("a stage that skips channels must keep the block length")
    picked = processed.info.ch_labels
    rest = [label for label in full.info.ch_labels if label not in selected]
    others = full.pick(rest) if rest else None
    if picked == selected and others is not None:
        data = np.array(full.data)
        index = {label: i for i, label in enumerate(full.info.ch_labels)}
        data[[index[label] for label in picked]] = processed.data
        bads = tuple(
            label
            for label in full.info.ch_labels
            if label in (processed.info.bads if label in picked else full.info.bads)
        )
        info = dataclasses.replace(full.info, bads=bads)
        return dataclasses.replace(
            full, data=data, info=info, bypassed=processed.bypassed
        )
    if others is None:
        return processed
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
        full.sample_index,
        full.timestamp,
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
