"""A configured processor in a pipeline, with calibration status (rule 7).

Staleness is derived, not tracked by hand. Each stage has a fingerprint over its
node identity, its parameters and the fingerprint of its input; a fitted model
records the fingerprint it was fitted under. If anything upstream changes (a
parameter, a refit, the input stream) the fingerprints differ and the stage reports
``NEEDS_REFIT``, so a stale model can never be used silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from meegxer.core.block import Block, SignalInfo
from meegxer.core.manifest import ParamValue
from meegxer.core.node import Processor, TransformResult
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
        self, processor: Processor, params: Mapping[str, object] | None = None
    ) -> None:
        self.processor = processor
        self.manifest = processor.manifest
        self.params: dict[str, ParamValue] = self.manifest.resolve_params(params)
        self.model: FittedModel | None = None
        self.state: State | None = None
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
        check_state(result.state)
        self.state = result.state
        return result


def chain_input_fingerprints(stages: Sequence[Stage], source_fp: str) -> list[str]:
    """Input fingerprint of each stage in a linear chain fed by ``source_fp``."""
    fps: list[str] = []
    current = source_fp
    for stage in stages:
        fps.append(current)
        current = stage.output_fingerprint(current)
    return fps
