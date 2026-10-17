"""Acceptance test 1 (docs/DESIGN.md §6): block-wise == one-pass.

For a stream processor, running a signal through ``transform`` in random block
sizes (state carried between blocks, and serialized and restored between blocks)
must give the same samples and events as one pass over the whole signal, to float
tolerance. For an analysis processor, it must give the same feature windows.
A node that reads rails (extra input ports) gets them cut to each block exactly
as the schedulers cut them (``meegxer.core.rails``), so the check covers rails.
Used by built-in tests now and the plugin conformance suite (M4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from meegxer.core.block import Block, Event, FeatureBlock, concat_blocks
from meegxer.core.node import Processor, RailInput, RailOutput, output_blocks
from meegxer.core.rails import cut
from meegxer.core.serialization import state_from_bytes, state_to_bytes
from meegxer.core.stage import Stage, signal_fingerprint


class ConformanceError(AssertionError):
    """A processor violates the block-wise equivalence contract."""


def random_block_sizes(
    n_samples: int, rng: np.random.Generator, max_block: int
) -> list[int]:
    """Random positive sizes summing to ``n_samples``, with single-sample blocks."""
    sizes: list[int] = []
    left = n_samples
    while left > 0:
        size = 1 if rng.random() < 0.2 else int(rng.integers(1, max_block + 1))
        size = min(size, left)
        sizes.append(size)
        left -= size
    return sizes


def _rails(
    stage: Stage, rails: Mapping[str, RailInput] | None, start: int, stop: int
) -> dict[str, RailInput] | None:
    if not stage.manifest.rail_inputs:
        return None
    if rails is None:
        raise ValueError(f"{stage.manifest.name} reads rails: pass rails=...")
    return cut(rails, stage.manifest.rail_inputs, start, stop)


def _written(
    written: dict[str, list[RailOutput]], result_rails: Mapping[str, RailOutput]
) -> None:
    for port, data in result_rails.items():
        written.setdefault(port, []).append(data)


def _same_rails(
    got: dict[str, list[RailOutput]],
    ref: dict[str, list[RailOutput]],
    rtol: float,
    atol: float,
) -> bool:
    """What the extra outputs wrote, compared as the readers would see it."""
    if set(got) != set(ref):
        return False
    for port in ref:
        a, b = _flat(got[port]), _flat(ref[port])
        if isinstance(a, Block) and isinstance(b, Block):
            if (a.sample_index, a.n_samples) != (b.sample_index, b.n_samples):
                return False
            if not np.allclose(a.data, b.data, rtol=rtol, atol=atol):
                return False
        elif isinstance(a, tuple) and isinstance(b, tuple):
            if [w.window for w in a] != [w.window for w in b]:
                return False
            for x, y in zip(a, b, strict=True):
                if not np.allclose(x.data, y.data, rtol=rtol, atol=atol,
                                   equal_nan=True):  # fmt: skip
                    return False
        else:
            return False
    return True


def _flat(parts: list[RailOutput]) -> Block | tuple[FeatureBlock, ...]:
    blocks = [p for p in parts if isinstance(p, Block) and p.n_samples]
    if blocks or all(isinstance(p, Block) for p in parts):
        return concat_blocks(blocks) if blocks else ()
    return tuple(w for p in parts if isinstance(p, tuple) for w in p)


def _run(
    stage: Stage,
    block: Block,
    sizes: list[int],
    fp: str,
    rails: Mapping[str, RailInput] | None = None,
    written: dict[str, list[RailOutput]] | None = None,
) -> tuple[Block, list[Event]]:
    outputs: list[Block] = []
    events: list[Event] = []
    written = {} if written is None else written
    start = block.sample_index
    for size in sizes:
        result = stage.transform(
            block.slice(start, start + size),
            fp,
            _rails(stage, rails, start, start + size),
        )
        if not isinstance(result.block, Block):
            raise ConformanceError("block-wise equivalence applies to Signal outputs")
        outputs.append(result.block)
        events.extend(result.events)
        _written(written, result.rails)
        if stage.state is not None:  # save/restore round trip between blocks
            stage.state = state_from_bytes(state_to_bytes(stage.state))
        start += size
    tail = stage.flush(fp)
    if tail is not None:
        if not isinstance(tail.block, Block):
            raise ConformanceError("block-wise equivalence applies to Signal outputs")
        outputs.append(tail.block)
        events.extend(tail.events)
        _written(written, tail.rails)
    return concat_blocks(outputs), events


def check_blockwise_equivalence(
    processor: Processor,
    block: Block,
    params: Mapping[str, object] | None = None,
    *,
    calibration: Block | None = None,
    events: Sequence[Event] = (),
    rails: Mapping[str, RailInput] | None = None,
    calibration_rails: Mapping[str, RailInput] | None = None,
    trials: int = 5,
    seed: int = 0,
    max_block: int | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> float:
    """Raise :class:`ConformanceError` on mismatch; return the max abs deviation.
    ``events`` label the calibration's trials for a node with ``fit_events``.
    ``rails`` hold, by extra input port, the rails over the whole ``block``
    (``calibration_rails`` over the calibration, if that is another stretch);
    what the extra outputs write must not depend on the block sizes either."""
    if processor.manifest.availability == "offline":
        raise ValueError(
            f"{processor.manifest.name} is offline-only: it always receives the "
            "whole recording, so block-wise equivalence does not apply"
        )
    if processor.manifest.kind == "analysis":
        return _check_feature_windows(
            processor, block, params, calibration=calibration, events=events,
            rails=rails, calibration_rails=calibration_rails,
            trials=trials, seed=seed, max_block=max_block, rtol=rtol, atol=atol,
        )  # fmt: skip
    fp = signal_fingerprint(block.info)
    reference_stage = Stage(processor, params)
    _fit(reference_stage, block, fp, calibration, events, rails, calibration_rails)
    ref_written: dict[str, list[RailOutput]] = {}
    reference, ref_events = _run(
        reference_stage, block, [block.n_samples], fp, rails, ref_written
    )
    if reference.n_samples != block.n_samples:
        raise ConformanceError("stream processors must return as many samples as given")

    rng = np.random.default_rng(seed)
    max_block = max_block or max(2, block.n_samples // 4)
    worst = 0.0
    for trial in range(trials):
        stage = Stage(processor, params)
        stage.model = reference_stage.model
        sizes = random_block_sizes(block.n_samples, rng, max_block)
        written: dict[str, list[RailOutput]] = {}
        out, events = _run(stage, block, sizes, fp, rails, written)
        if (out.sample_index, out.n_samples) != (
            reference.sample_index,
            reference.n_samples,
        ):
            raise ConformanceError(f"trial {trial}: output sample range differs")
        if out.info != reference.info:
            raise ConformanceError(f"trial {trial}: output channels differ")
        diff = float(np.max(np.abs(out.data - reference.data), initial=0.0))
        worst = max(worst, diff)
        if not np.allclose(out.data, reference.data, rtol=rtol, atol=atol):
            first_bad = int(
                np.argmax(
                    np.any(
                        ~np.isclose(out.data, reference.data, rtol=rtol, atol=atol),
                        axis=0,
                    )
                )
            )
            raise ConformanceError(
                f"trial {trial}: block-wise output deviates by up to {diff:.3g} "
                f"(first mismatch at sample {block.sample_index + first_bad}; "
                f"block sizes {sizes[:10]}{'...' if len(sizes) > 10 else ''})"
            )
        if events != ref_events:
            raise ConformanceError(f"trial {trial}: events differ from one pass")
        if not _same_rails(written, ref_written, rtol, atol):
            raise ConformanceError(f"trial {trial}: rails written differ from one pass")
    return worst


def _fit(
    stage: Stage,
    block: Block,
    fp: str,
    calibration: Block | None,
    events: Sequence[Event],
    rails: Mapping[str, RailInput] | None,
    calibration_rails: Mapping[str, RailInput] | None,
) -> None:
    if not stage.manifest.requires_fit:
        return
    data = calibration if calibration is not None else block
    given = calibration_rails if calibration_rails is not None else rails
    stage.fit(data, fp, events, _rails(stage, given, data.sample_index,
                                       data.stop_index))  # fmt: skip


def _run_windows(
    stage: Stage,
    block: Block,
    sizes: list[int],
    fp: str,
    rails: Mapping[str, RailInput] | None = None,
    written: dict[str, list[RailOutput]] | None = None,
) -> tuple[list[FeatureBlock], list[Event]]:
    windows: list[FeatureBlock] = []
    events: list[Event] = []
    written = {} if written is None else written
    start = block.sample_index
    for size in sizes:
        result = stage.transform(
            block.slice(start, start + size),
            fp,
            _rails(stage, rails, start, start + size),
        )
        _written(written, result.rails)
        for out in output_blocks(result.block):
            if not isinstance(out, FeatureBlock):
                raise ConformanceError("analysis processors must output Features")
            windows.append(out)
        events.extend(result.events)
        if stage.state is not None:  # save/restore round trip between blocks
            stage.state = state_from_bytes(state_to_bytes(stage.state))
        start += size
    return windows, events


def _check_feature_windows(
    processor: Processor,
    block: Block,
    params: Mapping[str, object] | None,
    *,
    calibration: Block | None,
    events: Sequence[Event],
    rails: Mapping[str, RailInput] | None,
    calibration_rails: Mapping[str, RailInput] | None,
    trials: int,
    seed: int,
    max_block: int | None,
    rtol: float,
    atol: float,
) -> float:
    """Analysis processors: the same windows (ranges, labels, values) whatever the
    block sizes."""
    fp = signal_fingerprint(block.info)
    reference_stage = Stage(processor, params)
    _fit(reference_stage, block, fp, calibration, events, rails, calibration_rails)
    ref_written: dict[str, list[RailOutput]] = {}
    reference, ref_events = _run_windows(
        reference_stage, block, [block.n_samples], fp, rails, ref_written
    )
    rng = np.random.default_rng(seed)
    max_block = max_block or max(2, block.n_samples // 4)
    worst = 0.0
    for trial in range(trials):
        stage = Stage(processor, params)
        stage.model = reference_stage.model
        sizes = random_block_sizes(block.n_samples, rng, max_block)
        written: dict[str, list[RailOutput]] = {}
        windows, events = _run_windows(stage, block, sizes, fp, rails, written)
        if [w.window for w in windows] != [w.window for w in reference]:
            raise ConformanceError(f"trial {trial}: feature windows differ")
        for got, ref in zip(windows, reference, strict=True):
            if got.dims != ref.dims or dict(got.coords) != dict(ref.coords):
                raise ConformanceError(f"trial {trial}: feature layout differs")
            if got.timestamp != ref.timestamp:
                raise ConformanceError(f"trial {trial}: window timestamps differ")
            # NaN marks a missing value (e.g. no spectral peak): same place, same
            diff = float(np.nanmax(np.abs(got.data - ref.data), initial=0.0))
            worst = max(worst, diff)
            if not np.allclose(got.data, ref.data, rtol=rtol, atol=atol,
                               equal_nan=True):  # fmt: skip
                raise ConformanceError(
                    f"trial {trial}: window {got.window} deviates by up to "
                    f"{diff:.3g} (block sizes {sizes[:10]}...)"
                )
        if events != ref_events:
            raise ConformanceError(f"trial {trial}: events differ from one pass")
        if not _same_rails(written, ref_written, rtol, atol):
            raise ConformanceError(f"trial {trial}: rails written differ from one pass")
    return worst
