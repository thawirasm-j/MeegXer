"""Detectors: strips that propose events, from published methods in MNE only
(docs/REVIEW.md, principle 6). Each passes the signal on unchanged and emits
its findings on the Events rail, where a Review strip can take them as
suggestions and a Reject segments strip can cut the ``BAD_…`` spans.

- Jumps and flat signal: MNE ``annotate_amplitude`` (``BAD_peak``,
  ``BAD_flat``), on the change between consecutive samples (what MNE calls
  peak-to-peak). Channels bad for much of the recording are proposed as
  ``bad_channel`` events, not applied.
- Muscle: MNE ``annotate_muscle_zscore`` (``BAD_muscle``).
- Blinks: MNE ``find_eog_events``, each peak as a short span (``blink``).

All three run offline, on the whole recording (each piece on its own after a
cut). A piece of one sample gives nothing, nor does one shorter than 10 s
for blinks (MNE's EOG filter is that long).

The methods are MNE's. What this module adds on top are MeegXer settings, not
part of the published methods, and are documented as such: a shortest span of
its own for flat signal (MNE takes one for both limits, so MNE is called once
per limit), a span either side of each blink's peak (MNE gives the peak only),
and skipping pieces shorter than MNE's 10 s EOG filter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.review import BAD_CHANNEL  # a channel proposed as bad
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest

UV = 1e-6  # thresholds are set in µV; signals are in volts
BLINK_MIN_SECONDS = 10.0  # MNE's EOG filter is 10 s long: shorter pieces skipped
_EOG_LABEL = re.compile(r"^(V|H)?EOG", re.IGNORECASE)


def _raw(block: Block, labels: Sequence[str], types: Sequence[str]) -> Any:
    """An MNE Raw of ``labels`` (typed ``types``) from ``block``."""
    import mne

    info = block.info
    idx = [info.ch_labels.index(lb) for lb in labels]
    raw_info = mne.create_info(list(labels), info.sfreq, list(types))
    return mne.io.RawArray(np.array(block.data[idx], dtype=np.float64), raw_info,
                           verbose="error")  # fmt: skip


def _good(info: SignalInfo) -> list[str]:
    return [lb for lb in info.ch_labels if lb not in info.bads]


def _spans(block: Block, annotations: Any) -> tuple[Event, ...]:
    """MNE annotations (seconds from the block's start) as events on the
    block's samples, clipped to it."""
    sfreq, lo, n = block.info.sfreq, block.sample_index, block.n_samples
    out = []
    found = zip(
        annotations.onset, annotations.duration, annotations.description, strict=True
    )
    for onset, duration, label in found:
        start = min(max(round(onset * sfreq), 0), n)
        stop = min(max(round((onset + duration) * sfreq), start), n)
        if stop > start:
            out.append(Event(lo + start, str(label), stop - start))
    return tuple(out)


class AmplitudeDetector(Processor):
    """Mark spans where the signal jumps or stays flat (MNE's peak-to-peak:
    the change between two consecutive samples)."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.detect_amplitude",
        "0.1.0",
        [
            {"name": "peak", "type": "float", "default": 100.0, "min": 0.0,
             "unit": "µV", "primary": True,
             "description": "Change between two consecutive samples above "
                            "which a span is BAD_peak (0: off)"},
            {"name": "flat", "type": "float", "default": 0.1, "min": 0.0,
             "unit": "µV", "primary": True,
             "description": "Change between consecutive samples below which a "
                            "span is BAD_flat, once it lasts flat_duration "
                            "(0: off)"},
            {"name": "flat_duration", "type": "float", "default": 0.1,
             "min": 0.0, "unit": "s",
             "description": "Shortest flat span counted (a MeegXer setting: "
                            "MNE takes one shortest span for both limits)"},
            {"name": "bad_percent", "type": "float", "default": 5.0, "min": 0.0,
             "max": 100.0, "unit": "%",
             "description": "A channel past a limit for more than this share "
                            "of the recording is proposed as bad instead of "
                            "marking spans"},
            {"name": "min_duration", "type": "float", "default": 0.005,
             "min": 0.0, "unit": "s", "advanced": True,
             "description": "Shortest BAD_peak span counted"},
        ],
        description="Mark spans where the signal jumps or is flat (BAD_peak, "
        "BAD_flat; MNE annotate_amplitude) and propose channels bad for much of "
        "the recording.",
        availability="offline",
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        from mne.preprocessing import annotate_amplitude

        limits = {
            "peak": (float(params["peak"]), float(params["min_duration"])),
            "flat": (float(params["flat"]), float(params["flat_duration"])),
        }
        if not any(value for value, _ in limits.values()):
            raise ValueError("set a peak or a flat limit (both are 0)")
        labels = _good(block.info)
        if not labels or block.n_samples < 2:
            return TransformResult(block, {})
        raw = _raw(block, labels, ["eeg"] * len(labels))
        proposed: dict[str, str] = {}  # channel -> reason, first reason kept
        events: list[Event] = []
        # one call per limit: each has its own shortest span (MNE takes one)
        for reason, (value, duration) in limits.items():
            if not value:
                continue
            annotations, bads = annotate_amplitude(
                raw, **{reason: value * UV},
                bad_percent=float(params["bad_percent"]),
                min_duration=min(duration, raw.times[-1]), picks="all",
                verbose="error",
            )  # fmt: skip
            events.extend(_spans(block, annotations))
            for ch in bads:
                proposed.setdefault(ch, reason)
        channels = tuple(
            Event(block.sample_index, BAD_CHANNEL,
                  payload={"channel": ch, "reason": reason})
            for ch, reason in sorted(proposed.items(),
                                     key=lambda kv: labels.index(kv[0]))
        )  # fmt: skip
        events.sort(key=lambda e: e.sample_index)
        return TransformResult(block, {}, (*channels, *events))


class MuscleDetector(Processor):
    """Mark spans with muscle activity (z-scored high-frequency envelope)."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.detect_muscle",
        "0.1.0",
        [
            {"name": "zscore", "type": "float", "default": 4.0, "min": 0.0,
             "primary": True,
             "description": "z-score of the band's envelope above which a span "
                            "is BAD_muscle"},
            {"name": "low", "type": "float", "default": 110.0, "min": 0.0,
             "unit": "Hz", "description": "Lower edge of the muscle band"},
            {"name": "high", "type": "float", "default": 140.0, "min": 0.0,
             "unit": "Hz",
             "description": "Upper edge of the muscle band (below half the "
                            "sampling rate)"},
            {"name": "min_length_good", "type": "float", "default": 0.1,
             "min": 0.0, "unit": "s", "advanced": True,
             "description": "Good stretches shorter than this between two "
                            "marked spans are marked too"},
        ],
        description="Mark spans with muscle activity (BAD_muscle; MNE "
        "annotate_muscle_zscore).",
        availability="offline",
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        from mne.preprocessing import annotate_muscle_zscore

        low, high = float(params["low"]), float(params["high"])
        nyquist = block.info.sfreq / 2
        if not 0 < low < high < nyquist:
            raise ValueError(
                f"the muscle band {low:g}–{high:g} Hz must lie below half the "  # noqa: RUF001
                f"sampling rate ({nyquist:g} Hz): lower low and high"
            )
        labels = _good(block.info)
        if not labels or block.n_samples < 2:
            return TransformResult(block, {})
        raw = _raw(block, labels, ["eeg"] * len(labels))
        annotations, _ = annotate_muscle_zscore(
            raw, threshold=float(params["zscore"]), ch_type="eeg",
            min_length_good=float(params["min_length_good"]),
            filter_freq=(low, high), verbose="error",
        )  # fmt: skip
        return TransformResult(block, {}, _spans(block, annotations))


def eog_channels(info: SignalInfo, eog: str) -> list[str]:
    """The EOG channels for the ``eog`` setting: ``auto`` (good channels typed
    or named EOG) or a comma-separated list of labels."""
    types = info.ch_types or ("",) * len(info.ch_labels)
    if eog.strip().lower() == "auto":
        found = [lb for lb, t in zip(info.ch_labels, types, strict=True)
                 if lb not in info.bads
                 and (t.lower() == "eog" or _EOG_LABEL.match(lb))]  # fmt: skip
    else:
        found = [lb.strip() for lb in eog.split(",") if lb.strip()]
        missing = [lb for lb in found if lb not in info.ch_labels]
        if missing:
            raise ValueError(f"no channel named {', '.join(missing)}")
    if not found:
        raise ValueError("no EOG channel found: name one in the eog setting "
                         "(e.g. Fp1), or mark its type as EOG (Recording "
                         "window)")  # fmt: skip
    return found


class BlinkDetector(Processor):
    """Mark blinks found on the EOG channels, each as a short span."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.detect_blinks",
        "0.1.0",
        [
            {"name": "eog", "type": "string", "default": "auto", "primary": True,
             "description": "EOG channels: auto (typed or named EOG) or labels, "
                            "comma-separated (e.g. Fp1)"},
            {"name": "mark_label", "type": "string", "default": "blink",
             "description": "Label of the blink spans (BAD_blink: a Reject "
                            "segments strip cuts them)"},
            {"name": "half_width", "type": "float", "default": 0.25, "min": 0.0,
             "max": 5.0, "unit": "s",
             "description": "How far each span reaches either side of the "
                            "blink's peak (a MeegXer setting: MNE gives the "
                            "peak only)"},
            {"name": "blink_threshold", "type": "float", "default": 0.0, "min": 0.0,
             "unit": "µV", "advanced": True,
             "description": "Peak amplitude of a blink on the filtered EOG "
                            "(0: MNE's, a quarter of its range)"},
            {"name": "low", "type": "float", "default": 1.0, "min": 0.0,
             "unit": "Hz", "advanced": True,
             "description": "Lower edge of the band the EOG is filtered to"},
            {"name": "high", "type": "float", "default": 10.0, "min": 0.0,
             "unit": "Hz", "advanced": True,
             "description": "Upper edge of the band the EOG is filtered to"},
        ],
        description="Mark blinks on the EOG channels as short spans (MNE "
        "find_eog_events).",
        availability="offline",
        channel_types=["eeg", "eog"],  # it needs to see the EOG channels
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        from mne.preprocessing import find_eog_events

        label = str(params["mark_label"]).strip()
        if not label:
            raise ValueError("the blink label is empty")
        eog = eog_channels(block.info, str(params["eog"]))
        if block.n_samples < BLINK_MIN_SECONDS * block.info.sfreq:
            return TransformResult(block, {})  # too short for MNE's EOG filter
        raw = _raw(block, eog, ["eog"] * len(eog))
        threshold = float(params["blink_threshold"]) * UV or None
        found = find_eog_events(
            raw, l_freq=float(params["low"]), h_freq=float(params["high"]),
            ch_name=eog, thresh=threshold, verbose="error",
        )  # fmt: skip
        half = round(float(params["half_width"]) * block.info.sfreq)
        n, lo = block.n_samples, block.sample_index
        events = []
        for peak in sorted(int(s) - raw.first_samp for s in found[:, 0]):
            start, stop = max(peak - half, 0), min(peak + half + 1, n)
            if stop > start:
                events.append(Event(lo + start, label, stop - start,
                                    payload={"peak": lo + peak}))  # fmt: skip
        return TransformResult(block, {}, tuple(events))
