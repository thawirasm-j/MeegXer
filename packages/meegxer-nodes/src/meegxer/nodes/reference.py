"""Re-referencing (common average or chosen reference channels) and picking
channels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from meegxer.core.block import CHANNEL_TYPES, Block, SignalInfo, match_channels
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest


def _labels(text: object) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


class Rereference(Processor):
    """Subtract the mean of the reference channels from every channel.

    ``mode=average`` averages all channels except those in ``exclude`` and, when
    ``exclude_bads`` is on, those marked bad in the stream (common average
    reference). ``mode=channels`` uses the comma-separated labels in ``channels``
    (e.g. ``"M1,M2"`` or ``"Cz"``); a bad reference channel is an error when
    ``exclude_bads`` is on. Every channel, bad or not, is re-referenced. Channels
    are matched by label. Stateless.
    """

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.rereference",
        "0.2.0",
        [
            {"name": "mode", "type": "enum", "default": "average", "primary": True,
             "choices": ["average", "channels"]},
            {"name": "channels", "type": "string", "default": "",
             "description": "Comma-separated reference labels (mode=channels)"},
            {"name": "exclude", "type": "string", "default": "", "advanced": True,
             "description": "Comma-separated labels left out of the average"},
            {"name": "exclude_bads", "type": "bool", "default": True,
             "advanced": True,
             "description": "Leave channels marked bad out of the reference"},
        ],
        description="Common average or channel re-reference.",
        live_params=["mode", "channels", "exclude", "exclude_bads"],
        input_space="sensor",  # electrodes only (spec 0.2.22)
    )  # fmt: skip

    @staticmethod
    def reference_channels(block: Block, params: Params) -> list[str]:
        bads = set(block.info.bads) if params["exclude_bads"] else set()
        if params["mode"] == "average":
            excluded = set(_labels(params["exclude"]))
            match_channels(block.info.ch_labels, sorted(excluded))  # labels must exist
            chosen = [
                c for c in block.info.ch_labels if c not in excluded and c not in bads
            ]
            if not chosen:
                raise ValueError("every channel is excluded from the average")
            return chosen
        chosen = _labels(params["channels"])
        if not chosen:
            raise ValueError("mode=channels needs at least one channel label")
        bad_refs = sorted(bads.intersection(chosen))
        if bad_refs:
            raise ValueError(f"reference channels are marked bad: {bad_refs}")
        return chosen

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        idx = match_channels(
            block.info.ch_labels, self.reference_channels(block, params)
        )
        reference = block.data[idx].mean(axis=0, keepdims=True)
        out = Block(
            np.asarray(block.data - reference),
            block.info,
            block.sample_index,
            block.timestamp,
        )
        return TransformResult(out, {})


class PickChannels(Processor):
    """Keep some channels, drop the rest.

    A channel is kept when its type is switched on (``keep_eeg``, ``keep_eog``,
    …; channels of unknown type count as EEG) and, when ``keep_labels`` lists
    labels, it is one of them; with ``drop_bads`` on, channels marked bad go too.
    A listed label the stream lacks, or whose type is off, is an error rather
    than a silent drop. Channels keep the stream's order. Stateless.
    """

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.pick_channels",
        "0.1.0",
        [
            {"name": "keep_labels", "type": "string", "default": "", "primary": True,
             "description": "Comma-separated labels to keep (empty: every "
                            "channel of the types switched on)"},
            *[{"name": f"keep_{t}", "type": "bool", "default": t == "eeg",
               "description": f"Keep {t.upper()} channels"} for t in CHANNEL_TYPES],
            {"name": "drop_bads", "type": "bool", "default": False,
             "description": "Drop channels marked bad"},
        ],
        description="Keep the chosen channels (by type, by name, without bad ones).",
        live_params=["keep_labels", *[f"keep_{t}" for t in CHANNEL_TYPES], "drop_bads"],
        channel_types=list(CHANNEL_TYPES),  # it sees every channel and chooses
    )  # fmt: skip

    @staticmethod
    def kept(info: SignalInfo, params: Params) -> list[str]:
        types = [t for t in CHANNEL_TYPES if params[f"keep_{t}"]]
        of_types = set(info.labels_of_types(types))
        named = _labels(params["keep_labels"])
        match_channels(info.ch_labels, named)  # every listed label must exist
        off = [label for label in named if label not in of_types]
        if off:
            raise ValueError(f"listed channels of a type switched off: {off}")
        wanted = set(named) if named else of_types
        bads = set(info.bads) if params["drop_bads"] else set()
        chosen = [c for c in info.ch_labels if c in wanted and c not in bads]
        if not chosen:
            raise ValueError("no channel is kept")
        return chosen

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block.pick(self.kept(block.info, params)), {})
