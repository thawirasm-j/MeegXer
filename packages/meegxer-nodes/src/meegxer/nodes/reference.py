"""Re-referencing: common average or a chosen set of reference channels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from meegxer.core.block import Block, match_channels
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
