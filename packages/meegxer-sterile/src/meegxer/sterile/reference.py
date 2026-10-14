"""Re-referencing: common average or a chosen set of reference channels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from meegxer.core.block import Block, match_channels
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.sterile._manifest import stream_manifest


class Rereference(Processor):
    """Subtract the mean of the reference channels from every channel.

    ``mode=average`` uses all channels (common average reference); ``mode=channels``
    uses the comma-separated labels in ``channels`` (e.g. ``"M1,M2"`` or ``"Cz"``).
    Channels are matched by label. Stateless.
    """

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.sterile.rereference",
        "0.1.0",
        [
            {"name": "mode", "type": "enum", "default": "average",
             "choices": ["average", "channels"]},
            {"name": "channels", "type": "string", "default": "",
             "description": "Comma-separated reference labels (mode=channels)"},
        ],
        description="Common average or channel re-reference.",
        live_params=["mode", "channels"],
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        if params["mode"] == "average":
            reference = block.data.mean(axis=0, keepdims=True)
        else:
            labels = [
                s.strip() for s in str(params["channels"]).split(",") if s.strip()
            ]
            if not labels:
                raise ValueError("mode=channels needs at least one channel label")
            idx = match_channels(block.info.ch_labels, labels)
            reference = block.data[idx].mean(axis=0, keepdims=True)
        out = Block(
            np.asarray(block.data - reference),
            block.info,
            block.sample_index,
            block.timestamp,
        )
        return TransformResult(out, {})
