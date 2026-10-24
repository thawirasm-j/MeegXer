"""Reject segments (docs/REVIEW.md): cut spans out of the signal by label.

The strip reads the Events rail (the input's events, a review's marks, a
detector's) and cuts out every span whose label starts with one of its
``reject`` words (default ``BAD``), overlapping spans merged. The samples after
a cut keep their index (rule 8); a ``boundary`` event marks each cut, and the
batch scheduler restarts every strip after it at each gap (``Stage.restart``),
so nothing leaks across. The cutting itself is done by the batch scheduler
(``cut_blocks``), over the whole recording: offline only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from meegxer.core.block import CHANNEL_TYPES, Block
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, RailInput, TransformResult
from meegxer.core.references import credits
from meegxer.core.serialization import State

REJECT_NODE = "meegxer.core.reject_segments"


class RejectSegments(Processor):
    """Passes the signal on; the batch run cuts the rejected spans out."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.20",
            "name": REJECT_NODE,
            "version": "0.1.0",
            "type": "stream",
            "description": "Cut out spans whose event label starts with a "
                           "reject word, with a boundary event at each cut.",
            "availability": "offline",
            "inputs": [{"name": "in", "type": "signal"},
                       {"name": "events", "type": "events", "rail": "events"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "channel_types": list(CHANNEL_TYPES),
            "params": [
                {"name": "reject", "type": "string", "default": "BAD",
                 "primary": True,
                 "description": "Spans whose label starts with one of these "
                                "comma-separated words are cut out (empty: "
                                "nothing is cut)"},
            ],
            **credits(REJECT_NODE),
        }
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        return TransformResult(block, {})

    def transform_rails(
        self,
        block: Block,
        model: State,
        state: State | None,
        params: Params,
        rails: Mapping[str, RailInput],
    ) -> TransformResult:
        return TransformResult(block, {})
