"""Built-in probe: keeps every block it sees, for overlay and difference views."""

from __future__ import annotations

from typing import ClassVar

from meegxer.core.block import Block, EventBlock, FeatureBlock, concat_blocks
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Probe


class MemoryProbe(Probe):
    """Stores the blocks of a Signal tap in memory (batch runs, tests)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.core.probe",
            "version": "0.1.0",
            "type": "stream",
            "role": "probe",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [],
            "params": [{"name": "label", "type": "string", "default": ""}],
        }
    )

    def __init__(self, params: dict[str, object] | None = None) -> None:
        super().__init__(params)
        self.blocks: list[Block | FeatureBlock | EventBlock] = []

    def tap(self, block: AnyBlock) -> None:
        self.blocks.append(block)

    def signal(self) -> Block:
        """All tapped Signal blocks joined (gaps or overlaps raise)."""
        signal_blocks = [b for b in self.blocks if isinstance(b, Block)]
        return concat_blocks(signal_blocks)

    def clear(self) -> None:
        self.blocks.clear()
