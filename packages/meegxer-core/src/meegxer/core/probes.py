"""Built-in probe: keeps the blocks it sees, for overlay and difference views."""

from __future__ import annotations

from typing import ClassVar

from meegxer.core.block import Block, EventBlock, FeatureBlock, concat_blocks
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Probe
from meegxer.core.references import credits


class MemoryProbe(Probe):
    """Stores the blocks of a tap in memory (batch runs, tests).

    Only the most recent ``keep_seconds`` of Signal data are kept (0 keeps
    everything), so a probe in a long live run does not grow without bound.
    """

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.core.probe",
            "version": "0.1.0",
            "type": "stream",
            "role": "probe",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [],
            "params": [
                {"name": "label", "type": "string", "default": "", "primary": True},
                {
                    "name": "keep_seconds",
                    "type": "float",
                    "default": 60.0,
                    "min": 0.0,
                    "unit": "s",
                    "advanced": True,
                    "description": "Signal history kept in memory (0 keeps all)",
                },
            ],
            **credits("meegxer.core.probe"),
        }
    )

    def __init__(self, params: dict[str, object] | None = None) -> None:
        super().__init__(params)
        self.blocks: list[Block | FeatureBlock | EventBlock] = []
        self._samples = 0

    def tap(self, block: AnyBlock) -> None:
        self.blocks.append(block)
        if not isinstance(block, Block):
            return
        self._samples += block.n_samples
        keep = float(self.params["keep_seconds"]) * block.info.sfreq
        while keep > 0:
            oldest = next(b for b in self.blocks if isinstance(b, Block))
            if self._samples - oldest.n_samples < keep:
                break
            self.blocks.remove(oldest)
            self._samples -= oldest.n_samples

    def signal(self) -> Block:
        """All tapped Signal blocks joined (gaps or overlaps raise)."""
        signal_blocks = [b for b in self.blocks if isinstance(b, Block)]
        return concat_blocks(signal_blocks)

    def clear(self) -> None:
        self.blocks.clear()
        self._samples = 0


class FeatureProbe(Probe):
    """Keeps the most recent feature windows of a Features tap (the feature
    views: a spectrum per window, or values over time)."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.core.feature_probe",
            "version": "0.1.0",
            "type": "analysis",
            "role": "probe",
            "inputs": [{"name": "in", "type": "features"}],
            "outputs": [],
            "params": [
                {"name": "label", "type": "string", "default": "", "primary": True},
                {
                    "name": "keep_windows",
                    "type": "int",
                    "default": 600,
                    "min": 0,
                    "advanced": True,
                    "description": "Feature windows kept in memory (0 keeps all)",
                },
            ],
            **credits("meegxer.core.feature_probe"),
        }
    )

    def __init__(self, params: dict[str, object] | None = None) -> None:
        super().__init__(params)
        self.windows: list[FeatureBlock] = []

    def tap(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            return
        self.windows.append(block)
        keep = int(self.params["keep_windows"])
        if keep and len(self.windows) > keep:
            del self.windows[: len(self.windows) - keep]

    def clear(self) -> None:
        self.windows.clear()
