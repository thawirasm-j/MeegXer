"""Built-in probes: the blocks a tap sees (overlay and difference views), the
feature windows of a Features tap, and the running ERP of an epochs tap."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from meegxer.core.block import (
    Block,
    EventBlock,
    FeatureBlock,
    FloatArray,
    concat_blocks,
)
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


class ErpProbe(Probe):
    """The running ERP of an epochs tap (docs/RAILS.md): the mean of every
    epoch of each label since the run started, its last reset, or a change
    upstream (``follow``). It keeps a sum and a count per label, so the mean
    is exact however many display frames are lost; the same mean as MNE's
    ``Epochs.average`` over those epochs."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.core.erp_probe",
            "version": "0.1.0",
            "type": "analysis",
            "role": "probe",
            "inputs": [{"name": "in", "type": "features"}],
            "outputs": [],
            "params": [
                {"name": "label", "type": "string", "default": "", "primary": True},
            ],
            **credits("meegxer.core.erp_probe"),
        }
    )

    def __init__(self, params: dict[str, object] | None = None) -> None:
        super().__init__(params)
        self.sums: dict[str, FloatArray] = {}
        self.counts: dict[str, int] = {}
        self.channels: tuple[str, ...] = ()
        self.times: tuple[float, ...] = ()
        self.unit = ""
        self.window: tuple[int, int] | None = None  # first to last epoch's samples
        self.changed = False  # since the display last sent it
        self._input = ""

    def tap(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock) or block.dims != (
            "epoch", "channel", "time"
        ):  # fmt: skip
            return
        channels = tuple(str(c) for c in block.coords.get("channel", ()))
        times = tuple(float(t) for t in block.coords.get("time", ()))
        if (channels, times, block.unit) != (self.channels, self.times, self.unit):
            self.reset()  # another epoch shape or montage: start over
            self.channels, self.times, self.unit = channels, times, block.unit
        label = str(block.coords["epoch"][0])
        if label in self.sums:
            self.sums[label] = self.sums[label] + block.data[0]
        else:
            self.sums[label] = np.array(block.data[0], dtype=np.float64)
        self.counts[label] = self.counts.get(label, 0) + 1
        lo = block.window[0] if self.window is None else self.window[0]
        self.window = (lo, block.window[1])
        self.changed = True

    def follow(self, input_fp: str) -> None:
        """Start over when what feeds the tap changes (rule 7: a mean of
        epochs from before and after a change upstream would mix the two)."""
        if input_fp != self._input:
            self._input = input_fp
            self.reset()

    def reset(self) -> None:
        if self.sums:
            self.changed = True
        self.sums, self.counts, self.window = {}, {}, None

    def clear(self) -> None:
        self.reset()

    def average(self) -> FeatureBlock | None:
        """The ERP of each label (sorted), ``dims`` ``(condition, channel,
        time)``; None before the first epoch. ``counts`` gives each one's n."""
        if not self.channels:
            return None
        sums, counts = self.sums, self.counts  # reset() replaces both
        labels = tuple(sorted(sums))
        shape = (len(labels), len(self.channels), len(self.times))
        data = np.zeros(shape)
        for k, label in enumerate(labels):
            data[k] = sums[label] / counts[label]
        return FeatureBlock(
            data, ("condition", "channel", "time"), self.window or (0, 0), 0.0,
            {"condition": labels, "channel": self.channels, "time": self.times},
            unit=self.unit,
        )  # fmt: skip
