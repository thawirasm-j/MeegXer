"""Built-in file sinks: EEGLAB .set, EDF and feature TSV.

Signal sinks collect blocks and write the file on ``close``; both formats need the
whole recording. Data are in volts, as MNE expects.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from meegxer.core.block import Block, FeatureBlock, concat_blocks
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink
from meegxer.core.ports import PortTypeError
from meegxer.core.references import credits


def _sink_manifest(name: str, input_type: str) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": "0.1.0",
            "type": "stream",
            "role": "sink",
            "inputs": [{"name": "in", "type": input_type}],
            "outputs": [],
            "params": [
                {"name": "path", "type": "string", "default": "", "primary": True},
                {"name": "overwrite", "type": "bool", "default": False},
            ],
            **credits(name),
        }
    )


def feature_columns(block: FeatureBlock) -> list[str]:
    """One name per feature value, in ``data.ravel()`` order, joined from its
    coordinates (``Fz|10.0``); indices stand in for unlabelled dimensions."""
    axes = []
    for dim, size in zip(block.dims, block.data.shape, strict=True):
        values = block.coords.get(dim, tuple(range(size)))
        axes.append([str(v) for v in values])
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1)
    return ["|".join(col) for col in grid.T]


def feature_units(block: FeatureBlock) -> list[str]:
    """The unit of each feature value, in ``data.ravel()`` order (``""``: none)."""
    if not block.unit_dim:
        return [block.unit] * block.data.size
    axis = block.dims.index(block.unit_dim)
    index = np.indices(block.data.shape)[axis].ravel()
    return [block.units[i] for i in index]


def tsv_value(value: float) -> str:
    """A feature value for a TSV cell: exact (repr), and BIDS's ``n/a`` for a
    missing (NaN) value."""
    v = float(value)
    return "n/a" if np.isnan(v) else repr(v)


class _RawFileSink(Sink):
    fmt: ClassVar[Literal["eeglab", "edf"]]

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        if not str(self.params["path"]):
            raise ValueError(f"{self.manifest.name}: 'path' is required")
        self._blocks: list[Block] = []

    @property
    def path(self) -> Path:
        return Path(str(self.params["path"]))

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError(f"{self.manifest.name} accepts Signal blocks only")
        self._blocks.append(block)

    def close(self) -> None:
        if not self._blocks or self.keep_existing(self.path):
            self._blocks.clear()
            return
        import mne

        whole = concat_blocks(self._blocks)
        info = mne.create_info(
            list(whole.info.ch_labels), whole.info.sfreq, ch_types="eeg"
        )
        info["bads"] = list(whole.info.bads)
        raw = mne.io.RawArray(np.array(whole.data), info, verbose="ERROR")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mne.export.export_raw(
            self.path,
            raw,
            fmt=self.fmt,
            overwrite=bool(self.params["overwrite"]),
            verbose="ERROR",
        )
        self._blocks.clear()


class SetSink(_RawFileSink):
    """EEGLAB .set (written with eeglabio through MNE)."""

    manifest: ClassVar[Manifest] = _sink_manifest("meegxer.core.set_writer", "signal")
    fmt = "eeglab"

    # The MAT-file header's descriptive text carries the platform and the time
    # of writing; replaced so the same data gives the same file (sessions,
    # acceptance test 4). Readers ignore this text.
    MAT_HEADER = b"MATLAB 5.0 MAT-file, written by MeegXer".ljust(116, b" ")

    def close(self) -> None:
        written = bool(self._blocks)
        super().close()
        if written and not self.skipped:
            with self.path.open("r+b") as fh:
                if fh.read(10) == b"MATLAB 5.0":  # v5 MAT-file (not HDF5 v7.3)
                    fh.seek(0)
                    fh.write(self.MAT_HEADER)


class EdfSink(_RawFileSink):
    """EDF (written with edfio through MNE). EDF stores 16-bit samples."""

    manifest: ClassVar[Manifest] = _sink_manifest("meegxer.core.edf_writer", "signal")
    fmt = "edf"


class FeatureTsvSink(Sink):
    """One row per feature window: window bounds, timestamp, then one column per
    feature value, named by its coordinates (``Fz|10.0``). The values' units go
    into a JSON sidecar next to it (``features.json``, BIDS style), when stated."""

    manifest: ClassVar[Manifest] = _sink_manifest(
        "meegxer.core.feature_tsv", "features"
    )

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        if not str(self.params["path"]):
            raise ValueError(f"{self.manifest.name}: 'path' is required")
        self._rows: list[list[str]] = []
        self._header: list[str] | None = None
        self._units: list[str] = []

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            raise PortTypeError(f"{self.manifest.name} accepts Features blocks only")
        columns = feature_columns(block)
        if self._header is None:
            self._header = ["window_start", "window_stop", "timestamp", *columns]
            self._units = feature_units(block)
        elif self._header[3:] != columns:
            raise ValueError("feature layout changed between blocks")
        values = [tsv_value(v) for v in block.data.ravel()]
        start, stop = block.window
        self._rows.append([str(start), str(stop), repr(block.timestamp), *values])

    def close(self) -> None:
        if self._header is None:
            return
        path = Path(str(self.params["path"]))
        if self.keep_existing(path):
            self._rows.clear()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(self._header)
            writer.writerows(self._rows)
        if any(self._units):
            sidecar = {
                "window_start": {"Units": "sample"},
                "window_stop": {"Units": "sample"},
                "timestamp": {"Units": "s"},
                **{
                    c: {"Units": u}
                    for c, u in zip(self._header[3:], self._units, strict=True)
                    if u
                },
            }
            path.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")  # fmt: skip
        self._rows.clear()
