"""Built-in file sinks: EEGLAB .set, EDF and feature TSV.

Signal sinks write the file on ``close``, as both formats need the whole recording.
Until then each block is appended to a spool next to the file (``<file>.part``,
described by ``<file>.part.json``) and flushed, so a crash loses at most the block
being written: ``recover`` (``meegxer recover``) writes the file from the spool, and
the next run of the same sink does so too, to ``<name>_recovered<suffix>``. Feature
TSV rows are appended and flushed as they come. Data are in volts, as MNE expects.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, ClassVar, Literal

import numpy as np

from meegxer.core.block import Block, FeatureBlock
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


SPOOL_SUFFIX = ".part"


def spool_paths(path: Path) -> tuple[Path, Path]:
    """The spool of the file at ``path``: its samples and its description."""
    data = path.with_name(path.name + SPOOL_SUFFIX)
    return data, data.with_name(data.name + ".json")


def write_raw(path: Path, fmt: Literal["eeglab", "edf"], data: np.ndarray,
              header: dict[str, Any], overwrite: bool) -> None:  # fmt: skip
    """Write channels x samples ``data`` (volts) as ``fmt``, with the labels, rate
    and bad channels from a spool ``header``."""
    import mne

    info = mne.create_info(list(header["ch_labels"]), header["sfreq"], ch_types="eeg")
    info["bads"] = list(header["bads"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(path, raw, fmt=fmt, overwrite=overwrite, verbose="ERROR")
    if fmt == "eeglab":
        with path.open("r+b") as fh:
            if fh.read(10) == b"MATLAB 5.0":  # v5 MAT-file (not HDF5 v7.3)
                fh.seek(0)
                fh.write(SetSink.MAT_HEADER)


def read_spool(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """A spool's header and its whole samples (channels x samples); a sample cut
    short by a crash is left out."""
    data_path, header_path = spool_paths(path)
    header = json.loads(header_path.read_text(encoding="utf-8"))
    n = len(header["ch_labels"])
    flat = np.fromfile(data_path, dtype="<f8") if data_path.exists() else np.empty(0)
    whole = flat[: flat.size - flat.size % n] if n else flat[:0]
    return header, np.array(whole.reshape(-1, n).T)


def recover(path: Path, output: Path | None = None, overwrite: bool = False) -> Path:
    """Write the file a crashed run left in its spool (``path`` is the file it was
    writing, or its ``.part``/``.part.json``), then remove the spool. Returns the
    file written (``output``, else the file the run was writing)."""
    for suffix in (SPOOL_SUFFIX + ".json", SPOOL_SUFFIX):
        if path.name.endswith(suffix):
            path = path.with_name(path.name[: -len(suffix)])
            break
    data_path, header_path = spool_paths(path)
    if not header_path.is_file():
        raise FileNotFoundError(f"no spool for {path} ({header_path.name})")
    header, data = read_spool(path)
    target = output if output is not None else path
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (overwrite replaces it)")
    if not data.shape[1]:
        raise ValueError(f"the spool of {path} holds no samples")
    write_raw(target, header["format"], data, header, overwrite=overwrite)
    data_path.unlink(missing_ok=True)
    header_path.unlink()
    return target


def _free_name(path: Path, tag: str) -> Path:
    candidate = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    k = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{tag}-{k}{path.suffix}")
        k += 1
    return candidate


class _RawFileSink(Sink):
    fmt: ClassVar[Literal["eeglab", "edf"]]

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        if not str(self.params["path"]):
            raise ValueError(f"{self.manifest.name}: 'path' is required")
        self._spool: IO[bytes] | None = None
        self._first: Block | None = None
        self._next_index = 0
        self._stopped = False  # skipped, or the stream changed: write no more

    @property
    def path(self) -> Path:
        return Path(str(self.params["path"]))

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError(f"{self.manifest.name} accepts Signal blocks only")
        if self._stopped or not block.n_samples:
            return
        if self._first is None:
            if self.keep_existing(self.path):
                self._stopped = True
                return
            self._open(block)
        elif block.info != self._first.info or block.sample_index != self._next_index:
            change = ("channels changed" if block.info != self._first.info
                      else f"samples jump to {block.sample_index}")  # fmt: skip
            self.notes.append(f"{change} at sample {self._next_index}: "
                              f"{self.path.name} keeps what came before")  # fmt: skip
            self._stopped = True
            return
        assert self._spool is not None
        self._spool.write(np.ascontiguousarray(block.data.T, dtype="<f8").tobytes())
        self._spool.flush()  # in the OS's hands: survives a crash of this process
        self._next_index = block.stop_index

    def _open(self, block: Block) -> None:
        data_path, header_path = spool_paths(self.path)
        if header_path.exists():  # left by a run that crashed
            if read_spool(self.path)[1].shape[1]:
                done = recover(self.path, _free_name(self.path, "recovered"))
                self.notes.append(f"recovered a crashed run's recording to {done.name}")
            else:  # it crashed before its first block: nothing to keep
                data_path.unlink(missing_ok=True)
                header_path.unlink()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        info = block.info
        header = {"format": self.fmt, "ch_labels": list(info.ch_labels),
                  "sfreq": info.sfreq, "bads": list(info.bads),
                  "sample_index": block.sample_index, "timestamp": block.timestamp,
                  "dtype": "<f8", "order": "sample-major"}  # fmt: skip
        with header_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._spool = data_path.open("wb")
        self._first = block

    def close(self) -> None:
        if self._spool is None:
            return
        self._spool.close()
        self._spool = None
        header, data = read_spool(self.path)
        write_raw(self.path, self.fmt, data, header,
                  overwrite=bool(self.params["overwrite"]))  # fmt: skip
        for spool_file in spool_paths(self.path):
            spool_file.unlink(missing_ok=True)


class SetSink(_RawFileSink):
    """EEGLAB .set (written with eeglabio through MNE)."""

    manifest: ClassVar[Manifest] = _sink_manifest("meegxer.core.set_writer", "signal")
    fmt = "eeglab"

    # The MAT-file header's descriptive text carries the platform and the time
    # of writing; replaced so the same data gives the same file (sessions,
    # acceptance test 4). Readers ignore this text.
    MAT_HEADER = b"MATLAB 5.0 MAT-file, written by MeegXer".ljust(116, b" ")


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
        self._file: IO[str] | None = None
        self._writer: Any = None
        self._columns: list[str] | None = None
        self._stopped = False  # skipped, or the layout changed: write no more

    @property
    def path(self) -> Path:
        return Path(str(self.params["path"]))

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            raise PortTypeError(f"{self.manifest.name} accepts Features blocks only")
        if self._stopped:
            return
        columns = feature_columns(block)
        if self._columns is None:
            if self.keep_existing(self.path):
                self._stopped = True
                return
            self._open(columns, feature_units(block))
        elif columns != self._columns:
            self.notes.append(f"feature layout changed at sample {block.window[0]}: "
                              f"{self.path.name} keeps the rows before")  # fmt: skip
            self._stopped = True
            return
        values = [tsv_value(v) for v in block.data.ravel()]
        start, stop = block.window
        self._writer.writerow([str(start), str(stop), repr(block.timestamp), *values])
        assert self._file is not None
        self._file.flush()  # in the OS's hands: survives a crash of this process

    def _open(self, columns: list[str], units: list[str]) -> None:
        self._columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(units):
            sidecar = {
                "window_start": {"Units": "sample"},
                "window_stop": {"Units": "sample"},
                "timestamp": {"Units": "s"},
                **{c: {"Units": u} for c, u in zip(columns, units, strict=True) if u},
            }
            self.path.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")  # fmt: skip
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, delimiter="\t", lineterminator="\n")
        self._writer.writerow(["window_start", "window_stop", "timestamp", *columns])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
