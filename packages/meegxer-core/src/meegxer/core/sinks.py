"""Built-in file sinks: EEGLAB .set, EDF and feature TSV.

Signal sinks write the file on ``close``, as both formats need the whole recording.
Until then each block is appended to a spool next to the file (``<file>.part``,
described by ``<file>.part.json``) and flushed, so a crash loses at most the block
being written: ``recover`` (``meegxer recover``) writes the file from the spool, and
the next run of the same sink does so too, to ``<name>_recovered<suffix>``. Feature
TSV rows are appended and flushed as they come. Data are in volts, as MNE expects.

Each spool also leaves a marker in the config folder (``recordings/``: the file's
path and the writing process), so a crashed run's recording is found again without
its pipeline (``unfinished_recordings``; the GUI offers to recover it at start).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, ClassVar, Literal

import numpy as np

from meegxer.core.block import Block, Event, FeatureBlock
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink
from meegxer.core.ports import PortTypeError
from meegxer.core.references import credits
from meegxer.core.settings import config_dir

BOUNDARY = "boundary"  # the event at a cut (meegxer.core.review)


def _sink_manifest(name: str, input_type: str, extension: str) -> Manifest:
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
                {
                    "name": "path",
                    "type": "string",
                    "default": "",
                    "primary": True,
                    "path": "save",
                    "extensions": [extension],
                },
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


def file_events(header: dict[str, Any], n_samples: int) -> list[tuple[int, int, str]]:
    """The spool's events (``events``: ``[sample index, samples, label]`` on
    the stream's clock) as ``(position in the file, samples, label)``: shifted
    over the cuts before them, those inside a cut left out, each clipped to the
    piece it starts in."""
    first = int(header.get("sample_index", 0))
    gaps: list[tuple[int, int, int]] = []  # stream index start, stop; cut before
    offset = 0
    for pos, cut in sorted(header.get("boundaries", [])):
        start = first + int(pos) + offset
        gaps.append((start, start + int(cut), offset))
        offset += int(cut)
    out = []
    for index, length, label in header.get("events", []):
        index, length = int(index), int(length)
        if index < first or any(a <= index < b for a, b, _ in gaps):
            continue
        before = sum(b - a for a, b, _ in gaps if b <= index)
        pos = index - first - before
        if pos >= n_samples:
            continue
        nxt = min((a for a, _, _ in gaps if a > index), default=None)
        if nxt is not None:
            length = min(length, nxt - index)
        out.append((pos, max(0, min(length, n_samples - pos)), str(label)))
    return out


def make_raw(
    data: np.ndarray,
    header: dict[str, Any],
    ch_types: Any = "eeg",
    positions: Mapping[str, Sequence[float]] | None = None,
) -> Any:
    """Channels x samples ``data`` (volts) as an ``mne.io.RawArray``, with the
    labels, rate and bad channels from a spool ``header``, its events (the
    input's and the strips', ``file_events``), and a ``boundary`` event at each
    cut a Reject segments strip made (``boundaries``: ``[position in data,
    samples cut]``); with the channels' ``positions`` (metres, head
    coordinates) where given."""
    import mne

    info = mne.create_info(list(header["ch_labels"]), header["sfreq"], ch_types)
    info["bads"] = list(header["bads"])
    if positions:
        montage = mne.channels.make_dig_montage(dict(positions), coord_frame="head")
        info.set_montage(montage, on_missing="ignore", verbose="error")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    sfreq = float(header["sfreq"])
    cuts = [pos for pos, _ in header.get("boundaries", []) if 0 < pos < data.shape[1]]
    marks = [(p, 0, "boundary") for p in cuts] + file_events(header, data.shape[1])
    if marks:
        marks.sort()
        onsets = [p / sfreq for p, _, _ in marks]
        durations = [n / sfreq for _, n, _ in marks]
        raw.set_annotations(
            mne.Annotations(onsets, durations, [label for _, _, label in marks])
        )
    return raw


def write_raw(path: Path, fmt: Literal["eeglab", "edf"], data: np.ndarray,
              header: dict[str, Any], overwrite: bool) -> None:  # fmt: skip
    """Write ``data`` and a spool ``header`` (``make_raw``) as ``fmt``."""
    import mne

    raw = make_raw(data, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(path, raw, fmt=fmt, overwrite=overwrite, verbose="ERROR")
    if fmt == "eeglab":
        stamp_mat_header(path)


def stamp_mat_header(path: Path) -> None:
    """Replace a v5 MAT-file's header text (it carries the platform and the
    time of writing) with ``SetSink.MAT_HEADER``: the same data, the same file."""
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
    header_path = spool_paths(path)[1]
    if not header_path.is_file():
        raise FileNotFoundError(f"no spool for {path} ({header_path.name})")
    header, data = read_spool(path)
    target = output if output is not None else path
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists (overwrite replaces it)")
    if not data.shape[1]:
        raise ValueError(f"the spool of {path} holds no samples")
    write_raw(target, header["format"], data, header, overwrite=overwrite)
    discard(path)
    return target


def _marker(path: Path) -> Path:
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return config_dir() / "recordings" / f"{key}.json"


def _register(path: Path) -> None:
    marker = _marker(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"path": str(path.resolve()), "pid": os.getpid()}),
                      encoding="utf-8")  # fmt: skip


def _unregister(path: Path) -> None:
    _marker(path).unlink(missing_ok=True)


def _process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process
    return True


def unfinished_recordings() -> list[dict[str, Any]]:
    """Recordings whose run ended before writing them (their spools are still
    there and no running process writes them): ``path``, ``format``,
    ``channels``, ``seconds`` and ``modified`` (ISO 8601, UTC) of each."""
    folder = config_dir() / "recordings"
    found = []
    for marker in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            entry = json.loads(marker.read_text(encoding="utf-8"))
            path, pid = Path(entry["path"]), int(entry["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            marker.unlink(missing_ok=True)
            continue
        data_path, header_path = spool_paths(path)
        if not header_path.is_file():  # written, recovered or deleted since
            marker.unlink(missing_ok=True)
            continue
        if pid == os.getpid() or _process_alive(pid):
            continue  # still being recorded
        header = json.loads(header_path.read_text(encoding="utf-8"))
        n = len(header["ch_labels"])
        size = data_path.stat().st_size if data_path.exists() else 0
        stamp = (data_path if data_path.exists() else header_path).stat().st_mtime
        modified = datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds")
        found.append({
            "path": str(path), "format": header["format"], "channels": n,
            "seconds": size // (8 * n) / header["sfreq"] if n else 0.0,
            "modified": modified,
        })  # fmt: skip
    return found


def discard(path: Path) -> None:
    """Delete a crashed run's spool (``path``: the file it was writing)."""
    for spool_file in spool_paths(path):
        spool_file.unlink(missing_ok=True)
    _unregister(path)


def free_name(path: Path, tag: str) -> Path:
    """``<stem>_<tag><suffix>`` next to ``path``, numbered until it is free."""
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
        self._written = 0  # samples in the file so far
        self._header: dict[str, Any] = {}
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
        elif block.info != self._first.info or block.sample_index < self._next_index:
            change = ("channels changed" if block.info != self._first.info
                      else f"samples jump back to {block.sample_index}")  # fmt: skip
            self.notes.append(f"{change} at sample {self._next_index}: "
                              f"{self.path.name} keeps what came before")  # fmt: skip
            self._stopped = True
            return
        elif block.sample_index > self._next_index:  # a cut (Reject segments)
            cut = block.sample_index - self._next_index
            self._header.setdefault("boundaries", []).append([self._written, cut])
            self._write_header()  # kept by a crash recovery too
        assert self._spool is not None
        self._spool.write(np.ascontiguousarray(block.data.T, dtype="<f8").tobytes())
        self._spool.flush()  # in the OS's hands: survives a crash of this process
        self._next_index = block.stop_index
        self._written += block.n_samples

    def write_events(self, events: Sequence[Event]) -> None:
        """Kept with the spool (a crash recovery writes them too); the cuts'
        boundaries are written from the cuts themselves."""
        if self._first is None or self._stopped:
            return
        self._header["events"] = [[e.sample_index, e.duration, e.label]
                                  for e in events if e.label != BOUNDARY]  # fmt: skip
        self._write_header()

    def _open(self, block: Block) -> None:
        data_path, header_path = spool_paths(self.path)
        if header_path.exists():  # left by a run that crashed
            if read_spool(self.path)[1].shape[1]:
                done = recover(self.path, free_name(self.path, "recovered"))
                self.notes.append(f"recovered a crashed run's recording to {done.name}")
            else:  # it crashed before its first block: nothing to keep
                discard(self.path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        info = block.info
        self._header = {"format": self.fmt, "ch_labels": list(info.ch_labels),
                        "sfreq": info.sfreq, "bads": list(info.bads),
                        "sample_index": block.sample_index,
                        "timestamp": block.timestamp,
                        "dtype": "<f8", "order": "sample-major"}  # fmt: skip
        self._write_header()
        self._spool = data_path.open("wb")
        _register(self.path)
        self._first = block

    def _write_header(self) -> None:
        """The spool's header, replaced whole (a crash leaves the old or the new)."""
        header_path = spool_paths(self.path)[1]
        tmp = header_path.with_name(header_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(self._header, indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, header_path)

    def close(self) -> None:
        if self._spool is None:
            return
        self._spool.close()
        self._spool = None
        header, data = read_spool(self.path)
        write_raw(self.path, self.fmt, data, header,
                  overwrite=bool(self.params["overwrite"]))  # fmt: skip
        discard(self.path)


class SetSink(_RawFileSink):
    """EEGLAB .set (written with eeglabio through MNE)."""

    manifest: ClassVar[Manifest] = _sink_manifest(
        "meegxer.core.set_writer", "signal", ".set"
    )
    fmt = "eeglab"

    # The MAT-file header's descriptive text carries the platform and the time
    # of writing; replaced so the same data gives the same file (sessions,
    # acceptance test 4). Readers ignore this text.
    MAT_HEADER = b"MATLAB 5.0 MAT-file, written by MeegXer".ljust(116, b" ")

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._graph: Any = None
        self._node_id = ""

    def bind(self, graph: Any, node_id: str) -> None:
        self._graph, self._node_id = graph, node_id

    def close(self) -> None:
        """The recording, then the ICA decomposition that unmixes it (the ICs
        rail's writer before this strip: the input's file, or an ICA strip,
        less the components it removed), as EEGLAB keeps one (M10.6)."""
        if self._spool is None:
            return
        labels = tuple(self._header["ch_labels"])
        super().close()
        if self._graph is None:
            return
        from meegxer.core.eeglab_ica import decomposition_for, write_decomposition

        dec, why = decomposition_for(self._graph, self._node_id)
        if dec is None:
            if why:
                self.skipped.append(why)  # the log shows it (GUI and CLI)
            return
        try:
            write_decomposition(self.path, dec, labels)
        except ValueError as exc:  # e.g. a channel it unmixes was dropped
            self.skipped.append(f"ICA decomposition not stored: {exc}")
            return
        stamp_mat_header(self.path)
        classes = "" if dec.probabilities is None else ", with its ICLabel classes"
        self.notes.append(f"stored the ICA decomposition ({dec.n_components} "
                          f"components{classes})")  # fmt: skip


class EdfSink(_RawFileSink):
    """EDF (written with edfio through MNE). EDF stores 16-bit samples."""

    manifest: ClassVar[Manifest] = _sink_manifest(
        "meegxer.core.edf_writer", "signal", ".edf"
    )
    fmt = "edf"


class FeatureTsvSink(Sink):
    """One row per feature window: window bounds, timestamp, then one column per
    feature value, named by its coordinates (``Fz|10.0``). The values' units go
    into a JSON sidecar next to it (``features.json``, BIDS style), when stated."""

    manifest: ClassVar[Manifest] = _sink_manifest(
        "meegxer.core.feature_tsv", "features", ".tsv"
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
