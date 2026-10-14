"""Built-in sources: a synthetic EEG generator and file/BIDS readers.

Signals are in volts (SI, as in MNE). Timestamps of file sources start at 0.0;
the synthetic source starts at ``start_time``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy import signal

from meegxer.core.block import Block, FloatArray, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Source

if TYPE_CHECKING:
    import mne
    import mne_bids

STANDARD_LABELS = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
)  # fmt: skip
_POSTERIOR = {"P3", "P4", "P7", "P8", "Pz", "O1", "O2", "Oz"}
_OCCIPITAL = {"O1", "O2", "Oz"}

# Paul Kellet / J.O. Smith pinking filter (-10 dB/decade).
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1.0, -2.494956002, 2.017265875, -0.522189400]


def _source_manifest(name: str, params: list[dict[str, object]]) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": "0.1.0",
            "type": "stream",
            "role": "source",
            "inputs": [],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": params,
        }
    )


class SyntheticSource(Source):
    """Pink noise + posterior alpha bursts + line noise + occipital SSVEP sine.

    Every component is a function of the absolute sample index, and the white
    noise is drawn sample-major from one generator, so the output does not depend
    on how the stream is cut into blocks.
    """

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.synthetic",
        [
            {"name": "n_channels", "type": "int", "default": 16, "min": 1,
             "max": len(STANDARD_LABELS)},
            {"name": "sfreq", "type": "float", "default": 500.0, "min": 1.0,
             "unit": "Hz"},
            {"name": "duration", "type": "float", "default": 60.0, "min": 0.0,
             "unit": "s", "description": "0 streams forever"},
            {"name": "seed", "type": "int", "default": 0, "min": 0, "advanced": True},
            *(
                {"name": name, "type": "float", "default": default, "min": 0.0,
                 "unit": unit, "advanced": True}
                for name, default, unit in (
                    ("noise_uv", 10.0, "uV"),
                    ("alpha_uv", 15.0, "uV"),
                    ("alpha_hz", 10.0, "Hz"),
                    ("line_uv", 5.0, "uV"),
                    ("line_hz", 50.0, "Hz"),
                    ("ssvep_uv", 3.0, "uV"),
                    ("ssvep_hz", 12.0, "Hz"),
                )
            ),
            {"name": "start_time", "type": "float", "default": 0.0, "unit": "s",
             "advanced": True},
        ],
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        p = self.params
        n_ch = int(p["n_channels"])
        self._info = SignalInfo(STANDARD_LABELS[:n_ch], float(p["sfreq"]), "synthetic")
        duration = float(p["duration"])
        self._total = round(duration * self._info.sfreq) if duration > 0 else None
        self._pos = 0
        seed = int(p["seed"])
        self._white = np.random.default_rng([seed, 0])
        phases = np.random.default_rng([seed, 1]).uniform(0, 2 * np.pi, (2, n_ch))
        self._burst_phase, self._alpha_phase = phases[:, :, None]
        labels = self._info.ch_labels
        self._alpha_weight = np.array(
            [1.0 if lab in _POSTERIOR else 0.3 for lab in labels]
        )[:, None]
        self._ssvep_weight = np.array(
            [1.0 if lab in _OCCIPITAL else 0.0 for lab in labels]
        )[:, None]
        self._pink_sos = signal.tf2sos(_PINK_B, _PINK_A)
        impulse = np.zeros(1 << 16)
        impulse[0] = 1.0
        self._pink_gain = float(
            np.sqrt(np.sum(signal.sosfilt(self._pink_sos, impulse) ** 2))
        )
        self._pink_zi = np.zeros((self._pink_sos.shape[0], n_ch, 2))

    @property
    def info(self) -> SignalInfo:
        return self._info

    @property
    def n_samples(self) -> int | None:
        return self._total

    def read(self, max_samples: int) -> Block | None:
        n = max_samples
        if self._total is not None:
            if self._pos >= self._total:
                return None
            n = min(n, self._total - self._pos)
        data = self._generate(self._pos, n)
        sfreq = self._info.sfreq
        block = Block(
            data,
            self._info,
            self._pos,
            float(self.params["start_time"]) + self._pos / sfreq,
        )
        self._pos += n
        return block

    def _generate(self, start: int, n: int) -> FloatArray:
        p = self.params
        uv = 1e-6
        t = (start + np.arange(n))[None, :] / self._info.sfreq
        white = self._white.standard_normal((n, self._info.n_channels)).T
        pink, self._pink_zi = signal.sosfilt(
            self._pink_sos, white, axis=-1, zi=self._pink_zi
        )
        out = float(p["noise_uv"]) * uv * pink / self._pink_gain
        envelope = np.maximum(0.0, np.sin(2 * np.pi * 0.15 * t + self._burst_phase))
        alpha = np.sin(2 * np.pi * float(p["alpha_hz"]) * t + self._alpha_phase)
        out += float(p["alpha_uv"]) * uv * self._alpha_weight * envelope**2 * alpha
        out += float(p["line_uv"]) * uv * np.sin(2 * np.pi * float(p["line_hz"]) * t)
        ssvep = np.sin(2 * np.pi * float(p["ssvep_hz"]) * t)
        out += float(p["ssvep_uv"]) * uv * self._ssvep_weight * ssvep
        result: FloatArray = np.ascontiguousarray(out, dtype=np.float64)
        return result


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


class _RawSource(Source):
    """Serves an in-memory MNE Raw block by block."""

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._pos = 0

    def _load(self) -> mne.io.BaseRaw:
        raise NotImplementedError

    def _files(self) -> list[Path]:
        raise NotImplementedError

    @cached_property
    def _raw(self) -> tuple[FloatArray, SignalInfo]:
        raw = self._load()
        raw.pick("data", exclude=())
        data: FloatArray = np.asarray(raw.get_data(), dtype=np.float64)
        info = SignalInfo(
            tuple(raw.ch_names),
            float(raw.info["sfreq"]),
            self._stream_id(),
            tuple(b for b in raw.info["bads"] if b in raw.ch_names),
        )
        return data, info

    def _stream_id(self) -> str:
        return self._files()[0].name

    @property
    def info(self) -> SignalInfo:
        return self._raw[1]

    @property
    def n_samples(self) -> int:
        return int(self._raw[0].shape[1])

    def read(self, max_samples: int) -> Block | None:
        data, info = self._raw
        if self._pos >= data.shape[1]:
            return None
        stop = min(self._pos + max_samples, data.shape[1])
        block = Block(
            data[:, self._pos : stop], info, self._pos, self._pos / info.sfreq
        )
        self._pos = stop
        return block

    def content_id(self) -> str:
        return _hash_files(self._files())


class FileSource(_RawSource):
    """EDF/BDF, EEGLAB .set, BrainVision, FIF: anything ``mne.io.read_raw`` reads."""

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.file",
        [{"name": "path", "type": "string", "default": ""}],
    )

    _COMPANIONS: ClassVar[dict[str, tuple[str, ...]]] = {
        ".set": (".fdt",),
        ".vhdr": (".eeg", ".vmrk"),
    }

    def _path(self) -> Path:
        path = Path(str(self.params["path"]))
        if not str(self.params["path"]) or not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
        return path

    def _files(self) -> list[Path]:
        path = self._path()
        companions = self._COMPANIONS.get(path.suffix.lower(), ())
        extra = [path.with_suffix(ext) for ext in companions]
        return [path, *[p for p in extra if p.is_file()]]

    def _load(self) -> mne.io.BaseRaw:
        import mne

        return mne.io.read_raw(self._path(), preload=True, verbose="ERROR")


class BidsSource(_RawSource):
    """One recording from a BIDS dataset, read with mne-bids."""

    manifest: ClassVar[Manifest] = _source_manifest(
        "meegxer.core.bids",
        [
            {"name": "root", "type": "string", "default": ""},
            {"name": "subject", "type": "string", "default": ""},
            {"name": "task", "type": "string", "default": ""},
            {"name": "session", "type": "string", "default": ""},
            {"name": "run", "type": "string", "default": ""},
            {"name": "datatype", "type": "string", "default": "eeg"},
        ],
    )

    def _bids_path(self) -> mne_bids.BIDSPath:
        import mne_bids

        p = self.params
        return mne_bids.BIDSPath(
            root=str(p["root"]),
            subject=str(p["subject"]) or None,
            task=str(p["task"]) or None,
            session=str(p["session"]) or None,
            run=str(p["run"]) or None,
            datatype=str(p["datatype"]),
        )

    def _files(self) -> list[Path]:
        fpath = self._bids_path().fpath
        return [Path(fpath)]

    def _load(self) -> mne.io.BaseRaw:
        import mne_bids

        raw: mne.io.BaseRaw = mne_bids.read_raw_bids(self._bids_path(), verbose="ERROR")
        raw.load_data()
        return raw
