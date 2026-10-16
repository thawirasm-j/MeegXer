"""Feature nodes: PSD, band power, connectivity, complexity (Features out) and the
Hilbert envelope (Signal out).

The analysis nodes share one windowing scheme: a window of ``window`` seconds
every ``step`` seconds, counted from the first sample of the stream. The input is
buffered in ``state`` and each call returns the windows its block completes, so
the windows never depend on block sizes (acceptance test 1). Each window is a
FeatureBlock with ``window`` = its sample range and ``timestamp`` = the LSL time of
its first sample. Bad channels are kept, so the feature layout stays fixed.

Spectra follow MATLAB's ``pwelch`` as EEGLAB's ``spectopo`` calls it: symmetric
Hamming (or Hann) segments, no detrending, one-sided density in V^2/Hz.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from scipy import signal as sps

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State

WINDOW_PARAMS: list[dict[str, object]] = [
    {"name": "window", "type": "float", "default": 2.0, "min": 0.05, "unit": "s",
     "description": "Length of each analysis window"},
    {"name": "step", "type": "float", "default": 0.5, "min": 0.01, "unit": "s",
     "description": "Time from one window to the next"},
]  # fmt: skip
SPECTRUM_PARAMS: list[dict[str, object]] = [
    {"name": "segment", "type": "float", "default": 1.0, "min": 0.05, "unit": "s",
     "advanced": True, "description": "Welch segment length (spectopo winsize)"},
    {"name": "overlap", "type": "float", "default": 0.5, "min": 0.0, "max": 0.95,
     "advanced": True, "description": "Welch segment overlap, as a fraction"},
    {"name": "taper", "type": "enum", "default": "hamming",
     "choices": ["hamming", "hann"], "advanced": True},
]  # fmt: skip


def analysis_manifest(
    name: str,
    params: list[dict[str, object]],
    description: str,
    *,
    window: float = 2.0,
    step: float = 0.5,
) -> Manifest:
    windowing = [dict(WINDOW_PARAMS[0], default=window),
                 dict(WINDOW_PARAMS[1], default=step)]  # fmt: skip
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": "0.1.0",
            "type": "analysis",
            "description": description,
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "features"}],
            "params": [*windowing, *params],
            "availability": "both",
        }
    )


def window_samples(params: Params, sfreq: float) -> tuple[int, int]:
    n = max(1, round(float(params["window"]) * sfreq))
    step = max(1, round(float(params["step"]) * sfreq))
    return n, step


def welch(x: np.ndarray, sfreq: float, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD (V^2/Hz) of each row, as MATLAB's pwelch computes it."""
    nper = min(x.shape[-1], max(2, round(float(params["segment"]) * sfreq)))
    noverlap = min(nper - 1, round(float(params["overlap"]) * nper))
    taper = sps.get_window(str(params["taper"]), nper, fftbins=False)  # symmetric
    freqs, psd = sps.welch(x, sfreq, window=taper, nperseg=nper, noverlap=noverlap,
                           detrend=False, scaling="density", axis=-1)  # fmt: skip
    return freqs, psd


class WindowedFeatures(Processor):
    """Base for analysis nodes: buffering and windowing; subclasses compute one
    window's features with ``features``."""

    manifest: ClassVar[Manifest]

    @abstractmethod
    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        """(data, dims, coords) for one window ``x`` (channels x samples)."""

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        info = block.info
        n, step = window_samples(params, info.sfreq)
        if state is None:  # times come from the first block, so splits agree
            state = {
                "buffer": np.zeros((info.n_channels, 0)),
                "buffer_start": block.sample_index,
                "next": block.sample_index,
                "t_ref": block.timestamp,
                "i_ref": block.sample_index,
            }
        buffer = np.asarray(state["buffer"])
        start = int(state["buffer_start"])  # type: ignore[arg-type]
        nxt = int(state["next"])  # type: ignore[arg-type]
        data = block.data
        if block.sample_index < start:  # a skipped stretch (step > window)
            data = data[:, start - block.sample_index :]
        buffer = np.concatenate([buffer, data], axis=1)
        t_ref, i_ref = float(state["t_ref"]), int(state["i_ref"])  # type: ignore[arg-type]
        out = []
        while nxt + n <= start + buffer.shape[1]:
            x = buffer[:, nxt - start : nxt - start + n]
            values, dims, coords = self.features(x, info, params)
            out.append(FeatureBlock(values, dims, (nxt, nxt + n),
                                    t_ref + (nxt - i_ref) / info.sfreq,
                                    coords, info.stream_id))  # fmt: skip
            nxt += step
        keep = min(max(nxt - start, 0), buffer.shape[1])
        buffer, start = buffer[:, keep:], start + keep
        if nxt > start:  # the next window starts beyond what we hold
            start = nxt
        new_state: State = {"buffer": buffer, "buffer_start": start, "next": nxt,
                            "t_ref": t_ref, "i_ref": i_ref}  # fmt: skip
        return TransformResult(tuple(out), new_state)


class PSD(WindowedFeatures):
    """Power spectral density per channel and window (Welch, pwelch-compatible)."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.psd",
        [
            *SPECTRUM_PARAMS,
            {"name": "fmin", "type": "float", "default": 1.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "fmax", "type": "float", "default": 45.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "db", "type": "bool", "default": False,
             "description": "Report 10*log10(power) instead of V^2/Hz"},
        ],
        "Power spectral density per channel (Welch).",
    )  # fmt: skip

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        freqs, psd = welch(x, info.sfreq, params)
        keep = (freqs >= float(params["fmin"])) & (freqs <= float(params["fmax"]))
        values = psd[:, keep]
        if params["db"]:
            values = 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
        rounded = tuple(round(float(f), 6) for f in freqs[keep])
        coords = {"channel": info.ch_labels, "frequency": rounded}
        return values, ("channel", "frequency"), coords


DEFAULT_BANDS = "delta:1-4, theta:4-8, alpha:8-13, beta:13-30, gamma:30-45"


def parse_bands(text: str) -> list[tuple[str, float, float]]:
    """``"alpha:8-13, beta:13-30"`` -> [("alpha", 8, 13), ("beta", 13, 30)]."""
    bands = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        try:
            name, rng = item.split(":")
            lo, hi = (float(v) for v in rng.split("-"))
        except ValueError:
            raise ValueError(f"bad band {item.strip()!r}; use name:low-high") from None
        if not 0 <= lo < hi:
            raise ValueError(f"band {name.strip()!r} needs 0 <= low < high")
        bands.append((name.strip(), lo, hi))
    if not bands:
        raise ValueError("no bands given")
    return bands


class BandPower(WindowedFeatures):
    """Power in named bands per channel and window: the Welch PSD summed over the
    band's bins [low, high) times the bin width. ``relative`` divides by the power
    over all the bands' span; ``db`` reports 10*log10."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.band_power",
        [
            *SPECTRUM_PARAMS,
            {"name": "bands", "type": "string", "default": DEFAULT_BANDS,
             "primary": True, "description": "name:low-high in Hz, comma-separated"},
            {"name": "relative", "type": "bool", "default": False, "primary": True},
            {"name": "db", "type": "bool", "default": False},
        ],
        "Power in frequency bands per channel.",
    )  # fmt: skip

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        bands = parse_bands(str(params["bands"]))
        freqs, psd = welch(x, info.sfreq, params)
        df = freqs[1] - freqs[0]
        values = np.stack(
            [psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df
             for _, lo, hi in bands], axis=1,
        )  # fmt: skip
        if params["relative"]:
            lo, hi = min(b[1] for b in bands), max(b[2] for b in bands)
            total = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df
            values = values / np.maximum(total, np.finfo(float).tiny)[:, None]
        if params["db"]:
            values = 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
        coords = {"channel": info.ch_labels, "band": tuple(b[0] for b in bands)}
        return values, ("channel", "band"), coords


class Connectivity(WindowedFeatures):
    """Channel-by-channel connectivity in one band per window (mne-connectivity):
    the window is cut into ``segment``-long pieces, which serve as the epochs the
    estimator averages over (multitaper spectra, band-averaged)."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.connectivity",
        [
            {"name": "method", "type": "enum", "default": "wpli", "primary": True,
             "choices": ["coh", "imcoh", "plv", "wpli"],
             "description": "Coherence, imaginary coherence, phase-locking value "
                            "or weighted phase-lag index"},
            {"name": "fmin", "type": "float", "default": 8.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "fmax", "type": "float", "default": 13.0, "min": 0.0,
             "unit": "Hz", "primary": True},
            {"name": "segment", "type": "float", "default": 1.0, "min": 0.05,
             "unit": "s", "advanced": True,
             "description": "Piece length; the window holds window/segment pieces "
                            "(give the lowest frequency 5 cycles per piece)"},
        ],
        "Connectivity between channel pairs in a frequency band.",
        window=4.0,
        step=1.0,
    )  # fmt: skip

    @classmethod
    def warm_up(cls) -> None:
        import mne_connectivity  # noqa: F401 - ~2 s to import

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        import mne
        from mne_connectivity import spectral_connectivity_epochs

        seg = max(2, round(float(params["segment"]) * info.sfreq))
        n_seg = x.shape[1] // seg
        if n_seg < 2:
            raise ValueError("connectivity needs at least two segments per window")
        epochs = x[:, : n_seg * seg].reshape(x.shape[0], n_seg, seg).transpose(1, 0, 2)
        method = str(params["method"])
        with mne.utils.use_log_level("error"):
            con = spectral_connectivity_epochs(
                epochs, method=method, sfreq=info.sfreq, fmin=float(params["fmin"]),
                fmax=float(params["fmax"]), faverage=True, verbose=False,
            )  # fmt: skip
        dense = con.get_data(output="dense")[:, :, 0]
        full = dense + dense.T  # the estimator fills the lower triangle
        if method == "imcoh":
            full = dense - dense.T  # antisymmetric
        np.fill_diagonal(full, 0.0 if method == "imcoh" else 1.0)
        coords = {"channel": info.ch_labels, "target": info.ch_labels}
        return np.asarray(full, dtype=float), ("channel", "target"), coords


COMPLEXITY = ("perm_entropy", "sample_entropy", "spectral_entropy",
              "hjorth_mobility", "hjorth_complexity", "lziv")  # fmt: skip


class Complexity(WindowedFeatures):
    """Complexity measures per channel and window (antropy): normalized
    permutation entropy (order 3), sample entropy (order 2), normalized spectral
    entropy (Welch), Hjorth mobility and complexity, normalized Lempel-Ziv
    complexity of the signal binarized at its median."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.complexity",
        [{"name": m, "type": "bool", "default": True,
          **({"advanced": True} if m not in ("perm_entropy", "lziv") else {})}
         for m in COMPLEXITY],
        "Entropy, Hjorth and Lempel-Ziv measures per channel.",
    )  # fmt: skip

    @classmethod
    def warm_up(cls) -> None:
        import antropy  # noqa: F401 - Numba compiles it on import (~9 s)

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        import antropy as ant

        chosen = [m for m in COMPLEXITY if params[m]]
        if not chosen:
            raise ValueError("choose at least one complexity measure")
        rows = [[_measure(ant, m, ch, info.sfreq) for m in chosen] for ch in x]
        coords = {"channel": info.ch_labels, "measure": tuple(chosen)}
        return np.array(rows), ("channel", "measure"), coords


def _measure(ant: Any, name: str, x: np.ndarray, sfreq: float) -> float:
    if name == "perm_entropy":
        return float(ant.perm_entropy(x, order=3, normalize=True))
    if name == "sample_entropy":
        return float(ant.sample_entropy(x, order=2))
    if name == "spectral_entropy":
        return float(ant.spectral_entropy(x, sf=sfreq, method="welch", normalize=True))
    if name in ("hjorth_mobility", "hjorth_complexity"):
        mobility, complexity = ant.hjorth_params(x)
        return float(mobility if name == "hjorth_mobility" else complexity)
    binary = (x > np.median(x)).astype(np.uint8)
    return float(ant.lziv_complexity(binary, normalize=True))


class HilbertEnvelope(Processor):
    """Amplitude envelope |x + i H(x)| at the full sample rate (Signal out).

    The analytic signal is computed piecewise: every ``hop`` samples, over the hop
    plus ``lookahead`` seconds on either side (the stream's ends mirrored), keeping
    the hop. So the same code runs live and offline, its output lags the input by
    the look-ahead (output carries its own sample index, rule 8), and it matches a
    whole-recording Hilbert transform closely for band-limited input whose period
    is short against the look-ahead (put a band-pass before this strip).
    """

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.nodes.hilbert_envelope",
            "version": "0.1.0",
            "type": "stream",
            "description": "Amplitude envelope (Hilbert), delayed by the look-ahead.",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": [
                {"name": "lookahead", "type": "float", "default": 0.5, "min": 0.01,
                 "unit": "s", "primary": True,
                 "description": "Context on each side; also the output delay"},
            ],
            "availability": "both",
        }
    )  # fmt: skip

    @staticmethod
    def _sizes(params: Params, sfreq: float) -> int:
        return max(1, round(float(params["lookahead"]) * sfreq))

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        look = self._sizes(params, block.info.sfreq)
        if state is None:
            state = {"buffer": np.zeros((block.info.n_channels, 0)), "padded": False,
                     "next": block.sample_index, "start": block.sample_index,
                     "t_ref": block.timestamp, "i_ref": block.sample_index}  # fmt: skip
        buffer = np.concatenate([np.asarray(state["buffer"]), block.data], axis=1)
        padded = bool(state["padded"])
        nxt = int(state["next"])  # type: ignore[arg-type]
        if not padded and buffer.shape[1] > look:  # mirror the stream's start
            buffer = np.concatenate([buffer[:, look:0:-1], buffer], axis=1)
            padded = True
        hops = []
        # buffer column 0 is sample nxt - look once padded
        while padded and buffer.shape[1] >= 3 * look:
            segment = buffer[:, : 3 * look]
            hops.append(np.abs(sps.hilbert(segment, axis=1))[:, look : 2 * look])
            buffer = buffer[:, look:]
        out = self._block(hops, nxt, look, state, block.info)
        nxt += sum(h.shape[1] for h in hops)
        new_state: State = {**state, "buffer": buffer, "padded": padded, "next": nxt}
        return TransformResult(out, new_state)

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        if state is None:
            return None
        look = self._sizes(params, info.sfreq)
        buffer = np.asarray(state["buffer"])
        nxt = int(state["next"])  # type: ignore[arg-type]
        if bool(state["padded"]):  # mirror the end, keep what follows the context
            tail = buffer[:, -2 : -look - 2 : -1]
            env = np.abs(sps.hilbert(np.concatenate([buffer, tail], axis=1), axis=1))
            hops = [env[:, look : env.shape[1] - tail.shape[1]]]
        else:  # shorter than the look-ahead: one transform over all of it
            hops = [np.abs(sps.hilbert(buffer, axis=1))] if buffer.shape[1] else []
        if not hops or hops[0].shape[1] == 0:
            return None
        out = self._block(hops, nxt, look, state, info)
        return TransformResult(out, {**state, "buffer": buffer[:, :0],
                                     "next": nxt + out.n_samples})  # fmt: skip

    @staticmethod
    def _block(
        hops: list[np.ndarray], nxt: int, look: int, state: State, info: SignalInfo
    ) -> Block:
        data = np.concatenate(hops, axis=1) if hops else np.zeros((info.n_channels, 0))
        t_ref, i_ref = float(state["t_ref"]), int(state["i_ref"])  # type: ignore[arg-type]
        return Block(data, info, nxt, t_ref + (nxt - i_ref) / info.sfreq)
