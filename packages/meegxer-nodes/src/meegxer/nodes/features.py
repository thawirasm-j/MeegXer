"""Feature nodes: PSD, band power, connectivity, complexity (Features out) and the
Hilbert envelope (Signal out).

The analysis nodes share one windowing scheme: a window of ``window`` seconds
every ``step`` seconds, counted from the first sample of the stream. The input is
buffered in ``state`` and each call returns the windows its block completes, so
the windows never depend on block sizes (acceptance test 1). Each window is a
FeatureBlock with ``window`` = its sample range and ``timestamp`` = the LSL time of
its first sample. Bad channels are kept, so the feature layout stays fixed.

Spectra follow MATLAB's ``pwelch`` as EEGLAB's ``spectopo`` calls it: symmetric
Hamming (or Hann) segments, no detrending, one-sided density in V^2/Hz. The
multitaper option is MNE's ``psd_array_multitaper`` (DPSS tapers, mean removed,
same density units).
"""

from __future__ import annotations

import functools
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from scipy import signal as sps

from meegxer.core.block import Block, FeatureBlock, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Mode, Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes.references import credits

WINDOW_PARAMS: list[dict[str, object]] = [
    {"name": "window", "type": "float", "default": 2.0, "min": 0.05, "unit": "s",
     "description": "Length of each analysis window"},
    {"name": "step", "type": "float", "default": 0.5, "min": 0.01, "unit": "s",
     "description": "Time from one window to the next"},
]  # fmt: skip
SPECTRUM_PARAMS: list[dict[str, object]] = [
    {"name": "method", "type": "enum", "default": "welch",
     "choices": ["welch", "multitaper"],
     "description": "Welch (averaged tapered segments) or multitaper (DPSS tapers "
                    "over the whole window)"},
    {"name": "bandwidth", "type": "float", "default": 0.0, "min": 0.0, "unit": "Hz",
     "advanced": True,
     "description": "Multitaper bandwidth; 0 uses 8 / window (MNE's default)"},
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
    **fields: object,
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
            **credits(name),
            **fields,
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


def multitaper(
    x: np.ndarray, sfreq: float, params: Params
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided multitaper PSD (V^2/Hz) of each row (MNE, DPSS tapers)."""
    import mne
    from mne.time_frequency import psd_array_multitaper

    bandwidth = float(params.get("bandwidth", 0.0)) or None
    with mne.utils.use_log_level("error"):
        psd, freqs = psd_array_multitaper(
            x, sfreq, bandwidth=bandwidth, normalization="full", verbose=False
        )
    return freqs, psd


def spectrum(
    x: np.ndarray, sfreq: float, params: Params
) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, PSD in V^2/Hz) of each row by the node's ``method``."""
    if params.get("method", "welch") == "multitaper":
        return multitaper(x, sfreq, params)
    return welch(x, sfreq, params)


class WindowedFeatures(Processor):
    """Base for analysis nodes: buffering and windowing; subclasses compute one
    window's features with ``features``."""

    manifest: ClassVar[Manifest]

    @abstractmethod
    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        """(data, dims, coords) for one window ``x`` (channels x samples)."""

    def fitted_features(
        self, x: np.ndarray, info: SignalInfo, params: Params, model: State
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        """As ``features``, for a node with a fitted model. Default: ``features``."""
        return self.features(x, info, params)

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        """The unit of the values (``FeatureBlock.unit``), or ``(dim, units)`` when
        each entry of ``dim`` has its own. Default: not stated."""
        return ""

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
            values, dims, coords = self.fitted_features(x, info, params, model)
            units = self.units(params, coords)
            unit, unit_dim, per = (units, "", ()) if isinstance(units, str) else (
                "", *units)  # fmt: skip
            out.append(FeatureBlock(values, dims, (nxt, nxt + n),
                                    t_ref + (nxt - i_ref) / info.sfreq,
                                    coords, info.stream_id, unit, unit_dim,
                                    tuple(per)))  # fmt: skip
            nxt += step
        keep = min(max(nxt - start, 0), buffer.shape[1])
        buffer, start = buffer[:, keep:], start + keep
        if nxt > start:  # the next window starts beyond what we hold
            start = nxt
        new_state: State = {"buffer": buffer, "buffer_start": start, "next": nxt,
                            "t_ref": t_ref, "i_ref": i_ref}  # fmt: skip
        return TransformResult(tuple(out), new_state)


class PSD(WindowedFeatures):
    """Power spectral density per channel and window (Welch, pwelch-compatible, or
    multitaper)."""

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
        "Power spectral density per channel (Welch or multitaper).",
    )  # fmt: skip

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        freqs, psd = spectrum(x, info.sfreq, params)
        keep = (freqs >= float(params["fmin"])) & (freqs <= float(params["fmax"]))
        values = psd[:, keep]
        if params["db"]:
            values = 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
        rounded = tuple(round(float(f), 6) for f in freqs[keep])
        coords = {"channel": info.ch_labels, "frequency": rounded}
        return values, ("channel", "frequency"), coords

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        return "dB" if params["db"] else "V^2/Hz"


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
    """Power in named bands per channel and window: the PSD summed over the
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
        freqs, psd = spectrum(x, info.sfreq, params)
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

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        if params["db"]:
            return "dB"
        return "1" if params["relative"] else "V^2"


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

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        return "1"


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

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        per = {"hjorth_mobility": "1/sample"}  # antropy: per sample, not per second
        return "measure", tuple(per.get(m, "1") for m in coords["measure"])


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


@functools.lru_cache(maxsize=32)
def hilbert_fir(delay: int, low: float, sfreq: float) -> np.ndarray:
    """A type III FIR Hilbert transformer of ``2 * delay + 1`` taps passing
    ``low`` to Nyquist minus ``low`` (Parks-McClellan, ``scipy.signal.remez``)."""
    nyquist = sfreq / 2
    if not 0 < low < nyquist / 2:
        raise ValueError(f"low must lie between 0 and a quarter of the sampling "
                         f"rate ({nyquist / 2:g} Hz)")  # fmt: skip
    taps = sps.remez(2 * delay + 1, [low, nyquist - low], [1.0], type="hilbert",
                     fs=sfreq, maxiter=200)  # fmt: skip
    taps.setflags(write=False)  # shared by every caller
    return np.asarray(taps)


class HilbertEnvelope(Processor):
    """Amplitude envelope |x + i H(x)| at the full sample rate (Signal out).

    Live (and offline with ``method`` ``fir``): a Parks-McClellan FIR Hilbert
    transformer (McClellan, Parks & Rabiner 1973) with carried state, exact
    block by block. Its output lags the input by the transformer's delay
    (``lookahead``, half its length); the output carries its own sample index
    (rule 8), and the first ``lookahead`` of the stream sees zeros before it.
    It passes ``low`` to Nyquist minus ``low``: put a band-pass before it.
    Offline with ``method`` ``auto``: the analytic signal over the whole
    recording (``HilbertEnvelopeAnalytic``).
    """

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.nodes.hilbert_envelope",
            "version": "0.2.0",
            "type": "stream",
            "description": "Amplitude envelope (Hilbert): an FIR Hilbert "
                           "transformer live, the analytic signal of the whole "
                           "recording offline (method auto).",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": [
                {"name": "lookahead", "type": "float", "default": 0.5, "min": 0.01,
                 "unit": "s", "primary": True,
                 "description": "Delay of the FIR transformer (half its length); "
                                "longer reaches lower frequencies"},
                {"name": "low", "type": "float", "default": 2.0, "min": 0.01,
                 "unit": "Hz",
                 "description": "Lowest frequency the FIR transformer passes (and "
                                "the same below Nyquist)"},
                {"name": "method", "type": "enum", "default": "auto",
                 "choices": ["auto", "fir", "analytic"],
                 "description": "auto: FIR transformer live, analytic signal of "
                                "the whole recording offline; analytic: offline "
                                "only"},
            ],
            "availability": "both",
            **credits("meegxer.nodes.hilbert_envelope"),
        }
    )  # fmt: skip

    def for_mode(self, params: Params, mode: Mode) -> Processor:
        method = params["method"]
        if mode == "online":
            if method == "analytic":
                raise ValueError("the analytic signal needs the whole recording; "
                                 "use method auto or fir live")  # fmt: skip
            return self
        return self if method == "fir" else HilbertEnvelopeAnalytic()

    @staticmethod
    def _taps(params: Params, sfreq: float) -> tuple[int, np.ndarray]:
        delay = max(1, round(float(params["lookahead"]) * sfreq))
        return delay, hilbert_fir(delay, float(params["low"]), float(sfreq))

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        delay, taps = self._taps(params, block.info.sfreq)
        n_ch = block.info.n_channels
        if state is None:  # zeros before the stream's first sample
            state = {"zi": np.zeros((n_ch, taps.size - 1)),
                     "tail": np.zeros((n_ch, delay)), "start": block.sample_index,
                     "seen": 0,
                     "t_ref": block.timestamp, "i_ref": block.sample_index}  # fmt: skip
        out, new_state = self._run(block.data, block.sample_index, delay, taps,
                                   state, block.info)  # fmt: skip
        return TransformResult(out, new_state)

    def flush(
        self, info: SignalInfo, model: State, state: State | None, params: Params
    ) -> TransformResult | None:
        """The last ``lookahead`` of the stream: zeros fed in after its end."""
        if state is None:
            return None
        delay, taps = self._taps(params, info.sfreq)
        n_ch = info.n_channels
        nxt = int(state["start"]) + int(state["seen"])  # type: ignore[arg-type]
        out, new_state = self._run(np.zeros((n_ch, delay)), nxt, delay, taps, state,
                                   info)  # fmt: skip
        return TransformResult(out, new_state) if out.n_samples else None

    @staticmethod
    def _run(
        x: np.ndarray,
        first: int,
        delay: int,
        taps: np.ndarray,
        state: State,
        info: SignalInfo,
    ) -> tuple[Block, State]:
        """The envelope of input samples ``first...``: the transformer's output
        for each, paired with the input ``delay`` samples before it."""
        start = int(state["start"])  # type: ignore[arg-type]
        t_ref, i_ref = float(state["t_ref"]), int(state["i_ref"])  # type: ignore[arg-type]
        if not x.shape[1]:  # an empty block (from a delaying node upstream)
            at = max(first - delay, start)
            return Block(x, info, at, t_ref + (at - i_ref) / info.sfreq), state
        imag, zi = sps.lfilter(taps, [1.0], x, axis=1, zi=np.asarray(state["zi"]))
        real = np.concatenate([np.asarray(state["tail"]), x], axis=1)
        env = np.hypot(real[:, : x.shape[1]], imag)
        index = first - delay  # the input sample env[:, 0] belongs to
        skip = min(max(start - index, 0), x.shape[1])  # before the stream's start
        at = max(index + skip, start)  # all before the start: an empty block there
        out = Block(env[:, skip:], info, at, t_ref + (at - i_ref) / info.sfreq)
        seen = int(state["seen"]) + x.shape[1]  # type: ignore[arg-type]
        new_state: State = {**state, "zi": zi, "tail": real[:, real.shape[1] - delay :],
                            "seen": seen}  # fmt: skip
        return out, new_state


class HilbertEnvelopeAnalytic(Processor):
    """Envelope from the analytic signal of the whole recording (offline only):
    ``abs(scipy.signal.hilbert(x))``, the FFT method (Marple 1999), as MATLAB's
    and Octave's ``hilbert``. No delay; the recording's ends have the
    transform's own edge effects."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": "meegxer.nodes.hilbert_envelope_analytic",
            "version": "0.1.0",
            "type": "stream",
            "description": "Amplitude envelope from the analytic signal of the "
                           "whole recording; offline only.",
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": [],
            "availability": "offline",
            **credits("meegxer.nodes.hilbert_envelope_analytic"),
        }
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        env = np.abs(sps.hilbert(block.data, axis=1)) if block.n_samples else block.data
        return TransformResult(
            Block(env, block.info, block.sample_index, block.timestamp), {}
        )
