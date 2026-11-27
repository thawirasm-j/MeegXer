"""A pipeline as a standalone MNE script (M11.7 step 2; ``docs/PYTHON_API.md``).

Decided 2026-11-27: each strip writes its own MNE/SciPy code (the
``export_code`` hook of ``Processor``, ``Source`` and ``Sink``), next to the
code it reproduces, and each strip's code is tested to give the strip's
offline output; fitted models are fitted again in the script; a strip without
code is named, and the thin export (``meegxer export``, which needs MeegXer)
runs any pipeline. Strips come PR by PR (``docs/ROADMAP.md``).

The script reads the recording into ``raw`` (an ``mne.io.Raw``, preloaded,
in volts) and runs each strip's lines in chain order. Before a strip's lines,
``picks`` holds the indices of the channels it processes (as MeegXer: the
strip's channel types, EEG by default; the rest pass through untouched). An
Epochs strip sets ``epochs``; a writer writes what is there at its place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from meegxer.core.block import CHANNEL_TYPES
from meegxer.core.ports import PortType


@dataclass(frozen=True)
class ExportCode:
    """A strip's code in a standalone MNE script."""

    comment: str  # what it does, in one line (with the method and library)
    lines: tuple[str, ...]  # statements on ``raw`` (and ``epochs``, ``picks``)
    imports: tuple[str, ...] = ()  # e.g. "from scipy import signal"
    helpers: tuple[str, ...] = ()  # names in HELPERS the lines call
    # module-level code of its own (a custom script's source, inlined), each
    # written once even when several strips give the same
    definitions: tuple[str, ...] = ()
    # what the script needs besides MNE, NumPy and SciPy (pip names)
    packages: tuple[str, ...] = ()
    # the code instead, when an ICA strip before it may hold a decomposition
    # (`ica`): a .set writer stores it, as MeegXer's does
    with_ica: ExportCode | None = None


class NotExportable(ValueError):
    """The pipeline has strips without standalone code (yet)."""


HELPERS: dict[str, str] = {
    "channels_of": '''def channels_of(raw, types):
    """The indices of the channels of these types (the channels a strip
    processes; the others pass through untouched)."""
    return [i for i, kind in enumerate(raw.get_channel_types()) if kind in types]
''',
    "causal_sos": '''def causal_sos(sos, x):
    """Causal IIR filtering, its delay line started at each channel's first
    sample (scipy.signal.sosfilt with sosfilt_zi)."""
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    return signal.sosfilt(sos, x, axis=-1, zi=zi)[0]
''',
    "causal_fir": '''def causal_fir(h, x):
    """Causal FIR filtering, its delay line started at each channel's first
    sample (scipy.signal.lfilter with lfilter_zi)."""
    zi = signal.lfilter_zi(h, [1.0])[None, :] * x[:, :1]
    return signal.lfilter(h, [1.0], x, axis=-1, zi=zi)[0]
''',
    "notch_sos": '''def notch_sos(freq, quality, harmonics, sfreq):
    """IIR notches at freq and its harmonics below Nyquist, as one cascade
    (scipy.signal.iirnotch)."""
    sections = []
    for k in range(1, harmonics + 1):
        if freq * k >= sfreq / 2:
            break
        b, a = signal.iirnotch(freq * k, quality, fs=sfreq)
        sections.append(signal.tf2sos(b, a))
    return np.vstack(sections)
''',
    "resampled": '''def resampled(raw, sfreq):
    """raw at another rate: scipy.signal.resample_poly by the rates' exact
    ratio (Kaiser window, beta 5); channels, bads, montage and annotations kept."""
    ratio = Fraction(str(sfreq)) / Fraction(str(raw.info["sfreq"]))
    data = signal.resample_poly(raw.get_data(), ratio.numerator,
                                ratio.denominator, axis=-1)
    info = mne.create_info(raw.ch_names, sfreq, raw.get_channel_types())
    out = mne.io.RawArray(data, info, verbose="error")
    out.set_meas_date(raw.info["meas_date"])
    montage = raw.get_montage()
    if montage is not None:
        out.set_montage(montage, on_missing="ignore", verbose="error")
    out.info["bads"] = list(raw.info["bads"])
    out.set_annotations(raw.annotations)
    return out
''',
    "spline_channels": '''NOT_EEG = {"eog", "ecg", "emg", "misc", "stim", "resp", "bio",
           "gsr", "temperature"}


def spline_channels(raw, picks):
    """The channels among picks that spherical splines use: EEG ones with a
    position (as MeegXer: other types and channels without one pass through)."""
    kinds = raw.get_channel_types()
    out = []
    for i in picks:
        xyz = np.asarray(raw.info["chs"][i]["loc"][:3], dtype=float)
        if kinds[i] not in NOT_EEG and np.all(np.isfinite(xyz)) and np.any(xyz):
            out.append(i)
    return out


def fit_sphere(raw, idx):
    """Centre and radius of the sphere best fitting these channels' positions
    (least squares): the sphere the spline methods use."""
    points = np.array([raw.info["chs"][i]["loc"][:3] for i in idx], dtype=float)
    a = np.column_stack([2 * points, np.ones(len(points))])
    sol = np.linalg.lstsq(a, (points**2).sum(axis=1), rcond=None)[0]
    centre = sol[:3]
    return centre, float(np.sqrt(max(sol[3] + centre @ centre, 1e-12)))


def spline_copy(raw, idx, bads):
    """These channels of raw, typed EEG (what MNE's spline methods take), with
    these bad channels."""
    sub = raw.copy().pick(idx)
    sub.set_channel_types(dict.fromkeys(sub.ch_names, "eeg"), verbose="error")
    sub.info["bads"] = list(bads)
    return sub
''',
    "run_process": '''def run_process(raw, picks, process, **params):
    """A custom script's process(raw, ...), called as MeegXer calls it: on a
    Raw of the channels it processes (typed EEG, bad channels marked, no
    events); what it returns (a Raw, or an array of the same shape) replaces
    them, and the annotations it adds become events of raw."""
    names = [raw.ch_names[i] for i in picks]
    sfreq = raw.info["sfreq"]
    info = mne.create_info(names, sfreq, "eeg")
    info["bads"] = [b for b in raw.info["bads"] if b in names]
    sub = mne.io.RawArray(raw.get_data(picks=picks), info, verbose="error")
    got = process(sub, **params)
    out = sub if got is None else got
    added = []
    if isinstance(out, mne.io.BaseRaw):
        if list(out.ch_names) != names or out.info["sfreq"] != sfreq:
            raise SystemExit(f"{process.__module__}: process must keep the "
                             "channels and the sampling rate")
        added = list(out.annotations)
        out = out.get_data()
    data = np.asarray(out, dtype=np.float64)
    if data.shape != (len(picks), raw.n_times):
        raise SystemExit(f"{process.__module__}: process must return "
                         f"{len(picks)} channels x {raw.n_times} samples")
    raw.apply_function(lambda x: data, picks=picks, channel_wise=False)
    for a in added:  # at whole samples, as MeegXer's events
        raw.annotations.append(round(a["onset"] * sfreq) / sfreq + raw.first_time,
                               round(a["duration"] * sfreq) / sfreq,
                               a["description"])
    return raw
''',
    "run_transform": '''def run_transform(raw, picks, transform, **params):
    """A live custom script's transform(data, state, ...) on the whole
    recording at once (what MeegXer's block by block run gives: the script
    passed acceptance test 1)."""
    got = transform(raw.get_data(picks=picks), None, **params)
    data = np.asarray(got[0] if isinstance(got, tuple) else got, dtype=np.float64)
    raw.apply_function(lambda x: data, picks=picks, channel_wise=False)
    return raw
''',
    "run_process_epochs": '''def run_process_epochs(epochs, process_epochs, **params):
    """A custom script's process_epochs(epochs, ...), called as MeegXer calls
    it: on the epochs kept, their channels typed EEG, labelled by the events
    they have (their data as they are: baseline=None); the epochs it returns
    get the recording's channel types and positions back."""
    sfreq = epochs.info["sfreq"]
    present = set(epochs.events[:, 2])
    view = mne.EpochsArray(epochs.get_data(copy=True),
                           mne.create_info(epochs.ch_names, sfreq, "eeg"),
                           epochs.events, tmin=epochs.tmin, baseline=None,
                           event_id={k: v for k, v in epochs.event_id.items()
                                     if v in present}, verbose="error")
    got = process_epochs(view, **params)
    out = view if got is None else got
    if not isinstance(out, mne.BaseEpochs) or out.info["sfreq"] != sfreq:
        raise SystemExit(f"{process_epochs.__module__}: process_epochs must "
                         "return mne.Epochs at the same rate")
    kinds = dict(zip(epochs.ch_names, epochs.get_channel_types()))
    out.set_channel_types({c: kinds[c] for c in out.ch_names if c in kinds},
                          verbose="error")
    montage = epochs.get_montage()
    if montage is not None:
        out.set_montage(montage, on_missing="ignore", verbose="error")
    return out
''',
    "fit_ica": '''ICLABEL_CLASSES = ("brain", "muscle", "eye", "heart", "line_noise",
                   "channel_noise", "other")  # ICLabel's output columns


def fit_ica(raw, picks, method, n_components, max_iter, seed):
    """ICA of the strip's good channels, as MeegXer fits it: MNE's ICA
    (extended Picard, or extended Infomax) on the data in microvolts, the
    components scaled to RMS microvolts as EEGLAB does, then classified by
    ICLabel (mne-icalabel): channels, weights, winv and probabilities."""
    good = [i for i in picks if raw.ch_names[i] not in raw.info["bads"]]
    if len(good) < 2:
        raise SystemExit("ICA needs at least two good channels")
    x = raw.get_data(picks=good)
    s = np.linalg.svd(x - x.mean(axis=1, keepdims=True), compute_uv=False)
    rank = int((s > s[0] * 1e-6).sum())
    n = n_components or rank
    if n > rank:
        raise SystemExit(f"n_components={n} exceeds the data rank {rank}")
    work = mne.io.RawArray(x * 1e6, mne.create_info(len(good), raw.info["sfreq"],
                                                    "eeg"), verbose="error")
    fit_params = {"extended": True}
    if method == "picard":
        fit_params["ortho"] = False  # extended Picard = Extended Infomax's model
    ica = mne.preprocessing.ICA(n_components=n, method=method,
                                fit_params=fit_params, max_iter=max_iter,
                                random_state=seed, verbose="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes
        ica.fit(work, verbose="error")
    weights = ica.unmixing_matrix_ @ ica.pca_components_[: ica.n_components_]
    weights = weights / ica.pre_whitener_.T
    scaling = np.sqrt(np.mean(np.linalg.pinv(weights) ** 2, axis=0))
    weights = weights * scaling[:, None]  # EEGLAB's option_scaleicarms
    return {"channels": [raw.ch_names[i] for i in good], "weights": weights,
            "winv": np.linalg.pinv(weights),
            "probabilities": iclabel(raw, good, weights)}


def iclabel(raw, good, weights):
    """ICLabel's probabilities (components x 7 classes, mne-icalabel) for
    weights in EEGLAB's convention, given to it as
    mne.preprocessing.read_ica_eeglab builds an ICA."""
    from mne_icalabel.iclabel import iclabel_label_components

    names = [raw.ch_names[i] for i in good]
    positions = {}
    for i in good:
        xyz = np.asarray(raw.info["chs"][i]["loc"][:3], dtype=float)
        if np.all(np.isfinite(xyz)) and np.any(xyz):
            positions[raw.ch_names[i]] = xyz
    missing = [c for c in names if c not in positions]
    if missing:
        raise SystemExit("ICLabel needs electrode positions; set a montage or "
                         f"positions for {', '.join(missing)}")
    sub = mne.io.RawArray(raw.get_data(picks=good),
                          mne.create_info(names, raw.info["sfreq"], "eeg"),
                          verbose="error")
    sub.set_montage(mne.channels.make_dig_montage(positions, coord_frame="head"),
                    verbose="error")
    n_comp, n_ch = weights.shape
    ica = mne.preprocessing.ICA(method="picard",
                                fit_params={"extended": True, "ortho": False})
    ica.current_fit = "raw"
    ica.ch_names = names
    ica.n_pca_components = None
    ica.n_components_ = n_comp
    ica.pre_whitener_ = np.ones((n_ch, 1))
    ica.pca_mean_ = np.zeros(n_ch)
    u, s, v = np.linalg.svd(weights, full_matrices=False)
    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica.info = sub.info
    ica._update_mixing_matrix()
    ica._update_ica_names()
    ica.reject_ = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reference and filter advice
        proba = iclabel_label_components(sub, ica, inplace=False, backend="onnx")
    return np.asarray(proba, dtype=np.float64)


def ica_rejected(ica, threshold, classes):
    """The components whose probability for any of these classes reaches the
    threshold."""
    cols = [ICLABEL_CLASSES.index(c) for c in classes]
    if not cols:
        return []
    hit = (ica["probabilities"][:, cols] >= threshold).any(axis=1)
    return [int(i) for i in np.flatnonzero(hit)]
''',
    "add_ica_to_set": '''def add_ica_to_set(path, raw, ica):
    """Store the ICA in the .set just written (mne.export leaves its fields
    empty), as MeegXer's .set writer does: icaweights (an identity
    icasphere), icawinv, icachansind and ICLabel's classes, the components
    removed from the data left out (as EEGLAB's pop_subcomp)."""
    from scipy.io import loadmat, savemat

    labels = list(raw.ch_names)
    missing = [c for c in ica["channels"] if c not in labels]
    if missing:
        print(f"ICA decomposition not stored in {path}: the file has no "
              f"channels {', '.join(missing)}")
        return
    keep = [k for k in range(len(ica["weights"])) if k not in ica["removed"]]
    names = np.empty((1, 7), dtype=object)
    names[0, :] = ["Brain", "Muscle", "Eye", "Heart", "Line Noise",
                   "Channel Noise", "Other"]
    fields = {
        "icaweights": ica["weights"][keep],
        "icasphere": np.eye(len(ica["channels"])),
        "icawinv": ica["winv"][:, keep],
        "icachansind": np.array([[labels.index(c) + 1 for c in ica["channels"]]],
                                dtype=np.float64),
        "icaact": np.zeros((0, 0)),
    }
    if ica["probabilities"] is not None:  # ICLabel's classes, as EEGLAB keeps them
        fields["etc"] = {"ic_classification": {"ICLabel": {
            "classes": names, "classification": ica["probabilities"][keep],
            "version": "default"}}}
    mat = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    if "EEG" in mat:  # saved as one struct
        eeg = mat["EEG"]
        mat["EEG"] = {**{n: eeg[n][0, 0] for n in eeg.dtype.names}, **fields}
    else:  # top-level fields (as mne.export writes them)
        mat.update(fields)
    savemat(path, mat, oned_as="row")
''',
    "detectors": '''def detector_raw(raw, names, kind):
    """These channels as a Raw of their own, typed kind and timed from 0:
    what MeegXer gives MNE's detectors."""
    return mne.io.RawArray(raw.get_data(picks=names),
                           mne.create_info(names, raw.info["sfreq"], kind),
                           verbose="error")


def add_spans(raw, found):
    """A detector's annotations (seconds from the data's start) added to
    raw as events, at whole samples and clipped to the data, as MeegXer's."""
    sfreq, n = raw.info["sfreq"], raw.n_times
    for onset, duration, label in zip(found.onset, found.duration,
                                      found.description):
        start = min(max(round(onset * sfreq), 0), n)
        stop = min(max(round((onset + duration) * sfreq), start), n)
        if stop > start:
            raw.annotations.append(start / sfreq + raw.first_time,
                                   (stop - start) / sfreq, str(label))


def good_channels(raw, picks):
    return [raw.ch_names[i] for i in picks if raw.ch_names[i] not in raw.info["bads"]]


def eog_channels(raw, picks, eog):
    """The EOG channels: auto (good channels typed or named EOG) or labels."""
    if eog.strip().lower() == "auto":
        kinds = raw.get_channel_types()
        found = [raw.ch_names[i] for i in picks
                 if raw.ch_names[i] not in raw.info["bads"]
                 and (kinds[i] == "eog"
                      or re.match(r"^(V|H)?EOG", raw.ch_names[i], re.IGNORECASE))]
    else:
        found = [c.strip() for c in eog.split(",") if c.strip()]
        names = [raw.ch_names[i] for i in picks]
        missing = [c for c in found if c not in names]
        if missing:
            raise SystemExit(f"no channel named {', '.join(missing)}")
    if not found:
        raise SystemExit("no EOG channel found: name one in the eog setting")
    return found
''',
    "windowed": '''def windowed(raw, picks, window, step, compute, unit):
    """A feature strip's windows, as MeegXer cuts them: window seconds every
    step seconds from the first sample, bad channels kept. compute(x, sfreq,
    names) gives one window's (values, dims, coords); unit is every value's
    unit, or (dim, units) when each entry of dim has its own."""
    sfreq = raw.info["sfreq"]
    n, hop = max(1, round(window * sfreq)), max(1, round(step * sfreq))
    x, names = raw.get_data(picks=picks), tuple(raw.ch_names[i] for i in picks)
    out = {"windows": [], "timestamps": [], "values": [], "dims": (),
           "coords": {}, "unit": unit}
    for start in range(0, x.shape[1] - n + 1, hop):
        values, out["dims"], out["coords"] = compute(x[:, start : start + n],
                                                     sfreq, names)
        out["windows"].append((start, start + n))
        out["timestamps"].append(0.0 + start / sfreq)  # seconds from the start
        out["values"].append(values)
    out["values"] = np.array(out["values"])  # windows x the window's dims
    if callable(unit):  # from the coordinates (a feature script's UNITS)
        out["unit"] = unit(out["coords"]) if out["windows"] else ""
    return out


def spectrum(x, sfreq, method, segment, overlap, taper, bandwidth):
    """(freqs, one-sided PSD in V^2/Hz) of each row: Welch as MATLAB's pwelch
    computes it (symmetric taper, no detrending), or MNE's multitaper."""
    if method == "multitaper":
        from mne.time_frequency import psd_array_multitaper

        psd, freqs = psd_array_multitaper(x, sfreq, bandwidth=bandwidth or None,
                                          normalization="full", verbose="error")
        return freqs, psd
    nper = min(x.shape[-1], max(2, round(segment * sfreq)))
    noverlap = min(nper - 1, round(overlap * nper))
    window = signal.get_window(taper, nper, fftbins=False)
    return signal.welch(x, sfreq, window=window, nperseg=nper, noverlap=noverlap,
                        detrend=False, scaling="density", axis=-1)


def decibels(values):
    return 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
''',
    "psd_window": '''def psd_window(x, sfreq, names, fmin, fmax, db, **spec):
    """One window's PSD per channel, fmin to fmax."""
    freqs, psd = spectrum(x, sfreq, **spec)
    keep = (freqs >= fmin) & (freqs <= fmax)
    values = decibels(psd[:, keep]) if db else psd[:, keep]
    rounded = tuple(round(float(f), 6) for f in freqs[keep])
    coords = {"channel": names, "frequency": rounded}
    return values, ("channel", "frequency"), coords
''',
    "band_power_window": '''\
def band_power_window(x, sfreq, names, bands, relative, db, **spec):
    """One window's power per channel and band: the PSD summed over the
    band's bins [low, high) times the bin width; relative to the power over
    all the bands' span, or in dB."""
    freqs, psd = spectrum(x, sfreq, **spec)
    df = freqs[1] - freqs[0]
    values = np.stack([psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df
                       for _, lo, hi in bands], axis=1)
    if relative:
        lo, hi = min(b[1] for b in bands), max(b[2] for b in bands)
        total = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) * df
        values = values / np.maximum(total, np.finfo(float).tiny)[:, None]
    if db:
        values = decibels(values)
    labels = tuple(b[0] for b in bands)
    coords = {"channel": names, "band": labels}
    return values, ("channel", "band"), coords
''',
    "morlet_window": '''\
def morlet_window(x, sfreq, names, fmin, fmax, fstep, n_cycles, db):
    """One window's Morlet wavelet power per channel and frequency (MNE's
    zero-mean wavelets), averaged where the whole wavelet lies in the window."""
    from mne.time_frequency import morlet

    freqs = np.arange(fmin, fmax + fstep / 2, fstep)
    wavelets = morlet(sfreq, freqs, n_cycles=n_cycles, zero_mean=True)
    if max(len(w) for w in wavelets) > x.shape[1]:
        raise SystemExit(f"the {freqs[0]:g} Hz wavelet is longer than the window: "
                         "lengthen the window, raise fmin or lower the cycles")
    power = np.empty((x.shape[0], len(freqs)))
    for j, w in enumerate(wavelets):
        conv = signal.fftconvolve(x, w[None, :], mode="valid", axes=1)
        power[:, j] = np.mean(np.abs(conv) ** 2, axis=1)
    if db:
        power = decibels(power)
    rounded = tuple(round(float(f), 6) for f in freqs)
    coords = {"channel": names, "frequency": rounded}
    return power, ("channel", "frequency"), coords
''',
    "specparam_window": '''\
def specparam_window(x, sfreq, names, fmin, fmax, aperiodic, bands, model,
                     **spec):
    """One window's spectral parameters per channel (specparam): the
    aperiodic fit, the strongest peak in each band (NaN without one), R² and
    the error; a failed fit leaves NaNs."""
    from specparam import SpectralModel

    freqs, psd = spectrum(x, sfreq, **spec)
    keep = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    if keep.sum() < 4:
        raise SystemExit("specparam needs at least 4 frequencies in its range")
    head = ["offset", "knee", "exponent"] if aperiodic == "knee" else ["offset",
                                                                      "exponent"]
    measures = (*head, *(f"{b}_{f}" for b, _, _ in bands
                         for f in ("frequency", "power", "bandwidth")),
                "r_squared", "error")
    rows = np.full((x.shape[0], len(measures)), np.nan)
    psd = np.maximum(psd, np.finfo(float).tiny)
    for i, power in enumerate(psd[:, keep]):
        fm = SpectralModel(aperiodic_mode=aperiodic, verbose=False, **model)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                fm.fit(freqs[keep], power)
            except Exception:  # a failed fit leaves NaNs
                continue
        if not fm.results.has_model:
            continue
        row = [float(v) for v in np.atleast_1d(fm.get_params("aperiodic"))]
        peaks = np.asarray(fm.get_params("periodic"), dtype=float).reshape(-1, 3)
        for _, lo, hi in bands:
            inside = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] < hi)]
            row.extend([float(v) for v in inside[np.argmax(inside[:, 1])]]
                       if len(inside) else [np.nan] * 3)
        rows[i] = [*row, float(fm.get_metrics("gof")), float(fm.get_metrics("error"))]
    return rows, ("channel", "measure"), {"channel": names, "measure": measures}
''',
    "write_features_tsv": '''def write_features_tsv(path, feats):
    """The windows as MeegXer's feature TSV: window_start, window_stop,
    timestamp, then one column per value named by its coordinates
    (Fz|10.0); the values' units in a JSON sidecar (BIDS style) when stated."""
    if not feats["windows"]:
        print(f"no windows: {path} not written")
        return
    shape = feats["values"].shape[1:]
    axes = [[str(v) for v in feats["coords"].get(dim, range(size))]
            for dim, size in zip(feats["dims"], shape)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1)
    columns = ["|".join(col) for col in grid.T]
    unit = feats["unit"]
    if isinstance(unit, str):
        units = [unit] * len(columns)
    else:
        axis = feats["dims"].index(unit[0])
        units = [unit[1][i] for i in np.indices(shape)[axis].ravel()]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if any(units):
        sidecar = {"window_start": {"Units": "sample"},
                   "window_stop": {"Units": "sample"}, "timestamp": {"Units": "s"},
                   **{c: {"Units": u} for c, u in zip(columns, units) if u}}
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(sidecar, indent=2) + "\\n")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\\t", lineterminator="\\n")
        out.writerow(["window_start", "window_stop", "timestamp", *columns])
        for (start, stop), ts, values in zip(feats["windows"], feats["timestamps"],
                                             feats["values"]):
            out.writerow([str(start), str(stop), repr(ts),
                          *("n/a" if np.isnan(v) else repr(float(v))
                            for v in values.ravel())])
    print(f"wrote {path}")
''',
    "events_of": '''\
def events_of(raw):
    """The events a strip sees, as MeegXer places them: (sample from the
    first, label, samples) at round(onset x rate), sorted by sample."""
    sfreq, out = raw.info["sfreq"], []
    for a in raw.annotations:
        onset = float(a["onset"] - raw.first_time)
        if onset < 0 or not str(a["description"]):
            continue
        out.append((round(onset * sfreq), str(a["description"]),
                    max(0, round(float(a["duration"]) * sfreq))))
    return sorted(out, key=lambda e: e[0])


def add_event(raw, sample, label):
    """An event at this sample (from the first), as an annotation of raw."""
    raw.annotations.append(sample / raw.info["sfreq"] + raw.first_time, 0.0, label)


def window_events(raw, feats, labels, least):
    """An event per window at its last sample, for its best value (its
    label from labels), when that value reaches least."""
    for (_, stop), values in zip(feats["windows"], feats["values"]):
        k = int(np.argmax(values))
        if values[k] >= least:  # at the window's last sample: decided then
            add_event(raw, max(stop - 1, 0), labels[k])
''',
    "connectivity_window": '''\
def connectivity_window(x, sfreq, names, method, fmin, fmax, segment):
    """One window's connectivity between channel pairs in a band
    (mne-connectivity): its segment-long pieces are the epochs averaged."""
    from mne_connectivity import spectral_connectivity_epochs

    seg = max(2, round(segment * sfreq))
    n_seg = x.shape[1] // seg
    if n_seg < 2:
        raise SystemExit("connectivity needs at least two segments per window")
    epochs = x[:, : n_seg * seg].reshape(x.shape[0], n_seg, seg).transpose(1, 0, 2)
    con = spectral_connectivity_epochs(epochs, method=method, sfreq=sfreq,
                                       fmin=fmin, fmax=fmax, faverage=True,
                                       verbose="error")
    dense = con.get_data(output="dense")[:, :, 0]
    full = dense - dense.T if method == "imcoh" else dense + dense.T
    np.fill_diagonal(full, 0.0 if method == "imcoh" else 1.0)
    coords = {"channel": names, "target": names}
    return np.asarray(full, dtype=float), ("channel", "target"), coords
''',
    "complexity_window": '''\
def complexity_window(x, sfreq, names, chosen):
    """One window's complexity measures per channel (antropy)."""
    import antropy as ant

    def measure(name, ch):
        if name == "perm_entropy":
            return float(ant.perm_entropy(ch, order=3, normalize=True))
        if name == "sample_entropy":
            return float(ant.sample_entropy(ch, order=2))
        if name == "spectral_entropy":
            return float(ant.spectral_entropy(ch, sf=sfreq, method="welch",
                                              normalize=True))
        if name in ("hjorth_mobility", "hjorth_complexity"):
            mobility, complexity = ant.hjorth_params(ch)
            return float(mobility if name == "hjorth_mobility" else complexity)
        binary = (ch > np.median(ch)).astype(np.uint8)
        return float(ant.lziv_complexity(binary, normalize=True))

    rows = [[measure(m, ch) for m in chosen] for ch in x]
    return np.array(rows), ("channel", "measure"), {"channel": names,
                                                    "measure": tuple(chosen)}
''',
    "ssvep_window": '''\
def ssvep_window(x, sfreq, names, bads, freqs, method, harmonics, subbands,
                 subband_low, subband_high):
    """One window's score per stimulus frequency: the largest canonical
    correlation with sine/cosine references (CCA), or filter-bank CCA (Chen
    et al. 2015: Chebyshev I sub-bands, weights m^-1.25 + 0.25)."""
    def basis(a):
        u, s, _ = np.linalg.svd(a - a.mean(axis=0), full_matrices=False)
        return u[:, :0] if not s.size or s[0] == 0 else u[:, s > s[0] * 1e-10]

    def correlation(a, b):
        qa, qb = basis(a), basis(b)
        if not qa.shape[1] or not qb.shape[1]:
            return 0.0
        return float(min(1.0, np.linalg.svd(qa.T @ qb, compute_uv=False)[0]))

    def references(freq, n):
        t, cols = np.arange(n) / sfreq, []
        for h in range(1, harmonics + 1):
            if h * freq >= sfreq / 2:
                break
            cols += [np.sin(2 * np.pi * h * freq * t), np.cos(2 * np.pi * h * freq * t)]
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    def subband(low, high):
        nyq = sfreq / 2
        stop_low = max(low - 2.0, low / 2)
        if high + 10 >= nyq:  # no room for an upper edge: high-pass only
            order, edge = signal.cheb1ord(low, stop_low, 3, 40, fs=sfreq)
            return signal.cheby1(order, 0.5, edge, btype="highpass", fs=sfreq,
                                 output="sos")
        order, edges = signal.cheb1ord([low, high], [stop_low, high + 10], 3, 40,
                                       fs=sfreq)
        return signal.cheby1(order, 0.5, edges, btype="bandpass", fs=sfreq,
                             output="sos")

    data = x[[i for i, ch in enumerate(names) if ch not in set(bads)]].T
    refs = [references(f, data.shape[0]) for f in freqs]
    if method == "cca":
        scores = [correlation(data, y) for y in refs]
    else:
        scores = [0.0] * len(freqs)
        for m in range(1, subbands + 1):
            if m * subband_low >= min(subband_high, 0.9 * sfreq / 2):
                break  # higher sub-bands lie beyond the top edge or Nyquist
            band = signal.sosfiltfilt(subband(m * subband_low, subband_high), data,
                                      axis=0)
            for k, y in enumerate(refs):
                scores[k] += (m**-1.25 + 0.25) * correlation(band, y) ** 2
    coords = {"frequency": tuple(round(f, 6) for f in freqs)}
    return np.asarray(scores, dtype=float), ("frequency",), coords
''',
    "csp": '''\
def fit_csp(raw, picks, classes, tmin, tmax, n_components, reg):
    """CSP filters as MeegXer fits them (MNE's CSP): on an epoch from tmin to
    tmax after each event of a class."""
    from mne.decoding import CSP

    x, sfreq = raw.get_data(picks=picks), raw.info["sfreq"]
    start, stop = round(tmin * sfreq), round(tmax * sfreq)
    if stop - start < 2:
        raise SystemExit("CSP: tmax must lie after tmin")
    epochs, labels = [], []
    for sample, label, _ in events_of(raw):
        lo = sample + start
        if label in classes and lo >= 0 and lo + stop - start <= x.shape[1]:
            epochs.append(x[:, lo : lo + stop - start])
            labels.append(classes.index(label))
    few = [c for i, c in enumerate(classes) if labels.count(i) < 2]
    if few:
        raise SystemExit(f"CSP needs at least 2 trials of each class; too few of {few}")
    if n_components > x.shape[0]:
        raise SystemExit(f"CSP: {n_components} components but {x.shape[0]} channels")
    csp = CSP(n_components=n_components, reg=reg, log=True)
    with mne.utils.use_log_level("error"):
        csp.fit(np.stack(epochs), np.array(labels))
    return np.asarray(csp.filters_[:n_components], dtype=float)


def csp_window(x, sfreq, names, filters):
    """One window's log power of each CSP component (as MNE's CSP.transform
    with log=True computes it for an epoch)."""
    power = np.mean((filters @ x) ** 2, axis=1)
    values = np.log(np.maximum(power, np.finfo(float).tiny))
    components = tuple(f"CSP{i + 1}" for i in range(filters.shape[0]))
    return values, ("component",), {"component": components}
''',
    "lda": '''\
def lda_windows(raw, feats, classes, tmin, tmax, shrinkage, least, prefix):
    """LDA as MeegXer fits it (scikit-learn, Ledoit-Wolf shrinkage or plain):
    on the windows of the feature strip it reads lying wholly from tmin to
    tmax after an event of one class; then each window's class
    probabilities, and a class event at its last sample."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    sfreq = raw.info["sfreq"]
    lo, hi = round(tmin * sfreq), round(tmax * sfreq)
    if hi <= lo:
        raise SystemExit("LDA: tmax must lie after tmin")
    trials = [(s + lo, s + hi, classes.index(label))
              for s, label, _ in events_of(raw) if label in classes]
    flat = np.asarray(feats["values"]).reshape(len(feats["windows"]), -1)
    rows, labels = [], []
    for (start, stop), row in zip(feats["windows"], flat):
        on = {k for a, b, k in trials if a <= start and stop <= b}
        if len(on) == 1:
            rows.append(row)
            labels.append(on.pop())
    few = [c for i, c in enumerate(classes) if labels.count(i) < 2]
    if few:
        raise SystemExit("LDA needs at least 2 windows of each class; too few "
                         f"of {few}")
    lda = LinearDiscriminantAnalysis(solver="lsqr",
                                     shrinkage="auto" if shrinkage else None)
    lda.fit(np.stack(rows), np.array(labels))
    out = dict(feats, values=lda.predict_proba(flat), dims=("class",),
               coords={"class": tuple(classes)}, unit="1")
    window_events(raw, out, [prefix + c for c in classes], least)
    return out
''',
    "run_features": '''\
def run_features(x, sfreq, names, bads, features, **params):
    """A custom script's features(data, ...) on one window, as MeegXer calls
    it: the settings its signature takes, and info if it takes it; what it
    returns (name -> one value per channel, or channels x features) as a
    table."""
    accepted = inspect.signature(features).parameters
    kwargs = {k: v for k, v in params.items() if k in accepted}
    if "info" in accepted:
        kwargs["info"] = {"ch_labels": list(names), "sfreq": sfreq, "stream_id": "",
                          "bads": [b for b in bads if b in names],
                          "sample_index": 0, "timestamp": 0.0}
    got = features(x.copy(), **kwargs)
    if isinstance(got, dict):
        columns = [np.asarray(v, dtype=np.float64).reshape(-1) for v in got.values()]
        if any(c.size != len(names) for c in columns):
            raise SystemExit(f"{features.__module__}: give one value per channel")
        values, labels = np.stack(columns, axis=1), [str(k) for k in got]
    else:
        values = np.asarray(got, dtype=np.float64)
        values = values[:, None] if values.ndim == 1 else values
        labels = [f"feature_{i}" for i in range(values.shape[1])]
    return values, ("channel", "feature"), {"channel": names, "feature": tuple(labels)}


def script_unit(module):
    """A feature script's UNITS: one for every feature, or one per name."""
    units = getattr(module, "UNITS", None)
    if isinstance(units, str):
        return units
    if isinstance(units, dict):
        return lambda coords: ("feature", tuple(str(units.get(n, ""))
                                                for n in coords["feature"]))
    return ""
''',
    "input_ica": '''\
def input_ica(raw, path):
    """The ICA a .set recording carries, as MeegXer reads it (MNE's
    read_ica_eeglab), with the file's ICLabel classes, or ICLabel's when it
    has none; None without one."""
    from scipy.io import loadmat

    if not str(path).lower().endswith(".set"):
        return None
    try:  # a quick look first: most files carry no decomposition
        mat = loadmat(path, squeeze_me=True, struct_as_record=False,
                      variable_names=["EEG", "icaweights"])
        eeg = mat.get("EEG", mat)
        weights = eeg.get("icaweights") if isinstance(eeg, dict) else getattr(
            eeg, "icaweights", None)
        if weights is None or np.size(weights) == 0:
            return None
    except (NotImplementedError, ValueError, OSError):
        pass  # MATLAB v7.3: let MNE try
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            read = mne.preprocessing.read_ica_eeglab(path, verbose="error")
    except Exception:  # no decomposition, or one MNE cannot read
        return None
    unmixing = np.asarray(read.unmixing_matrix_ @ read.pca_components_, dtype=float)
    ica = {"channels": list(read.ch_names), "weights": unmixing,
           "winv": np.linalg.pinv(unmixing), "removed": [],
           "probabilities": file_classes(path, unmixing.shape[0])}
    if ica["probabilities"] is None:
        try:
            idx = [raw.ch_names.index(c) for c in ica["channels"]]
            ica["probabilities"] = iclabel(raw, idx, unmixing)
        except (SystemExit, ValueError) as exc:  # e.g. no electrode positions
            print(f"ICLabel not run on the recording's ICA: {exc}")
    return ica


def file_classes(path, n):
    """The .set's ICLabel probabilities (n x 7, ICLABEL_CLASSES' order), or
    None."""
    from scipy.io import loadmat

    def field(obj, name):
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except (NotImplementedError, ValueError, OSError):
        return None
    label = field(field(field(mat.get("EEG", mat), "etc"), "ic_classification"),
                  "ICLabel")
    table, names = field(label, "classification"), field(label, "classes")
    if table is None or names is None or np.size(table) == 0:
        return None
    table = np.atleast_2d(np.asarray(table, dtype=np.float64))
    names = [str(c).strip().lower().replace(" ", "_") for c in np.atleast_1d(names)]
    if table.shape != (n, len(names)) or sorted(names) != sorted(ICLABEL_CLASSES):
        return None
    return table[:, [names.index(c) for c in ICLABEL_CLASSES]]
''',
}
HELPER_IMPORTS = {
    "causal_sos": ("from scipy import signal",),
    "causal_fir": ("from scipy import signal",),
    "notch_sos": ("import numpy as np", "from scipy import signal"),
    "resampled": ("from fractions import Fraction", "from scipy import signal"),
    "spline_channels": ("import numpy as np",),
    "run_process": ("import numpy as np",),
    "run_transform": ("import numpy as np",),
    "fit_ica": ("import numpy as np", "import warnings"),
    "add_ica_to_set": ("import numpy as np",),
    "detectors": ("import re",),
    "windowed": ("import numpy as np", "from scipy import signal"),
    "morlet_window": ("from scipy import signal",),
    "specparam_window": ("import warnings",),
    "write_features_tsv": ("import csv", "import json", "import os"),
    "events_of": (),
    "input_ica": ("import numpy as np", "import warnings"),
    "ssvep_window": ("from scipy import signal",),
    "run_features": ("import inspect",),
}

SCRIPT = '''#!/usr/bin/env python3
"""MeegXer pipeline {name} as a standalone MNE script (MeegXer {version}).

It does what MeegXer's offline run of the pipeline does, without MeegXer
(it needs: pip install {packages}).{fitted}

    python {file} RECORDING [-o OUT] [--epochs EPOCHS_OUT]

RECORDING: a file mne.io.read_raw reads (.set, .edf, .bdf, .vhdr, .fif, ...).
OUT: the signal at the chain's end (.fif, .set or .edf); EPOCHS_OUT: the
epochs of its Epochs strip (-epo.fif or .set).

Strips:
{strips}
"""

{imports}

{helpers}
def pipeline(path):
    """The pipeline on the recording at path: (raw, epochs or None, features:
    each feature strip's windows by its id)."""
    epochs, features = None, {{}}
{body}
    return raw, epochs, features


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", help="the recording to process")
    parser.add_argument("-o", "--out", help="write the signal here")
    parser.add_argument("--epochs", help="write the epochs here")
    args = parser.parse_args()
    raw, epochs, _ = pipeline(args.recording)
    if args.out:
        save(raw, args.out)
    if args.epochs:
        if epochs is None:
            raise SystemExit("no epochs: the pipeline has no Epochs strip")
        save(epochs, args.epochs)


def save(data, path):
    if path.endswith(".fif"):
        data.save(path, overwrite=True, verbose="error")
    else:
        data.export(path, overwrite=True, verbose="error")
    print(f"wrote {{path}}")


if __name__ == "__main__":
    main()
'''


def in_docstring(text: str) -> str:
    """Text as written in the script's docstring: a backslash (a Windows path
    in a writer's setting, ``C:\\Users``) escaped, so it is no escape
    sequence, and three quotes escaped, so they do not end it."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def picks_line(types: tuple[str, ...]) -> str:
    """``picks`` for a strip processing channels of ``types``."""
    if set(CHANNEL_TYPES) <= set(types):
        return "picks = list(range(len(raw.ch_names)))  # every channel"
    return f"picks = channels_of(raw, {tuple(types)!r})"


def standalone_script(
    document: dict[str, Any], name: str, version: str, file: str = "run_mne.py"
) -> str:
    """The pipeline (a session document) as a standalone MNE script, or
    ``NotExportable`` naming the strips without code."""
    from meegxer.core.node import Probe
    from meegxer.core.registry import Registry
    from meegxer.core.session import load_session
    from meegxer.core.stage import Stage

    graph = load_session(document, Registry.from_entry_points()).graph
    missing: list[str] = []
    parts: list[tuple[str, str, ExportCode, tuple[str, ...] | None]] = []
    last_signal = None  # the node the chain's signal comes from
    epochs_from = None  # the Epochs strip whose epochs `epochs` holds
    feature_strips: set[str] = set()  # their windows: features[id]
    for node in graph.nodes:
        impl = node.impl
        if isinstance(impl, Probe):
            continue  # what the GUI shows
        reads_chain = node.input in (last_signal, f"{epochs_from}:epochs",
                                     *feature_strips)  # fmt: skip
        gives_features = bool(node.manifest.outputs) and (
            node.manifest.outputs[0].type == PortType.FEATURES
        )
        if node.input is not None and not reads_chain:
            missing.append(f"{node.id} (reads {node.input}, not the chain)")
            continue
        if node.manifest.name == "meegxer.nodes.epochs":
            epochs_from = node.id
        code: ExportCode | None
        types: tuple[str, ...] | None = None
        if isinstance(impl, Stage):
            stage = impl.for_mode("offline")
            code = stage.processor.export_code(stage.params)
            types = stage.channel_types
            if gives_features:  # a side branch: the signal goes on unchanged
                feature_strips.add(node.id)
            else:
                last_signal = node.id
        else:
            code = impl.export_code()
            if node.input is None:
                last_signal = node.id
        if code is None:
            missing.append(f"{node.id} ({node.manifest.name})")
            continue
        if any(line.startswith("epochs = ") for line in code.lines):
            epochs_from = node.id  # its epochs are what `epochs` holds now
        if gives_features:  # its windows, for the writers reading them
            code = replace(code, lines=(*code.lines, f"features[{node.id!r}] = feats"))
        elif node.input in feature_strips:  # a feature writer
            code = replace(code, lines=(f"feats = features[{node.input!r}]",
                                        *code.lines))  # fmt: skip
        parts.append((node.id, node.manifest.name, code, types))
    if missing:
        raise NotExportable(
            "these strips have no standalone MNE code yet: "
            + ", ".join(missing)
            + "; `meegxer export` (needs MeegXer) runs any pipeline"
        )
    from meegxer.core.eeglab_ica import SPATIAL

    imports = {"import argparse", "import mne"}
    helpers: list[str] = []
    definitions: list[str] = []
    packages = ["mne", "numpy", "scipy"]
    # an ICA strip's decomposition, which a .set writer after it stores
    stores_ica = any(code.with_ica is not None for _, _, code, _ in parts)
    if stores_ica and parts and parts[0][1] == "meegxer.core.file":
        # a .set recording's own ICA, which a .set writer stores again
        node_id, name_, code, types = parts[0]
        parts[0] = (
            node_id,
            name_,
            replace(
                code,
                lines=(*code.lines, "ica = input_ica(raw, path)"),
                helpers=(*code.helpers, "fit_ica", "input_ica"),
                packages=(*code.packages, "mne-icalabel", "onnxruntime"),
            ),
            types,
        )
    uses_ica = any(line.startswith("ica = ") for _, _, code, _ in parts
                   for line in code.lines)  # fmt: skip
    body: list[str] = ["    ica = None"] if uses_ica else []
    strips: list[str] = []
    for node_id, name_, code, types in parts:
        if uses_ica and code.with_ica is not None:
            code = code.with_ica
        packages += [p for p in code.packages if p not in packages]
        imports.update(code.imports)
        wanted = [*code.helpers, *(["channels_of"] if types is not None else [])]
        for helper in wanted:
            if helper not in helpers:
                helpers.append(helper)
                imports.update(HELPER_IMPORTS.get(helper, ()))
        definitions += [d for d in code.definitions if d not in definitions]
        strips.append(f"  {node_id:<12} {code.comment}")
        body.append(f"    # {node_id}: {code.comment}")
        if types is not None:
            body.append(f"    {picks_line(types)}")
        body.extend(f"    {line}" for line in code.lines)
        if uses_ica and name_ in SPATIAL:
            body.append(
                "    ica = None  # the channels changed: it unmixes them no more"
            )
    if "channels_of" in helpers and not any(
        "channels_of(" in line for line in body
    ):  # every strip took every channel
        helpers.remove("channels_of")
    std = sorted(i for i in imports if i.startswith("import ") and i != "import mne")
    std += sorted(i for i in imports if i.startswith("from fractions"))
    third = ["import mne", *sorted(i for i in imports
                                   if i not in std and i != "import mne")]  # fmt: skip
    fitted = any("fit" in code.comment.lower() for _, _, code, _ in parts)
    return SCRIPT.format(
        name=in_docstring(repr(name)), version=version, file=in_docstring(file),
        packages=" ".join(packages),
        fitted=("\nModels are fitted again on the recording it reads." if fitted
                else ""),
        strips=in_docstring("\n".join(strips)) or "  (none)",
        imports="\n".join(std) + "\n\n" + "\n".join(third),
        helpers="".join(f"\n{HELPERS[h]}\n" for h in helpers)
        + "".join(f"\n{d.rstrip()}\n\n" for d in definitions),
        body="\n".join(body),
    )  # fmt: skip
