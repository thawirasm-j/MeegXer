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

from dataclasses import dataclass
from typing import Any

from meegxer.core.block import CHANNEL_TYPES


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
        "etc": {"ic_classification": {"ICLabel": {
            "classes": names, "classification": ica["probabilities"][keep],
            "version": "default"}}},
    }
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
    """The pipeline on the recording at path: (raw, epochs or None)."""
    epochs = None
{body}
    return raw, epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", help="the recording to process")
    parser.add_argument("-o", "--out", help="write the signal here")
    parser.add_argument("--epochs", help="write the epochs here")
    args = parser.parse_args()
    raw, epochs = pipeline(args.recording)
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
    for node in graph.nodes:
        impl = node.impl
        if isinstance(impl, Probe):
            continue  # what the GUI shows
        reads_chain = node.input in (last_signal, f"{epochs_from}:epochs")
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
