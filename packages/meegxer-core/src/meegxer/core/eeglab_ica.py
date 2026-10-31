"""An EEGLAB file's own ICA decomposition (M10.6, docs/RAILS.md "ICs rail").

A ``.set`` with ``icaweights`` carries a decomposition: the File input starts
the ICs rail with it (an ICA strip after the input rewrites the rail).

- **The decomposition:** MNE's ``read_ica_eeglab`` reads it (``icaweights @
  icasphere``, or ``pinv(icawinv)`` where components were removed, on the
  channels of ``icachansind``). Its unmixing matrix is EEGLAB's, applied to the
  data as the ICA strip applies its own: activations ``weights @ x``, x in
  volts.
- **ICLabel classes:** the file's own when it has them
  (``EEG.etc.ic_classification.ICLabel``, read with SciPy), else computed by
  the ICA strip's ICLabel (``meegxer.nodes.ica``, reached through the node
  registry: the core imports no node package), when a run first reads them.
- **Writing:** the ``.set`` writer stores the decomposition that unmixes the
  data it writes (``decomposition_for``, ``write_decomposition``): the ICs
  rail's writer before it, less the components that strip removed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from meegxer.core.block import FloatArray, SignalInfo

# ICLabel's classes, in the order of its output columns (as meegxer.nodes.ica)
CLASSES = ("brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other")
ICA_NODE = "meegxer.nodes.ica"


@dataclass(frozen=True, eq=False)
class Decomposition:
    """An ICA decomposition on ``channels``: ``weights`` (components x
    channels) unmixes, ``winv`` mixes; ``probabilities`` (components x
    ``CLASSES``) are ICLabel's, None until classified."""

    channels: tuple[str, ...]
    weights: FloatArray
    winv: FloatArray
    probabilities: FloatArray | None = None
    classified_by: str = ""  # "the file" or "ICLabel (computed by MeegXer)"
    notes: tuple[str, ...] = field(default=())

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    def model(self) -> dict[str, Any]:
        """As the ICA strip's model (``get_model``, the IC head plot)."""
        out: dict[str, Any] = {"channels": list(self.channels),
                               "weights": self.weights, "winv": self.winv,
                               "classes": list(CLASSES)}  # fmt: skip
        if self.probabilities is not None:
            out["probabilities"] = self.probabilities
        return out


def component_names(n: int) -> tuple[str, ...]:
    width = max(2, len(str(n)))
    return tuple(f"IC{i + 1:0{width}d}" for i in range(n))


def components_info(dec: Decomposition, info: SignalInfo) -> SignalInfo:
    """The ICs rail's channels: one per component, with ICLabel's most likely
    class and its probability when classified."""
    classes: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    if dec.probabilities is not None:
        best = dec.probabilities.argmax(axis=1)
        classes = tuple(CLASSES[int(b)] for b in best)
        scores = tuple(float(dec.probabilities[i, b]) for i, b in enumerate(best))
    return SignalInfo(component_names(dec.n_components), info.sfreq, info.stream_id,
                      ch_classes=classes, ch_scores=scores,
                      filters=info.filters)  # fmt: skip


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _file_classes(path: Path, n: int) -> tuple[FloatArray | None, str]:
    """The file's ICLabel probabilities in ``CLASSES`` order, or None and why."""
    from scipy.io import loadmat

    try:
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except (NotImplementedError, ValueError, OSError) as exc:  # e.g. MATLAB v7.3
        return None, f"its classes could not be read ({type(exc).__name__})"
    eeg = mat.get("EEG", mat)
    label = _field(_field(_field(eeg, "etc"), "ic_classification"), "ICLabel")
    table = _field(label, "classification")
    names = _field(label, "classes")
    if table is None or names is None or np.size(table) == 0:
        return None, ""
    table = np.atleast_2d(np.asarray(table, dtype=np.float64))
    names = [str(c).strip().lower().replace(" ", "_") for c in np.atleast_1d(names)]
    if table.shape != (n, len(names)) or sorted(names) != sorted(CLASSES):
        return None, "its classes do not match its components"
    return table[:, [names.index(c) for c in CLASSES]], ""


def read_decomposition(path: str | Path) -> Decomposition | None:
    """The decomposition a ``.set`` carries, with the file's ICLabel classes
    if it has them; None when it has none."""
    import mne
    from scipy.io import loadmat

    path = Path(path)
    try:  # a quick look first: most files carry no decomposition
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False,
                      variable_names=["EEG", "icaweights"])  # fmt: skip
        weights = _field(mat.get("EEG", mat), "icaweights")
        if weights is None or np.size(weights) == 0:
            return None
    except (NotImplementedError, ValueError, OSError):
        pass  # MATLAB v7.3: let MNE try
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ica = mne.preprocessing.read_ica_eeglab(str(path), verbose="error")
        except Exception:  # no decomposition, or one MNE cannot read
            return None
    unmixing = np.asarray(ica.unmixing_matrix_ @ ica.pca_components_, dtype=np.float64)
    notes = []
    if any("icawinv" in str(w.message) for w in caught):
        notes.append("icaweights and icawinv disagree (components were removed in "
                     "EEGLAB): icawinv is used, as MNE does")  # fmt: skip
    proba, why = _file_classes(path, unmixing.shape[0])
    if why:
        notes.append(f"ICLabel: {why}")
    return Decomposition(
        tuple(ica.ch_names), unmixing, np.linalg.pinv(unmixing), proba,
        "the file" if proba is not None else "", tuple(notes),
    )  # fmt: skip


def classify(dec: Decomposition, x: FloatArray, info: SignalInfo) -> Decomposition:
    """ICLabel on the decomposition (``x``, volts, and ``info``: the
    decomposition's channels with their positions), by the ICA strip found
    through the node registry. Unclassified, with a note, when it cannot."""
    from meegxer.core.registry import Registry

    try:
        strip: Any = Registry.from_entry_points().get(ICA_NODE)
        run = strip.classify  # the ICA strip's (meegxer-nodes), if installed
    except Exception:
        why = "the ICA strip (meegxer-nodes) is not installed"
    else:
        try:
            proba = np.asarray(run(x, info, dec.weights), dtype=np.float64)
        except ValueError as exc:  # e.g. no electrode positions
            why = str(exc)
        else:
            return replace(dec, probabilities=proba,
                           classified_by="ICLabel (computed by MeegXer)")  # fmt: skip
    return replace(dec, notes=(*dec.notes, f"ICLabel not run: {why}"))


EEGLAB_CLASSES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise",
                  "Other")  # ICLabel's names in EEG.etc.ic_classification  # fmt: skip


def write_decomposition(
    path: str | Path, dec: Decomposition, labels: tuple[str, ...]
) -> None:
    """Put ``dec`` into a ``.set`` written without one (eeglabio leaves the ICA
    fields empty): ``icaweights`` (with an identity ``icasphere``), ``icawinv``,
    ``icachansind`` (``labels``: the file's channels, in order), and its
    ICLabel classes as ``etc.ic_classification.ICLabel``, as EEGLAB keeps
    them."""
    from scipy.io import loadmat, savemat

    path = Path(path)
    missing = [c for c in dec.channels if c not in labels]
    if missing:
        raise ValueError(f"the file has no channels {', '.join(missing)}")
    mat = {k: v for k, v in loadmat(str(path)).items() if not k.startswith("__")}
    n_ch = len(dec.channels)
    fields: dict[str, Any] = {
        "icaweights": dec.weights,
        "icasphere": np.eye(n_ch),
        "icawinv": dec.winv,
        "icachansind": np.array(
            [[labels.index(c) + 1 for c in dec.channels]], dtype=np.float64
        ),
        "icaact": np.zeros((0, 0)),
    }
    if dec.probabilities is not None:
        names = np.empty((1, len(EEGLAB_CLASSES)), dtype=object)
        names[0, :] = list(EEGLAB_CLASSES)
        fields["etc"] = {"ic_classification": {"ICLabel": {
            "classes": names, "classification": dec.probabilities,
            "version": "default"}}}  # fmt: skip
    if "EEG" in mat:  # saved as one struct
        eeg = mat["EEG"]
        merged = {name: eeg[name][0, 0] for name in eeg.dtype.names}
        mat["EEG"] = {**merged, **fields}
    else:  # saved as top-level fields (eeglabio, EEGLAB's flat files)
        mat.update(fields)
    savemat(str(path), mat, oned_as="row")


# strips whose output is a spatial transform of their input: a decomposition
# fitted before one no longer unmixes the data after it
SPATIAL = frozenset({"meegxer.nodes.rereference", "meegxer.nodes.interpolate_bads",
                     "meegxer.nodes.csd", "meegxer.nodes.eog_regression"})  # fmt: skip


def decomposition_for(graph: Any, node_id: str) -> tuple[Decomposition | None, str]:
    """The decomposition a ``.set`` writer at ``node_id`` stores with its data:
    the ICs rail's nearest writer before it (the input's file, or an ICA
    strip), less the components that strip removed from the data (as
    EEGLAB's ``pop_subcomp``). None, and why, when there is none to store."""
    from meegxer.core.node import Source
    from meegxer.core.stage import Stage

    nodes = list(graph.nodes)
    here = [n.id for n in nodes].index(node_id)
    if ":" in str(graph[node_id].input or ""):
        return None, ""  # it writes a rail, not the signal
    producer = next((n for n in reversed(nodes[:here])
                     if any(p.rail == "ics" for p in n.manifest.rail_outputs)),
                    None)  # fmt: skip
    if producer is None:
        return None, ""
    between = nodes[[n.id for n in nodes].index(producer.id) + 1 : here]
    spatial = [n.id for n in between if n.manifest.name in SPATIAL]
    if spatial:
        return None, (f"ICA decomposition not stored: {', '.join(spatial)} "
                      "transforms the channels after it")  # fmt: skip
    impl = producer.impl
    if isinstance(impl, Source):
        found = getattr(impl, "decomposition", lambda: None)()
        return (found, "") if isinstance(found, Decomposition) else (None, "")
    if not isinstance(impl, Stage) or impl.model is None:
        return None, ""
    describe = getattr(impl.processor, "decomposition", None)
    if describe is None:
        return None, ""
    d = describe(impl.model.data, impl.params)
    keep = [k for k in range(len(d["weights"])) if k not in set(d["removed"])]
    weights = np.asarray(d["weights"])[keep]
    proba = d.get("probabilities")
    return Decomposition(
        tuple(d["channels"]), weights, np.asarray(d["winv"])[:, keep],
        None if proba is None else np.asarray(proba)[keep],
        "ICLabel (computed by MeegXer)",
    ), ""  # fmt: skip
