"""ICA with ICLabel classification and component rejection.

``fit`` decomposes the calibration data with extended Picard (default) or Extended
Infomax, then classifies every component with ICLabel (mne-icalabel, the same
network and features as EEGLAB's ICLabel plugin). ``transform`` removes the
components whose probability for any of the selected artifact classes reaches the
threshold: ``x - winv[:, r] @ weights[r] @ x``. That is a fixed spatial filter, so it
streams block by block with no state. The threshold and the classes act in
``transform`` only (live parameters), so changing them needs no refit.

The strip also writes the **ICs rail** (docs/RAILS.md): the component activations
``weights @ x``, one channel per component (``IC01``...), in the component space,
each carrying the class ICLabel finds most likely and that probability as channel
attributes (``SignalInfo.ch_classes``, ``ch_scores``). Later strips read it, e.g.
a spectrum of the components (``input: "ica1:ics"``).

Conventions follow EEGLAB so the ICLabel features match it: ``weights`` apply to
data in microvolts (``icaweights * icasphere`` with an identity sphere) and the
components are scaled to RMS microvolts the way ``eeg_checkset`` does. ICLabel was
trained on average-referenced data band-passed at 1-100 Hz; put those strips
before this one. Bad channels are left out of the decomposition and pass through
unchanged.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest

# ICLabel's classes, in the order of its output columns
CLASSES = ("brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other")
REJECTABLE = ("eye", "muscle", "heart", "line_noise", "channel_noise")
RANK_TOL = 1e-6  # singular values below this fraction of the largest are rank loss


def data_rank(x: np.ndarray) -> int:
    s = np.linalg.svd(x - x.mean(axis=1, keepdims=True), compute_uv=False)
    return int((s > s[0] * RANK_TOL).sum())


def decompose(
    x: np.ndarray, sfreq: float, method: str, n_components: int, *, max_iter: int,
    seed: int,
) -> np.ndarray:  # fmt: skip
    """Unmixing matrix (components x channels) for data in microvolts."""
    import mne

    info = mne.create_info(x.shape[0], sfreq, "eeg")
    raw = mne.io.RawArray(x, info, verbose=False)
    fit_params = {"extended": True}
    if method == "picard":
        fit_params["ortho"] = False  # extended Picard = Extended Infomax's model
    ica = mne.preprocessing.ICA(
        n_components=n_components, method=method, fit_params=fit_params,
        max_iter=max_iter, random_state=seed, verbose=False,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes go to the model instead
        ica.fit(raw, verbose=False)
    unmixing = ica.unmixing_matrix_ @ ica.pca_components_[: ica.n_components_]
    return np.asarray(unmixing / ica.pre_whitener_.T)


def scale_to_rms(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EEGLAB's ``option_scaleicarms``: each component's map gets RMS 1 (uV)."""
    winv = np.linalg.pinv(weights)
    scaling = np.sqrt(np.mean(winv**2, axis=0))
    weights = weights * scaling[:, None]
    return weights, np.linalg.pinv(weights)


def eeglab_ica(info: Any, weights: np.ndarray) -> Any:
    """An MNE ICA object holding EEGLAB-convention ``weights`` (as
    ``mne.preprocessing.read_ica_eeglab`` builds it), for mne-icalabel."""
    import mne

    n_comp, n_ch = weights.shape
    fit_params = {"extended": True, "ortho": False}
    ica = mne.preprocessing.ICA(method="picard", fit_params=fit_params)
    ica.current_fit = "raw"
    ica.ch_names = list(info["ch_names"])
    ica.n_pca_components = None
    ica.n_components_ = n_comp
    ica.pre_whitener_ = np.ones((n_ch, 1))
    ica.pca_mean_ = np.zeros(n_ch)
    u, s, v = np.linalg.svd(weights, full_matrices=False)
    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica.info = info
    ica._update_mixing_matrix()
    ica._update_ica_names()
    ica.reject_ = None
    return ica


def require_positions(info: SignalInfo) -> None:
    """ICLabel needs a position for every channel it labels."""
    missing = [lb for lb, p in zip(info.ch_labels, info.positions or (), strict=True)
               if p is None] if info.positions else list(info.ch_labels)  # fmt: skip
    if missing:
        raise ValueError(
            "ICLabel needs electrode positions; set a montage or positions for "
            f"{', '.join(missing)} (Recording window, Channels tab)"
        )


def iclabel(x: np.ndarray, info: SignalInfo, weights: np.ndarray) -> np.ndarray:
    """ICLabel probabilities (components x 7 classes) for data in volts."""
    import mne
    from mne_icalabel.iclabel import iclabel_label_components

    require_positions(info)
    mne_info = mne.create_info(list(info.ch_labels), info.sfreq, "eeg")
    raw = mne.io.RawArray(x, mne_info, verbose=False)
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_labels, info.positions, strict=True)), coord_frame="head"
    )
    raw.set_montage(montage, verbose=False)
    ica = eeglab_ica(raw.info, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # reference/filter advice; see module doc
        proba = iclabel_label_components(raw, ica, inplace=False, backend="onnx")
    return np.asarray(proba, dtype=np.float64)


def component_info(model: State, info: SignalInfo) -> SignalInfo:
    """The ICs rail's channels: one per component, with ICLabel's most likely
    class and its probability as attributes."""
    proba = np.asarray(model["probabilities"])
    best = proba.argmax(axis=1)
    width = max(2, len(str(proba.shape[0])))
    return SignalInfo(
        tuple(f"IC{i + 1:0{width}d}" for i in range(proba.shape[0])),
        info.sfreq,
        info.stream_id,
        ch_classes=tuple(CLASSES[int(b)] for b in best),
        ch_scores=tuple(float(proba[i, b]) for i, b in enumerate(best)),
        filters=info.filters,  # components mix the filtered channels
    )


def rejected(model: State, params: Params) -> list[int]:
    """Components to remove: any selected class at or above the threshold."""
    proba = np.asarray(model["probabilities"])
    cols = [CLASSES.index(c) for c in REJECTABLE if params[f"reject_{c}"]]
    if not cols:
        return []
    hit = (proba[:, cols] >= params["threshold"]).any(axis=1)
    return [int(i) for i in np.flatnonzero(hit)]


class ICAClean(Processor):
    """ICA → ICLabel → rejection of artifact components (see the module doc)."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.ica",
        "0.1.0",
        [
            {"name": "method", "type": "enum", "default": "picard", "primary": True,
             "choices": ["picard", "infomax"],
             "description": "Extended Picard (fast) or Extended Infomax (EEGLAB "
                            "runica); both fit the model ICLabel was trained on"},
            {"name": "threshold", "type": "float", "default": 0.9, "min": 0.0,
             "max": 1.0, "primary": True,
             "description": "Remove a component when ICLabel gives any selected "
                            "artifact class at least this probability"},
            {"name": "reject_eye", "type": "bool", "default": True,
             "advanced": True},
            {"name": "reject_muscle", "type": "bool", "default": True,
             "advanced": True},
            {"name": "reject_heart", "type": "bool", "default": True,
             "advanced": True},
            {"name": "reject_line_noise", "type": "bool", "default": True,
             "advanced": True},
            {"name": "reject_channel_noise", "type": "bool", "default": True,
             "advanced": True},
            {"name": "n_components", "type": "int", "default": 0, "min": 0,
             "advanced": True,
             "description": "Number of components; 0 = the data's rank"},
            {"name": "max_iter", "type": "int", "default": 1000, "min": 10,
             "advanced": True},
            {"name": "seed", "type": "int", "default": 97, "min": 0,
             "advanced": True,
             "description": "Random seed, so a refit gives the same components"},
        ],
        description="ICA, ICLabel classification and removal of eye, muscle, "
                    "heart, line-noise and channel-noise components.",
        live_params=["threshold", "reject_eye", "reject_muscle", "reject_heart",
                     "reject_line_noise", "reject_channel_noise"],
        requires_fit=True,
        requires_positions=True,  # ICLabel
        rail_outputs=[{"name": "ics", "type": "signal", "rail": "ics",
                       "space": "component"}],
    )  # fmt: skip

    def fit(self, calibration: Block, params: Params) -> State:
        info = calibration.info
        good = [i for i, lb in enumerate(info.ch_labels) if lb not in info.bads]
        if len(good) < 2:
            raise ValueError("ICA needs at least two good channels")
        x = calibration.data[good]
        rank = data_rank(x)
        n_comp = int(params["n_components"]) or rank
        if n_comp > rank:
            raise ValueError(f"n_components={n_comp} exceeds the data rank {rank}")
        good_info = info.subset([info.ch_labels[i] for i in good])
        weights = decompose(
            x * 1e6, info.sfreq, str(params["method"]), n_comp,
            max_iter=int(params["max_iter"]), seed=int(params["seed"]),
        )  # fmt: skip
        weights, winv = scale_to_rms(weights)
        proba = iclabel(x, good_info, weights)
        return {
            "channels": list(good_info.ch_labels),
            "weights": weights,
            "winv": winv,
            "probabilities": proba,
            "classes": list(CLASSES),
        }

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        channels = model["channels"]
        assert isinstance(channels, list)
        idx = [block.info.ch_labels.index(str(lb)) for lb in channels]
        weights = np.asarray(model["weights"])
        # weights take uV, so on data in volts they give the activations in volts
        sources = weights @ block.data[idx]
        rails = {"ics": Block(sources, component_info(model, block.info),
                              block.sample_index, block.timestamp)}  # fmt: skip
        drop = rejected(model, params)
        if not drop:
            return TransformResult(block, {}, (), rails)
        winv = np.asarray(model["winv"])[:, drop]
        data = block.data.copy()
        data[idx] -= winv @ sources[drop]
        return TransformResult(
            Block(data, block.info, block.sample_index, block.timestamp,
                  block.bypassed), {}, (), rails
        )  # fmt: skip
