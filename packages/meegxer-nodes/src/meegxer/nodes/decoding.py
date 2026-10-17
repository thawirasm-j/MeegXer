"""Decoding nodes that learn from labelled trials (manifest ``fit_events``).

CSP (common spatial patterns): spatial filters whose output variance differs most
between two (or more) classes of trials, e.g. left- and right-hand motor imagery.
Fit (MNE ``CSP``) cuts an epoch from ``tmin`` to ``tmax`` seconds after every
event whose label names a class; transform reports, per window, the log of the
mean power of each CSP component, as MNE's ``CSP.transform`` with ``log=True``
computes it for an epoch. Band-pass the signal first (e.g. 8-30 Hz for mu and
beta rhythms).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, Event, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params
from meegxer.core.serialization import State, StateValue
from meegxer.nodes.features import WindowedFeatures, analysis_manifest


def parse_classes(text: str) -> list[str]:
    classes = [c.strip() for c in str(text).split(",") if c.strip()]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("name at least two different classes (event labels)")
    return classes


def trials(
    calibration: Block, events: Sequence[Event], params: Params
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Epochs (trials x channels x samples) from ``tmin`` to ``tmax`` after each
    event of a class, their class numbers, and the classes."""
    classes = parse_classes(str(params["classes"]))
    sfreq = calibration.info.sfreq
    start = round(float(params["tmin"]) * sfreq)
    stop = round(float(params["tmax"]) * sfreq)
    if stop - start < 2:
        raise ValueError("tmax must lie after tmin")
    epochs, labels = [], []
    for e in events:
        if e.label not in classes:
            continue
        lo = e.sample_index + start - calibration.sample_index
        if lo < 0 or lo + stop - start > calibration.n_samples:
            continue  # the trial runs off the calibration data
        epochs.append(calibration.data[:, lo : lo + stop - start])
        labels.append(classes.index(e.label))
    counts = [labels.count(i) for i in range(len(classes))]
    missing = [c for c, n in zip(classes, counts, strict=True) if n < 2]
    if missing:
        found = sorted({e.label for e in events})
        raise ValueError(
            f"CSP needs at least 2 trials of each class; too few of {missing} "
            f"(the calibration's events: {found})"
        )
    return np.stack(epochs), np.array(labels), classes


class CSP(WindowedFeatures):
    """Log power of each CSP component per window (Features out)."""

    manifest: ClassVar[Manifest] = analysis_manifest(
        "meegxer.nodes.csp",
        [
            {"name": "classes", "type": "string", "default": "left, right",
             "primary": True,
             "description": "Event labels of the classes, comma-separated"},
            {"name": "tmin", "type": "float", "default": 0.5, "unit": "s",
             "description": "Trial start after its event"},
            {"name": "tmax", "type": "float", "default": 2.5, "unit": "s",
             "description": "Trial end after its event"},
            {"name": "n_components", "type": "int", "default": 4, "min": 1,
             "max": 64, "primary": True,
             "description": "CSP components kept (the most discriminative)"},
            {"name": "reg", "type": "enum", "default": "none",
             "choices": ["none", "ledoit_wolf", "oas"], "advanced": True,
             "description": "Covariance shrinkage (helps with few trials or "
                            "many channels)"},
        ],
        "Common spatial patterns: log power of the components that separate "
        "the classes best.",
        requires_fit=True,
        fit_events=True,
    )  # fmt: skip

    def fit(self, calibration: Block, params: Params) -> State:
        raise ValueError("CSP learns from labelled trials: fit with events")

    def fit_labelled(
        self, calibration: Block, events: Sequence[Event], params: Params
    ) -> State:
        import mne
        from mne.decoding import CSP as MNECSP

        x, y, classes = trials(calibration, events, params)
        n = int(params["n_components"])
        if n > x.shape[1]:
            raise ValueError(f"CSP: {n} components but only {x.shape[1]} channels")
        reg = None if params["reg"] == "none" else str(params["reg"])
        csp = MNECSP(n_components=n, reg=reg, log=True)
        with mne.utils.use_log_level("error"):
            csp.fit(x, y)
        names: list[StateValue] = [*classes]
        return {
            "filters": np.asarray(csp.filters_[:n], dtype=float),
            "patterns": np.asarray(csp.patterns_[:n], dtype=float),
            "classes": names,
            "trials": [int((y == i).sum()) for i in range(len(classes))],
        }

    def features(
        self, x: np.ndarray, info: SignalInfo, params: Params
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        raise ValueError("CSP needs its fitted model")

    def fitted_features(
        self, x: np.ndarray, info: SignalInfo, params: Params, model: State
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, tuple[Any, ...]]]:
        filters = np.asarray(model["filters"])
        power = np.mean((filters @ x) ** 2, axis=1)
        values = np.log(np.maximum(power, np.finfo(float).tiny))
        names = tuple(f"CSP{i + 1}" for i in range(filters.shape[0]))
        return values, ("component",), {"component": names}
