"""Decoding nodes that learn from labelled trials (manifest ``fit_events``).

CSP (common spatial patterns): spatial filters whose output variance differs most
between two (or more) classes of trials, e.g. left- and right-hand motor imagery.
Fit (MNE ``CSP``) cuts an epoch from ``tmin`` to ``tmax`` seconds after every
event whose label names a class; transform reports, per window, the log of the
mean power of each CSP component, as MNE's ``CSP.transform`` with ``log=True``
computes it for an epoch. Band-pass the signal first (e.g. 8-30 Hz for mu and
beta rhythms).

LDA (linear discriminant analysis, Fisher 1936) classifies the windows of a
feature strip (CSP, band power, ...; its main input is Features, spec 0.2.23).
It fits on the windows lying wholly from ``tmin`` to ``tmax`` after an event of
a class, with Ledoit-Wolf shrinkage of the covariance by default (Blankertz et
al. 2011): scikit-learn's ``LinearDiscriminantAnalysis(solver="lsqr",
shrinkage="auto")``. Each window then gives the probability of each class
(scikit-learn's ``predict_proba``) and a class event at its last sample, named
``event_prefix`` + class, for the Events rail (an action sink, a file
writer). The ``min_probability`` and ``event_prefix`` settings are MeegXer's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, Event, FeatureBlock, SignalInfo, Windows
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State, StateValue
from meegxer.nodes.features import WindowedFeatures, analysis_manifest
from meegxer.nodes.references import credits


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

    def units(
        self, params: Params, coords: dict[str, tuple[Any, ...]]
    ) -> str | tuple[str, tuple[str, ...]]:
        return "log(V^2)"  # natural log of the filtered signal's variance


def labelled_windows(
    windows: Windows, events: Sequence[Event], params: Params
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The training set: each window lying wholly from ``tmin`` to ``tmax``
    after an event of one class (windows on trials of two classes are left
    out), as a row of its values, with its class number, and the classes."""
    classes = parse_classes(str(params["classes"]))
    lo = round(float(params["tmin"]) * windows.sfreq)
    hi = round(float(params["tmax"]) * windows.sfreq)
    if hi <= lo:
        raise ValueError("tmax must lie after tmin")
    trials = [(e.sample_index + lo, e.sample_index + hi, classes.index(e.label))
              for e in events if e.label in classes]  # fmt: skip
    rows, labels = [], []
    for w in windows.windows:
        on = {k for a, b, k in trials if a <= w.window[0] and w.window[1] <= b}
        if len(on) == 1:
            rows.append(np.ravel(w.data))
            labels.append(on.pop())
    counts = [labels.count(i) for i in range(len(classes))]
    few = [c for c, n in zip(classes, counts, strict=True) if n < 2]
    if few:
        found = sorted({e.label for e in events})
        raise ValueError(
            f"LDA needs at least 2 windows of each class from {params['tmin']:g} to "
            f"{params['tmax']:g} s after its events; too few of {few} (windows per "
            f"class: {dict(zip(classes, counts, strict=True))}; the calibration's "
            f"events: {found}). Use shorter windows on the feature strip, or a "
            "longer trial (tmin, tmax)"
        )
    return np.stack(rows), np.array(labels), classes


def _estimator(model: State) -> Any:
    """scikit-learn's LDA with the fitted weights, to compute probabilities."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()
    lda.coef_ = np.asarray(model["coef"], dtype=float)
    lda.intercept_ = np.asarray(model["intercept"], dtype=float)
    lda.classes_ = np.arange(len(model["classes"]))  # type: ignore[arg-type]
    lda.n_features_in_ = lda.coef_.shape[1]
    return lda


class LDA(Processor):
    """Class probabilities of each feature window (Features in and out), and a
    class event per window."""

    manifest: ClassVar[Manifest] = Manifest.from_dict({
        "spec_version": "0.2.23",
        "name": "meegxer.nodes.lda",
        "version": "0.1.0",
        "type": "analysis",
        "description": "Linear discriminant analysis: the class of each window "
                       "of a feature strip (e.g. CSP), with shrinkage.",
        "inputs": [{"name": "in", "type": "features"}],
        "outputs": [{"name": "out", "type": "features"}],
        "params": [
            {"name": "classes", "type": "string", "default": "left, right",
             "primary": True,
             "description": "Event labels of the classes, comma-separated"},
            {"name": "tmin", "type": "float", "default": 0.5, "unit": "s",
             "description": "Trial start after its event: the windows lying "
                            "wholly in the trial are its training windows"},
            {"name": "tmax", "type": "float", "default": 4.0, "unit": "s",
             "description": "Trial end after its event"},
            {"name": "shrinkage", "type": "bool", "default": True,
             "primary": True,
             "description": "Ledoit-Wolf shrinkage of the covariance (helps with "
                            "few trials or many features); off: plain LDA"},
            {"name": "min_probability", "type": "float", "default": 0.0, "min": 0.0,
             "max": 1.0, "advanced": True,
             "description": "A class event only when that class's probability "
                            "reaches this (MeegXer's setting; 0: every window)"},
            {"name": "event_prefix", "type": "string", "default": "pred_",
             "advanced": True,
             "description": "Start of the class events' labels, so they never "
                            "mix with the cues (MeegXer's setting)"},
        ],
        "live_params": ["min_probability", "event_prefix"],
        "availability": "both",
        "requires_fit": True,
        "fit_events": True,
        **credits("meegxer.nodes.lda"),
    })  # fmt: skip

    def fit(self, calibration: Block, params: Params) -> State:
        raise ValueError("LDA reads a feature strip: choose one under Reads")

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        raise ValueError("LDA reads a feature strip: choose one under Reads")

    def fit_features(
        self, windows: Windows, events: Sequence[Event], params: Params
    ) -> State:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x, y, classes = labelled_windows(windows, events, params)
        shrinkage = "auto" if params["shrinkage"] else None
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        lda.fit(x, y)
        names: list[StateValue] = [*classes]
        return {
            "coef": np.asarray(lda.coef_, dtype=float),
            "intercept": np.asarray(lda.intercept_, dtype=float),
            "classes": names,
            "windows": [int((y == i).sum()) for i in range(len(classes))],
        }

    def transform_features(
        self, windows: Windows, model: State, state: State | None, params: Params
    ) -> TransformResult:
        if not windows.windows:
            return TransformResult((), state or {})
        classes = tuple(str(c) for c in model["classes"])  # type: ignore[union-attr]
        x = np.stack([np.ravel(w.data) for w in windows.windows])
        proba = _estimator(model).predict_proba(x)
        threshold = float(params["min_probability"])
        prefix = str(params["event_prefix"])
        out, events = [], []
        for w, p in zip(windows.windows, proba, strict=True):
            out.append(FeatureBlock(p, ("class",), w.window, w.timestamp,
                                    {"class": classes}, w.stream_id, "1"))  # fmt: skip
            k = int(np.argmax(p))
            if p[k] >= threshold:  # at the window's last sample: decided then
                payload = {"probability": round(float(p[k]), 6)}
                at = max(w.window[1] - 1, 0)
                events.append(Event(at, prefix + classes[k], payload=payload))
        return TransformResult(tuple(out), state or {}, tuple(events))
