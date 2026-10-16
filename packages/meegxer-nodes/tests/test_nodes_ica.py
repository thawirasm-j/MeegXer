"""ICA → ICLabel → rejection, against EEGLAB references (acceptance test 6).

The recording (``tests/data/m6_ica.npz``) mixes known brain and artifact sources;
the EEGLAB results (``m6_ref_ica.npz``) come from ``tools/make_m6_ica_references.py``.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from meegxer.core.block import Block, SignalInfo
from meegxer.core.conformance import check_blockwise_equivalence
from meegxer.core.stage import Stage, StageStatus
from meegxer.nodes.ica import (
    CLASSES,
    ICAClean,
    decompose,
    eeglab_ica,
    iclabel,
    rejected,
    scale_to_rms,
)

DATA = Path(__file__).resolve().parents[3] / "tests" / "data"
REC = np.load(DATA / "m6_ica.npz")
REF = np.load(DATA / "m6_ref_ica.npz")
FP = "m6-ica"


def recording(**changes: Any) -> Block:
    info = SignalInfo(
        tuple(str(s) for s in REC["labels"]),
        float(REC["sfreq"]),
        "m6",
        positions=tuple(tuple(p) for p in REC["positions"]),
    )
    return Block(REC["data"].astype(np.float64), dataclasses.replace(info, **changes),
                 0, 0.0)  # fmt: skip


def matched_correlations(unmixing: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """|r| of each component with its best one-to-one match among ``sources``."""
    acts = unmixing @ (REC["data"].astype(np.float64) * 1e6)
    n = len(acts)
    r = np.abs(np.corrcoef(acts, sources)[:n, n:])
    rows, cols = linear_sum_assignment(-r)
    return np.asarray(r[rows, cols])


@pytest.fixture(scope="module")
def fitted() -> Stage:
    stage = Stage(ICAClean())
    stage.fit(recording(), FP)
    return stage


def test_iclabel_matches_eeglab_on_the_same_decomposition() -> None:
    weights, _ = scale_to_rms(REF["weights_in"])
    assert np.allclose(weights, REF["icaweights"], rtol=1e-9)  # eeg_checkset's RMS
    rec = recording()
    import mne
    from mne_icalabel.iclabel.features import get_iclabel_features

    info = mne.create_info(list(rec.info.ch_labels), rec.info.sfreq, "eeg")
    raw = mne.io.RawArray(rec.data, info, verbose=False)
    raw.set_montage(mne.channels.make_dig_montage(
        dict(zip(rec.info.ch_labels, rec.info.positions, strict=True)),
        coord_frame="head"))  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        topo, psd, autocorr = get_iclabel_features(raw, eeglab_ica(raw.info, weights))
    assert np.abs(topo - REF["topo"]).max() < 1e-5
    assert np.abs(psd - REF["psd"]).max() < 1e-5
    assert np.abs(autocorr - REF["autocorr"]).max() < 1e-5
    proba = iclabel(rec.data, rec.info, weights)
    assert np.abs(proba - REF["probabilities"]).max() < 1e-4


def test_picard_matches_eeglab_runica() -> None:
    x = REC["data"].astype(np.float64) * 1e6
    w = decompose(x, float(REC["sfreq"]), "picard", 18, max_iter=1000, seed=97)
    r = matched_correlations(w, REF["runica_unmixing"] @ x)
    assert np.median(r) > 0.99 and r.min() > 0.9


def test_infomax_matches_eeglab_runica() -> None:
    x = REC["data"].astype(np.float64) * 1e6
    w = decompose(x, float(REC["sfreq"]), "infomax", 18, max_iter=1000, seed=97)
    r = matched_correlations(w, REF["runica_unmixing"] @ x)
    assert np.median(r) > 0.999 and r.min() > 0.95


def test_strip_recovers_the_sources_and_removes_eye_and_muscle(fitted: Stage) -> None:
    model = fitted.model
    assert model is not None
    truth = np.linalg.pinv(REC["mixing"]) @ REC["data"].astype(np.float64)
    assert np.median(matched_correlations(np.asarray(model.data["weights"]),
                                          truth)) > 0.97  # fmt: skip
    drop = rejected(model.data, fitted.params)
    acts = np.asarray(model.data["weights"]) @ (REC["data"] * 1e6)
    kinds = REC["kinds"]
    best = np.abs(np.corrcoef(acts, truth)[: len(acts), len(acts) :]).argmax(axis=1)
    removed = sorted(str(kinds[best[i]]) for i in drop)
    assert removed == ["eye", "eye", "muscle"]  # the 0.9-confident artifacts

    rec = recording()
    out = fitted.transform(rec, FP).block
    assert isinstance(out, Block)
    blink = truth[list(kinds).index("eye")]
    fp1 = rec.info.ch_labels.index("Fp1")
    assert abs(np.corrcoef(rec.data[fp1], blink)[0, 1]) > 0.5
    assert abs(np.corrcoef(out.data[fp1], blink)[0, 1]) < 0.2  # r~0.99 separation
    brain = kinds == "brain"  # the true brain-only signal on each channel
    signal = np.asarray(REC["mixing"])[:, brain] @ truth[brain]

    def r(x: np.ndarray) -> np.ndarray:
        return np.array(
            [np.corrcoef(a, b)[0, 1] for a, b in zip(x, signal, strict=True)]
        )

    before, after = r(rec.data), r(out.data)
    assert (after >= before).all()  # closer to the brain signal on every channel
    assert after.mean() > before.mean() + 0.2


def test_threshold_and_classes_are_live(fitted: Stage) -> None:
    stage = Stage(ICAClean())
    stage.model = fitted.model
    stage.set_params(threshold=0.999, reject_muscle=False)
    assert stage.status(FP) is StageStatus.READY  # no refit for these
    assert stage.model is not None
    proba = np.asarray(stage.model.data["probabilities"])
    eye = CLASSES.index("eye")
    expected = [i for i in range(len(proba)) if proba[i, eye] >= 0.999]
    assert rejected(stage.model.data, stage.params) == expected
    stage.set_params(method="infomax")
    assert stage.status(FP) is StageStatus.NEEDS_REFIT


def test_blockwise_output_equals_one_pass(fitted: Stage) -> None:
    rec = recording()
    short = rec.slice(0, 2000)
    proc = ICAClean()
    proc.fit = lambda calibration, params: fitted.model.data  # type: ignore[method-assign, union-attr]
    check_blockwise_equivalence(proc, short, calibration=rec)


def test_bad_channels_are_left_out_and_pass_through() -> None:
    rec = recording(bads=("O2",))
    stage = Stage(ICAClean())
    stage.fit(rec, FP)
    assert stage.model is not None
    channels = stage.model.data["channels"]
    assert isinstance(channels, list) and "O2" not in channels
    out = stage.transform(rec, FP).block
    assert isinstance(out, Block)
    o2 = rec.info.ch_labels.index("O2")
    assert np.array_equal(out.data[o2], rec.data[o2])


def test_iclabel_needs_positions() -> None:
    with pytest.raises(ValueError, match="electrode positions"):
        Stage(ICAClean()).fit(recording(positions=()), FP)
