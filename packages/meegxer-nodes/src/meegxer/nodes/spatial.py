"""Spatial transforms from electrode positions: interpolate bad channels and the
surface Laplacian (current source density).

Both are linear in the data and depend only on the channels, their positions and
which are bad, so each is computed once as a channel-mixing matrix (by MNE, on an
identity "recording": column j of the result is channel j's influence) and then
applied to every block as ``matrix @ data``. That makes them exact block by block,
cheap live, and refreshed whenever the positions or the bad channels change.

Only EEG channels with a position take part (channel type ``eeg``, or no type
given); others pass through. The sphere both methods need is fitted to the
electrode positions themselves.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from meegxer.core.block import Block, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State
from meegxer.nodes._manifest import stream_manifest

Matrix = npt.NDArray[np.float64]
_NOT_EEG = {"eog", "ecg", "emg", "misc", "stim", "resp", "bio", "gsr", "temperature"}


def eeg_with_positions(info: SignalInfo) -> list[int]:
    types = info.ch_types or ("",) * len(info.ch_labels)
    positions = info.positions or (None,) * len(info.ch_labels)
    return [i for i, (t, p) in enumerate(zip(types, positions, strict=True))
            if p is not None and t.lower() not in _NOT_EEG]  # fmt: skip


def fit_sphere(points: Matrix) -> tuple[Matrix, float]:
    """Centre and radius of the sphere best fitting the points (least squares)."""
    a = np.column_stack([2 * points, np.ones(len(points))])
    sol, *_ = np.linalg.lstsq(a, (points**2).sum(axis=1), rcond=None)
    centre = sol[:3]
    radius = float(np.sqrt(max(sol[3] + centre @ centre, 1e-12)))
    return centre, radius


def _identity_raw(labels: tuple[str, ...], positions: Matrix, bads: tuple[str, ...]):  # type: ignore[no-untyped-def]
    import mne

    info = mne.create_info(list(labels), 1000.0, "eeg")
    raw = mne.io.RawArray(np.eye(len(labels)), info, verbose="error")
    montage = mne.channels.make_dig_montage(
        dict(zip(labels, positions, strict=True)), coord_frame="head"
    )
    raw.set_montage(montage, verbose="error")
    raw.info["bads"] = [b for b in bads if b in labels]
    return raw


def _key(
    info: SignalInfo,
) -> tuple[tuple[str, ...], Matrix, tuple[str, ...], list[int]]:
    idx = eeg_with_positions(info)
    labels = tuple(info.ch_labels[i] for i in idx)
    positions = np.array([info.positions[i] for i in idx], dtype=float)
    return labels, positions, tuple(b for b in info.bads if b in labels), idx


@lru_cache(maxsize=32)
def _interpolation(labels: tuple[str, ...], flat: tuple[float, ...],
                   bads: tuple[str, ...]) -> Matrix:  # fmt: skip
    positions = np.array(flat).reshape(-1, 3)
    raw = _identity_raw(labels, positions, bads)
    centre, _ = fit_sphere(positions[[labels.index(lb) for lb in labels
                                      if lb not in bads]])  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.interpolate_bads(reset_bads=False, origin=tuple(centre), verbose="error")
    return np.asarray(raw.get_data(), dtype=np.float64)


@lru_cache(maxsize=32)
def _csd(
    labels: tuple[str, ...],
    flat: tuple[float, ...],
    bads: tuple[str, ...],
    lambda2: float,
    stiffness: float,
    terms: int,
) -> Matrix:
    """CSD over the good channels (MNE refuses bad ones); bad rows stay."""
    import mne

    positions = np.array(flat).reshape(-1, 3)
    good = [i for i, lb in enumerate(labels) if lb not in bads]
    good_labels = tuple(labels[i] for i in good)
    raw = _identity_raw(good_labels, positions[good], ())
    centre, radius = fit_sphere(positions[good])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mne.preprocessing.compute_current_source_density(
            raw, sphere=(*centre, radius), lambda2=lambda2, stiffness=stiffness,
            n_legendre_terms=terms, verbose="error",
        )  # fmt: skip
    matrix = np.eye(len(labels))
    matrix[np.ix_(good, good)] = np.asarray(out.get_data(), dtype=np.float64)
    return matrix


def _apply(block: Block, idx: list[int], matrix: Matrix) -> TransformResult:
    data = block.data.copy()
    data[idx] = matrix @ block.data[idx]
    return TransformResult(
        Block(data, block.info, block.sample_index, block.timestamp,
              block.bypassed), {}
    )  # fmt: skip


class InterpolateBads(Processor):
    """Replace bad channels by a spherical-spline interpolation of the good ones."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.interpolate_bads",
        "0.1.0",
        [],
        description="Spherical-spline interpolation of the channels marked bad "
        "(MNE); needs electrode positions.",
        requires_positions=True,
        input_space="sensor",  # electrodes only (spec 0.2.22)
    )

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        labels, positions, bads, idx = _key(block.info)
        if not bads:
            return TransformResult(block, {})
        if len(labels) - len(bads) < 3:
            raise ValueError("interpolation needs at least three good channels "
                             "with positions")  # fmt: skip
        matrix = _interpolation(labels, tuple(positions.ravel()), bads)
        return _apply(block, idx, matrix)


class CSD(Processor):
    """Surface Laplacian (current source density) by spherical splines."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.csd",
        "0.1.0",
        [
            {"name": "lambda2", "type": "float", "default": 1e-5, "min": 0.0,
             "advanced": True,
             "description": "Regularization of the spline fit"},
            {"name": "stiffness", "type": "float", "default": 4.0, "min": 2.0,
             "max": 10.0, "advanced": True,
             "description": "Spline flexibility (m); 4 is typical"},
            {"name": "n_legendre_terms", "type": "int", "default": 50, "min": 10,
             "max": 200, "advanced": True,
             "description": "Terms of the Legendre series"},
        ],
        description="Surface Laplacian (current source density, MNE): sharper, "
        "reference-free topographies; output in V/m². Needs electrode positions.",
        requires_positions=True,
        input_space="sensor",  # electrodes only (spec 0.2.22)
    )  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        labels, positions, bads, idx = _key(block.info)
        if len(labels) - len(bads) < 4:
            raise ValueError("CSD needs at least four good channels with positions")
        matrix = _csd(labels, tuple(positions.ravel()), bads,
                      float(params["lambda2"]), float(params["stiffness"]),
                      int(params["n_legendre_terms"]))  # fmt: skip
        return _apply(block, idx, matrix)
