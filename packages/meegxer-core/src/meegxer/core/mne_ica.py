"""MNE ICA files (``-ica.fif``) on the ICs rail (M10.7b, docs/RAILS.md).

- **Reading** (the File input's ``ica``, ``meegxer.api.run(..., ica=)``): an
  ``mne.preprocessing.ICA`` becomes a ``Decomposition`` measured with MNE's
  own ``ICA.get_sources`` on a probe recording (a zero column, then one unit
  column per channel), so its projectors, pre-whitener and PCA mean are
  applied by MNE, not re-implemented: ``get_sources(x) = W x + b`` gives the
  unmixing ``W`` and the offset ``b``. MeegXer's weights take µV, as
  EEGLAB's (``eeglab_ica``), so the rail carries MNE's sources in µV read as
  volts: ``W / 1e6``. ICLabel's classes are computed (an MNE file keeps
  none: ``labels_`` names a class, without its probabilities).
- **Writing** (the *MNE ICA (.fif)* writer): the decomposition goes into MNE
  through its own EEGLAB import (``read_ica_eeglab`` on a small ``.set``
  holding it), then ``ICA.save``: ``exclude`` names the components marked
  for removal and ``labels_`` the ICLabel classes, as mne-icalabel's
  ``label_components`` sets them.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import Block, SignalInfo
from meegxer.core.eeglab_ica import CLASSES, Decomposition, write_decomposition
from meegxer.core.epochs_out import channel_positions
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink
from meegxer.core.ports import PortTypeError
from meegxer.core.sinks import _sink_manifest

ICA_WRITER = "meegxer.core.ica_writer"

UV = 1e6  # µV per volt: MNE's weights take volts, MeegXer's (EEGLAB's) µV
FIF_ENDINGS = ("-ica.fif", "_ica.fif", "-ica.fif.gz", "_ica.fif.gz")
# mne-icalabel's names of ICLabel's classes, in CLASSES order
ICLABEL_NAMES = ("brain", "muscle artifact", "eye blink", "heart beat",
                 "line noise", "channel noise", "other")  # fmt: skip


def from_mne(ica: Any) -> Decomposition:
    """``ica`` (fitted) as a decomposition on its channels."""
    import mne

    channels = list(ica.ch_names)
    n = len(channels)
    types = [ica.info.get_channel_types(picks=[c])[0] for c in channels]
    info = mne.create_info(channels, float(ica.info["sfreq"]), types)
    probe = mne.io.RawArray(np.hstack([np.zeros((n, 1)), np.eye(n)]), info,
                            verbose="error")  # fmt: skip
    with mne.utils.use_log_level("error"):  # its projectors' log lines
        sources = np.asarray(ica.get_sources(probe).get_data(), dtype=np.float64)
    offset = sources[:, 0]
    unmixing = sources[:, 1:] - offset[:, None]  # W, taking volts
    weights = unmixing / UV
    notes = []
    if ica.exclude:
        marked = ", ".join(str(k + 1) for k in sorted(ica.exclude))
        which = "component" if len(ica.exclude) == 1 else "components"
        notes.append(f"the file marks {which} {marked} for removal "
                     "(kept on the rail: an ICA strip removes components)")  # fmt: skip
    return Decomposition(
        tuple(channels), weights, np.linalg.pinv(weights), None, "", tuple(notes),
        offset=offset / UV if np.any(offset) else None,
        excluded=tuple(sorted(int(k) for k in ica.exclude)),
    )  # fmt: skip


def read_mne_ica(path: str | Path) -> Decomposition:
    """The decomposition an MNE ICA file holds."""
    import mne

    return from_mne(mne.preprocessing.read_ica(str(path), verbose="error"))


def to_mne(
    dec: Decomposition,
    sfreq: float,
    positions: Mapping[str, Sequence[float]] | None = None,
) -> Any:
    """``dec`` as an ``mne.preprocessing.ICA`` on its channels (``positions``:
    metres, head coordinates, where known), through MNE's EEGLAB import."""
    import mne

    channels = list(dec.channels)
    info = mne.create_info(channels, sfreq, "eeg")
    known = {c: positions[c] for c in channels if positions and c in positions}
    if known:
        montage = mne.channels.make_dig_montage(known, coord_frame="head")
        info.set_montage(montage, on_missing="ignore", verbose="error")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "decomposition.set"
        raw = mne.io.RawArray(np.zeros((len(channels), 2)), info, verbose="error")
        raw.export(path, fmt="eeglab", verbose="error")
        scaled = replace(dec, weights=dec.weights * UV, winv=dec.winv / UV)
        write_decomposition(path, scaled, tuple(channels))
        ica = mne.preprocessing.read_ica_eeglab(str(path), verbose="error")
    if dec.offset is not None:  # MNE centres before unmixing: W (x - m) = W x + b
        ica.pca_mean_ = -np.linalg.pinv(dec.weights * UV) @ (dec.offset * UV)
    ica.exclude = list(dec.excluded)
    if dec.probabilities is not None:  # as mne-icalabel's label_components
        best = dec.probabilities.argmax(axis=1)
        ica.labels_ = {ICLABEL_NAMES[c]: [int(k) for k in np.flatnonzero(best == c)]
                       for c in range(len(CLASSES))}  # fmt: skip
    return ica


def write_mne_ica(
    path: str | Path,
    dec: Decomposition,
    sfreq: float,
    positions: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """``dec`` as an MNE ICA file (``ICA.save``)."""
    ica = to_mne(dec, sfreq, positions)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ica.save(str(path), overwrite=True, verbose="error")


class IcaWriter(Sink):
    """The ICA decomposition of the signal it reads, as an MNE ICA file: the
    ICs rail's nearest writer before it (the input's file or ICA, or an ICA
    strip), with the components that strip removed marked ``exclude``."""

    # the decomposition of electrodes: a component rail is no input (spec 0.2.22)
    manifest: ClassVar[Manifest] = _sink_manifest(ICA_WRITER, "signal", ".fif",
                                                  input_space="sensor")  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        if not str(self.params["path"]):
            raise ValueError(f"{self.manifest.name}: 'path' is required")
        if not self.path.name.endswith(FIF_ENDINGS):
            raise ValueError(f"{self.manifest.name}: name an ICA file …-ica.fif "
                             "(MNE's convention)")  # fmt: skip
        self._graph: Any = None
        self._node_id = ""
        self._info: SignalInfo | None = None

    @property
    def path(self) -> Path:
        return Path(str(self.params["path"]))

    def bind(self, graph: Any, node_id: str) -> None:
        self._graph, self._node_id = graph, node_id

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError(f"{self.manifest.name} accepts Signal blocks only")
        if self._info is None:
            self._info = block.info  # the channels the decomposition must be on

    def close(self) -> None:
        info, self._info = self._info, None
        if info is None or self._graph is None or self.keep_existing(self.path):
            return
        from meegxer.core.eeglab_ica import decomposition_for

        dec, why = decomposition_for(self._graph, self._node_id, drop_removed=False)
        if dec is None:
            none = ("no ICA decomposition before it (an ICA strip, or an input "
                    "carrying one): not written")  # fmt: skip
            self.skipped.append(why or none)
            return
        missing = [c for c in dec.channels if c not in info.ch_labels]
        if missing:
            self.skipped.append(f"ICA not written: the signal has no channels "
                                f"{', '.join(missing)}")  # fmt: skip
            return
        write_mne_ica(self.path, dec, info.sfreq, channel_positions(self._graph))
        marked = f", {len(dec.excluded)} marked excluded" if dec.excluded else ""
        self.notes.append(f"wrote {self.path.name}: {dec.n_components} "
                          f"components{marked}")  # fmt: skip
