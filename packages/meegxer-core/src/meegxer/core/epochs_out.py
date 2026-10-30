"""Epochs and ERPs into files: the windows of an ``epochs`` rail (docs/RAILS.md)
as an ``mne.EpochsArray``, written as ``-epo.fif`` or an epoched EEGLAB ``.set``,
and their averages per label (the ERPs) as ``-ave.fif`` evoked files.

The writers read the nearest ``epochs`` rail, so after a Review epochs strip
they get the epochs kept, with the labels given (rejected ones are left out:
they stay in the review's file and as ``rejected_epoch`` events). Everything
here is MNE: ``EpochsArray``, ``Epochs.save``, ``mne.export.export_epochs``,
``Epochs.average`` and ``mne.write_evokeds``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import FeatureBlock
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink
from meegxer.core.ports import PortTypeError
from meegxer.core.references import credits

EPOCHS_WRITER = "meegxer.core.epochs_writer"
FIF_ENDINGS = ("-epo.fif", "_epo.fif")  # what MNE expects of an epochs file
MNE_TYPES = {"eeg", "eog", "ecg", "emg", "misc", "stim", "resp", "bio", "gsr",
             "temperature", "seeg", "ecog", "dbs"}  # fmt: skip


def epochs_array(windows: Sequence[FeatureBlock], ch_types: Mapping[str, str]) -> Any:
    """The windows of an epochs rail (``epoch x channel x time``, times in
    seconds from the event) as an ``mne.EpochsArray``, each epoch's event at
    its sample on the stream's clock and labelled as on the rail."""
    import mne

    first = windows[0]
    channels = [str(c) for c in first.coords["channel"]]
    times = np.asarray(first.coords["time"], dtype=np.float64)
    if times.size < 2:
        raise ValueError("epochs need at least two samples to know their rate")
    sfreq = round(float((times.size - 1) / (times[-1] - times[0])), 6)
    for w in windows:
        if [str(c) for c in w.coords["channel"]] != channels or w.data.shape[2] != (
            times.size
        ):
            raise ValueError("the epochs' channels or length changed during the run")
    labels = [str(w.coords["epoch"][0]) for w in windows]
    event_id = {label: k + 1 for k, label in enumerate(dict.fromkeys(labels))}
    zero = round(-times[0] * sfreq)  # samples from the window's start to its event
    events = np.array([[w.window[0] + zero, 0, event_id[label]]
                       for w, label in zip(windows, labels, strict=True)])  # fmt: skip
    # a channel with no type is EEG, as everywhere in MeegXer
    types = [t if (t := (ch_types.get(c) or "eeg").lower()) in MNE_TYPES
             else "misc" for c in channels]  # fmt: skip
    info = mne.create_info(channels, sfreq, types)
    data = np.concatenate([w.data for w in windows], axis=0)
    return mne.EpochsArray(data, info, events, tmin=float(times[0]),
                           event_id=event_id, baseline=None,
                           verbose="error")  # fmt: skip


def erps(epochs: Any) -> list[Any]:
    """The average of each label's epochs (``Epochs.average``), named by it."""
    out = []
    for label in epochs.event_id:
        evoked = epochs[label].average()
        evoked.comment = label
        out.append(evoked)
    return out


def write_epochs(epochs: Any, path: Path, erp_path: Path | None) -> list[Path]:
    """``epochs`` to ``path`` (``-epo.fif``, or an epoched EEGLAB ``.set``),
    and their ERPs to ``erp_path`` (``-ave.fif``) when given."""
    import mne

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".set":
        mne.export.export_epochs(path, epochs, fmt="eeglab", overwrite=True,
                                 verbose="error")  # fmt: skip
    else:
        epochs.save(path, overwrite=True, verbose="error")
    written = [path]
    if erp_path is not None:
        mne.write_evokeds(erp_path, erps(epochs), overwrite=True, verbose="error")
        written.append(erp_path)
    return written


def erp_path_for(path: Path) -> Path:
    """Where the ERPs of the epochs file ``path`` go (``-ave.fif`` beside it)."""
    name = path.name
    for ending in FIF_ENDINGS:
        if name.endswith(ending):
            return path.with_name(name[: -len(ending)] + ending[0] + "ave.fif")
    return path.with_name(f"{path.stem}-ave.fif")


def channel_types(graph: Any) -> dict[str, str]:
    """The input's channel types, by label (empty without a graph)."""
    if graph is None:
        return {}
    info = graph.source.impl.info
    return dict(zip(info.ch_labels, info.ch_types or (), strict=False))


class EpochsWriter(Sink):
    """The kept epochs of the run, as ``-epo.fif`` or an epoched ``.set``, and
    their ERPs per label as ``-ave.fif`` beside it."""

    manifest: ClassVar[Manifest] = Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": EPOCHS_WRITER,
            "version": "0.1.0",
            "type": "stream",
            "role": "sink",
            "availability": "offline",
            "description": "Epochs into a file (-epo.fif or .set), with their "
                           "ERPs per label (-ave.fif).",
            "inputs": [{"name": "in", "type": "features"}],
            "outputs": [],
            "params": [
                {"name": "path", "type": "string", "default": "", "primary": True,
                 "path": "save", "extensions": [".fif", ".set"],
                 "description": "The epochs file: name-epo.fif (MNE) or name.set "
                                "(EEGLAB)"},
                {"name": "erp", "type": "bool", "default": True, "primary": True,
                 "description": "Also write the average of each label's epochs "
                                "(the ERPs) as name-ave.fif"},
                {"name": "overwrite", "type": "bool", "default": False},
            ],
            **credits(EPOCHS_WRITER),
        }
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        path = self.path
        if not str(self.params["path"]):
            raise ValueError(f"{self.manifest.name}: 'path' is required")
        if path.suffix not in (".fif", ".set"):
            raise ValueError(f"{self.manifest.name}: the file must be .fif or .set")
        if path.suffix == ".fif" and not path.name.endswith(FIF_ENDINGS):
            raise ValueError(f"{self.manifest.name}: name an epochs file "
                             "…-epo.fif (MNE's convention)")  # fmt: skip
        self._windows: list[FeatureBlock] = []
        self._graph: Any = None
        self.written: list[Path] = []

    @property
    def path(self) -> Path:
        return Path(str(self.params["path"]))

    def bind(self, graph: Any, node_id: str) -> None:
        self._graph = graph

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock) or block.dims[:1] != ("epoch",):
            raise PortTypeError(f"{self.manifest.name} reads an epochs rail")
        self._windows.append(block)

    def close(self) -> None:
        if not self._windows:
            return
        windows, self._windows = self._windows, []
        if self.keep_existing(self.path):
            return
        epochs = epochs_array(windows, channel_types(self._graph))
        erp = erp_path_for(self.path) if bool(self.params["erp"]) else None
        self.written = write_epochs(epochs, self.path, erp)
