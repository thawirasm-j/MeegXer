"""EOG regression: subtract the part of each EEG channel explained by the EOG
channels (Gratton et al. 1983; Croft & Barry 2000).

``fit`` estimates the propagation coefficients by least squares on calibration
data (MNE's ``EOGRegression``); ``transform`` subtracts ``coef @ eog`` from the
EEG channels, a linear operation that is exact block by block. The coefficients
are fitted on the data as given, so filter (high-pass) and re-reference before
this strip (MNE recommends an average reference).
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np

from meegxer.core.block import Block, SignalInfo
from meegxer.core.manifest import Manifest
from meegxer.core.node import Params, Processor, TransformResult
from meegxer.core.serialization import State, StateValue
from meegxer.nodes._manifest import stream_manifest

_EOG_LABEL = re.compile(r"^(V|H)?EOG", re.IGNORECASE)
_NOT_EEG = {"eog", "ecg", "emg", "misc", "stim", "resp", "bio", "gsr", "temperature"}


def split_channels(info: SignalInfo, eog: str) -> tuple[list[str], list[str]]:
    """(EEG channels to correct, EOG channels) for the ``eog`` setting: ``auto``
    (channels typed or named EOG) or a comma-separated list of labels."""
    types = info.ch_types or ("",) * len(info.ch_labels)
    if eog.strip().lower() == "auto":
        artifact = [lb for lb, t in zip(info.ch_labels, types, strict=True)
                    if t.lower() == "eog" or _EOG_LABEL.match(lb)]  # fmt: skip
    else:
        artifact = [lb.strip() for lb in eog.split(",") if lb.strip()]
        missing = [lb for lb in artifact if lb not in info.ch_labels]
        if missing:
            raise ValueError(f"no channel named {', '.join(missing)}")
    if not artifact:
        raise ValueError("no EOG channel found: name them in the eog setting, or "
                         "mark their type as EOG (Recording window)")  # fmt: skip
    eeg = [lb for lb, t in zip(info.ch_labels, types, strict=True)
           if lb not in artifact and lb not in info.bads
           and t.lower() not in _NOT_EEG]  # fmt: skip
    return eeg, artifact


class EOGRegression(Processor):
    """Remove eye artifacts by regressing out the EOG channels."""

    manifest: ClassVar[Manifest] = stream_manifest(
        "meegxer.nodes.eog_regression",
        "0.1.0",
        [
            {"name": "eog", "type": "string", "default": "auto", "primary": True,
             "description": "EOG channels: auto (typed or named EOG) or labels, "
                            "comma-separated"},
        ],
        description="Subtract the EOG channels' contribution from the EEG "
        "(regression fitted on calibration data, MNE).",
        requires_fit=True,
        channel_types=["eeg", "eog"],  # it needs to see the EOG channels
        input_space="sensor",  # electrodes only (spec 0.2.22)
    )  # fmt: skip

    def fit(self, calibration: Block, params: Params) -> State:
        import mne

        info = calibration.info
        eeg, eog = split_channels(info, str(params["eog"]))
        if not eeg:
            raise ValueError("no EEG channel to correct")
        labels = [*eeg, *eog]
        raw_info = mne.create_info(labels, info.sfreq,
                                   ["eeg"] * len(eeg) + ["eog"] * len(eog))  # fmt: skip
        idx = [info.ch_labels.index(lb) for lb in labels]
        raw = mne.io.RawArray(np.array(calibration.data[idx], dtype=np.float64),
                              raw_info, verbose="error")  # fmt: skip
        # the data come as referenced upstream (e.g. a Re-reference strip):
        # tell MNE so, without changing them
        raw.set_eeg_reference([], verbose="error")
        model = mne.preprocessing.EOGRegression(
            picks="eeg", picks_artifact="eog", proj=False
        ).fit(raw)
        eeg_labels: list[StateValue] = list(eeg)
        eog_labels: list[StateValue] = list(eog)
        return {"eeg": eeg_labels, "eog": eog_labels,
                "coef": np.asarray(model.coef_, dtype=np.float64)}  # fmt: skip

    def transform(
        self, block: Block, model: State, state: State | None, params: Params
    ) -> TransformResult:
        labels = block.info.ch_labels
        eeg = [labels.index(str(lb)) for lb in model["eeg"]]  # type: ignore[union-attr]
        eog = [labels.index(str(lb)) for lb in model["eog"]]  # type: ignore[union-attr]
        data = block.data.copy()
        data[eeg] -= np.asarray(model["coef"]) @ block.data[eog]
        return TransformResult(
            Block(data, block.info, block.sample_index, block.timestamp,
                  block.bypassed), {}
        )  # fmt: skip
