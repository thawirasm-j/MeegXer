"""Templates for custom scripts (``meegxer-plugin new``; ``docs/PORTING.md``).

Each kind is a complete script that works as it is (it passes the conformance
suite), with comments where your own code goes, so porting is replacing the
marked lines with what you already have.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

CREDIT = """
# -- credit (optional, shown under Credits, recorded in sessions) ---------------
# __author__ = ["Your Name"]
# __license__ = "BSD-3-Clause"
# REFERENCES = [{"citation": "Author A (2024). Title. Journal 1:1.", "doi": "10.0/x"}]
# CATEGORY = "filters"  # where the add-strip menu lists it (or your own word)
# REQUIRES = ["pyriemann"]  # packages it imports that MeegXer does not ship
"""

PROCESS = '''"""{title}: what it does, in one line (the strip's description)."""

# A script on the whole recording, as MNE gives it: paste the body of your MNE
# script into process(). Offline only (it sees all of the recording at once);
# for live use, see the transform kind (meegxer-plugin new NAME --kind transform).
{credit}

def process(raw, low: float = 1.0, high: float = 40.0):
    """Band-pass the recording (replace with your own processing).

    Parameters
    ----------
    low : float
        Low cut-off (Hz).
    high : float
        High cut-off (Hz).
    """
    # `raw` is an mne.io.Raw: the channels this strip processes (EEG by
    # default), in volts, bad channels marked. Every setting above (with its
    # default, type and the docstring line) becomes a field of the strip.
    #
    # -- your code: anything that changes `raw` and keeps its channels, rate
    # -- and length (filters, ICA applied, re-referencing, interpolation...)
    raw.filter(low, high, verbose="error")
    # raw.annotations.append(onset, duration, "label")  # these become events
    return raw
'''

TRANSFORM = '''"""{title}: what it does, in one line (the strip's description)."""

# A script block by block: the same code runs live (from an amplifier) and
# offline. Put whatever must survive from one block to the next in `state`;
# the result must not depend on how the stream is cut into blocks
# (meegxer-plugin check tests it).
{credit}
import numpy as np


def transform(data, state, info, gain: float = 1.0, smooth: int = 5):
    """Scale, then a causal moving average (replace with your own).

    Parameters
    ----------
    gain : float
        Multiplier.
    smooth : int
        Samples averaged (1: none).
    """
    # `data` is one block, (channels, samples), in volts; `info` holds
    # ch_labels, sfreq, bads and sample_index. `state` is None on the first
    # block: start it there.
    if state is None:
        state = {{"tail": np.zeros((data.shape[0], max(smooth - 1, 0)))}}
    # -- your code: compute `out` (same shape as `data`) from `data` and state
    joined = np.concatenate([state["tail"], data * gain], axis=1)
    kernel = np.ones(smooth) / smooth
    out = np.stack([np.convolve(row, kernel, mode="valid") for row in joined])
    tail = joined[:, joined.shape[1] - (smooth - 1):] if smooth > 1 else state["tail"]
    return out, {{"tail": tail}}
'''

CALIBRATED = '''"""{title}: what it does, in one line (the strip's description)."""

# A script that learns from calibration data first (fit), then processes
# block by block with what it learned (transform): live, you calibrate it for
# a few seconds; offline, it fits on the whole recording.
{credit}

def fit(raw, clip: float = 3.0):
    """Learn each channel's mean and spread (replace with your own model).

    Parameters
    ----------
    clip : float
        Limit, in standard deviations.
    """
    # `raw` is the calibration recording as an mne.io.Raw. Return the model
    # as a dict of numbers and arrays (MeegXer keeps it, saves it in sessions).
    x = raw.get_data()
    mean, std = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    return {{"mean": mean, "std": std}}


def transform(data, state, model, clip: float = 3.0):
    """Standardize each channel with the fitted model, then clip."""
    # -- your code: apply `model` to the block `data` (channels, samples)
    z = (data - model["mean"]) / model["std"]
    return z.clip(-clip, clip), state
'''

FEATURES = '''"""{title}: what it does, in one line (the strip's description)."""

# A script that computes values per analysis window (a feature extraction):
# MeegXer cuts the stream into windows (the window and step settings) and
# hands you each whole window; no state to keep. Works live and offline.
{credit}
import numpy as np

UNITS = {{"line_length": "V", "variance": "V^2"}}  # one unit per feature


def features(data, info, window: float = 2.0, step: float = 0.5):
    """Line length and variance per channel (replace with your own).

    Parameters
    ----------
    window : float
        Window length (s).
    step : float
        Step between windows (s).
    """
    # `data` is one window, (channels, samples), in volts. Return a dict of
    # feature name -> one value per channel.
    # -- your code: compute your measures from `data`
    return {{
        "line_length": np.abs(np.diff(data, axis=1)).sum(axis=1),
        "variance": data.var(axis=1),
    }}
'''

EPOCHS = '''"""{title}: what it does, in one line (the strip's description)."""

# A script on the epochs an Epochs strip cut, as MNE gives them: paste the body
# of your script on mne.Epochs into process_epochs(). Put it after an Epochs
# strip; the strips after it (the ERP tap, the epochs writer) get the epochs it
# returns. Offline only (it sees all of the epochs at once).
{credit}

def process_epochs(epochs, threshold: float = 100.0):
    """Drop epochs by peak-to-peak amplitude (replace with your own).

    Parameters
    ----------
    threshold : float
        Largest peak-to-peak amplitude kept (µV).
    """
    # `epochs` is an mne.Epochs: one per event, labelled by its event (its
    # event_id), in volts, times from the event. Return the epochs you keep:
    # fewer of them, other data, other labels, but the same rate.
    #
    # -- your code: anything on `epochs` (rejection, baseline, relabelling...)
    epochs.drop_bad(reject={{"eeg": threshold * 1e-6}}, verbose="error")  # µV -> V
    return epochs
'''

TEMPLATES = {
    "process": PROCESS,
    "transform": TRANSFORM,
    "calibrated": CALIBRATED,
    "features": FEATURES,
    "epochs": EPOCHS,
}
KINDS = tuple(TEMPLATES)


def script_text(name: str, kind: str) -> str:
    """The template of ``kind`` for a script called ``name``."""
    if kind not in TEMPLATES:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    title = name.replace("_", " ").capitalize()
    return TEMPLATES[kind].format(title=title, credit=CREDIT)


def new_script(name: str, kind: str, folder: Path = Path()) -> Path:
    """Write ``folder/name.py`` from the template of ``kind``; never over an
    existing file. ``name`` must be a Python identifier (it becomes the
    strip's name, ``script.<name>``)."""
    stem = name.removesuffix(".py")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stem) or keyword.iskeyword(stem):
        raise ValueError(
            f"{name!r}: use letters, digits and underscores (e.g. my_filter)"
        )
    path = folder / f"{stem}.py"
    if path.exists():
        raise FileExistsError(f"{path} exists: choose another name")
    path.write_text(script_text(stem, kind), encoding="utf-8")
    return path
