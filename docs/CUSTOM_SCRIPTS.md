# Custom scripts

A custom script turns your own MNE-Python code (or code from a paper) into a
MeegXer strip, without containers or packaging. Write an ordinary script with one
function, `process`:

```python
"""Band-pass with MNE."""  # becomes the strip description

from typing import Literal


def process(
    raw, low: float = 1.0, high: float = 40.0, method: Literal["fir", "iir"] = "fir"
):
    """Band-pass the recording.

    Parameters
    ----------
    low : float
        Low cut-off (Hz).
    high : float
        High cut-off (Hz).
    method : str
        Filter design.
    """
    return raw.filter(low, high, method=method)
```

## Conventions

- `process(raw, ...)` gets the whole recording as an `mne.io.Raw`: the channels
  the strip processes (EEG by default), in volts, with bad channels marked.
- It returns an `mne.io.Raw` (the same object is fine) or an array, with the
  **same channels, sampling rate and length**. Annotations it adds come out as
  events (markers).
- **Settings come from the signature.** Every parameter after `raw` needs a
  default. Its type comes from the annotation or the default: `bool`, `int`,
  `float`, `str`, or `Literal[...]` for a choice list.
- **The docstring** (numpy style, `Parameters` section) gives each setting's
  description. A trailing `(unit)` such as `(Hz)` becomes its unit.
- Optional, at module level: `NAME = "lab.my_filter"` (default
  `script.<file name>`), `__version__ = "1.2.0"` (default `0.1.0`).
- **Credit** (optional, encouraged): `__author__` (a name or a list),
  `__license__` (e.g. `"BSD-3-Clause"`) and `REFERENCES`, the papers to cite when
  someone uses the script, as strings or as dicts with `citation` and a `doi` or
  `url`:

  ```python
  __author__ = ["Jane Doe", "TrimLAB"]
  __license__ = "BSD-3-Clause"
  REFERENCES = [
      {
          "citation": "Doe J (2025). A better blink remover. J Neural Eng 1:1.",
          "doi": "10.1234/example",
      },
  ]
  ```

  The strip shows them under *Credits*; sessions record them and BIDS derivatives
  list them in `dataset_description.json`. `meegxer-plugin check` notes when there
  is none, but never refuses a script for it.
- **Where it is listed** (optional): `CATEGORY = "artifacts"` puts the strip under
  that heading of its group in the add-strip menu (filters, reference, artifacts;
  spectral, time-frequency, connectivity, complexity, decoding, events; or your
  own word). The group comes from what the script returns; without a category it
  is listed under *Other*. Scripts are marked ✎ (they run unsandboxed).
- **Offline only.** `process` sees the whole recording at once, so it runs in
  offline runs (files, BIDS), not live. For live use, write `transform` instead
  (below).

## Live scripts

A script with `transform` instead of `process` runs block by block, so the same
code works live and offline:

```python
"""Causal high-pass designed with MNE."""

import mne
from scipy import signal


def transform(data, state, info, cutoff: float = 1.0, order: int = 4):
    """High-pass, keeping the filter state between blocks.

    Parameters
    ----------
    cutoff : float
        Cut-off frequency (Hz).
    order : int
        Butterworth order.
    """
    if state is None:  # the first block
        iir = mne.filter.construct_iir_filter(
            {"order": order, "ftype": "butter", "output": "sos"},
            cutoff,
            None,
            info["sfreq"],
            "highpass",
            return_copy=False,
        )
        sos = iir["sos"]
        zi = signal.sosfilt_zi(sos)[:, None, :] * data[:, :1][None]
        state = {"sos": sos, "zi": zi}
    out, zf = signal.sosfilt(state["sos"], data, axis=-1, zi=state["zi"])
    return out, {"sos": state["sos"], "zi": zf}
```

- `data` is one block, `(channels, samples)`, in volts. Return the processed block
  (same shape) and the new `state`.
- `state` is `None` on the first block. Put everything that must survive between
  blocks in it (a dict of numbers and arrays): MeegXer keeps it, saves it and
  restores it if the script's process restarts. Return a new state rather than
  changing the old one in place.
- Optional parameters `info` (channel labels `ch_labels`, `sfreq`, `bads`,
  `sample_index`) and `model`; all other parameters are settings, as above.
- **Calibration:** add `fit(raw, ...)`. It gets the calibration recording as an
  `mne.io.Raw` (live: the seconds you calibrate on; offline: the whole recording)
  and returns the model as a dict; `transform` receives it as `model`:

  ```python
  def fit(raw):
      x = raw.get_data()
      return {"mean": x.mean(axis=1, keepdims=True), "std": x.std(axis=1, keepdims=True)}


  def transform(data, state, model):
      return (data - model["mean"]) / model["std"], state
  ```
- **Labelled trials:** set `FIT_EVENTS = True` and `fit` finds the calibration's
  events in `raw.annotations` (onset in seconds from the start of `raw`,
  duration, label), e.g. for a classifier trained on cued trials. The fit is
  refused when the calibration has no events (offline: the recording's events or
  an events file; live: the LSL input's marker stream, or a recording played
  live).
- **Block by block must equal one pass.** Live, blocks arrive in whatever sizes the
  stream delivers; the result must not depend on them (acceptance test 1). A script
  that, say, removes each block's own mean fails this. `meegxer-plugin check`
  tests it, and adding a script that fails is refused with the reason.
- No look-ahead yet: the output covers the same samples as the input.

## Feature-extraction scripts

A script with `features` computes values per analysis window instead of changing
the signal; it lands under *Custom feature extraction* and works live and offline:

```python
"""Alpha power and line length per channel."""

import mne
import numpy as np


def features(data, info, fmin: float = 8.0, fmax: float = 13.0):
    """Band power (Welch) and line length.

    Parameters
    ----------
    fmin : float
        Lower band edge (Hz).
    fmax : float
        Upper band edge (Hz).
    """
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        n_fft=min(256, data.shape[1]),
        verbose="error",
    )
    return {
        "alpha_power": psd.mean(axis=1),
        "line_length": np.abs(np.diff(data, axis=1)).sum(axis=1),
    }
```

- `data` is **one whole window**, `(channels, samples)` in volts. MeegXer cuts the
  stream into windows and keeps the unfinished one between blocks, so there is no
  state to manage and the result never depends on block sizes.
- Return a dict of feature name → one value per channel (or a `channels × features`
  array; its columns are then named `feature_0`, `feature_1`, …). The output is a
  Features stream with `channel` and `feature` labels, which the feature writers
  (CSV/TSV, BIDS derivatives) record per window.
- Every such strip gets **window** (default 2 s) and **step** (0.5 s) settings.
  Declare them in the signature (`window: float = 4.0`) to choose other defaults;
  `features` then receives them too.
- Optional `info`, as for `transform`. A window that is still incomplete at the end
  of a recording is dropped, as the built-in feature strips do.

Octave/EEGLAB scripts (`function EEG = process(EEG, ...)`) are planned next.

## Add one

In the GUI: **+ Add strip → ✎ Add a custom script…**, pick the file, read the notice.
From the command line:

```bash
meegxer-plugin check --script my_filter.py     # does it follow the conventions?
meegxer-plugin add --script my_filter.py       # notice, then the same checks
meegxer-plugin list
meegxer-plugin remove script.my_filter
```

## The notice: it runs on your computer

A custom script runs as a local process **with your permissions**, outside any
sandbox: it can read and change your files and send data over the network. If you
work with identifiable patient data, running it may break your institution's data
rules (HIPAA, GDPR, IRB approval). MeegXer records that you accepted this, and
sessions record a SHA-256 of the script: reusing a session after the script
changed is refused, so a result always names the code that made it.

Sterile mode (planned) will run scripts and every other strip isolated in a
container, away from the network and the rest of the computer.
