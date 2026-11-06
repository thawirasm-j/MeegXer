# Custom scripts

A custom script turns your own MNE-Python code (or code from a paper) into a
MeegXer strip, without containers or packaging. **New to this? Start with
`docs/PORTING.md`**: which entry point fits your code, a template to paste it
into (`meegxer-plugin new NAME --kind process|transform|calibrated|features|epochs`),
before/after recipes, and `meegxer-plugin run --script FILE -i REC` to try it
on a recording. This page is the reference.

Write an ordinary script with one function, `process`:

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
- Optional `UNITS` at the top of the script states the values' units, shown in the
  feature view and recorded by the writers: one for every feature
  (`UNITS = "V^2"`) or one per feature name
  (`UNITS = {"alpha_power": "V^2/Hz", "line_length": "V"}`). Use `"1"` for a
  dimensionless value.

## Epochs scripts

A script with `process_epochs` works on the epochs an Epochs strip cut, as
`mne.Epochs`. Put it after an Epochs strip; the strips after it (an ERP tap, the
epochs writers, the Python API's `output=`) get the epochs it returns. Offline
only, like `process`.

```python
"""Drop epochs by peak-to-peak amplitude, then baseline-correct."""


def process_epochs(epochs, threshold: float = 100.0):
    """Peak-to-peak rejection (MNE).

    Parameters
    ----------
    threshold : float
        Largest peak-to-peak amplitude kept (µV).
    """
    epochs.drop_bad(reject={"eeg": threshold * 1e-6}, verbose="error")  # µV -> V
    return epochs.apply_baseline((None, 0), verbose="error")
```

- `epochs` holds one epoch per event, labelled by its event (`event_id`, the
  labels on the epochs rail), in volts, with times from the event as the Epochs
  strip set them. Settings come from the signature, as for `process`.
- Return `mne.Epochs` (the same object is fine): fewer epochs, other data
  (a baseline, a spatial filter), other labels (`epochs.event_id`), other times
  (`crop`). The **sampling rate stays the same**; resample before the Epochs
  strip instead.
- The strip passes the signal on unchanged and writes its own epochs rail
  (`docs/RAILS.md`, rule 8), as *Review epochs* does: the strips after it read
  its epochs, and the Epochs strip's own rail stays as it was cut.
- Where a Reject segments strip has cut the recording, `process_epochs` is
  called once for each stretch between cuts (with that stretch's epochs).
- `meegxer-plugin check` gives the script test epochs (around an event every
  second, labelled `go` and `stop`); `meegxer-plugin run --script FILE -i REC`
  cuts them with an Epochs strip first (`--epochs tmin=-0.5` sets its settings)
  and says how many it kept, per label; `-o kept-epo.fif` writes them.

## Script folders

A method that is more than one file (helper modules, a lookup table, trained
weights, a channel template) is added as its folder:

```text
lab_method/
├── main.py          # defines process (or transform, features, process_epochs)
├── lib/
│   ├── __init__.py
│   └── filters.py   # import lib.filters, as usual
└── data/
    └── weights.npy  # np.load(Path(__file__).parent / "data" / "weights.npy")
```

- **The entry file** is the one `.py` file at the folder's top that defines
  `process`, `transform`, `features` or `process_epochs` (found by reading the
  files, not running them). Its name is yours to choose; two such files is an
  error that names them.
- **Helpers import as usual**: the folder comes first on the import path, so
  `import helpers` or `from lib.filters import notch` work.
- **Data files** are read relative to the entry file, `Path(__file__).parent`,
  never by an absolute path (in Sterile mode the folder is somewhere else).
- **The strip is named after the folder** (`script.lab_method`), unless the
  entry file sets `NAME`.
- **Everything in it is the script**: every file but hidden ones (`.git`,
  `.venv`, `.env`, …), `__pycache__` and compiled Python, at most 100 MB in
  2000 files. Sessions record one SHA-256 over all of them, so a changed
  helper or data file is noticed like a changed script; Sterile mode sends
  them all. Keep recordings and outputs out of the folder.

```bash
meegxer-plugin run --script lab_method/ -i rec.set   # try it
meegxer-plugin add --script lab_method/              # add it
```

In the GUI: **+ Add strip → ✎ Add a script folder…**.

## Extra packages (`REQUIRES`)

A script may import any package MeegXer ships (MNE, SciPy, NumPy,
scikit-learn, mne-connectivity, antropy, …). For one it does not, name it at
the top of the script (or of a folder's entry file):

```python
REQUIRES = ["pyriemann", "mne-connectivity>=0.7"]
```

- Each item is a PyPI requirement: a name, optionally with extras and a
  version range (no URLs or paths). MeegXer reads it without running the
  script.
- **It must be installed in MeegXer's own Python.** `meegxer-plugin add`,
  `check` and `run` (and *Add a custom script…*) refuse a script whose
  packages are missing or outside their range, naming them and the command
  that installs them: `uv tool install --prerelease allow --with pyriemann
  meegxer` for a uv tool install (list every extra package your scripts use:
  a new `--with` list replaces the last), or `pip install pyriemann` where
  MeegXer runs.
- **Sessions record the versions** used (`requires` in the node's
  provenance); reloading with other versions warns, as for library versions.
- **In Sterile mode** the image needs them too: `meegxer sterile build`
  installs the added scripts' packages at the versions installed here, and a
  run whose scripts need one the image lacks is refused, saying to build
  again (as after updating MeegXer).

Octave/EEGLAB scripts (`function EEG = process(EEG, ...)`) are planned next.

## Add one

In the GUI: **+ Add strip → ✎ Add a custom script…**, pick the file, read the notice.
From the command line:

```bash
meegxer-plugin new my_filter --kind process    # a template to start from
meegxer-plugin run --script my_filter.py -i rec.set [-o out.set]  # try it
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

**Opening a pipeline or session runs no code.** Pipelines and sessions only
name strips; a script runs only if *you* added it with the notice accepted. A
pipeline or session from someone else that names a script you have not added
stops with "not installed", and the script command a session records is never
started from it (`packages/meegxer-plugins/tests/test_plugins_no_code_from_pipelines.py`
checks this). To use their script, read it, then add it yourself.

## In Sterile mode

With Sterile mode on (`docs/STERILE.md`), a custom script does not run on your
computer: it runs in a container of the Sterile image, with no network, no
capabilities and a read-only filesystem, like the built-in strips. Nothing is
mounted: when the run starts, MeegXer sends the script's code into the
container (the file, or every file of a script folder, each checked by its
SHA-256), which serves it exactly as outside Sterile
mode, so its output is the same. Editing the script needs no image rebuild;
the next run sends the new code. The script can use the packages MeegXer
ships (MNE, SciPy, scikit-learn, …) and those it declares in `REQUIRES`, which
`meegxer sterile build` installs into the image. A local plugin that is a program of its
own (`meegxer-plugin add --local`) cannot run in the image and is refused in
Sterile mode.
