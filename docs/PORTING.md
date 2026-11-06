# Porting your code to MeegXer

You have analysis code that works: an MNE script, a NumPy function, a
classifier, a feature you compute per window. This guide turns it into a
MeegXer strip with as little change as possible: most of the time you paste
the body of your code into one function and delete the lines that read and
write files. The reference for every convention is `docs/CUSTOM_SCRIPTS.md`.

## 1. Pick the entry point from what your code works on

| Your code works on… | Kind | Function you write | Live? |
|---|---|---|---|
| a whole recording, as `mne.io.Raw` (filters, ICA, re-referencing, interpolation) | `process` | `process(raw, ...)` returns the Raw | offline |
| blocks of samples, as a NumPy array (a causal filter, a running normalisation) | `transform` | `transform(data, state, info, ...)` returns `(data, state)` | live and offline |
| a model learnt first, then applied (a spatial filter, a classifier, a z-score) | `calibrated` | `fit(raw, ...)` returns the model; `transform(data, state, model, ...)` | live and offline |
| one value (or a few) per channel per analysis window (band power, entropy, a ratio) | `features` | `features(data, info, ...)` returns `{name: per-channel values}` | live and offline |

Coming next: `mne.Epochs` (`process_epochs`), scripts made of several files
with data files, and declared extra packages (`REQUIRES`).

## 2. Five steps

```bash
meegxer-plugin new my_filter --kind process       # a working template
# paste your code where it says "your code"
meegxer-plugin run --script my_filter.py -i rec.set            # try it
meegxer-plugin run --script my_filter.py -i rec.set -o out.set # and write it
meegxer-plugin add --script my_filter.py          # make it a strip
```

`run` uses your script exactly as the strip will (offline, through the same
runner) and says what came out: the channels, rate and length, and the events
it added, or the feature windows. `--param low=0.5` tries another setting.
`add` checks it again (the conformance suite: settings, block by block equal
to one pass, a restart) and asks you to accept the local-code notice.

## 3. Recipes: before and after

### An MNE script on a recording → `process`

Before: a script that reads a file, cleans it and saves it.

```python
import mne

raw = mne.io.read_raw_fif("sub-01_raw.fif", preload=True)
raw.filter(1.0, 40.0)
raw.notch_filter(50.0)
raw.set_eeg_reference("average")
raw.save("sub-01_clean_raw.fif")
```

After: the same lines in `process`; MeegXer does the reading and writing
(any input it opens, any writer after the strip), and the numbers you might
change become settings.

```python
def process(raw, low: float = 1.0, high: float = 40.0, line: float = 50.0):
    """Band-pass, line-noise notch and average reference.

    Parameters
    ----------
    low : float
        Low cut-off (Hz).
    high : float
        High cut-off (Hz).
    line : float
        Line frequency (Hz).
    """
    raw.filter(low, high)
    raw.notch_filter(line)
    raw.set_eeg_reference("average")
    return raw
```

### A NumPy function on blocks → `transform`

Before: a running baseline you subtract from a stream.

```python
alpha = 0.01
baseline = None
for chunk in stream:  # (channels, samples)
    if baseline is None:
        baseline = chunk[:, :1].copy()
    out = np.empty_like(chunk)
    for i in range(chunk.shape[1]):
        baseline = (1 - alpha) * baseline + alpha * chunk[:, i : i + 1]
        out[:, i : i + 1] = chunk[:, i : i + 1] - baseline
    send(out)
```

After: the loop over chunks is MeegXer's; what survived between chunks
(`baseline`) goes into `state`.

```python
import numpy as np


def transform(data, state, alpha: float = 0.01):
    """Subtract an exponential running baseline.

    Parameters
    ----------
    alpha : float
        Smoothing factor per sample.
    """
    baseline = data[:, :1].copy() if state is None else state["baseline"]
    out = np.empty_like(data)
    for i in range(data.shape[1]):
        baseline = (1 - alpha) * baseline + alpha * data[:, i : i + 1]
        out[:, i : i + 1] = data[:, i : i + 1] - baseline
    return out, {"baseline": baseline}
```

A per-sample Python loop is slow live; `scipy.signal.lfilter` with `zi` in
the state does the same at C speed (the `transform` template shows the
pattern).

### A model you train, then apply → `calibrated`

Before: fit on calibration data, then apply to new data.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=8).fit(calibration.get_data().T)
cleaned = pca.inverse_transform(pca.transform(data.T)).T
```

After: `fit` returns what the model *is* (arrays and numbers, so MeegXer can
save it in sessions and send it to Sterile mode), `transform` applies it.

```python
from sklearn.decomposition import PCA


def fit(raw, n_components: int = 8):
    """Keep the leading principal components.

    Parameters
    ----------
    n_components : int
        Components kept.
    """
    x = raw.get_data()
    pca = PCA(n_components=min(n_components, x.shape[0])).fit(x.T)
    return {"mean": pca.mean_, "components": pca.components_}


def transform(data, state, model, n_components: int = 8):
    centred = data.T - model["mean"]
    scores = centred @ model["components"].T
    return (scores @ model["components"] + model["mean"]).T, state
```

A model is a dict of NumPy arrays, numbers and strings, never an object
(a fitted scikit-learn estimator, an MNE ICA): keep its arrays. Trials with
labels (a classifier on cued trials): set `FIT_EVENTS = True` and read them in
`fit` from `raw.annotations`.

### A measure per window → `features`

Before: you cut epochs or windows yourself and compute a value on each.

```python
for start in range(0, n_samples - win, step):
    seg = data[:, start : start + win]
    f, p = scipy.signal.welch(seg, fs)
    ratio = p[:, (f >= 8) & (f < 13)].mean(1) / p[:, (f >= 4) & (f < 8)].mean(1)
    rows.append(ratio)
```

After: MeegXer cuts the windows (`window` and `step` settings, the same live
and offline) and hands you each one.

```python
import scipy.signal

UNITS = {"alpha_theta": "1"}


def features(data, info, window: float = 2.0, step: float = 0.5):
    """Alpha/theta power ratio per channel."""
    f, p = scipy.signal.welch(data, info["sfreq"], nperseg=min(256, data.shape[1]))
    alpha = p[:, (f >= 8) & (f < 13)].mean(axis=1)
    theta = p[:, (f >= 4) & (f < 8)].mean(axis=1)
    return {"alpha_theta": alpha / theta}
```

## 4. What changes when you port

| In your code | In a MeegXer script |
|---|---|
| reading and writing files, paths | none: the input strip reads, writers after it write |
| numbers you tune (cut-offs, thresholds) | settings: parameters with a default, typed, described in the docstring (`(Hz)` gives the unit) |
| channel picks (`pick_types`, lists of names) | MeegXer gives the channel types the strip processes (EEG by default; *Advanced → Channel types*); keep the same channels in what you return |
| plots, `print` | remove plots; `print` and logging go to MeegXer's log |
| variables that live across chunks | `state` (transform) or the model (fit) |
| units | volts, as in MNE (the GUI shows µV) |
| random numbers | seed them (`np.random.default_rng(0)`): results must repeat |
| events you mark | `raw.annotations.append(...)` in `process`: they become events on the Events rail |

## 5. When the check says no

`meegxer-plugin check --script FILE` (and `add`) explain each failure in words;
the usual ones:

- **"block-wise output deviates by up to …"**: something depends on where a
  block starts (a per-block mean, a filter restarted each block). Keep what
  carries over in `state`.
- **"setting 'x' needs a default value"** or **"setting 'x': use bool, int,
  float, str or Literal[...]"**: every parameter after the data needs a
  default of one of those types.
- **"process must keep the channels (names and order)"**, **"process must
  return N channels x M samples"**, **"stream processors must return as many
  samples as given"**: `process` and `transform` return the same channels and
  the same number of samples; drop channels with the built-in *Pick channels*
  strip instead.

In Sterile mode your script runs in a container, away from the network, with
the same output (`docs/STERILE.md`).
