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
- **Offline only for now.** The script sees the whole recording at once, so it
  runs in offline runs (files, BIDS), not live. Live scripts (fit/transform, block
  by block) come later.

## Add one

In the GUI: **+ Add strip → Custom script…**, pick the file, read the notice.
From the command line:

```bash
meegxer-plugin check --script my_filter.py     # does it follow the conventions?
meegxer-plugin add --script my_filter.py       # asks you to accept the notice
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
