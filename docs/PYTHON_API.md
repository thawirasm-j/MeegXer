# Python API (M10.7)

For MNE users: run a MeegXer pipeline from Python, on an `mne.io.Raw` in
memory or a recording file, and get MNE objects back.

```python
import mne
from meegxer.api import run

raw = mne.io.read_raw("sub-01.set", preload=True)
clean = run("pipeline.json", raw)  # mne.io.Raw: the chain's end
epochs = run("erp.json", raw, output="ep1")  # mne.Epochs: an Epochs strip's
ics = run("ica.json", raw, output="ica1:ics")  # the ICA strip's components
```

`run` is a thin layer over the batch scheduler (architecture rule 4: the same
node code as the GUI's offline runs and `meegxer run`). It ships with
`meegxer-core` as the `meegxer.api` module and, like the core, imports no GUI
and no node package (an import-linter contract): the strips come through the
node registry.

## `run(pipeline, raw, *, output=None, block_size=None, allow_version_mismatch=False)`

| Argument | What it takes |
|---|---|
| `pipeline` | A pipeline file (`.json`) or session file (`.meegxer.json`), or the same as a dict. A session's calibrated models are used where they fit this input; otherwise they are refitted (with a warning). |
| `raw` | An `mne.io.Raw`, or a recording file the File input reads (`.set`, `.edf`, `.bdf`, `.vhdr`, `.fif`, `.xdf`, …). |
| `output` | Where the result is read; see below. |
| `block_size` | Feed the strips in blocks of this many samples (as `meegxer run --block-size`). |
| `allow_version_mismatch` | Run a session whose node versions or plugin images differ (their models are refitted). |

**The input.** The pipeline's input strip is replaced by `raw`.
- An `mne.io.Raw` is read as a copy, so yours is left as it is. Its annotations are the input's events, on the Events rail. It has no review at import, since that belongs to a file.
- A file is read as the File input reads it, with what it carries, e.g. a `.set`'s ICA on the ICs rail (`docs/RAILS.md`).

**The output.**

| `output` | Returns |
|---|---|
| `None` | The signal at the chain's end, as `mne.io.Raw`. |
| a strip's id, e.g. `"bp1"` | That strip's signal, as `mne.io.Raw`. |
| an Epochs or Review epochs strip's id, e.g. `"ep1"` | Its epochs, as `mne.Epochs` (`EpochsArray`, as the epochs writer makes them); None when no event was cut. |
| `"strip:port"`, e.g. `"ica1:ics"` or `"input:ics"` | That rail output: the ICs as `mne.io.Raw` with `misc` channels, or an epochs rail as `mne.Epochs`. |

A feature strip's output (spectra, band power, …) is not returned; write it
with the feature writer (`.tsv`) in the pipeline.

**What the Raw holds.** It holds what a `.set` written at the same place would hold (`make_raw`, shared with the file writers):
- the events on the Events rail there, as annotations: the input's, a review's marks, a detector's spans;
- a `boundary` annotation at each cut a Reject segments strip made;
- the channels' types and positions, and the bad channels;
- no measurement date and no subject information.

**The run** goes as `meegxer run` goes:
- The strips fit on the whole recording.
- The pipeline's own writers write their files (a shared session's writers without a file name are left out).
- A Review strip not yet reviewed stops the run (`meegxer.core.review.ReviewNeeded`).
- What the run noted goes to the `meegxer` logger (`logging`). Saves it skipped, and models it could not reuse, are Python warnings.

## Next

MNE ICA files (`-ica.fif`) on the ICs rail (M10.7b):
- the File input reads one onto the ICs rail;
- an *MNE ICA (.fif)* writer stores the rail's decomposition;
- `run(..., ica=...)` takes an `mne.preprocessing.ICA`.
