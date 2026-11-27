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

## `run(pipeline, raw, *, output=None, ica=None, block_size=None, allow_version_mismatch=False, sterile=None)`

| Argument | What it takes |
|---|---|
| `pipeline` | A pipeline file (`.json`) or session file (`.meegxer.json`), or the same as a dict. A session's calibrated models are used where they fit this input; otherwise they are refitted (with a warning). |
| `raw` | An `mne.io.Raw`, or a recording file the File input reads (`.set`, `.edf`, `.bdf`, `.vhdr`, `.fif`, `.xdf`, …). |
| `output` | Where the result is read; see below. |
| `ica` | An `mne.preprocessing.ICA` of `raw`, or its file (`-ica.fif`): its components start the ICs rail at the input (`output="input:ics"`, or an ICA-reading strip), instead of a `.set`'s own. |
| `block_size` | Feed the strips in blocks of this many samples (as `meegxer run --block-size`). |
| `allow_version_mismatch` | Run a session whose node versions or plugin images differ (their models are refitted). |
| `sterile` | Run the processing in the Sterile container (`docs/STERILE.md`). None: the `general.sterile` setting (`MEEGXER_STERILE`). |

**The input.** The pipeline's input strip is replaced by `raw`.
- An `mne.io.Raw` is read as a copy, so yours is left as it is. Its annotations are the input's events, on the Events rail. It has no review at import, since that belongs to a file.
- A file is read as the File input reads it, with what it carries, e.g. a `.set`'s ICA on the ICs rail (`docs/RAILS.md`).

**The output.**

| `output` | Returns |
|---|---|
| `None` | The signal at the chain's end, as `mne.io.Raw`. |
| a strip's id, e.g. `"bp1"` | That strip's signal, as `mne.io.Raw`. |
| an Epochs, Review epochs or epochs-script strip's id, e.g. `"ep1"` | Its epochs, as `mne.Epochs` (`EpochsArray`, as the epochs writer makes them); None when no event was cut. |
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
- A BIDS writer that cannot name where it writes is left out with a warning. This happens when a pipeline made for a BIDS input, such as the *ERP study* template, runs on a Raw in memory or on a file not named like `sub-01_task-erp_eeg.set`; `output` still returns the result.
- A Review strip not yet reviewed stops the run (`meegxer.core.review.ReviewNeeded`).
- What the run noted goes to the `meegxer` logger (`logging`). Saves it skipped, and models it could not reuse, are Python warnings.

## `export_script(pipeline, path=None)`: the pipeline as a script

```python
from meegxer.api import export_script

export_script("erp.json", "erp.py")  # or a session, with its fitted models
```

```bash
python erp.py sub-01.set -o clean.set              # the chain's end
python erp.py sub-01.set --output ep1 -o ep-epo.fif  # an Epochs strip's epochs
```

The script embeds the pipeline (a session with its fitted models), runs it
with `run`, and writes the result (`.fif`, `.set`, `.edf`; `-epo.fif` or
`.set` for epochs): exactly what MeegXer computes (decided 2026-11-27).
Its docstring lists the strips and the plugins or custom scripts it needs,
which must be added on the computer that runs it (`meegxer-plugin add …`); it
needs MeegXer (the version that exported it). Also `meegxer export PIPELINE
[-o SCRIPT]`, and *Save → Export as Python script…* in the GUI (with the
models the last offline run fitted, or calibrated live).

## `export_mne(pipeline, path=None)`: as MNE code, no MeegXer

```python
from meegxer.api import export_mne

export_mne("erp.json", "erp_mne.py")
```

```bash
python erp_mne.py sub-01.set -o clean.fif --epochs erp-epo.fif
```

A standalone script: MNE, NumPy and SciPy only (decided 2026-11-27). Each
strip writes its own code, next to the code it reproduces in MeegXer, and a
test per strip checks that it gives the strip's offline output; fitted models
are fitted again on the recording the script reads. Its `pipeline(path)`
returns `(raw, epochs)`. Strips with such code so far: the File input, the
IIR and FIR filters (causal and zero-phase), notch, re-reference, pick
channels, resample, Epochs, and the `.set`, `.edf` and epochs writers; more
come strip by strip (`docs/ROADMAP.md`, M11.7). Any other strip (an Octave
script, a plugin image) is named and the script is not written:
`export_script` runs any pipeline. Also `meegxer export PIPELINE
--standalone`, and *Save → Export as standalone MNE script…* in the GUI.

## MNE ICA files

An MNE ICA (`-ica.fif`) goes both ways (`meegxer.core.mne_ica`, `docs/RAILS.md`):
- **In:** `run(..., ica=ica)` here, or the File input's *MNE ICA file* in the GUI and pipelines (`ica`), starts the ICs rail with MNE's own sources.
- **Out:** the *MNE ICA (.fif)* writer stores the ICs rail's decomposition for `mne.preprocessing.read_ica`. The components an ICA strip removed are marked `exclude`, and ICLabel's classes go in `labels_`.

```python
ica = mne.preprocessing.read_ica("sub-01-ica.fif")
ics = run("pipeline.json", raw, ica=ica, output="input:ics")
```
