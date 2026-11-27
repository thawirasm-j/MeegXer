# Octave and EEGLAB scripts

MeegXer runs Octave/EEGLAB code (M11.5) in one image, built on your computer:
GNU Octave with the signal package and a pinned EEGLAB. Every Octave script
runs in it, sandboxed like any plugin container (no network, read-only, no
capabilities), so a script gives the same result on every computer and a
session names the EEGLAB it ran with. You need Docker or Podman; you do not
need Octave or EEGLAB installed.

## A script

An Octave script is a `.m` file with one function, named as its file, that
takes and returns an EEGLAB `EEG` structure. Paste the body of your EEGLAB
script into it:

```matlab
function EEG = lab_bandpass (EEG, varargin)
% Band-pass with EEGLAB's firfilt (the strip's description).
%
% Parameters
% ----------
% low : float = 1
%     Low cut-off (Hz).
% high : float = 40
%     High cut-off (Hz).
  opt = struct (varargin{:});  % the settings, by name: opt.low, opt.high
  EEG = pop_eegfiltnew (EEG, opt.low, opt.high);
end
```

- **Settings come from the help block** right after the `function` line:
  `name : type = default`, the type `float`, `int`, `bool`, `str` or a choice
  `{'fir', 'iir'}`, a description on the lines below; a trailing `(Hz)`
  becomes the unit. MeegXer reads it without running Octave. Every setting
  needs a default.
- **Called EEGLAB-style**: every setting as a name-value pair,
  `EEG = lab_bandpass (EEG, 'low', 1, 'high', 40)`; `struct (varargin{:})`
  turns them into fields. A script with settings takes `varargin`.
- **`EEG` is the recording in microvolts**, as EEGLAB has it (MeegXer works
  in volts and converts both ways): `EEG.data` (channels × samples),
  `EEG.srate`, `EEG.nbchan`, `EEG.pnts`, `EEG.chanlocs(k).labels`,
  `EEG.event` (the recording's events, when the run has them); the channels
  MeegXer marked bad are in `EEG.etc.meegxer.bads` (EEGLAB has no flag of its
  own). EEGLAB's functions, firfilt's filters and ICLabel are on the path.
- **Return the same channels and number of samples** (`EEG.data`); events
  you add to `EEG.event` (`type`, `latency` in samples from 1, `duration`)
  come out as events. To drop channels, use the *Pick channels* strip.
- **Offline**: this script sees the whole recording at once. For live use,
  write a live script (below).
- **Sandboxed, always**: the script runs in a container of the Octave image
  (no network, read-only, no capabilities), its code sent in when it starts
  (nothing is mounted), so adding one asks for no local-code notice, and an
  edited script needs no rebuild. In Sterile mode it stays in its own
  container. Sessions record its SHA-256, the image and the EEGLAB version;
  a changed script is refused like any changed plugin.

```bash
meegxer-plugin new lab_bandpass --kind octave            # a template
meegxer-plugin run --script lab_bandpass.m -i rec.set     # try it
meegxer-plugin check --script lab_bandpass.m              # the conformance suite
meegxer-plugin add --script lab_bandpass.m                # make it a strip
```

In the GUI: **+ Add strip → ✎ Add a custom script…**, pick the `.m` file.

## A live script

A script that also takes and returns its **state** is called block by block,
live and offline alike (decided 2026-11-25):

```matlab
function [EEG, state] = lab_highpass (EEG, state, varargin)
% High-pass block by block (Butterworth, the signal package's butter).
%
% Parameters
% ----------
% cutoff : float = 1
%     High-pass cut-off (Hz).
% order : int = 2
%     Butterworth order.
  opt = struct (varargin{:});
  [b, a] = butter (opt.order, opt.cutoff / (EEG.srate / 2), "high");
  if (isempty (state))  % the first block
    state = struct ("zi", zeros (max (numel (a), numel (b)) - 1, EEG.nbchan));
  endif
  % along time (dimension 2), even for a block of one sample
  [EEG.data, state.zi] = filter (b, a, double (EEG.data), state.zi, 2);
end
```

- **`EEG` is one block**, in microvolts, with the same fields as above;
  `EEG.xmin` is its start in seconds since the first sample. A block can be
  a single sample: work along dimension 2 (time), not on `EEG.data'`.
- **`state` is `[]` on the first block**; whatever the script returns comes
  back with the next block (numbers, arrays, text, structs and cells of
  them). MeegXer keeps it, not Octave, so a restarted container carries on
  exactly.
- **Causal, and the same block by block as in one pass**: the script sees
  no samples after the block. `meegxer-plugin check --script` runs MeegXer's
  acceptance test 1 (random block sizes, down to one sample); a script that
  forgets its state fails it.
- **Settings** come from the help block as for any script and change live
  (`meegxer set NODE cutoff=3`, or the strip in the GUI): read them on every
  block, as above.
- **Cost**: each block is a round trip into the container, about 5 ms here
  at 16 channels and 500 Hz (well under the block's own duration). When a
  block holds only a sample or two, that round trip is longer than the
  block, and the status counts it as a late block; the next block carries
  more samples, so nothing piles up. The container's start counts a few
  more at the beginning.

```bash
meegxer-plugin new lab_highpass --kind octave-live   # a template
meegxer-plugin check --script lab_highpass.m         # block-wise == one pass
```

## The image

```bash
meegxer-plugin octave build     # once, and again after updating MeegXer
meegxer-plugin octave check     # is it there and current? does EEGLAB work?
```

`check` starts the image in its sandbox, filters a test signal with EEGLAB's
`pop_eegfiltnew`, and says what it holds, e.g. `ok: meegxer-octave:0.5.0b3 is
up to date (GNU Octave 8.4.0, signal 1.4.5, EEGLAB 2026.0.0; pop_eegfiltnew
works)`. Building needs the network once (Ubuntu's packages and EEGLAB's
repositories, about 1.3 GB); running it never does. `--tag` names another
image, `--engine` picks docker or podman; `MEEGXER_OCTAVE_IMAGE` makes scripts run
in another tag.

What it holds:

| Part | Version | Licence |
|---|---|---|
| GNU Octave, octave-signal | Ubuntu 24.04's (Octave 8.4) | GPL-3.0-or-later |
| EEGLAB core (`eeglab.m`, `functions/`) | 2026.0.0, commit `bcc710a8` | BSD-2-Clause |
| firfilt (`pop_eegfiltnew` and the FIR filters) | as EEGLAB 2026.0.0 pins it, `ff8227f8` | GPL-2.0-or-later |
| ICLabel | as EEGLAB 2026.0.0 pins it, `644578bd` | GPL-2.0-or-later |

Left out of the base: EEGLAB's sample data and its other plugins (a script
declares the ones it needs, below). clean_rawdata's ASR files are under the UC
Regents licence (non-commercial); dipfit bundles FieldTrip.

The image is labelled with a SHA-256 of its recipe and of MeegXer's Octave
files (the plugin SDK and the runner, `meegxer/plugins/octave/`), so an image
built from other code (another MeegXer version, an edited checkout) is refused
with what to do: build it again.

## EEGLAB plugins

A script that needs an EEGLAB plugin beyond firfilt and ICLabel says so in
its help block (decided 2026-11-25):

```matlab
function EEG = lab_clean (EEG, varargin)
% Remove drifts and bursts with clean_rawdata (ASR).
%
% Requires: clean_rawdata
  EEG = clean_drifts (EEG, [0.25 0.75]);
  EEG = clean_asr (EEG, 20);
end
```

- **A name MeegXer knows**, case-insensitive, several per line
  (`% Requires: clean_rawdata, dipfit`), or **any git repository at a full
  commit**: `% Requires: zapline_plus = https://github.com/... @ 40-HEX-COMMIT`
  (an `https://` URL of letters, digits and `._~%+-`; the name is its
  folder under EEGLAB's `plugins/`).
- **Fetched when the image is built, never while a script runs.**
  `meegxer-plugin octave build` fetches every plugin the added scripts
  declare (and any `--with NAME`), each at its pinned commit, and lists them
  in the image's label; `octave check` says which it holds. Adding a script
  whose plugins the image lacks records it, not checked yet, and says to run
  `octave build`, then `meegxer-plugin check --script`; running it is
  refused, saying to build again, until the image has them. The base image
  and its code hash do not change.
- **Each script gets only its own plugins** on its path, so two scripts'
  plugins never clash. This keeps them apart, it is not a boundary: a
  script can `addpath` any plugin in the image, and every plugin runs in the
  same sandbox (no network, read-only). One image holds one copy of each plugin: two
  added scripts asking for one at different commits make the build refuse,
  naming both.
- Sessions record each plugin's commit with the script.

The plugins MeegXer knows, at the commits EEGLAB 2026.0.0 pins:

| Name | Commit | Licence |
|---|---|---|
| `clean_rawdata` | `d4b143f2` | GPL-3.0; `asr_calibrate.m`, `asr_process.m`: UC Regents, non-commercial use only |
| `dipfit` | `b0b660e7` | GPL-2.0-or-later (bundles FieldTrip) |
| `EEG-BIDS` | `8486b1dd` | GPL-2.0-or-later |

A plugin's licence is yours to accept when you declare it: clean_rawdata's ASR
files allow non-commercial use only.

The image also holds `yulewalk.m`, MeegXer's tested port (the one the EEGLAB
ASR plugin uses): `asr_calibrate` designs its filter with it at rates
EEGLAB has no precomputed filter for (anything but 100, 128, 200, 256, 300,
500 and 512 Hz), as Octave's signal package has no `yulewalk`.

## Mixing MNE and EEGLAB

One language per script, mixed by chaining strips (decided 2026-11-27): an
MNE strip (a Python custom script, or a built-in) and an Octave/EEGLAB
script are strips of one pipeline, in any order. MeegXer passes the data
between them (MNE sees an `mne.io.Raw` in volts, EEGLAB an `EEG` in µV;
events both ways). From Python, the whole chain is one script:

```python
from meegxer.api import run

PIPELINE = {
    "spec_version": "0.2.23",
    "nodes": [
        {"id": "input", "node": "meegxer.core.file"},
        {"id": "mne", "node": "script.my_mne"},  # my_mne.py, added
        {"id": "asr", "node": "script.lab_asr"},  # lab_asr.m, added
        {"id": "ref", "node": "meegxer.nodes.rereference"},
    ],
}
clean = run(PIPELINE, "sub-01.set")
```

`meegxer export` (or *Save → Export as Python script…*) writes such a script
for a pipeline built in the GUI (`docs/PYTHON_API.md`). Calling Octave from
inside a Python script (e.g. `oct2py`) is not supported: it would run your
own Octave and EEGLAB, outside the image and its sandbox.
