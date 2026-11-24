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
- **Offline only for now**: the script sees the whole recording at once.
  Live Octave scripts come next.
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

Left out: EEGLAB's sample data and its other plugins. clean_rawdata's ASR
files are under the UC Regents licence (non-commercial); the separate EEGLAB
ASR plugin image (`packages/meegxer-plugins/plugins/eeglab_asr`) fetches it for
those who accept that. dipfit bundles FieldTrip.

The image is labelled with a SHA-256 of its recipe and of MeegXer's Octave
files (the plugin SDK and the runner, `meegxer/plugins/octave/`), so an image
built from other code (another MeegXer version, an edited checkout) is refused
with what to do: build it again.
