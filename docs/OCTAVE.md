# Octave and EEGLAB scripts

MeegXer runs Octave/EEGLAB code (M11.5) in one image, built on your computer:
GNU Octave with the signal package and a pinned EEGLAB. Every Octave script
runs in it, sandboxed like any plugin container (no network, read-only, no
capabilities), so a script gives the same result on every computer and a
session names the EEGLAB it ran with. You need Docker or Podman; you do not
need Octave or EEGLAB installed.

The scripts themselves (`function EEG = my_filter(EEG, 'low', 1, ...)`) come
in the next step; this page is the image they run in.

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
image, `--engine` picks docker or podman.

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
