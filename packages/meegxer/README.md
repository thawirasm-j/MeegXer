# MeegXer

MeegXer (pronounced "Meek-Ser") is a user-friendly mixer for EEG processing:
Voicemeeter/VST-style live and offline pipelines with audited built-in nodes.

```
pip install meegxer
meegxer-gui
```

This installs the sterile build: `meegxer-core` (engine, schedulers, I/O, control
API), `meegxer-sterile` (filters, re-referencing, ICA/ICLabel, features),
`meegxer-gui` and `meegxer-cli`. Containerized research plugins (EEGLAB ASR, your
own Octave or Python plugins) need the separate `meegxer-runtime` package and
Docker or Podman.

Status: pre-release. Source, documentation and issues:
https://github.com/thawirasm-j/MeegXer

Licence: GPL-3.0-or-later.
