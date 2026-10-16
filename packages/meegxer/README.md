# MeegXer

MeegXer (pronounced "Meek-Ser") is a user-friendly mixer for EEG processing:
Voicemeeter/VST-style live and offline pipelines with audited built-in nodes.

```
pip install meegxer
meegxer-gui
```

This installs everything: `meegxer-core` (engine, schedulers, I/O, control API),
`meegxer-nodes` (built-in filters, re-referencing, ICA/ICLabel, features),
`meegxer-plugins` (Python/Octave plugins as strips, e.g. EEGLAB ASR),
`meegxer-gui` and `meegxer-cli`. Processing runs on your machine. Plugins currently
run in Docker or Podman; a one-click Sterile mode that isolates all processing in
a container is planned.

Status: pre-release. Source, documentation and issues:
https://github.com/thawirasm-j/MeegXer

Licence: GPL-3.0-or-later.
