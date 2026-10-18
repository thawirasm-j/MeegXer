# Release notes

Every package (`meegxer`, `meegxer-core`, `meegxer-nodes`, `meegxer-plugins`,
`meegxer-gui`, `meegxer-cli`) is released together with one version
(`docs/RELEASING.md`). Install or upgrade: `uv tool install --python 3.14
--prerelease allow meegxer` (or `uv tool upgrade meegxer`).

## 0.3.0b2 (not released yet)

- BIDS derivatives inherit the raw dataset's credits: `Authors`, `License`,
  funding and acknowledgements, its name, links and DOI (as the source's).
  The validator's TOO_FEW_AUTHORS warning goes away when the raw dataset lists
  its authors.
- CI also type-checks as on Windows (`mypy --platform win32`).
- **The engine only talks to you.** The daemon writes an access token to a file
  in your config folder that only you can read; the GUI, CLI and scripts present
  it on the control and display channels, so other users and programs on the
  computer cannot drive or watch it (control API 0.2.4). Set `MEEGXER_TOKEN` to
  connect from elsewhere. The token is not encrypted: bind the engine to another
  network interface only on a trusted network. After upgrading, end any daemon
  an older version started: the new GUI and CLI cannot reach it.
- **Share-safe sessions:** `meegxer session share IN OUT` (or *Share session…*
  in the GUI) writes a copy that names no file or folder (`<input>.set`); models
  and hashes stay, and the name and description you wrote are shown first, with
  the offer to remove them. Rerun it with `-i RECORDING -o RESULT`. New guide:
  `docs/DEIDENTIFICATION.md`.
- **Supply chain:** every PyPI file carries a PEP 740 attestation (built by this
  repository's release workflow); each release has a CycloneDX SBOM; CI audits
  the locked dependencies for known vulnerabilities and runs CodeQL, also weekly.

## 0.3.0b1: first beta (2026-10-17)

The first beta: every feature planned for v1 (`docs/DESIGN.md` §5) is in. It
carries the code of 0.2.1a8; what changes is the promise. From here to 1.0.0 the
pipeline, session and plugin formats change only in compatible ways, and the
work is fixing what the beta turns up. The manual release checks passed
(`docs/TESTING.md`: acceptance test 5 on Windows with Docker, and the check on
real BIDS data, ds004504); the EEGLAB read-back is still to be done, and
anything it finds goes into 0.3.0b2.

### What MeegXer does at this release

- **One chain, live or offline.** The same strips run on an LSL stream, a
  recording played back, or offline over whole recordings and BIDS datasets
  (rule 4: one node implementation, two schedulers).
- **Inputs:** LSL (signal plus an optional marker stream), recordings (EEGLAB
  `.set`, EDF/BDF, BrainVision, FIF), BIDS, and a synthetic EEG generator with
  ground-truth artefact events. The Recording window fixes channel names,
  types, bad channels, positions and events without touching the file.
- **Built-in strips:** IIR and FIR filters (zero-phase offline), notch,
  re-referencing, bad-channel interpolation, CSD, EOG regression, ICA + ICLabel;
  features: PSD and band power (Welch or multitaper), specparam, Morlet wavelet
  power, Hilbert envelope, connectivity, complexity, SSVEP detection (CCA,
  filter-bank CCA) and CSP. Every strip lists the methods to cite (*Credits*).
- **Rails:** the chain view draws what travels beside the signal. The Events
  rail starts at the input (the recording's events or LSL markers) and feeds
  labelled calibration (CSP). ICA writes the ICs rail, which later strips can
  read. Each feature strip starts its own rail.
- **Views:** probe taps anywhere on the chain (overlay, difference, spectrum
  with a head plot); click a rail's name for its view: a feature strip's
  spectrum or values over time, or the ICA components coloured by ICLabel
  class ("IC03 · eye 0.97"). Live and offline; offline views scroll through
  the whole recording.
- **Your own code:** a custom MNE-Python script becomes a strip (offline
  `process`, live `transform`/`fit`, windowed `features`) after you accept the
  local-code notice. Plugin images run in containers (Docker or Podman).
- **Reproducible output:** a session file keeps the pipeline, node versions and
  fitted models (ICA), and a rerun gives a byte-identical file. BIDS
  derivatives come with provenance, one recording or a whole dataset from the
  CLI.
- **Settings:** preferences and a remembered view (`docs/SETTINGS.md`), never
  written into pipelines or sessions.

### Upgrading from 0.2.0 or earlier

- `meegxer-sterile` is now `meegxer-nodes` and `meegxer-runtime` is now
  `meegxer-plugins` (since 0.2.1a1). Old node names (`meegxer.sterile.*`) still
  load, so old pipelines and sessions open unchanged. A `uv tool install`
  reinstall drops the old packages; with pip,
  `pip uninstall meegxer-sterile meegxer-runtime`.
- "Sterile" now names a planned isolation mode, not a package: every install
  ships every package.
- Pipelines and sessions from 0.2.0 load as they are. A session's models are
  reused only while their fingerprints match the input; otherwise the strip
  asks for a refit (rule 7).

### Known limitations

- **Sterile mode is not there yet.** Built-in strips and custom scripts run on
  your computer with your permissions; only plugin images run in a container.
- **Feature values have no units yet.** The feature view shows plain values
  (e.g. band power as "value (×1e-12)" in V²).
- **ICA needs positions for every channel it processes.** A recording that
  stores EOG channels as EEG must mark them EOG (Recording window) or give them
  positions before ICA will start.
- The head plot cannot map components (they have no positions), so the ICs'
  spectrum view shows no head plot.
- A strip whose channel layout changes mid-stream stops the LSL feature outlet.
- Acceptance test 3 (latency) is measured and reported (`meegxer status`,
  `tools/stability_test.py`) but its target is still to be agreed.

## Alpha releases

### 0.2.1a8 (2026-10-17)

- **Feature views:** click a rail's name to show it in place of the signal:
  a feature strip's spectrum per channel or its values over time (*Show*
  picks further dimensions, e.g. the band), live and offline. The ICs rail
  shows the components coloured by ICLabel class. Display protocol 0.2 sends
  feature windows; the `offline_features` control call serves an offline run's.
- Chain view: the signal line is grey again, arrows hop over it, and rails are
  fainter past their last reader.
- Fixed: a small window no longer flashes up on Windows when a run starts or a
  parameter is edited.

### 0.2.1a7 (2026-10-17)

- **Rails:** one chain on the signal line, with parallel rails above it for
  events, ICs and features; feature and output strips hang below the signal.
- **Events rail in the core:** the input's events, and LSL markers placed on
  the signal's sample index at the inlet (`marker_stream`, `marker_type`,
  `marker_wait`); live labelled calibration waits for late markers.
- **Rail ports** (spec 0.2.17): extra inputs read rails, extra outputs write
  them, in both schedulers, the plugin protocol and the Python SDK.
- **ICs rail:** ICA writes the component activations with ICLabel's class and
  probability; a strip can take a rail output as its input (`"ica1:ics"`).

### 0.2.1a6 (2026-10-17)

- Preprocessing built-ins: FIR filters, bad-channel interpolation, CSD, EOG
  regression; the add-strip menu is grouped by category (spec 0.2.15).
- Feature built-ins: multitaper PSD and band power, specparam, Morlet wavelet
  power, SSVEP CCA and filter-bank CCA.
- Labelled calibration (spec 0.2.16) and a CSP strip.
- Live time axis setting: seconds before now, or since the start.
- Labels use proper terms and case throughout.

### 0.2.1a5 (2026-10-17)

- Spectrum head plot: band power across the scalp as a topomap, with four
  colour schemes and a held scale.
- Settings: preferences (`settings.json`), the remembered view (`state.json`)
  and a settings dialog.

### 0.2.1a4 (2026-10-17)

- Feature-extraction custom scripts: `features(data, ...)` per window (spec
  0.2.14).
- GUI: the inspector keeps its place, number fields step sensibly, the live
  window is fixed.

### 0.2.1a3 (2026-10-16)

- Live custom scripts: `transform(data, state, ...)` with an optional
  `fit(raw, ...)`.
- Every node lists its references (spec 0.2.12); custom strips are grouped by
  type.

### 0.2.1a2 (2026-10-16)

- Local plugins with consent, and custom scripts as offline strips
  (`process(raw, ...)`).

### 0.2.1a1 (2026-10-16)

- Packages renamed: `meegxer-sterile` → `meegxer-nodes`, `meegxer-runtime` →
  `meegxer-plugins`; "sterile" now names an isolation mode. No change in
  behaviour.
