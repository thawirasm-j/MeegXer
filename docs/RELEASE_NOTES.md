# Release notes

Every package (`meegxer`, `meegxer-core`, `meegxer-nodes`, `meegxer-plugins`,
`meegxer-gui`, `meegxer-cli`) is released together with one version
(`docs/RELEASING.md`). Install or upgrade: `uv tool install --python 3.14
--prerelease allow meegxer` (or `uv tool upgrade meegxer`).

## Next (not released yet)

- **Suggestions in the Review strip:** name a detector's labels in its
  *Suggestions to review* (`points`) and each one gets a verdict, yes, no or
  edited, kept with the strip that proposed it in the marks file (rejected
  ones too, so a detector's misses can be counted). Later strips see only the
  verdicts: a rejected `BAD_peak` is no longer cut. *Decision* *Verify* is a
  yes/no queue (keys Y, N, E) that must be finished before saving; *Free*
  opens each suggestion as a mark to keep, change or delete. A detector's
  proposed bad channels (`bad_channel`) are answered the same way. Control
  API 0.2.17.

- **Fixed: recovering a long recording.** Recovering an unfinished recording
  wrote the whole file inside one control call, and the GUI gives up on a
  reply after 3 s: a recording of an hour or more (40 s to write at 64
  channels) was reported as failed while it was still being written, and the
  daemon did not answer anything else meanwhile. The daemon now writes it in
  the background, and the GUI says when it is done (control API 0.2.16:
  `recover_recording` returns at once, `recoveries` reports).

- **Detectors:** three new strips under *Event detection* mark artifacts on
  the Events rail, each wrapping a published MNE method: *Detect amplitude &
  flat* (`BAD_peak`, `BAD_flat`, and channels proposed as bad), *Detect
  muscle* (`BAD_muscle`) and *Detect blinks* (`blink` spans around each
  blink's peak, from the EOG). A Reject segments strip after them cuts the
  `BAD_…` spans. Offline only.

- **Epoch review:** a new *Review epochs* strip after an Epochs strip pauses
  the offline run so a person can accept, reject or relabel each epoch (a
  page of epochs side by side, keys A/R/L and 1 to 9), with the ERP of the
  kept epochs of each label below. Later strips read the reviewed epochs; the
  verdicts are saved per recording (`epoch_status` in a BIDS events file) and
  asked for again when anything upstream changes. Control API 0.2.15.

- **Review at import:** *Review…* in the Recording window reviews the raw
  recording itself, before any pipeline: edit its events, mark spans and bad
  channels, and crop its start and end. The review is saved with the
  recording (`<file>_desc-import_events.tsv`, or in its BIDS derivatives),
  never in the raw file, and every pipeline reading the recording uses it
  (the input's *Use the review made at import* setting turns it off).

- **Review points:** the Review strip's new `points` setting names event
  labels to step through (e.g. a detector's candidates). The review window
  goes from point to point (◀/▶, ←/→), selects each to label and retime, keeps
  each answer linked to its point (`review_point` in the marks file), and can
  move on after each answer. A counter shows how many are answered.
- **The review display is never locked:** the Review strip's display
  settings are starting values, each *Reviewer's choice* by default (no more
  -1/0 codes), and the reviewer can change them.

- **Fixed: events reach the output files and the views.** A review's marks
  (and any strip's events) were put on the Events rail but never written:
  `.set` and EDF files only got cut boundaries, and the BIDS writer only the
  input file's own events. Now every file writer saves the events on the
  rail where it sits (the input's and its upstream strips'), on the
  processed recording's time (events inside a cut are left out). The offline
  views mark the run's events once it pauses for a review or ends
  (`offline_events`, control API 0.2.13).

- **Review window display** (Design v2, step 2 in `docs/REVIEW.md`):
  - A montage: as recorded, monopolar against the **average** reference (the
    default), longitudinal ("double banana") or transverse bipolar, or a
    Laplacian (Hjorth, by electrode position). Display filters: high-pass,
    low-pass and notch. The core computes each page through them from the
    raw samples (control API 0.2.12); nothing in the pipeline changes.
  - Clinical scale by default: sensitivity in µV/mm and speed in mm/s on the
    screen's real size (7 µV/mm, 30 mm/s), or channel spacing in µV and
    seconds per page. Defaults are on the new **Review** page in Settings.
  - **Display presets** in the Review strip (Advanced): montage, filters,
    sensitivity and page length that the researcher fixes for the reviewer.
  - **Bad channels:** marked in the window, drawn grey and dashed, left out
    of the average and the Laplacian, saved with the review, and the stream's
    bad channels after the strip (so interpolation and re-reference strips
    act on them).
  - Keys 1 to 9 mark the band with the first nine labels; **Undo** (Ctrl+Z).
- **Marking in the review window:** drag on the traces from the start to the
  end of a span (a click picks an instant); drag the tabs on its edges to
  adjust it, past the page if need be (a tab at the border shows where it
  ends). Click a mark to move its edges, relabel it or delete it.
- **Channel names are clickable pills:** a click marks a channel bad or good
  again; a right-click opens its menu.

- **Reject segments** strip (Artifacts, offline): cuts out spans whose label
  starts with one of its `reject` words (default `BAD`), with a `boundary`
  event at each cut and a fresh start for every strip after it. It reads the
  Events rail, so it cuts the recording's own `BAD…` events, a review's marks
  and a detector's alike.
  - **The Review strip now produces events only** (Design v2 in
    `docs/REVIEW.md`); its `reject` setting names the labels the window's
    *Reject* key gives. **A pipeline from 0.3.1b3 with a Review strip no longer
    cuts:** add a Reject segments strip after it to keep cutting.
  - A GUI offline run with a Review strip asks the reviewer's name once, when
    it starts. The name is saved with each review (sidecar JSON) and in the
    session (`nodes.<id>.review`, with when and on what input); a shared
    session leaves it out (control API 0.2.11).
  - Sharing a session now also clears a Review strip's `file` (it could name
    the subject).

- **Epochs** strip (Epochs, offline): cuts the signal around events onto an
  `epochs` side rail and passes the signal on unchanged.
  - Choose which events to cut around and the span (`tmin`/`tmax`, default
    -0.2 to 0.8 s).
  - The baseline before the event is subtracted by default.
  - Epochs overlapping a `BAD…` mark, crossing a cut, or running off either
    end of the recording are left out.
  - Each epoch is a Features window `(epoch, channel, time)` labelled with its
    event, so it can be tapped and read by later strips (`docs/RAILS.md`).
  - A Features rail output now gets a Features tap in the GUI.

- **Stacked review window:** the review window is now the stacked view of the
  recording, with the review tools below it:
  - A coverage bar shows the parts already looked at in green, rejected spans
    in red and annotations in purple; click it to go there. *Next unreviewed*
    (N) jumps to the first part not yet shown.
  - *Annotate* (A) and *Reject* (R) mark the span under the band. Marks are
    shaded on the traces with their label, and listed below (double-click to
    show one, Delete to remove it). ←/→ page through the recording.
  - Saving asks first if part of the recording was never shown. The spans
    looked at are kept with the marks and shown again next time (control API
    0.2.9).

## 0.3.1b3 (2026-10-20)

- **Rejected spans are cut out** (Review strip, `reject` parameter, default
  `BAD`). The samples after a cut keep their index and a `boundary` event marks
  the cut. Every strip after it restarts per piece, so filters don't leak
  across a cut, and fitted strips learn from the kept samples only. `.set`,
  EDF and BIDS outputs save the joined pieces with boundary events; BIDS
  `events.tsv` onsets follow the processed recording's time
  (`docs/REVIEW.md`).
- **Review strip** (Event detection): mark a recording by hand: bad segments
  (`BAD_muscle`, …), states, any label over a span. The marks go on the Events
  rail for the strips after it, and are saved next to each recording (in its
  BIDS derivatives for a BIDS input) as a BIDS events file, with what they
  were made on.
  - An offline run in the GUI pauses at the strip: the views show everything
    before it, and a review window marks spans dragged on the signal view. In
    an all-subjects run, it pauses at each recording not yet reviewed.
  - If something upstream changes later, the next run asks you to confirm the
    marks again.
  - The CLI stops an unreviewed recording at the strip, says so, and exits with
    code 2 (`docs/REVIEW.md`).
  - Control API 0.2.8: `submit_review`, `cancel_review`, offline `state`
    `review`.

- **Resample** strip (Filters): change the sampling rate by an exact ratio
  of the rates (e.g. 1000 → 250 Hz, 500 → 256 Hz) with a linear-phase
  polyphase FIR, the same samples as `scipy.signal.resample_poly`, live and
  offline alike, with no delay. Events keep their time: a strip after the
  Resample strip sees them counted at its rate, and the views line up taps at
  different rates. Ratios needing factors above 1000 are refused with a
  message. Known limitation: a Features tap after a rate change, or a Signal
  rail read across one, is not mapped yet (the second stops with a message).
- Offline taps after upsampling now hold the whole recording (they were sized
  by the input's sample count).

- **Pick channels** strip (Reference & channels): keep channels by type (EEG
  only by default; EOG, ECG, … on request) and/or by name, and optionally drop
  the ones marked bad. A listed name the stream lacks, or whose type is off,
  stops with a message instead of being dropped silently.

- **Browse buttons:** file and folder settings (the input file, a BIDS
  dataset, an output file or folder) have a *Browse…* button next to the text
  box that opens the right dialog: a file to open, a folder, or where to save
  (the extension is added when you leave it out). Plugins get it by marking a
  string parameter as a path (spec 0.2.20).
- **Every subject of a BIDS dataset from the GUI:** leave *Subject* empty on
  the BIDS input (it reads *All subjects*) and *Run* processes every matching
  recording one after another, as `meegxer run … --bids` does: each into the
  dataset's `derivatives/meegxer`, finished ones skipped, a failure not stopping
  the rest. The bar shows `3/9 sub-03_task-rest`, the views follow the recording
  being run, and at the end show the last subject; the log sums up what was
  done, skipped or failed. The Recording window shows the first subject.
- In a dataset run (GUI or `--bids`), a .set, EDF or feature-table writer with
  one fixed file name is left out with a note (each recording would overwrite
  the last); the BIDS writer saves every recording.

### Known limitations

- **The review window is a first version:** it marks spans on the main
  signal view. The stacked review window (annotate and reject tools,
  reviewed vs not-yet-reviewed colouring) and epoch review come next.
- A strip that reads rails cannot fit on a stream a review cut (it says so),
  and strips after a cut are not cached.
- A Features tap after a Resample strip counts its windows in the resampled
  samples; the views do not map it yet.
- The limitations of 0.3.1b2 below still hold.

## 0.3.1b2 (2026-10-19)

- **Project hygiene:** a security policy (`SECURITY.md`: report problems
  privately through GitHub; only the latest release gets fixes), a contributing
  guide (`CONTRIBUTING.md`), a code of conduct (Contributor Covenant 2.1), and
  issue and pull request templates.
- **A layout change no longer ends a live run at the LSL outputs.** When a
  strip's channels or feature values change mid-stream (a channel dropped, a
  band added), the LSL output and the LSL feature output close their stream and
  open it again with the new channel list, under the same name; `meegxer status`
  says so. Programs reading the stream see it end and must look it up again.
  Before, the whole run stopped.
- **Live recordings survive a crash.** The .set and EDF writers no longer keep
  the recording in memory until the end: each block goes to a spool next to the
  file (`rec.set.part`, with `rec.set.part.json`) and is flushed at once. The
  file is written from it when the run ends, the same file as before. If
  MeegXer crashes first, `meegxer recover rec.set` writes the file from the
  spool, losing at most the block being written (after a power cut, the last
  seconds the system had not yet saved); the next run of the same pipeline also
  does it on its own, to `rec_recovered.set`, before it starts recording. When
  the GUI starts, it offers to recover any recording a crash left unfinished,
  even from a pipeline you never saved: *Recover* (a file already there is
  kept; the recording goes next to it as `…_recovered`), *Delete it* or *Ask me
  later* (control API 0.2.6). Feature tables (TSV) are written row by row, so
  they are readable as they grow. Memory use no longer grows with the length of
  a recording.
- A file writer whose input changes channels (or feature values) mid-run keeps
  what it received before the change and says so (`meegxer status`, or the
  end of `meegxer run`), instead of ending the run or writing nothing.
- Checked on real data (OpenNeuro ds004504, `docs/TESTING.md` B4 and B6): the
  cleaned derivative loads in EEGLAB and MNE-BIDS with every channel, drift and
  mains line removed; ICA removed the three eye components and with them the
  blinks.

### Known limitations

- **Sterile mode is not there yet.** Built-in strips and custom scripts run on
  your computer with your permissions; only plugin images run in a container.
- **ICA needs positions for every channel it processes.** A recording that
  stores EOG channels as EEG must mark them EOG (Recording window) or give them
  positions before ICA will start.
- A file writer stops at a change of channels mid-run (it keeps what came
  before); it does not start a second file.
- EEGLAB notes "upper time limit (xmax) adjusted" when it opens a .set that
  MeegXer wrote: eeglabio stores the end time one sample late. EEGLAB corrects
  it as it loads; the data are not affected.
- Acceptance test 3 (latency) is measured and reported (`meegxer status`,
  `tools/stability_test.py`) but its target is still to be agreed.

## 0.3.1b1 (2026-10-18)

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
- **Opening a pipeline or session runs no code:** a test now proves that a
  pipeline or session naming a custom script or local plugin you have not added
  (with the notice accepted) starts nothing, from the CLI, the daemon or a session.
- **The 1/f fit skips filtered frequencies.** Filter strips now record what
  they did on the stream (high-pass, low-pass, band-stop, notches; spec 0.2.19),
  and *Remove 1/f* in the spectrum view and head plot leaves those bands out of
  the fit: below 1.5x a high-pass cutoff, above 0.9x a low-pass cutoff and within
  2 Hz of a notch. The specparam strip can do the same: *Fit unfiltered
  frequencies only* (off by default) trims its range and bridges notches.
  The offline cache recomputes outputs it kept from earlier versions once (they
  do not record their filters).
- **Connectivity map:** a connectivity strip's feature view adds a head map
  with a line from the chosen channel (in *Show*, or click an electrode) to
  every other, coloured and thickened by its value.
- **IC head plot:** the ICs' spectrum view now has a head plot: the band power
  of the components selected in the channel menu, mixed back on the electrodes
  (one component: its own scalp power; all of them: the sensors'). Control API
  0.2.5 adds `get_model`.
- **Feature view hover:** near a line, the feature view draws it bold, marks
  its nearest point and names the line, where the point is and its value with
  the unit.
- **The head plot maps every EEG channel,** whichever curves the channel menu
  shows (hiding curves no longer leaves holes in the map); EOG, EMG and other
  non-EEG channels stay off it.
- **An existing output no longer stops the run.** A file writer whose file
  already exists (without `overwrite`) keeps it and skips its save, with a
  note (`skipped: …` in the CLI, *Not saved* in the GUI log); every other node
  and writer still runs to the end.
- **Feature units:** feature strips state the unit of their values (PSD
  `V^2/Hz`, band power `V^2`, relative power and connectivity dimensionless,
  specparam per measure). The feature view labels its value axis with it, *Show*
  names the unit of each measure, and feature tables write it into their JSON
  sidecar (`Units`), in BIDS derivatives too. Custom scripts and plugins may
  state theirs (`UNITS` in a script; spec 0.2.19, display protocol 0.3).
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
anything it finds goes into 0.3.1b1.

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
