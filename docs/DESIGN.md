# MeegXer — Design Document (v0.1 draft)

*MeegXer ("Meek-Ser"): a user-friendly mixer for EEG processing.*

## 1. Vision

A mixing desk for EEG pipelines: route inputs through chains of processing nodes,
turn knobs, and see the effect live — on a streaming amp or on a recorded dataset.

- **Plugins as executable methods sections.** A research pipeline becomes a runnable,
  versioned object with declared parameters, instead of prose plus an unrunnable repo.
- **Code to data.** A peer's plugin runs on *your* local dataset; data never leaves
  your machine.
- **Knobs make analytic flexibility visible.** Turning a parameter and watching the
  result is a hands-on multiverse/robustness analysis.

### Honest scope of the claims
- MeegXer provides **computational reproducibility** (same code + data → same result).
  It does not by itself ensure **replication** (a finding holding in new populations);
  it makes replication failures visible.
- Research plugins are **not clinically validated tools**. Clinical use of a pipeline
  requires regulatory validation (SaMD). Each plugin should carry a provenance card
  (intended population, training/validation data, performance, failure modes,
  research-only vs validated status) — planned, not in v1.

## 2. Modes and builds

- **Online** = live processing on the local machine (LSL in → pipeline → outputs).
  Nothing is streamed over the internet.
- **Offline** = the same pipeline run on recorded files / BIDS datasets.
- **One install:** every package ships in every install (core, built-in nodes,
  plugins, GUI, CLI). Processing runs on the local machine.
- **Sterile mode** (planned, 0.2.1 decision): a runtime switch, not a package or a
  build. All processing (built-in nodes and plugins) runs inside a container with no
  network and read-only mounts; the host keeps only trusted I/O (sources and LSL,
  file writers, device sinks, the GUI). Turned on with a "Sterilize" button, later
  optionally at startup; needs Docker or Podman. See §3.5.
- **Sterile means isolation only.** It keeps patient data away from the network and
  the rest of the machine; it does not certify which code runs. Auditing the
  pipeline (and compliance: HIPAA, GDPR, IRB, clinical validation) stays the user's
  responsibility. Until 0.2.1 "sterile" named a package set (audited built-ins, no
  plugin runtime); that meaning is retired.
- Target: research environments first, not hospitals.

## 3. Architecture

```
┌────────────────── meegxer-core (daemon process) ──────────────────┐
│  Sources ─▶ Graph scheduler ─▶ Sinks (LSL out, files, actions)    │
│      │        │  fit/transform, probes, plugin workers (Docker)   │
│      │        └─▶ Probe taps ─▶ Display decimator                 │
│      ▼                                   │                        │
│  Control API (JSON-RPC)          Display stream (binary, lossy)   │
└──────────┬───────────────────────────────┬────────────────────────┘
           │ commands / status             │ decimated frames ~30–60 fps
     ┌─────▼──────┐   ┌──────────┐   ┌─────▼──────┐
     │ meegxer-gui│   │ meegxer- │   │ remote GUI │
     │   (Qt)     │   │   cli    │   │            │
     └────────────┘   └──────────┘   └────────────┘
```

### 3.1 Core daemon
- Owns: input pulling, block scheduling, graph execution, sample index + timestamp
  propagation, calibration state, plugin worker lifecycle, all outputs.
- Never blocks on the GUI. Display data is decimated to screen-sized min/max envelopes
  before sending; frames drop if the client lags.
- Separate process from the GUI for GIL isolation, crash isolation, headless use
  (CLI, batch servers, Raspberry Pi), and a possible future Rust rewrite of the core.

### 3.2 Graph model
- Internally a DAG; v1 UI exposes a linear drag-and-drop chain with insert-anywhere probes.
- Later: ComfyUI-style node editor on the same model (candidate: NodeGraphQt).
- **Port types:** Signal, Features, Events.
- **Node roles:** Source, Processor, Probe, Sink.
- **Schedulers:** real-time (block-wise, deadlines, overrun handling) and batch
  (whole recordings, parallel subjects). Same node implementations for both.
- **Caching (batch):** per-node cache keyed by hash(input, node version, params), so a
  change at stage N only recomputes stage N onward.
- **Batch scheduling is node-major:** each node consumes its whole input (fed block by
  block through the same `transform`) before the next node runs, so fitting sees the
  full input and every node output is cacheable. The real-time scheduler is
  block-major over the same nodes.
- **Pipeline file:** `spec/pipeline.schema.json`, an ordered node list where each
  node's input names an earlier node. Nodes are discovered through the
  `meegxer.nodes` entry-point group, so the core never imports node or plugin code.
  Built-ins were named `meegxer.sterile.*` before 0.2.1; those names still resolve.
- **Units:** signals are in volts (SI, as in MNE) throughout the core.
- **Real-time scheduling is block-major:** each loop pulls what the source has ready
  and pushes it through the graph; parameter changes and calibration requests are
  queued and applied by the scheduler thread between blocks. Recorded and synthetic
  sources are paced to their sampling rate (about 10 ms chunks) and re-stamped on the
  LSL clock. A block overruns when its processing takes longer than its duration;
  latency is the sink-side LSL clock minus the timestamp of the block's last sample.
- **Online calibration:** a stage that needs fitting passes nothing downstream until
  `calibrate(node, seconds)` has buffered its input and fitted it. Per node, a user can
  instead opt in to passing data through unprocessed while there is no valid model;
  such data carries a `bypassed` mark (blocks and display frames) downstream.
- **Fitted models are kept** across pipeline restarts and dropped only explicitly
  (`clear_calibration`). Kept models are still checked: an upstream or source change
  makes them needs-refit, and a channel or rate mismatch is refused.
- **Specs:** control API in `spec/control-api.md`, display stream in
  `spec/display-protocol.md`.

### 3.3 Processor contract
```
manifest:  spec_version, name, version, type (stream | analysis), image (external only),
           inputs/outputs [{name, type: signal | features | events}],
           params [{name, type, min, max, default, unit, choices}],
           latency_samples, requires_fit (bool), fit_events (bool, 0.2.16),
           authors, references [{citation, doi | url}], license (credit, 0.2.12)
fit(calibration_data | dataset, params)       -> model
  (fit_events: the calibration's events too)
transform(block, model, state, params)        -> block_out, state, events
```
- Every node (source, processor, probe, sink) has a manifest with a `role`;
  `live_params` lists parameters that may change mid-stream without resetting state;
  on a node that needs fitting they act in `transform` only, so changing one never
  asks for a refit (e.g. the ICA rejection threshold).
- **Availability:** each node declares `online`, `offline` or `both`. The batch
  scheduler refuses online-only nodes and hands offline-only nodes the whole recording
  as one block; the real-time scheduler refuses offline-only nodes. Offline filtering
  is zero-phase (forward-backward IIR nodes); online filtering is causal.
- **Friendly and advanced modes:** each parameter may be marked `advanced`; friendly
  mode hides those and uses their defaults.
- **Bad channels:** `SignalInfo.bads` marks bad channels (read from files where the
  format stores them). They stay in the data; pooling nodes such as the average
  reference leave them out by default (`exclude_bads`).
- **Channel types:** `SignalInfo.ch_types` (eeg, eog, emg, ecg, resp, stim, misc;
  unknown counts as EEG). A stage hands its processor only the channel types in its
  `channel_types` (pipeline field, default EEG only) and puts the result back in
  place; the other channels pass through untouched. So no node, including the
  average reference, touches EOG/ECG/EMG unless its strip includes them, and the
  processor code needs no type handling of its own.
- **Delayed output (spec 0.2.5):** a processor with look-ahead (ASR) may return
  output that lags its input: the output block carries the sample index of its first
  sample, which may be earlier than the input's, and may be shorter or empty (the
  schedulers do not hand an empty result on, so no node ever receives an empty
  block, spec 0.2.8). It returns what it still holds from `flush(info, model, state, params)` at the end of
  the stream (the batch scheduler and the conformance helper call it). The stage
  delays the channels it does not process by the same amount, so every channel and
  probe stays aligned by sample index (rule 8); online, the delay is plain latency.
- **Electrode positions (spec 0.2.11):** a node that needs them declares
  `requires_positions` in its manifest (ICA, for ICLabel). Both schedulers check
  the input before running: every good channel such a node processes needs a
  position, or the run refuses to start and names the node and channels (live:
  once the stream is open). The GUI checks on Start and opens the Recording
  window on its Channels tab instead of starting.
- Schema: `spec/manifest.schema.json`. `state` is `None` on the first block of a
  stream and after a parameter change or refit; the processor initialises it then.
  Model and state must be serializable (arrays, scalars, strings, lists, dicts).
- Staleness is derived from fingerprints (node identity, params, upstream
  fingerprint and model content), so a model fitted under different upstream
  conditions reports needs-refit and refuses to run.
- **Stream processors:** block in → block out at the same rate (filters with carried
  state, re-ref, ASR apply, ICA apply).
- **Analysis processors:** windowed input → Features/Events at a lower rate (PSD,
  band power, connectivity, CCA, classifiers). They buffer input in `state` and
  return every window a block completes as a tuple of FeatureBlocks (none when no
  window is complete), each with its sample range and the time of its first sample,
  so windows never depend on block sizes (acceptance test 1 compares the windows).
  In a linear chain a Features node branches off: a node without an explicit
  `input` reads from the nearest earlier node whose output type matches (spec
  0.2.7), so Signal strips after it continue from the last Signal node.
- **Calibration (online):** record N seconds, fit, then go live. Per-stage status:
  uncalibrated / calibrating / ready / needs-refit.
- Some processors are dataset-level only (ComBat, group statistics) and never stream.
- Live parameter changes without restarting workers.

### 3.4 Probes
- A probe is a non-modifying tap node. Example chain:
  `Input > Highpass > Probe R1 > ASR > Probe R2 > ICA > ICLabel > ICA rejection > Probe Main`
- Viewer shows Main as solid and selected comparison probes as faded background.
- **Display modes:** overlay; difference (e.g., Main − R1 = what was removed).
- Aligned by **sample index**, channels matched by **label**. The index counts the
  source's samples from its first (rule 8); a rate-changing node maps it by the
  exact ratio of the rates, so index / rate is the same time on either side and
  probes at different rates still line up. Markers from a separate stream (the
  LSL input's `marker_stream`) are placed on the signal's sample index once, at
  the inlet, from their LSL timestamp; downstream, events align by index only.
- **Time axis:** offline, time since the recording's start; live, seconds before
  the newest data (fixed ticks, 0 at the right edge; the default) or time since
  the stream started (setting `visualization.time_axis`).
- Feature-domain probes: overlay PSDs across probes.

### 3.5 Plugins and Sterile mode
- Plugins are untrusted code (rule 9). Today they run in Docker/Podman:
  `docker run -i --network none --read-only --cap-drop ALL`
  (no network, devices or writable filesystem except a small `/tmp`); data arrives on
  stdin only. Protocol: `spec/plugin-protocol.md`.
- Transport: framed binary over stdin/stdout (JSON header + typed arrays; signal as
  float32); ZeroMQ is the upgrade path. No shared memory across the container boundary.
- Persistent worker per plugin node (Octave + EEGLAB loaded once). The host owns model
  and state: every `transform` carries the state and gets the new state back, models
  are loaded into the worker once by content hash. A crashed or hung worker is killed
  and relaunched with bounded backoff and continues exactly where it stopped.
- Plugins appear as ordinary Processor nodes (an adapter class built from the
  handshake manifest), discovered through `meegxer-plugin add IMAGE` and a node-provider
  entry point that exists only when `meegxer-plugins` is installed.
- Unsandboxed local processes are allowed only for plugin development and tests
  (`MEEGXER_ALLOW_UNSANDBOXED=1`) without recorded consent. Users add local plugins
  and custom scripts after accepting a notice (`meegxer.core.notices`, one wording
  for CLI and GUI): it runs with the user's permissions and can reach the network
  and files, which can break institutional data rules for identifiable patient data
  (HIPAA, GDPR, IRB). The plugin index records the acceptance; session provenance
  records the command and a SHA-256 of the code files, and a changed script refuses
  the session like another image would. Each runs in its own process over the
  same protocol.
- **Custom scripts** (`docs/CUSTOM_SCRIPTS.md`): an MNE-Python file with
  `process(raw, ...)`; `meegxer.plugins.script` builds the manifest from the
  signature (types, defaults, `Literal` choices) and the numpy docstring
  (descriptions, `(unit)`), and runs it through the bundled copy of the Python SDK.
  Offline tier: `process(raw, ...)` sees the whole recording as one block; the
  result must keep channels, rate and length; added annotations become events.
  Live tier: `transform(data, state, ...)` block by block (availability both), with
  optional `info`/`model` and an optional `fit(raw, ...)` (requires_fit); state is
  host-owned like any plugin's. Feature tier (spec 0.2.14): `features(data, ...)`
  per analysis window returns name → per-channel values; the runner buffers the
  stream in host-owned state (as `WindowedFeatures` does), adds `window`/`step`
  settings and emits Features (`channel` × `feature`) through the protocol's
  `features` result. Adding a script runs the conformance suite (for a
  live script including acceptance test 1 and a restart mid-stream) and refuses it
  on failure. Next: the Octave/EEGLAB tier (`function EEG = process(EEG, ...)`,
  then a live one), committed but after the MNE side.
- **Credit and grouping (0.2.12):** every manifest may carry `authors`,
  `references` and `license`; built-ins cite their methods and software (checked
  DOIs, `meegxer.core.references`, `meegxer.nodes.references`), scripts use
  `__author__`, `__license__` and `REFERENCES`. The inspector shows them (Credits),
  sessions record each node's references, BIDS derivatives gather them
  (`ReferencesAndLinks`). Informative only: `meegxer-plugin check` mentions missing
  credit but never fails on it. `list_nodes` reports each node's `origin`
  (builtin, plugin, custom).
- **Menu categories and processed channel types (0.2.15):** a manifest may name a
  `category` (filters, reference, artifacts, rate, epochs; spectral,
  time-frequency, connectivity, complexity, decoding, events; files, streams,
  actions; or its own). The add-strip menu lists each group (by data type) by
  category, built-ins first, then what the user added, marked ✎ (custom script or
  local plugin: unsandboxed) or ⬡ (container plugin); scripts set `CATEGORY`. A
  manifest may also declare `channel_types`, the types it processes unless the
  pipeline names others (default EEG; EOG regression takes EEG and EOG).
- **More preprocessing built-ins:** FIR filter (MNE design; minimum-phase live,
  zero-phase offline with phase auto), bad-channel interpolation and surface
  Laplacian (CSD) as channel-mixing matrices computed once from the positions
  (spherical splines, sphere fitted to the electrodes), EOG regression (fit on
  calibration, linear subtraction). All are exact block by block.
- **More feature built-ins:** PSD, band power and spectral parameterization take a
  spectrum `method`: Welch (pwelch-compatible, the default) or multitaper (MNE,
  DPSS). Spectral parameterization (specparam 2.0) reports the aperiodic offset,
  exponent (and knee) and the strongest peak per named band (NaN when a band has
  none; feature tables write `n/a`). Morlet wavelet power averages |x * w|^2 over
  the samples where the whole wavelet fits in the window (no zero padding). SSVEP
  detection scores each stimulus frequency by CCA or filter-bank CCA with sine and
  cosine references, no calibration; bad channels are left out. The conformance
  check counts NaNs in the same place as equal.
- **Labelled calibration (0.2.16):** a node with `fit_events` learns from labelled
  trials. Its fit gets the events inside the calibration stretch at absolute
  sample indices (built-ins: `fit_labelled(calibration, events, params)`; plugins:
  `events` in the `fit` message; scripts with `FIT_EVENTS`: `raw.annotations`).
  Offline the events are the recording's (its annotations or the events file,
  renamed and dropped as the recording overrides say); they are part of the node's
  input fingerprint, so changed events ask for a refit. With no events the fit is
  refused with the reason. Live, the events come from the **Events rail**
  (`docs/RAILS.md`): the LSL input's marker stream (`marker_stream`), or a
  recording played live. `calibrate NODE --seconds N` collects N s, waits
  `marker_wait` (default 0.5 s, Advanced) for markers still on their way while
  processing goes on, then fits on the events inside the stretch; a failed fit
  (too few trials) is reported and the node stays uncalibrated. Status counts
  `input_events` and `events_late` (markers that landed in a stretch already
  fitted on: missed trials). A live fit's model fingerprint names the input, not
  the events (later markers are new trials, not a changed calibration). A fit
  sees the input's events and those of the strips upstream of it.
  First user: **CSP** (MNE), the log power of each component per window, from
  `tmin`-`tmax` trials of the named classes.
- Plugins emit Events only; trusted host Sinks perform actions.
- **Sterile mode** (planned): one container runs the whole processing chain (built-in
  nodes and plugins) as a single worker over the plugin protocol, so the added
  delay is one hop per block; the host keeps sources, LSL, file writers, device
  sinks and the GUI. Measure the delay with the stability test before shipping.
- Base images: Octave with pinned EEGLAB; slim Python with pinned MNE. Multi-arch
  (amd64 + arm64) eventually.
- Host and LSL run natively; only plugin workers are containerized (LSL multicast
  discovery does not cross Docker Desktop's VM boundary).

### 3.6 Sinks
- LSL outlets (signal; features as an irregular-rate stream, one sample per window,
  channels labelled like the feature TSV columns), file writers (.set, EDF,
  BIDS-derivatives, feature TSV), session JSON.
- Action sinks (later): keyboard/mouse (pynput), virtual gamepad (vgamepad/uinput),
  serial (pyserial), OSC/MIDI. Trusted host-side only; user maps Events → actions.
  Safety: arm/disarm toggle, confidence thresholds, dwell time.

### 3.7 GUI (v1)
The GUI is a client of the daemon (control API + display stream); closing or crashing
it never interrupts processing.

- **Layout:** top bar (transport, friendly/advanced toggle, CPU/latency/overrun
  meters, theme); signal view; a horizontal chain of strips; inspector and log below.
- **Strips:** one strip per graph node, left to right (the later node editor reuses
  strips as its nodes). A strip shows its name, a status LED (uncalibrated,
  calibrating, ready, needs-refit, bypassed) and parameters:
  - friendly mode: the node's crucial parameters (manifest `primary`; if none, the
    first non-advanced parameter);
  - advanced mode: up to 3 parameters the user picks per strip, saved in the
    pipeline file. The inspector always shows every parameter.
  - **+ Add strip** at the end of the chain and in each gap; strips drag to before or
    after any other strip. Probe taps sit between strips.
  - **Groups:** each strip's top band, shape and label give its group, from its role
    and ports: ■ input, ● preprocessing (Signal to Signal), ◆ feature extraction
    (Signal to Features), ▶ output (sinks). Theme tokens `groups`.
  - **One chain, rails above it** (`docs/RAILS.md`): every strip sits on the
    signal, a thick line; a feature strip passes the signal on unchanged. Feature
    and output strips hang below the signal, with a down arrow for each line they
    pull (the signal, a rail). Thinner parallel rails above the strips show what
    else travels with it: each feature strip pushes up into a rail named after it
    (PSD, Band power·2; a filled dot where it starts, a hollow one where a strip
    adds to an existing rail), and the input's events start the Events rail. Each
    rail has its own row, solid to its last reader, then dashed and fainter to the
    end of the chain with an arrow: it is still there for later strips. A strip that needs
    a rail the chain lacks (CSP without events) shows "needs Events". By default a
    strip reads the nearest earlier node of its input type (spec 0.2.7); an output
    can pick another feature rail in the inspector (*Reads*), saved as the node's
    `input`.
- **Mode comes from the Input strip:** an LSL source is always live; a file or BIDS
  source is offline by default and a synthetic source live, and either can be
  switched on the Input strip (live playback or offline processing). The transport
  follows (Start/Stop live; Run offline, with files written by output strips).
  Compatibility is strict: a strip whose availability does not fit the mode turns
  reddish ("Offline only" / "Live only") and the pipeline will not start. The one
  adaptable node is the band-pass filter (which also covers high- and low-pass): its
  `phase` parameter is `auto` (causal live, zero-phase offline), `causal` or `zero`;
  `auto` is resolved to the existing causal or zero-phase node when the pipeline is
  loaded, so each variant keeps one implementation (rule 4). The notch stays strict.
- **Signal view:** a single trace area with a channel selector (all channels by
  default, i.e. butterfly; one or several selectable). The Main probe is drawn thick,
  comparison probes thin and faded; overlay, difference and spectrum modes; bypassed
  data hatched. A button opens stacked traces (EEGLAB-scroll style) in a separate
  window that follows the same probes.
- **Settings:** preferences in `settings.json` and the remembered view in
  `state.json`, in the user's config folder (`meegxer.core.settings`, pure Python so
  the daemon and CLI read them too; `docs/SETTINGS.md`). A dialog (⚙, Ctrl+,) with
  General, Appearance (theme incl. "system" and theme files, text size),
  Visualization (colour maps, fixed in every theme unless flipped on light themes;
  view defaults), Data & storage (offline folder and cache, dialog start folders)
  and Plugins & privacy (the plugin list through `meegxer-plugin list --json`, rule 9).
  Changes preview live; OK keeps, Cancel reverts. A view's last state wins over its
  default until that default changes or the remembered view is forgotten. Settings
  never enter pipelines, sessions or provenance.
- **Spectrum head plot:** spectrum mode puts a square head plot left of the
  spectra (`gui/headplot.py`), for the Main tap, drawn as a topomap: filled contour
  bands with lines. The band is a shaded region on the spectrum (drag it or its
  edges, or pick δ/θ/α/β/γ); its power is the integral of the PSD (µV², absolute or
  relative to 1–45 Hz). The map is a smooth (Clough–Tocher) interpolation over the
  electrodes, extrapolated to the head's edge through a ring of virtual points
  (each the mean of its nearest electrodes, like MNE's `extrapolate="head"`) and
  clipped to the electrodes' range. Maps: *Band power* (in 2 dB bands over
  10, 20 or 30 dB below the top; live, the top rises at once with the head's
  maximum and decays with the smoothing time constant, so the map fades instead of
  flickering), *Power change* (Main vs. the first comparison tap in dB, diverging, the
  0 dB line bold) and *Peak frequency* (where the band peaks; electrodes whose
  maximum is on a band edge are left out). **Remove 1/f** divides each channel's
  spectrum by its aperiodic fit over 1–45 Hz (specparam's robust fixed-mode fit,
  vectorized: a least-squares line in log-log, then one through the points below
  it): the spectrum shows dB above the fit, *Band power* the band's mean dB above
  it from 0 dB (on the 1/f line) up to the held top, *Power change* compares those
  values and *Peak frequency* finds the peak above the slope. Electrodes are drawn in their channel
  colours inside a black-and-white ring, apart from the map's colours; hover names
  one and thickens its spectrum line, a click shows or hides the channel. (A first
  "spatial" scheme, the scalp colour wheel with power as opacity, was dropped:
  opacity shows magnitude poorly and the wheel's own lightness biased it.) Live,
  spectra (lines and head) are smoothed between refreshes by an exponential moving
  average with a chosen time constant (off, 1–10 s); offline views are not
  smoothed, and an unchanged offline view is not mapped again. These are view
  settings, kept out of the pipeline and its sessions.
- **Channel colours (colour wheel on the scalp):** electrodes are projected
  azimuthally from Cz. Hue follows the angle around a perceptually uniform OKLCH
  wheel (front magenta-red, right orange, back green, left blue; left and right stay
  distinct for red-green colour-blind viewers) at constant lightness; chroma follows
  distance from Cz, so the vertex is neutral: white in the dark theme, near-black in
  the light theme, full colour at the circumference. Bipolar channels (`A-B` labels) take the mix of both electrodes'
  colours; non-EEG or unknown channels are neutral grey, dashed. Positions come from
  the recording when it has them, otherwise from a standard 10-05 table shipped with
  the GUI (no MNE import in the GUI).
- **Themes:** colour tokens (surfaces, text, accent, status colours, and the channel
  wheel's lightness, chroma and vertex lightness) drive both the Qt stylesheet and
  the plots. Dark (default) and light ship first. A theme is a small file, so more
  can be added.
- **Recording window** (M5.1): opened from the Input strip. Tabs for channels
  (label, type, bad, position, and an EEGLAB-style lookup that fills positions by
  channel name from a standard montage or a locations file, for recordings that
  store names only), montage (a standard montage or a file, with a head
  map that previews the channel colours), events (an events file instead of the
  recording's own, renames, drops) and info (sampling rate, length: read from the
  recording, not editable). The edits are the source's `recording` overrides in the
  pipeline file (spec 0.2.3); they describe the channels and never change the
  samples. Channel colours then use the recording's own positions (projected from a
  sphere fitted to the electrodes), falling back to the standard table.
- **Offline view** (M5.1): probes write their samples and min/max zoom levels to
  memory-mapped files while the run goes, so a probe scrolls as soon as it has data;
  the operating system keeps those files in RAM when memory allows and on disk when
  not. The batch scheduler runs node by node, so taps fill in chain order. Events
  are marked on the offline time axis.

## 4. Packages and languages

| Package | Language now | Later |
|---|---|---|
| meegxer-core | Python (numpy, pylsl, pyzmq) | Rust behind the same protocol, if measured need |
| meegxer-nodes | Python (MNE, SciPy, python-picard, mne-icalabel, mne-connectivity, antropy) | stays Python |
| meegxer-plugins | Python (docker SDK; Podman-compatible) | follows core |
| Plugin SDKs | Python package + Octave `.m` template | MATLAB-compatible variant |
| meegxer-gui | PySide6 + pyqtgraph | stays Python |
| meegxer-cli | Typer/Click | stays Python |
| Spec/protocol | JSON Schema + protocol doc + JSON-RPC | language-neutral always |
| Tests | pytest; Octave plugins tested via protocol | — |

If the core moves to Rust, built-ins either run as trusted uncontainerized Python
workers over the plugin protocol, or are embedded via PyO3. Keep built-ins behind the
same node interface plugins use so both remain possible.

## 5. v1 scope

**In:**
1. Inputs: LSL inlet, synthetic generator, file playback; BIDS, EDF/BDF, EEGLAB .set.
   The synthetic generator (node 0.2.0) is made of parts switched on and off:
   neural (pink 1/f background + posterior alpha bursts), line noise, SSVEP, a pure
   sine, DC offset per channel, and artefacts (blinks, saccades, slow eye
   movements, ECG, EMG bursts, electrode pops, drift). Artefact events come from a
   seeded schedule in absolute time, so the output is block-size invariant and
   every event is written to the recording's events as ground truth (`blink`,
   `saccade`, `ecg_r`, `emg`, `pop`) for scoring cleaning methods. Defaults and
   their sources (blink rate: Bentivoglio et al. 1997; EMG spectra and topography:
   Goncharova et al. 2003; EOG ~20 uV per degree) are listed in
   `meegxer/core/sources.py`; values without a source are marked heuristic.
2. Engine: DAG core with linear chain view, fit/transform + calibration, sample-index
   alignment, real-time and batch schedulers.
3. Built-in nodes (`meegxer-nodes`): spectral filters, spatial filters, referencing,
   ICA → ICLabel; features: PSD and band power (Welch or multitaper), spectral
   parameterization (specparam), Morlet wavelet power, Hilbert, connectivity,
   complexity, SSVEP detection (CCA, filter-bank CCA).
   ASR is the EEGLAB plugin `eeglab.asr`: EEGLAB's ASR code is licensed for
   non-commercial use only, so no copy of it lives in MeegXer and the built-in
   nodes have no ASR.
4. External plugin runtime + two reference plugins (Octave/EEGLAB stateful IIR
   bandpass; MNE notch or re-ref).
5. GUI: drag-and-drop chain, probes (overlay + difference), auto-generated parameter
   controls from manifests, PSD/feature views, per-stage status, CPU/latency/overrun meters.
6. Outputs: LSL outlet, .set/EDF export, feature tables (BIDS-derivatives layout),
   session JSON (the unit of reproducibility; format in `spec/session.md`);
   BIDS derivatives layout in `spec/bids-derivatives.md`.

**Out (later):** markers/epoching, buses/routing matrix, sidechains, node editor,
provenance cards, plugin registry/signing, multi-stream sync, action sinks, Rust core,
CLAP native plugins, installers.

## 6. Acceptance tests

1. **Correctness:** block-wise streamed output of each stateful plugin matches the same
   operation applied offline in one pass (causal reference), to float tolerance; no
   block-boundary artifacts.
2. **Stability:** 30 min at 64 ch / 1 kHz with two plugins, zero dropped or duplicated
   samples (verified via a sample counter channel).
3. **Latency:** end-to-end latency (input timestamp → output push) measured and reported;
   target to be agreed (≈ < 150 ms for monitoring; tighter for closed-loop BCI).
4. **Reproducibility:** save session → reload on a second machine → offline run gives
   an identical output file.
5. **Cross-platform:** passes on Linux and Windows (Docker Desktop / WSL2); macOS nice-to-have.
6. **Built-in concordance:** Python built-ins (esp. ICLabel) and the EEGLAB ASR
   plugin compared against EEGLAB reference implementations on the same data, with
   the references stored in the repository so CI needs no EEGLAB; report per node.

## 7. Roadmap extensions (design must not block these)

1. **Live BCI controller:** decoder nodes → Events → action sinks (game controllers,
   boards).
2. **Mass preprocessing GUI:** BIDS browser, job queue, parallel workers, resume,
   per-subject QC reports, BIDS-derivatives with provenance, CLI for HPC.
3. **Plugin specification as a community standard:** semver spec, conformance suite,
   reference SDKs, alignment with BIDS derivatives. Call it "the MeegXer plugin
   specification" until adopted by other groups.
4. **Mini-PC / Raspberry Pi:** headless engine on device, optional remote GUI over a
   local connection; verify liblsl/pylsl and PySide6 on ARM64; multi-arch images; ARM CI runner.

## 8. Settled questions

- **Python floor:** 3.13 (Debian 13 / Raspberry Pi OS Trixie system Python); primary
  development on 3.14 (Ubuntu 26.04 LTS). Revisit when SPEC 0 drops 3.13 (~Oct 2027).
  Workspace tool: uv.
- **License:** this repository is GPL-3.0 (`LICENSE`).
- **Online calibration UX:** one-shot calibration at session start for v1; periodic
  refits later.
- **Feature outputs in v1:** Features → Sink (view, export, LSL feature stream) only;
  Features → Processor (e.g. classifiers) later.
- **wICA:** dropped from v1 (non-standard method); may return as a plugin.
- **Latency target:** < 150 ms end-to-end for monitoring; closed-loop BCI target set
  when that roadmap item starts.
- **Data and state:** float64 inside the core, float32 on the plugin wire; node state
  must be serializable.
- **Transports:** JSON-RPC control and display frames both over ZeroMQ (REQ/REP and
  PUB/SUB); shared memory for same-machine display later.
