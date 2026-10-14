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
- **Research build:** includes containerized external plugins.
- **Sterile build:** built-in audited nodes only, run uncontainerized; no plugin runtime
  shipped. Intended for environments where unvalidated code must never run
  (e.g., future hospital use). Research plugins must not reach hospital computers by design.
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

### 3.3 Processor contract
```
manifest:  name, version, type (stream | analysis), image (external only),
           params [{name, type, min, max, default, unit}],
           latency_samples, requires_fit (bool)
fit(calibration_data | dataset, params)       -> model
transform(block, model, state, params)        -> block_out, state, events
```
- **Stream processors:** block in → block out at the same rate (filters with carried
  state, re-ref, ASR apply, ICA apply).
- **Analysis processors:** windowed input → Features/Events at a lower rate (PSD,
  band power, connectivity, CCA, classifiers).
- **Calibration (online):** record N seconds, fit, then go live. Per-stage status:
  uncalibrated / calibrating / ready / needs-refit.
- Some processors are dataset-level only (ComBat, group statistics) and never stream.
- Live parameter changes without restarting workers.

### 3.4 Probes
- A probe is a non-modifying tap node. Example chain:
  `Input > Highpass > Probe R1 > ASR > Probe R2 > ICA > ICLabel > ICA rejection > Probe Main`
- Viewer shows Main as solid and selected comparison probes as faded background.
- **Display modes:** overlay; difference (e.g., Main − R1 = what was removed).
- Aligned by **sample index**, channels matched by **label**.
- Feature-domain probes: overlay PSDs across probes.

### 3.5 External plugins
- Run in Docker/Podman: `docker run -i --network none`, read-only mounts.
- Transport: framed binary over stdin/stdout (small header + float32 channels×samples);
  ZeroMQ is the upgrade path. No shared memory across the container boundary.
- Persistent worker per plugin (Octave + EEGLAB loaded once).
- Plugins emit Events only; they never access devices or networks.
- Base images: Octave with pinned EEGLAB; slim Python with pinned MNE. Multi-arch
  (amd64 + arm64) eventually.
- Host and LSL run natively; only plugin workers are containerized (LSL multicast
  discovery does not cross Docker Desktop's VM boundary).

### 3.6 Sinks
- LSL outlets, file writers (.set, EDF, BIDS-derivatives, feature TSV), session JSON.
- Action sinks (later): keyboard/mouse (pynput), virtual gamepad (vgamepad/uinput),
  serial (pyserial), OSC/MIDI. Trusted host-side only; user maps Events → actions.
  Safety: arm/disarm toggle, confidence thresholds, dwell time.

## 4. Packages and languages

| Package | Language now | Later |
|---|---|---|
| meegxer-core | Python (numpy, pylsl, pyzmq) | Rust behind the same protocol, if measured need |
| meegxer-sterile | Python (MNE, SciPy, mne-icalabel, asrpy/meegkit, mne-connectivity, antropy) | stays Python |
| meegxer-runtime | Python (docker SDK; Podman-compatible) | follows core |
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
1. Inputs: LSL inlet, synthetic generator (pink noise + alpha bursts + 50 Hz line
   noise + SSVEP sine), file playback; BIDS, EDF/BDF, EEGLAB .set.
2. Engine: DAG core with linear chain view, fit/transform + calibration, sample-index
   alignment, real-time and batch schedulers.
3. Sterile built-ins: spectral filters, spatial filters, referencing, ASR,
   ICA → ICLabel; features: PSD, band power, Hilbert, connectivity, complexity.
4. External plugin runtime + two reference plugins (Octave/EEGLAB stateful IIR
   bandpass; MNE notch or re-ref).
5. GUI: drag-and-drop chain, probes (overlay + difference), auto-generated parameter
   controls from manifests, PSD/feature views, per-stage status, CPU/latency/overrun meters.
6. Outputs: LSL outlet, .set/EDF export, feature tables (BIDS-derivatives layout),
   session JSON (the unit of reproducibility).

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
6. **Built-in concordance:** Python built-ins (esp. ASR, ICLabel) compared against
   EEGLAB reference implementations on the same data; report per built-in.

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
