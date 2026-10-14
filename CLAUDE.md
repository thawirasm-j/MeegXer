# MeegXer

MeegXer (pronounced "Meek-Ser") — a user-friendly mixer for EEG processing.
Voicemeeter/VST-style live and offline EEG pipelines, with built-in audited
processing nodes and containerized research plugins (Octave/EEGLAB or MNE-Python).

Status: pre-MVP. Full design and rationale: `docs/DESIGN.md` — read it before
making architectural changes.

## Architecture rules (do not break these)

1. **Headless core.** `meegxer-core` runs as a separate daemon process and owns all
   timing, routing, and outputs. It must never import Qt or any GUI code.
2. **The core never waits for the GUI.** Display frames are decimated in the core
   and sent over a lossy channel; if the GUI lags, frames drop, processing continues.
3. **Separate channels:** control = JSON-RPC (request/reply); display = binary frames
   (ZeroMQ, or shared memory on the same machine).
4. **One node implementation, two schedulers.** Real-time (block-wise) and batch
   (whole recordings, many subjects) run the same node code. No separate offline path.
5. **Graph internally, linear UI for now.** The pipeline is a DAG; the v1 UI shows a
   single chain with probe taps. A node editor comes later on the same model.
6. **Typed ports:** Signal (time series), Features (windowed), Events (markers/commands).
   **Node roles:** Source, Processor, Probe, Sink.
7. **Fit/transform contract** for every processor: `fit(calibration) -> model`,
   `transform(block, model, state) -> block, state`. Upstream changes mark downstream
   models "needs refit"; never silently reuse a stale model.
8. **Align by sample index, match channels by label.** Every block carries a sample
   index and LSL timestamp. Probes overlay by index, never by arrival time.
9. **Plugins never touch devices or the network.** External plugins run in Docker/Podman
   with `--network none`, read-only mounts, stdin/stdout framed binary protocol. They emit
   Events only; trusted host-side Sinks perform actions (keyboard, gamepad, serial).
10. **Sterile build is a packaging property.** Sterile = core + sterile + gui + cli.
    `meegxer-runtime` (Docker plugins, SDKs) is never shipped in it, not merely disabled.
11. **The plugin spec is language-neutral** (JSON Schema + protocol doc), versioned with
    semver. Python/Octave SDKs implement the spec; they do not define it.

## Packages

| Package | Contents |
|---|---|
| `meegxer-core` | engine, graph, schedulers, plugin spec, I/O, control API (no GUI) |
| `meegxer-sterile` | audited built-ins: filters, re-ref, ASR, ICA/ICLabel/wICA, features, action sinks |
| `meegxer-runtime` | Docker plugin runtime + Python/Octave SDKs (research builds only) |
| `meegxer-gui` | PySide6 + pyqtgraph client |
| `meegxer-cli` | headless runs, batch jobs, Pi deployment |

Canonical code name is lowercase `meegxer` everywhere (imports, repo, image prefixes).

## Stack

- Python first for everything. Only the core is a candidate for a later Rust rewrite,
  behind the same protocol.
- Core: numpy, pylsl, pyzmq. Built-ins: MNE, SciPy, mne-icalabel, asrpy/meegkit,
  mne-connectivity, antropy. GUI: PySide6, pyqtgraph. CLI: Typer or Click.
- I/O: BIDS (mne-bids), EDF/BDF, EEGLAB .set in; .set (eeglabio), EDF, BIDS-derivatives out.
- Performance escalation order: vectorize (preallocated ring buffers, no per-sample
  loops) → Numba → PyO3 Rust extension → full Rust core. Measure before escalating.

## Benchmarks

- Online: 16 ch / 500 Hz. Offline/batch: 64 ch / 1 kHz.

## Working style

- When fixing bugs: diagnose the precise root cause first, apply the smallest fix,
  state what the cause was and why the fix works. No unsolicited refactors.
- For architecture or method decisions: flag alternatives briefly and ask before
  committing to one.
- Ask before changing anything listed under "Architecture rules".
