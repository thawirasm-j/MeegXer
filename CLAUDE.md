# MeegXer

MeegXer (pronounced "Meek-Ser") — a user-friendly mixer for EEG processing.
Voicemeeter/VST-style live and offline EEG pipelines, with built-in processing
nodes and plugins (Octave/EEGLAB or MNE-Python). Sterile mode isolates processing
in a container, away from the machine and the network.

Status: pre-MVP. Full design and rationale: `docs/DESIGN.md` — read it before
making architectural changes. Milestones and sequencing: `docs/WORKPLAN.md`.

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
7. **Fit/transform contract** for every processor: `fit(calibration, params) -> model`,
   `transform(block, model, state, params) -> block, state, events` (events may be
   empty; full manifest in `docs/DESIGN.md` §3.3). Upstream changes mark downstream
   models "needs refit"; never silently reuse a stale model.
8. **Align by sample index, match channels by label.** Every block carries a sample
   index and LSL timestamp. Probes overlay by index, never by arrival time.
9. **Plugins are untrusted code; only host Sinks act.** A plugin runs in its own
   process and speaks the stdin/stdout framed binary protocol. It emits Events only;
   trusted host-side Sinks perform actions (keyboard, gamepad, serial). Other packages
   reach plugins only through the node-provider entry point. Outside Sterile mode a
   plugin may run as a local process with the user's permissions (network, files), so
   adding one asks the user to acknowledge that risk (patient data: HIPAA, GDPR, IRB).
   In a container it runs with `--network none` and read-only mounts. (Local plugins
   and custom scripts need the accepted notice: `meegxer-plugin add --local/--script`
   or *Add a custom script…* in the GUI; `docs/CUSTOM_SCRIPTS.md`.)
10. **Sterile = isolation, a runtime mode.** Sterile mode runs all processing (built-in
    nodes and plugins) in a container with no network and read-only mounts; the host
    keeps only trusted I/O (sources, LSL, file writers, device sinks, the GUI). It
    means isolation only: which code is trusted (auditing) is the user's
    responsibility. Every install ships every package; "sterile" never names a
    package or a build. (Planned; needs Docker or Podman.)
11. **The plugin spec is language-neutral** (JSON Schema + protocol doc), versioned with
    semver. Python/Octave SDKs implement the spec; they do not define it.

## Packages

| Package | Contents |
|---|---|
| `meegxer-core` | engine, graph, schedulers, plugin spec, I/O, control API (no GUI) |
| `meegxer-nodes` | built-in nodes: filters, re-ref, ICA/ICLabel, features, action sinks |
| `meegxer-plugins` | plugin runtime (container runner) + Python/Octave SDKs, EEGLAB ASR plugin |
| `meegxer-gui` | PySide6 + pyqtgraph client |
| `meegxer-cli` | headless runs, batch jobs, Pi deployment |
| `meegxer` | bundle: installs every package (core + nodes + plugins + gui + cli); no code of its own |

Renamed in 0.2.1: `meegxer-sterile` → `meegxer-nodes`, `meegxer-runtime` →
`meegxer-plugins`. Node names `meegxer.sterile.*` still load as `meegxer.nodes.*`
(`meegxer.core.registry.canonical_name`).

Canonical code name is lowercase `meegxer` everywhere (imports, repo, image prefixes).

## Stack

- Python first for everything; floor is Python 3.13 (develop on 3.14;
  CI tests 3.13 + 3.14; revisit the floor when SPEC 0 drops 3.13, ~Oct 2027).
  Monorepo managed as a uv workspace. Only the core is a candidate for a later Rust rewrite,
  behind the same protocol.
- Core: numpy, scipy, jsonschema, pylsl, pyzmq (JSON-RPC control and display both over
  ZeroMQ); file I/O through MNE (mne-bids, eeglabio, edfio). Signals are in volts. Built-ins: MNE, SciPy, python-picard, mne-icalabel,
  mne-connectivity, antropy (ASR is the EEGLAB plugin). GUI: PySide6, pyqtgraph. CLI: Typer or Click.
- I/O: BIDS (mne-bids), EDF/BDF, EEGLAB .set in; .set (eeglabio), EDF, BIDS-derivatives out.
- Performance escalation order: vectorize (preallocated ring buffers, no per-sample
  loops) → Numba → PyO3 Rust extension → full Rust core. Measure before escalating.

## Benchmarks

- Online: 16 ch / 500 Hz. Offline/batch: 64 ch / 1 kHz.
- The streaming stability acceptance test (`docs/DESIGN.md` §6.2) intentionally runs at
  the batch size (64 ch / 1 kHz) as headroom over the online target.

## Commands

- `uv sync --all-packages`, then `uv run pytest`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`, `uv run lint-imports`.
- `python tools/smoke_deps.py --python 3.14`: planned third-party deps install/import.
- `uv run meegxer nodes`; `uv run meegxer run pipeline.json -i rec.set -o out.set`.
- Sessions (`spec/session.md`): `meegxer run … --session out.meegxer.json`,
  `meegxer run out.meegxer.json`, `meegxer session check FILE`,
  `meegxer session save FILE` (from the daemon, with calibrated models).
- BIDS derivatives (`spec/bids-derivatives.md`): `meegxer run … --bids-out DIR`,
  `meegxer run PIPELINE --bids RAW_ROOT [--subjects 01,02] [--task T] [--overwrite]`.
- Live: `uv run meegxer daemon --pipeline live.json --start`, then `meegxer status`,
  `meegxer set hp low=0.5`, `meegxer calibrate NODE --seconds 60`, `meegxer stop`,
  `meegxer clear-calibration [NODE]`, `meegxer passthrough NODE --on/--off`.
- `uv run python tools/stability_test.py --minutes 30`: acceptance tests 2 and 3
  (`--plugin-image IMAGE` to include plugins).
- GUI: `uv run meegxer-gui` (connects to the daemon on the default endpoint, or
  starts one); `uv run python tools/gui_screenshot.py OUT_DIR` renders it offscreen.
  Offline runs write to `MEEGXER_OFFLINE_DIR` (default: the system temp folder),
  with a processor-output cache capped by `MEEGXER_OFFLINE_CACHE_GB` (default 5).
- Plugins: `uv run meegxer-plugin check IMAGE` (conformance),
  `meegxer-plugin add IMAGE`, `meegxer-plugin list`. Custom scripts
  (`process(raw, ...)` offline, or `transform(data, state, ...)` [+ `fit(raw, ...)`]
  live and offline): `meegxer-plugin check|add --script FILE` (`add` asks to accept
  the local-code notice, or `--accept-risk`, then runs the conformance suite). SDKs in
  `packages/meegxer-plugins/sdk/`; reference plugins in `packages/meegxer-plugins/plugins/`.
- New nodes register in the `meegxer.nodes` entry-point group (see package
  pyproject.toml) and declare `availability` (online/offline/both). Stream
  processors that run online must pass
  `meegxer.core.conformance.check_blockwise_equivalence` (acceptance test 1).
- Releases: push a `v*` tag (TestPyPI), then run the Release workflow from the
  tag with target `pypi`; see `docs/RELEASING.md`. `tools/set_version.py`
  stamps one version on every package and pins siblings exactly.
- Architecture rules 1 and 9 (plugins only through entry points) are import-linter
  contracts in the root `pyproject.toml`;
  `tests/test_architecture.py` proves each contract catches a violation.

## Working style

- When fixing bugs: diagnose the precise root cause first, apply the smallest fix,
  state what the cause was and why the fix works. No unsolicited refactors.
- For architecture or method decisions: flag alternatives briefly and ask before
  committing to one.
- Ask before changing anything listed under "Architecture rules".
