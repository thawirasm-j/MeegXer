# MeegXer — Work Plan (v0.1)

Sequencing for the v1 scope in `DESIGN.md` §5. Each milestone ends with something
runnable and a named acceptance test (`DESIGN.md` §6) where one applies.
Settled decisions are logged at the end; none are currently open.

## Guiding principles

- **Batch before real-time.** The batch scheduler is the simplest harness for proving
  the node contract. The real-time scheduler then reuses the same nodes, and
  acceptance test 1 (streamed == one-pass) becomes the gate between them.
- **Contract before content.** Freeze `Block`, ports, manifest schema and the
  fit/transform signature before writing many nodes; every node written earlier is
  rework if the contract moves.
- **Thin vertical slice first.** Synthetic source → one filter → probe → file sink,
  end to end, before breadth (ASR, ICA, GUI polish).
- **Architecture rules are enforced by tooling, not memory** (import contracts, a
  packaging check for the sterile build).

## Dependency graph

```
M0 Scaffolding ─▶ M1 Core contract ─▶ M2 Batch slice ─┬─▶ M3 Real-time + daemon ─┬─▶ M5 GUI
                                                       │                          │
                                                       │                          └─▶ M4 Plugin runtime
                                                       └─▶ M6 Heavy built-ins (offline first, then online)
                                                                         all ─▶ M7 Reproducibility & release gate
```

M4 and M5 can run in parallel after M3. M6 offline work can start right after M2.

---

## M0 — Repository scaffolding

**Goal:** an empty-but-real monorepo where the architecture rules are checked by CI.

**Status:** done locally (Linux, 3.13 + 3.14); GitHub Actions unverified until first PR run.
Linux CI needs `libegl1 libgl1 libxkbcommon0` for Qt/pyqtgraph imports.

- Layout: `packages/meegxer-{core,sterile,runtime,gui,cli}/`, each with its own
  `pyproject.toml` and `src/meegxer/<subpkg>/` (namespace package `meegxer`).
- `spec/` at repo root: JSON Schemas + protocol docs (language-neutral, rule 11),
  versioned independently with semver.
- Tooling: ruff (lint + format), mypy (strict on core), pytest, pre-commit.
- **Import contracts** (import-linter): `meegxer.core` must not import Qt/PySide6,
  `meegxer.gui`, `meegxer.runtime`; `meegxer.sterile` must not import `meegxer.runtime`.
- **Sterile packaging check:** CI builds the sterile wheel set and asserts no
  `meegxer.runtime`/docker module is present (rule 10).
- uv workspace at the root; `requires-python = ">=3.13"` in every package.
- CI matrix: Linux + Windows × Python 3.13, 3.14 (develop on 3.14).
- Early smoke job: install every third-party dependency on 3.13 and 3.14 (see risks).

**Exit:** `pytest` green on an empty core; import-linter and sterile-packaging checks
fail on a deliberately bad import.

## M1 — Core contract

**Goal:** the data model and node interface everything else builds on.

**Status:** implemented in `meegxer.core` (block, ports, serialization, manifest,
node, stage, ringbuffer); spec 0.1.0. Source/Probe/Sink ABCs are provisional until M2.

- `Block`: `data (n_ch × n_samp)`, `sfreq`, `ch_labels`, `sample_index` (int64, first
  sample), `lsl_timestamps` (or first-timestamp + sfreq), `stream_id`.
- `FeatureBlock`, `EventBlock` (sample-indexed markers/commands with payload).
- Port types (Signal/Features/Events) and node roles (Source/Processor/Probe/Sink)
  with typed connection validation.
- Processor ABC implementing `fit(calibration, params) -> model` and
  `transform(block, model, state, params) -> block, state, events`.
- Stage status state machine: uncalibrated → calibrating → ready → needs-refit;
  upstream param/model change propagates needs-refit downstream (rule 7).
- Manifest JSON Schema (`spec/manifest.schema.json`), validated in Python.
- Preallocated ring buffer utility (no per-sample loops).
- Channel matching by label; sample-index alignment helpers (rule 8).

**Decided**
- float64 inside the core (IIR state precision); float32 only on the plugin wire.
- Node `state` must be serializable (session save/resume, future Rust core).

**Exit:** unit tests for Block slicing/concat by sample index, label matching,
status propagation, manifest validation.

## M2 — Batch vertical slice

**Goal:** first end-to-end pipeline, offline.

- Graph (DAG) with topological execution; linear-chain builder on top (rule 5).
- Batch scheduler: whole recording, optionally fed block-wise to prove the contract.
- Sources: synthetic generator (pink noise + alpha bursts + 50 Hz + SSVEP sine),
  file reader (EDF/BDF, EEGLAB .set, BIDS via mne-bids).
- First sterile processors: stateful IIR band-pass/high-pass (SOS + carried `zi`),
  notch, common-average / custom re-reference.
- Probe node (non-modifying tap, stores aligned copies).
- Sinks: .set (eeglabio), EDF, feature TSV.
- Per-node cache keyed by hash(input, node version, params).
- **Acceptance test 1 harness:** for every stateful node, block-wise output with
  random block sizes == one-pass causal output (float tolerance), no boundary
  artifacts. This harness is reused for every later node and every plugin.

**Exit:** `meegxer run pipeline.json --input sub-01.set` produces a filtered .set;
test 1 passes for filter, notch, re-ref.

## M3 — Real-time scheduler, daemon, control and display

**Goal:** the same nodes running live in a headless daemon.

- Real-time scheduler: block-wise pull, deadlines, overrun detection/reporting,
  CPU/latency metrics per stage.
- LSL inlet source and LSL outlet sink (pylsl); file playback source at real-time rate.
- Daemon process (`meegxer-core` entry point), no GUI imports (rule 1).
- Control API: JSON-RPC (load graph, set param, start/stop, calibrate, status).
  Spec in `spec/control-api.md`.
- Display stream: decimator to min/max envelopes per probe, lossy pub/sub; frames
  dropped when the client lags, never back-pressuring the core (rule 2).
- Online calibration flow: record N s → fit → ready.
- CLI (Typer): `meegxer daemon`, `meegxer run`, `meegxer status`.
- **Acceptance test 2 (stability, no plugins yet)** with a sample-counter channel;
  **acceptance test 3 (latency)** measured and reported.

**Decided**
- Control: JSON-RPC over ZeroMQ REQ/REP. Display: ZeroMQ PUB/SUB; shared memory later.
- Latency target: < 150 ms end-to-end for monitoring.
- Calibration: one-shot at session start for v1; periodic refit later.

**Exit:** daemon streams synthetic LSL → filter → LSL out for 30 min with zero
dropped/duplicated samples; a scripted client can tune params live.

## M4 — Plugin specification and runtime (research build only)

**Goal:** containerized external nodes behaving like built-ins.

- `spec/plugin-protocol.md` + schemas: framing (header + float32 ch×samples),
  handshake (manifest, spec version), fit/transform/param messages, error and
  shutdown semantics. Semver'd (rule 11).
- `meegxer-runtime`: Docker/Podman worker lifecycle (`--network none`, read-only
  mounts, persistent worker, restart/backoff, timeouts), adapter exposing a plugin as
  a normal Processor node.
- SDKs: Python package and Octave `.m` template implementing the spec.
- Reference plugins: Octave/EEGLAB stateful IIR band-pass; MNE notch or re-ref.
- Conformance suite: runs any plugin image through the protocol + test 1 harness.
- Rerun **test 2** with two plugins at 64 ch / 1 kHz.

**Exit:** both reference plugins pass the conformance suite and test 1; sterile
build still contains no runtime code.

## M5 — GUI (v1)

**Goal:** the "mixer" experience, as a pure client of M3's APIs.

- PySide6 + pyqtgraph client connecting via control + display channels only.
- Linear drag-and-drop chain; insert-anywhere probes.
- Auto-generated parameter controls from manifests.
- Probe viewer: Main solid, comparisons faded; overlay and difference modes; PSD view.
- Per-stage status badges, CPU/latency/overrun meters.

**Exit:** a user can build Synthetic → HP → Probe → Notch → Probe, see the 50 Hz
line disappear in difference mode, and kill the GUI without interrupting processing.

## M6 — Heavy built-ins and features

**Goal:** the processing that makes MeegXer useful for real data.

- ASR (asrpy/meegkit), ICA → ICLabel → ICA rejection. (wICA dropped for v1.)
- Features (analysis processors, Features port): PSD, band power, Hilbert envelope,
  connectivity (mne-connectivity), complexity (antropy).
- Each: offline first, then streaming where meaningful; test 1 where stateful.
- **Acceptance test 6:** concordance vs EEGLAB references on shared data; report
  per built-in.

**Decided**
- Features → Sink only in v1 (view, export, LSL feature stream); Features → Processor
  later, so the v1 scheduler needs no multi-rate processor edges.

## M7 — Reproducibility and release gate

- Session JSON (graph, node versions, params, fitted models or their hashes, spec
  version, image digests): the unit of reproducibility.
- BIDS-derivatives output layout with provenance.
- **Acceptance test 4:** save → reload on a second machine → identical output file.
- **Acceptance test 5:** full suite on Linux and Windows (Docker Desktop / WSL2).
- v1 tag once tests 1–6 are green.

---

## Risks to watch

| Risk | Where | Mitigation |
|---|---|---|
| ASR/ICA not truly causal/blockwise | M6 | Define "online" variants explicitly; test 1 against causal reference only |
| GIL contention at 64 ch / 1 kHz + display | M3 | Daemon separate from GUI; measure, then Numba → PyO3 per CLAUDE.md order |
| Docker Desktop latency/IPC overhead on Windows | M4 | Measure early with a passthrough plugin; ZeroMQ upgrade path |
| LSL discovery across container boundary | M4 | Host-only LSL (already in design) |
| pylsl has no bundled liblsl on Linux ARM64 (Pi) | Pi roadmap | Build liblsl from a pinned tag (as the ARM64 CI job does) or use conda-forge `liblsl`; set `PYLSL_LIB` |
| Contract churn after many nodes exist | M1–M2 | Freeze contract at M1 exit; version the spec |
| Python 3.14 wheels missing for a dependency (e.g. pylsl, numba, onnxruntime for mne-icalabel, PySide6 on ARM64) | M0, M6 | Dependency smoke job in M0; 3.13 floor as fallback; pin known-good versions |
| Scientific stack drops 3.13 under SPEC 0 (~Oct 2027) | ongoing | Bump floor to 3.14 then |
| GPL-3.0 scope for externally distributed plugins | M4 | Repo is GPL-3.0; plugin licensing not yet prescribed, revisit before bundling any plugin |

## Decision log

| # | Decision | Outcome | Affects |
|---|---|---|---|
| 1 | Python floor | 3.13 floor, develop on 3.14 | M0 |
| 2 | Workspace tool | uv | M0 |
| 3 | Internal dtype; node state | float64 core / float32 wire; serializable state | M1 |
| 4 | Control/display transports | ZeroMQ for both (REQ/REP, PUB/SUB) | M3 |
| 5 | Latency target; calibration UX | < 150 ms monitoring; one-shot calibration | M3 |
| 6 | License | GPL-3.0 for this repo | all |
| 7 | wICA; Features downstream | wICA dropped (non-standard); Features → Sink only | M6 |
