# MeegXer — Work Plan (v0.1)

Sequencing for the v1 scope in `DESIGN.md` §5. Each milestone ends with something
runnable and a named acceptance test (`DESIGN.md` §6) where one applies.
Decisions marked **[D]** need an owner answer before that milestone starts.

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

- Layout: `packages/meegxer-{core,sterile,runtime,gui,cli}/`, each with its own
  `pyproject.toml` and `src/meegxer/<subpkg>/` (namespace package `meegxer`).
- `spec/` at repo root: JSON Schemas + protocol docs (language-neutral, rule 11),
  versioned independently with semver.
- Tooling: ruff (lint + format), mypy (strict on core), pytest, pre-commit.
- **Import contracts** (import-linter): `meegxer.core` must not import Qt/PySide6,
  `meegxer.gui`, `meegxer.runtime`; `meegxer.sterile` must not import `meegxer.runtime`.
- **Sterile packaging check:** CI builds the sterile wheel set and asserts no
  `meegxer.runtime`/docker module is present (rule 10).
- CI matrix: Linux + Windows; Python floor **[D]**.

**[D] decisions**
- Python version floor (suggest 3.11: needed by recent MNE/PySide6, good perf gains).
- Workspace tool: uv workspaces vs hatch. (Suggest uv.)

**Exit:** `pytest` green on an empty core; import-linter and sterile-packaging checks
fail on a deliberately bad import.

## M1 — Core contract

**Goal:** the data model and node interface everything else builds on.

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

**[D] decisions**
- Internal dtype: float64 in-process (IIR state precision) with float32 only on the
  plugin wire, vs float32 everywhere. (Suggest float64 internal / float32 wire.)
- Is `state` opaque per node, or must it be serializable (needed for session
  save/resume mid-stream and for a future Rust core)? (Suggest: serializable.)

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

**[D] decisions**
- Control transport: JSON-RPC over ZeroMQ REQ/REP vs over WebSocket (the latter eases a
  future web/remote GUI). Display: ZeroMQ PUB/SUB first, shared memory later.
- Latency target number (`DESIGN.md` §8).
- Calibration UX: one-shot at start only for v1? (Suggest yes; periodic refit later.)

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

- ASR (asrpy/meegkit), ICA → ICLabel → ICA rejection, wICA.
- Features (analysis processors, Features port): PSD, band power, Hilbert envelope,
  connectivity (mne-connectivity), complexity (antropy).
- Each: offline first, then streaming where meaningful; test 1 where stateful.
- **Acceptance test 6:** concordance vs EEGLAB references on shared data; report
  per built-in.

**[D] decisions**
- wICA ground truth: existing implementation vs re-implement from the paper.
- Do v1 Features feed downstream nodes (classifier, LSL feature stream) or are they
  view/export only? Affects whether the scheduler must handle multi-rate edges in v1.
  (Suggest: allow Features → Sink in v1, Features → Processor later.)

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
| Contract churn after many nodes exist | M1–M2 | Freeze contract at M1 exit; version the spec |
| GPL-3.0 implications for third-party plugins | M4 | Decide and document whether container/stdio plugins are separate works **[D]** |

## Open decisions (summary)

| # | Decision | Needed by |
|---|---|---|
| 1 | Python floor | M0 |
| 2 | Workspace tool (uv / hatch) | M0 |
| 3 | Internal dtype; serializable node state | M1 |
| 4 | Control/display transports | M3 |
| 5 | Latency target; calibration UX | M3 |
| 6 | Plugin licensing stance under GPL-3.0 | M4 |
| 7 | wICA reference; Features downstream in v1 | M6 |
