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

- Layout: `packages/meegxer-{core,nodes,plugins,gui,cli}/` (nodes and plugins were
  sterile and runtime until 0.2.1), each with its own `pyproject.toml` and
  `src/meegxer/<subpkg>/` (namespace package `meegxer`).
- `spec/` at repo root: JSON Schemas + protocol docs (language-neutral, rule 11),
  versioned independently with semver.
- Tooling: ruff (lint + format), mypy (strict on core), pytest, pre-commit.
- **Import contracts** (import-linter): `meegxer.core` must not import Qt/PySide6,
  `meegxer.gui`, `meegxer.plugins`; `meegxer.nodes` must not import `meegxer.plugins`.
- **Sterile packaging check:** CI builds the sterile wheel set and asserts no
  `meegxer.plugins`/docker module is present (rule 10).
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

**Status:** implemented. `meegxer.core`: graph + pipeline format, entry-point registry,
node-major batch scheduler, per-node cache (memory/directory), synthetic/file/BIDS
sources, memory probe, .set/EDF/feature-TSV sinks, `conformance` (test 1 harness).
`meegxer.nodes`: IIR filter, notch, re-reference (test 1 passes for all three).
CLI: `meegxer run`, `meegxer nodes`. Spec 0.2.0 (manifest `role`, `live_params`;
`pipeline.schema.json`).

- Graph (DAG) with topological execution; linear-chain builder on top (rule 5).
- Batch scheduler: whole recording, optionally fed block-wise to prove the contract.
- Sources: synthetic generator (pink noise + alpha bursts + 50 Hz + SSVEP sine),
  file reader (EDF/BDF, EEGLAB .set, BIDS via mne-bids).
- First sterile processors: stateful IIR band-pass/high-pass (SOS + carried `zi`),
  notch, common-average / custom re-reference.
- Probe node (non-modifying tap, stores aligned copies).
- Sinks: .set (eeglabio), EDF, feature TSV.
- Per-node cache keyed by hash(input, node version, params); reuse the M1 stage
  fingerprints.
- Opt-in for processors to keep streaming state across a parameter change when it
  stays valid (e.g. gain), so live knob turns don't reset filters. Default stays reset.
- **Acceptance test 1 harness:** for every stateful node, block-wise output with
  random block sizes == one-pass causal output (float tolerance), no boundary
  artifacts. This harness is reused for every later node and every plugin.

**Exit:** `meegxer run pipeline.json --input sub-01.set` produces a filtered .set;
test 1 passes for filter, notch, re-ref.

## M3 — Real-time scheduler, daemon, control and display

**Goal:** the same nodes running live in a headless daemon.

**Status:** implemented. `meegxer.core.realtime` (block-major scheduler, paced playback,
overruns, per-stage timing, latency, one-shot calibration gating), `lsl` (LSL in/out
with `SAMPLE_INDEX` counter), `display` (index-aligned min/max decimation, lossy PUB),
`control` + `daemon` (JSON-RPC over ZeroMQ, `meegxer-daemon`), CLI daemon commands,
`spec/control-api.md`, `spec/display-protocol.md`, `tools/stability_test.py`
(acceptance tests 2 and 3).
Exit run (2026-10-14, 30 min, 64 ch / 1 kHz, Linux container, 59 live parameter
changes): 0 gaps and 0 duplicates at the daemon input and output, 2 overruns in
163,403 blocks, daemon latency p95 4.0 ms, end-to-end (generator -> daemon -> receiver
over LSL) p95 14.1 ms, max 247 ms. Report: `docs/results/m3-stability-30min.json`.

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
- Refuse offline-only nodes (manifest `availability`) in the real-time scheduler.
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

**Status:** implemented. `spec/plugin-protocol.md` (`meegxer.plugin/0.1`);
`meegxer.plugins` (wire format, worker lifecycle with timeouts/restarts, adapter,
plugin index + node provider, conformance suite, `meegxer-plugin` CLI); Python SDK
(`sdk/python/meegxer_plugin.py`, numpy only) and Octave SDK (`sdk/octave/`); reference
plugins `examples.average_reference` (Python) and `examples.octave_iir_bandpass`
(Octave), both conformant inside the Docker sandbox. The reference Octave image does
not include EEGLAB yet (the IIR filter does not need it); a pinned-EEGLAB base image is
a follow-up. Exit run (acceptance tests 2 and 3 with both reference plugins in the live
chain, Docker, 64 ch / 1 kHz, 30 min): 0 gaps and 0 duplicates, 460 overruns in 163k
blocks (0.28%, steady rate), daemon latency p95 8.9 ms, end-to-end p95 18.5 ms
(max 632 ms). Mean plugin cost per block: 1.2 ms (Python), 3.2 ms (Octave). Report:
`docs/results/m4-stability-30min-two-plugins.json`.

- `spec/plugin-protocol.md` + schemas: framing (header + float32 ch×samples),
  handshake (manifest, spec version), fit/transform/param messages, error and
  shutdown semantics. Semver'd (rule 11).
- `meegxer-plugins`: Docker/Podman worker lifecycle (`--network none`, read-only
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

**Status:** implemented (`meegxer.gui`, `meegxer-gui`). Spec 0.2.2 added `primary`
parameters, per-node `strip_params` and the adaptable band-pass (`phase`); the
daemon gained `run_offline` / `offline_window` so offline mode stays a pure client.
Every source and processor gets a probe tap, so showing a tap never reloads the
pipeline. Exit criteria are automated in `packages/meegxer-gui/tests/test_gui_exit.py`
(50 Hz line in difference mode; SIGKILL of the GUI leaves the daemon processing).
Screenshots: `uv run python tools/gui_screenshot.py OUT_DIR`. Not yet: electrode
positions from the recording (the standard 10-05 table is used), an incremental
offline view for very long recordings, and a check on real displays and Windows.
Found and fixed on the way: the memory probe kept every block in live runs.

- PySide6 + pyqtgraph client connecting via control + display channels only
  (layout and behaviour: `docs/DESIGN.md` §3.7).
- Horizontal chain of strips: + Add strip (end and gaps), drag to reorder,
  insert-anywhere probe taps. Strips show name, status LED and parameters: crucial
  (`primary`) ones in friendly mode, up to 3 user-picked ones in advanced mode.
- Auto-generated parameter controls from manifests in the inspector, with a friendly
  mode (hides `advanced` params) and an advanced mode.
- Mode set by the Input strip (LSL = live; file/BIDS/synthetic played live or
  processed offline, default offline for files); incompatible strips
  reddish and blocking; adaptable band-pass (`phase`: auto/causal/zero).
- Signal view: single trace area, channel selector (all by default), colour wheel
  channel colours, Main thick / comparisons faded; overlay, difference and spectrum
  modes; stacked traces in a separate window.
- Per-stage status LEDs, CPU/latency/overrun meters.
- Per-node "pass through while uncalibrated" checkbox, a clear-calibration action, and
  a visible mark on probe views showing `bypassed` data.
- Themes from colour tokens; dark (default) and light.

**Exit:** a user can build Synthetic → HP → Probe → Notch → Probe, see the 50 Hz
line disappear in difference mode, and kill the GUI without interrupting processing.

## M5.1 — Recording settings and long offline recordings

**Goal:** describe real recordings correctly and view long ones without waiting.

**Status:** implemented. Spec 0.2.3 `recording` overrides on the source (channel
rename, type, bad flag, position; montage; events file, renames, drops);
`SignalInfo.ch_types` and `positions`; daemon `describe_source` and `list_montages`;
the GUI's Recording window, with a channel-location lookup by name (standard
montages or files, including EEGLAB `.ced`; EDF-style names such as `EEG Fp1-REF`
and old 10-20 names match); channel colours from the recording's positions; events
on the offline view. Offline probes write samples and zoom levels to memory-mapped
files as the run goes (`MEEGXER_OFFLINE_DIR`), with a size-capped disk cache for
processor outputs (`MEEGXER_OFFLINE_CACHE_GB`). Not yet: events routed through the
pipeline (Events port) and written by the file sinks; a sampling-rate override for
formats without one.

**Non-EEG channels** (spec 0.2.4, after M5.1): files keep EOG, ECG, EMG, misc and
respiration channels (trigger channels stay out; events come from annotations).
Every processor works on EEG only unless its `channel_types` lists more types; the
others pass through it untouched, in place. Channels without a type count as EEG.
The average reference therefore ignores non-EEG channels, and ICA (M6) includes
EOG/ECG/EMG only when its strip says so. The signal view hides non-EEG channels by
default.

## M6 — Heavy built-ins and features

**Goal:** the processing that makes MeegXer useful for real data.

- ASR as the EEGLAB plugin `eeglab.asr`; ICA → ICLabel → ICA rejection. (wICA
  dropped for v1.)
- Features (analysis processors, Features port): PSD, band power, Hilbert envelope,
  connectivity (mne-connectivity), complexity (antropy).
- Each: offline first, then streaming where meaningful; test 1 where stateful.
- **Acceptance test 6:** concordance vs EEGLAB references on shared data; report
  per built-in.

**Delivery:** three PRs. M6a: ASR plugin plus delayed-output support in the core;
M6b: ICA, ICLabel and component rejection; M6c: features.

**M6a status:** implemented. `packages/meegxer-plugins/plugins/eeglab_asr/` runs
EEGLAB's `asr_calibrate`/`asr_process` (clean_rawdata, fetched at a pinned commit when
the image is built, never stored in this repository) in Octave. Octave's signal
package has no `yulewalk`, so the plugin carries its own modified Yule–Walker
(`mx_yulewalk.m`, GPL), which matches EEGLAB's precomputed filter tables to 6e-11.
The plugin processes fixed `step_size` chunks, so its output does not depend on block
sizes, and reports output with ASR's look-ahead delay (half a window) through the
new result `sample_index` and `flush` (plugin protocol 0.2, spec 0.2.5); EEGLAB
prints to stdout, so the image moves the protocol to fd 3 (`MEEGXER_PROTOCOL_FD`).
Acceptance test 6 for ASR: streamed in 500-sample blocks and flushed, the plugin's
output equals `clean_asr` run in EEGLAB on `tests/data/m6_eeg.npz` (stored reference
`tests/data/m6_ref_asr.npz`, made by `tools/make_m6_references.py`); conformance
(test 1, restart) passes in the sandbox.

**M6b status:** implemented. `meegxer.nodes.ica` ("ICA + ICLabel" strip) fits
extended Picard (default) or Extended Infomax through MNE, classifies the components
with ICLabel (mne-icalabel, ONNX backend) and removes those with eye, muscle, heart,
line-noise or channel-noise probability ≥ 0.9. Removal is a fixed spatial filter, so
the strip streams after calibration (`availability: both`); threshold and classes
are live parameters and need no refit (for fitted nodes, live parameters now stay
out of the model's fingerprint, spec 0.2.6). Bad channels stay out of the
decomposition; positions are required (ICLabel's topographies). Components are
scaled to RMS microvolts as EEGLAB's `eeg_checkset` does, which ICLabel's spectrum
feature depends on. Acceptance test 6, on `tests/data/m6_ica.npz` (19 channels,
known brain and artifact sources, `tools/make_m6_ica_testdata.py`) against
`tests/data/m6_ref_ica.npz` (`tools/make_m6_ica_references.py`, EEGLAB in Octave):
on the same decomposition, ICLabel features equal EEGLAB's to float32 precision and
probabilities to 4e-6; Extended Infomax matches `runica` (matched components,
median |r| 1.000, min 0.98) and extended Picard nearly so (median 0.996, min 0.93).
The strip writes the ICs rail (activations with ICLabel classes, `docs/RAILS.md`), which later strips can read as their input. Tapping the ICs rail shows the
components, coloured by ICLabel class.

**M6c status:** implemented. Analysis processors return the windows a block
completes as a tuple of FeatureBlocks; both schedulers route them to Features sinks
(feature TSV and the new `meegxer.core.lsl_features_out`), and acceptance test 1 has
a windows variant. Nodes in `meegxer.nodes.features`: PSD (Welch, as MATLAB's
pwelch runs it for EEGLAB's spectopo), band power (named bands, absolute or
relative, optional dB), connectivity (mne-connectivity: coh, imcoh, PLV, wPLI in one
band, the window cut into 1 s pieces), complexity (antropy: permutation, sample and
spectral entropy, Hjorth mobility and complexity, Lempel-Ziv), and the Hilbert
envelope as a Signal strip computed piecewise with look-ahead (output delayed by it,
same code live and offline). Acceptance test 6: PSD equals Octave's pwelch with
MATLAB's no-detrend setting to 3e-16 (`tests/data/m6_ref_features.npz`,
`tools/make_m6_feature_references.py`); the envelope stays within 1 % of a
whole-recording `hilbert` away from the recording's ends (median 0.03 %);
connectivity and complexity equal direct mne-connectivity/antropy calls and give
the expected answers on signals with known coupling and regularity. Tapping a feature
rail shows its feature view (a spectrum, or values over time; live and offline).
Not yet: a strip's channel layout changing mid-stream stops the feature outlets.

**Decided**
- Features → Sink only in v1 (view, export, LSL feature stream); Features → Processor
  later, so the v1 scheduler needs no multi-rate processor edges.

## M7 — Reproducibility and release gate

- Session JSON (graph, node versions, params, fitted models or their hashes, spec
  version, image digests): the unit of reproducibility. **Done:** `spec/session.md`
  (format 1.0, spec 0.2.9): models as readable JSON, plugin provenance (image,
  image ID, registry digests), path-independent file fingerprints, deterministic
  `.set` writer; CLI `run --session`, `session check/save`; daemon
  `get_session`/`load_session`; GUI *Open…* reads sessions and *Save session…*
  writes them (offline runs reuse the previous run's or the session's models).
- BIDS-derivatives output layout with provenance. **Done:**
  `spec/bids-derivatives.md` (spec 0.2.10): `meegxer.core.bids_writer` and
  `meegxer.core.bids_features` (offline sinks; entities from a BIDS input or a
  BIDS-style file name), sidecars, `dataset_description.json` with `GeneratedBy`,
  the run's session file per recording; CLI `--bids-out` and a sequential
  `--bids` loop over a dataset (skips finished recordings, continues past
  failures). Parallel workers, a job queue and QC reports: roadmap item 2.
- **Acceptance test 4:** save → reload on a second machine → identical output file.
  **Passing** (`packages/meegxer-cli/tests/test_cli_session.py`, ICA model reused,
  byte-identical `.set`).
- **Acceptance test 5:** full suite on Linux and Windows (Docker Desktop / WSL2). Procedure and
  checklist: `docs/TESTING.md` (with the real-data BIDS check). **Windows part A
  passed** (2026-10-17): static checks, tests, plugin conformance and ASR
  concordance in Docker, GUI with a plugin; 30 min at 64 ch / 1 kHz with a plugin,
  zero gaps or duplicates, end-to-end p95 15.4 ms
  (`docs/results/a7-windows-30min-plugin.json`).
- Publish to PyPI: the five `meegxer-*` packages plus the `meegxer` bundle (it also
  carries the CLI, GUI and plugin commands so `uv tool install meegxer` exposes
  them). Since 0.2.1 the bundle installs every package, plugins included
  (decision 40; before, it was the "sterile set" without the plugin runtime).
- Release path: 0.2.1a8 is the last alpha (everything in `docs/DESIGN.md` §5
  is in). **0.3.0b1**, the first beta, once the manual checks of
  `docs/TESTING.md` pass (test 5 on Windows with Docker, real BIDS data); then
  fixes from the beta, and **1.0.0** once tests 1–6 are green. Release notes:
  `docs/RELEASE_NOTES.md`.

---

## M8 — Sterile mode and local plugins

0.2.1a1 (done): the rename and rule rewrite of decision 40, with no change in
behaviour (plugins still run in containers).

Next, in order:
1. **Local plugins and custom scripts by default.** Done (0.2.1a2): consent in the
   plugin index, notice shared by CLI and GUI, `sha256` provenance, the offline
   Python tier (`docs/CUSTOM_SCRIPTS.md`), *Add a custom script…* in the GUI. Live Python
   tier done (0.2.1a3): `transform(data, state, ...)` + optional `fit(raw, ...)`,
   conformance checked when added. Credit and custom grouping done (decision 41).
   Feature-extraction scripts done: `features(data, ...)` per window, with Features
   results in the plugin protocol (spec 0.2.14; Python SDK and adapter, the Octave
   SDK gains them with the Octave tier). Committed next: the Octave/EEGLAB tier
   (`function EEG = process(EEG, ...)`, then live); "Enable Sterile mode" in the
   prompt once Sterile mode exists.
2. **Sterile mode.** A processing-worker image with the built-in nodes; the whole
   chain runs in one container (one hop per block); the "Sterilize" switch in the
   GUI and the CLI; delay measured with `tools/stability_test.py`.
3. Later: "sterilize at startup" in the options.

**Exit:** a pipeline with a custom script runs locally after the prompt, and the
same pipeline runs in Sterile mode with no network inside the container and
identical output.

---

## Beta to 1.0: M9–M12

Full plan, decisions and item lists: `docs/ROADMAP.md`. One PR per item, by ID.

| Version | Milestone | Exit |
|---|---|---|
| 0.3.0bN | **M9 Fix and harden**: beta triage, Windows type check in CI, control-channel token, supply-chain checks, share-safe sessions, feature units, robustness, project hygiene | No known data-loss or crash bugs; security basics done |
| 0.4.0bN | **M10 Research workflows**: resample and pick, epoching, events writer, ERP view, bad segments, XDF input, EEGLAB decompositions (4b), MNE Python API, batch workers and QC reports, templates, undo | An ERP and a resting-state study end to end, GUI and CLI |
| 0.5.0bN | **M11 Isolation and BCI**: Sterile mode (M8 item 2), plugin hardening, LDA, armed keyboard sink, Octave script tier, shortcut creation | Sterile output identical to local; SSVEP → keypress demo |
| 1.0.0rcN | **M12 Freeze and validate**: spec freeze, validation report, ERP CORE replication, interoperability and soak tests, cross-OS reproducibility, test 5 in CI, docs site, usability round, JOSS | Acceptance tests 1–6 green on Linux and Windows |

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
| 8 | Param change vs streaming state | Reset by default; opt-in carry-over in M2 | M2 |
| 9 | Manifest additions | Typed `inputs`/`outputs`, `spec_version`, enum `choices` | M1 |
| 10 | Initial streaming state | `state` is `None` on first block; no `initial_state()` | M1 |
| 11 | Distribution (superseded by 40) | Five `meegxer-*` names + `meegxer` sterile bundle; research builds install `meegxer-plugins` by name (no `[full]` extra: rule 10 forbids the bundle declaring the runtime) | M7 |
| 12 | Filter start-up | Steady-state initialisation from the first sample | M2 |
| 13 | Offline filtering | Zero-phase (forward-backward) nodes; every node declares `availability` | M2 |
| 14 | Friendly/advanced UI | Per-parameter `advanced` flag; friendly mode uses defaults | M2, M5 |
| 15 | Bad channels | `SignalInfo.bads`; average reference `exclude`/`exclude_bads` options | M2 |
| 16 | Calibration across restarts | Keep fitted models by default; drop explicitly (`clear_calibration`) | M3 |
| 17 | Uncalibrated nodes | Block by default; per-node opt-in to pass through, marked `bypassed` (checkbox in M5) | M3, M5 |
| 18 | Plugin state ownership | Host owns model and state; workers are restartable; models loaded once by hash | M4 |
| 19 | Container engine access | Docker/Podman CLI (no Docker SDK dependency); locked-down `run` flags | M4 |
| 20 | Unsandboxed plugins | Development and tests only, behind `MEEGXER_ALLOW_UNSANDBOXED=1` | M4 |
| 21 | GUI chain layout | Horizontal strips, one per graph node (reused by the later node editor) | M5 |
| 22 | Strip parameters | Friendly: crucial params (manifest `primary`); advanced: up to 3 user-picked, saved in the pipeline | M5 |
| 23 | Live/offline in the GUI | Set by the Input strip; incompatible strips reddish and blocking; only the band-pass adapts (`phase` auto) | M5 |
| 24 | Signal view and colours | Single trace area (all channels by default) + stacked window; uniform OKLCH colour wheel on the scalp (neutral at Cz: white dark theme, near-black light theme) | M5 |
| 25 | Themes | Colour-token themes; dark default, light included | M5 |
| 26 | Recording settings | Separate Recording window from the Input strip; stored as the source's `recording` overrides (spec 0.2.3) | M5.1 |
| 27 | Long offline recordings | Probes write memory-mapped samples + zoom levels while the run goes; size-capped disk cache; no chunk-major batch order | M5.1 |
| 28 | Non-EEG channels | Kept; every strip processes EEG only unless it lists more types (`channel_types`, spec 0.2.4); reference ignores them; hidden in the view by default | M5.1 |
| 29 | ASR | EEGLAB's ASR as a research-build plugin (`eeglab.asr`); its code is UC-licensed for non-commercial use only, so it is fetched at image build and never shipped in MeegXer; no ASR in the sterile build | M6a |
| 30 | Look-ahead processors | Output may lag its input by sample index, plus an end-of-stream `flush`; the stage delays untouched channels to match (spec 0.2.5, plugin protocol 0.2) | M6a |
| 31 | ICA | Picard (extended) by default, Extended Infomax as a strip setting; EEG only unless the strip includes EOG/ECG/EMG | M6b |
| 32 | Component rejection | Reject when ICLabel's eye, muscle, heart, line-noise or channel-noise probability is ≥ 0.9 | M6b |
| 33 | EEGLAB references | Stored in `tests/data/` with the generating script, so CI needs no EEGLAB | M6 |
| 34 | Live params on fitted nodes | Act in `transform` only and stay out of the model fingerprint, so e.g. the ICA threshold changes without a refit (spec 0.2.6) | M6b |
| 35 | Feature windows | An analysis processor returns a tuple of FeatureBlocks, one per completed window (or none); sinks get them one by one | M6c |
| 36 | Features in a linear chain | Branch off: an omitted `input` takes the nearest earlier node with a matching output type (spec 0.2.7); a feature strip's tap is on its rail (a Features probe), shown alone in the feature view | M6c |
| 37 | Hilbert envelope | Signal strip, piecewise analytic signal with look-ahead (delayed output), one implementation live and offline | M6c |
| 38 | First feature measures | Connectivity: coh, imcoh, PLV, wPLI; complexity: permutation/sample/spectral entropy, Hjorth, Lempel-Ziv | M6c |
| 39 | PyPI releases | Six projects (five packages + `meegxer` bundle = sterile build), one shared version from a `v*` tag, Trusted Publishing; tag → TestPyPI, PyPI by manual approved run (`docs/RELEASING.md`) | — |
| 40 | Sterile = isolation (0.2.1) | "Sterile" names a runtime mode, never a package: Sterile mode runs all processing (built-ins and plugins) in one no-network container, the host keeps trusted I/O; auditing stays the user's responsibility. Packages renamed `meegxer-sterile` → `meegxer-nodes`, `meegxer-runtime` → `meegxer-plugins`; the `meegxer` bundle installs everything; `meegxer.sterile.*` node names still resolve. Rules 9 and 10 rewritten; the sterile wheel check is gone, an import-linter contract keeps plugins reachable only through entry points | M8 |
| 41 | Credit and custom grouping (0.2.12) | Manifest `authors`, `references` (citation + DOI or URL), `license`; built-ins cite methods and software with checked DOIs; shown in the inspector, recorded in sessions, gathered into BIDS `ReferencesAndLinks`; informative, never a gate. Plugins and scripts are grouped as "Custom <group>" by their data types (`origin` in `list_nodes`) | M8 |
| 42 | Scope of 1.0 (`docs/ROADMAP.md`) | Sterile mode and epoching/ERPs are in 1.0; install is shortcut creation only; LSL is the only device path; a JOSS paper with 1.0, aiming for a methods paper; versions 0.3.0bN → 0.4.0bN → 0.5.0bN → 1.0.0rcN | M9–M12 |
