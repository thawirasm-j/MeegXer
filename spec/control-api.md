# MeegXer control API (0.2.20, unstable)

JSON-RPC 2.0 over a ZeroMQ REQ/REP socket (CLAUDE.md rule 3). The daemon binds
`tcp://127.0.0.1:5555` by default. Every request is one JSON object; every reply is
one JSON object with either `result` or `error`.

## Authentication (0.2.4)

The daemon accepts a connection, on this socket and on the display socket
(`display-protocol.md`), only with its token, checked by ZeroMQ's PLAIN mechanism
(ZAP): username `meegxer`, password the token. The messages themselves carry no
credentials.

- When it starts, the daemon makes a random token (32 bytes, URL-safe base64) and
  writes it to `<config folder>/daemon/<endpoint>.token`, where `<endpoint>` is
  the bound control endpoint with every run of non-alphanumeric characters
  replaced by `_` (`tcp_127_0_0_1_5555.token`). The file is readable by the user
  only (mode 0600 in a 0700 folder; on Windows the config folder is private to
  the account). The daemon removes it when it exits.
- Clients take the token from `MEEGXER_TOKEN` if set, else from that file, and
  read it again when they reconnect (a restarted daemon has a new token).
- A client without the right token is not let in: its request times out. MeegXer's
  clients then say whether they had no token file.

PLAIN is not encrypted: the token keeps other users and programs on the same
computer out, but on another interface anyone on the network path could read it.
Bind to a non-local interface only on a trusted network (CurveZMQ comes with the
remote GUI).

```json
{"jsonrpc": "2.0", "method": "set_params", "params": {"node": "hp", "params": {"low": 0.5}}, "id": 7}
{"jsonrpc": "2.0", "id": 7, "result": {"kind": "highpass", "low": 0.5, "high": 40.0, "order": 4}}
```

`params` is always an object (named parameters).

## Methods

| Method | Params | Result |
|---|---|---|
| `ping` | – | `{version, spec_version, display}`: `display` is the display endpoint |
| `rescan_nodes` | – | installed node manifests (as `list_nodes`), after looking again for nodes installed since the daemon started (e.g. a plugin just added). A running pipeline keeps the classes it was built with |
| `list_nodes` | – | installed node manifests: `name, version, role, type, availability, requires_fit, requires_positions, fit_events, latency_samples, description, authors, references, license, category, channel_types, live_params, inputs, outputs, params[]`, plus `origin`: `builtin`, `plugin` (a container image) or `custom` (a local plugin or custom script) (each param: `name, type, default, min, max, step, choices, unit, description, advanced, primary`) |
| `load_pipeline` | `pipeline` (see `pipeline.schema.json`) | the normalised pipeline. Refused while running, or if any node cannot run live (offline-only, or a band-pass with `phase` `zero`). Fitted models of nodes with the same id, name and version are kept (as on a restart) |
| `get_pipeline` | – | current pipeline, including live parameter changes |
| `get_model` (0.2.5) | `node`, `offline` (optional) | `{node, model}`: the node's fitted model as readable JSON (as in a session, `session.md`), or `model: null` without one; of the loaded pipeline, or with `offline` true of the last offline run (the GUI reads ICA's unmixing from it for the IC head plot) |
| `get_session` | `name`, `pipeline`, `offline` (all optional) | a session (`session.md`) of the loaded pipeline with the models calibrated live, or with `offline` true of the last finished offline run (its fitted models and written files). With `pipeline` (e.g. the GUI's), that pipeline is saved with the models of the nodes it shares. Paths absolute (the client writes the file, making paths relative to it) |
| `load_session` | `session`, `allow_version_mismatch` (default false) | `{pipeline, models, dropped_models, warnings}`: loads the pipeline with its fitted models. Paths must be absolute (the client resolves them against the session file). Refused while running, and on a node version or plugin image mismatch unless allowed (that node's model is then dropped). An offline-only pipeline loads too (`start` then says why it cannot run live). The models seed the next `run_offline`; a model is used, live or offline, only if its fingerprint matches the input |
| `start` | – | status. A restart builds fresh sources and sinks from the current pipeline and keeps fitted models (drop them with `clear_calibration`). A node with `requires_positions` whose good channels lack electrode positions stops the run before any data flows, with the reason in `errors` (offline runs: `state` `error`) |
| `stop` | – | status |
| `status` | – | see below |
| `set_params` | `node`, `params` | the node's full parameter set after the change. Processors only while running. Parameters in the node's `live_params` keep streaming state; others restart it |
| `calibrate` | `node`, `seconds` | status. Buffers `seconds` of the node's input, fits, then the node runs. An input that ends before `seconds` is reported in `errors` and leaves the node uncalibrated |
| `clear_calibration` | `node` (optional) | `{cleared, status}`: drops the fitted model of one node, or of every node when `node` is omitted |
| `reset_probe` (0.2.20) | `probe` | status. Starts a live ERP tap's running averages over (the ERP view's *Reset*; `docs/RAILS.md`). Refused when the pipeline is not running or the probe is not an ERP tap |
| `set_pass_uncalibrated` | `node`, `enabled` | status. Per-node opt-in: while the node has no valid model, pass data through unprocessed (marked as bypassed) instead of blocking |
| `set_strip_params` | `node`, `params` (up to 3 names) | the pipeline. Display only: which parameters the node's strip shows in the UI's advanced mode |
| `describe_source` | `node` (a source entry of a pipeline, with optional `recording`) | `{sfreq, n_samples, duration_s, stream_id, channels, events, event_labels, events_origin, recording}`. `channels`: per channel `{label, original, type, bad, position}` after the overrides (`original` is the label the source reports; `position` is `[x, y, z]` in metres or `null`). `events`: `{onset, duration, label}` (seconds) after renaming and dropping; `event_labels`: the event names before renaming; `events_origin`: `recording`, `file` or `none`. Opens the file or stream to read it. A BIDS input with an empty `subject` (every subject, 0.2.7) describes its first recording |
| `list_montages` | – | names of the standard montages `recording.montage` accepts |
| `lookup_positions` | `labels`, `montage` (standard name or montage file, incl. EEGLAB `.ced`) | `{positions: {label: [x, y, z]}, unmatched: [label]}`: positions by channel name, like EEGLAB's channel-location lookup. Matching tries the label, then case-insensitively, then without a type prefix or reference suffix (`EEG Fp1-REF` → `Fp1`), then the 10-20 name of old labels (T3 → T7). Bipolar labels do not match |
| `run_offline` | `pipeline`; `reviewer` (0.2.11, optional: who reviews at its review strips, saved with each review and in the session) | offline status (below). Runs the pipeline over its whole, finite input in the background with the batch scheduler (offline variants such as the zero-phase band-pass apply). Its sinks write files. Its probes write their samples and min/max zoom levels to memory-mapped files in a temporary folder (`MEEGXER_OFFLINE_DIR`) as the run goes, so they can be viewed before it ends; the previous run's files are deleted. Processor outputs go to a disk cache capped at `MEEGXER_OFFLINE_CACHE_GB` (default 5). Fitted models of the previous offline run (or of the session loaded since) are reused where their fingerprint matches, otherwise refitted. Refused while the live pipeline runs or another offline run is in progress. **All subjects (0.2.7):** a BIDS input (`meegxer.core.bids`) with an empty `subject` stands for every recording of its dataset that matches its task, session and run; they run one after another as `meegxer run --bids` does (`spec/bids-derivatives.md`): each with a fresh graph and its own fitted models, its outputs in the dataset's BIDS derivatives (a BIDS writer is added if the chain has none; other file writers are left out, with a note in `skipped`), a recording whose session file exists skipped, one that fails not stopping the others. The probes show the recording being run, and at the end the last one (run even when finished, its files kept). The status's `recordings` is then `{index, total, current, done, skipped, failed}` (`index` counts from 1; `failed` entries carry the error). **0.2.18:** with the `general.batch_jobs` setting (or `MEEGXER_BATCH_JOBS`) above 1 (0: one per CPU), all recordings but the last run in that many worker processes first, as `meegxer run --bids --jobs N` does (`index` then counts those finished); one that stops at a review strip without saved marks runs again in the daemon, where the review can be answered, and the last runs in the daemon for the views. Each recording run gets a QC report, and at the end `recordings.summary` names the dataset's `qc.html` (`spec/bids-derivatives.md`) |
| `offline_window` | `probe`, `start`, `n_samples`, `max_bins` (default 2000, ≤ 20000) | `{probe, ch_labels, sfreq, bin_size, start_bin, mins, maxs}`: min/max of samples `[start, start + n_samples)` of one probe of the current offline run in at most `max_bins` bins (`bin_size` 1 when zoomed in to single samples), clipped to what has been written; `mins`/`maxs` are channel × bin lists in volts (`null` where empty). Bin `k` covers samples `[k * bin_size, (k + 1) * bin_size)`, as on the display stream. **`display` (0.2.12, optional):** `{montage, highpass, lowpass, notch, bads}` (any of them): the page through a display montage (`as recorded`, `average`, `longitudinal bipolar`, `transverse bipolar`, `laplacian`) and zero-phase display filters in Hz (0: off), computed from the raw samples, padded so the filter settles, each stretch between cuts on its own; `bads` (labels) stand for the stream's bad channels, left out of the average and the Laplacian. `ch_labels` are the montage's (`Fp1-F7` for a bipolar pair). The reply adds `derived` (true); a page longer than 120 s is the plain envelope with `derived` false |
| `import_review_open` | `node` (a file or BIDS source entry) | (0.2.14) the review window's request to review that recording at import (`docs/REVIEW.md`): `{node: "import", file, reason: "import", marks, labels, duration_s, reject, display, points, viewed, bads, crop, reviewer, reviewed}`: the raw file's events and bad channels, or those of the review saved before with its crop (`[start, end]` s, end `null` to the end). Refused for inputs without a file |
| `save_import_review` | `node`, `marks`, `bads`, `crop`, `viewed`, `reviewer` | `{file}` (0.2.14): saves the review at import; file and BIDS inputs apply it from then on (their `import_review` setting) |
| `offline_events` | `node` (optional) | `{events}` (0.2.13): the events of the current offline run so far, `{onset, duration, label, origin}` with times in seconds from the recording's start: the input's and those of every strip that has run (a review's marks: `origin` `review`), or with `node` those that node sees (the input's and its upstream strips'). The views mark them; a run paused at a review has those before the strip |
| `offline_features` | `probe`, `start` (default 0), `stop` (default: the end), `max_windows` (default 2000), `reduce` (`none` or `mean`) | `{probe, dims, coords, n_windows, windows, values}` (plus `unit`, or `unit_dim` and `units`, when the node states them; spec 0.2.19): the windows of one Features probe of the current offline run that end in `(start, stop]` (source samples). `n_windows` counts them; `windows` lists `[start, stop)` and `values` the windows' values (nested lists by `dims`, `null` for NaN): at most `max_windows` of them, evenly picked, or with `mean` one entry, their mean over the whole range ; epochs (0.2.10): `labels` gives each returned window's event label |
| `offline_erp` (0.2.20) | `probe` | `{probe, dims, coords, values, counts, unit}`: the running ERP of one ERP tap of the current offline run, as its display frame carries it (`dims` `["condition", "channel", "time"]`, `values` nested by `dims`, `counts` the epochs of each condition); `dims` is `[]` before the first epoch |
| `submit_review` | `marks` (list of `{onset, duration, label}`, seconds from the recording's start; 0.2.17: `suggestion` on a mark answering one, and a suggestion answered no as a mark with it and `verdict` `rejected`); `viewed` (0.2.9, optional: the spans looked at, `[[start, stop], …]` in seconds); `bads` (0.2.12, optional: channels marked bad, labels); `epochs` (0.2.15, an epoch review: `{onset, duration, label, keep, event}` per epoch, as in the request; `marks` is then `[]`) | offline status. **0.2.8:** answers the review strip the offline run waits at (`state` `review`): the marks are saved to its sidecar (`docs/REVIEW.md`) with the fingerprint of the strip's input, put on the Events rail, and the run goes on. `viewed` is saved too and comes back in the next request for that recording. `bads` are saved too and become the stream's bad channels after the strip. **0.2.17:** each suggestion's verdict is worked out and saved (`docs/REVIEW.md` step 4b: `accepted`, `edited`, `rejected`; a `bad_channel` one is accepted when its channel is in `bads`); with `decision` `verify` a suggestion without an answer is refused, with `free` it is rejected. Refused when no review is waiting |
| `offline_review_epochs` | `indices` (optional: epochs by their index in the request's `epochs`; default all), `reduce` (`none` or `mean`), `max_samples` (default 1000) | (0.2.15) the data of the epochs a Review epochs strip waits on: `{times, ch_labels, unit, epochs: [{index, data}]}` (`data` channels x times, every k-th sample so at most `max_samples` per epoch), or with `reduce="mean"` `{times, ch_labels, unit, mean, n}` (the ERP of those epochs). Refused when no epoch review is waiting |
| `cancel_review` | – | offline status. **0.2.8:** stops the offline run waiting at a review strip (`state` `error`, `error` says it was cancelled); nothing is saved |
| `stop_offline` (0.2.19) | – | offline status, at once, with `stopping` true. Ends the offline run: the recording being run stops before its next strip (a strip runs over the whole recording, so a file is never half written; that recording has no session file and is done again by a later run), worker processes are stopped, and no other recording starts; a run waiting at a review stops too. The state turns `stopped` (not `error`) when the strip it was on is done; what finished before is kept, with the QC summary. Refused when there has been no offline run |
| `unfinished_recordings` (0.2.6) | – | the recordings a run ended before writing (it crashed): `[{path, format, channels, seconds, modified}]`, from the spools the file writers leave (`path`: the file it was writing; `format`: `eeglab` or `edf`; `modified`: ISO 8601, UTC). A spool a running process still writes is not listed, nor (0.2.16) one being recovered. The GUI offers to recover them when it starts |
| `recover_recording` (0.2.6) | `path`, `discard` (default false) | `{path, state}`: starts writing an unfinished recording to `path`, or to `<name>_recovered<suffix>` when `path` exists, and removing its spool; `path` is the file being written. **0.2.16:** it is written in the background (a long recording takes longer than a client waits for a reply): `state` is `running`, and `recoveries` says when it is done. With `discard` deletes the spool at once (`path: null`, `state` `deleted`). Refused for a path `unfinished_recordings` does not list (so also while it is being recovered) |
| `recoveries` (0.2.16) | – | the recordings `recover_recording` started writing since the daemon started: `[{path, output, state, error}]`, `state` `running`, `done` or `failed` (`error` says why; the spool is kept, so `unfinished_recordings` lists it again) |
| `shutdown` | – | `"bye"`; the daemon exits |

## Status

```json
{
  "running": true, "uptime_s": 12.3, "blocks": 1180, "samples": 12300,
  "overruns": 0, "events": 0, "errors": [],
  "latency": {"mean_ms": 2.1, "p95_ms": 3.4, "max_ms": 9.8},
  "display": {"sent": 370, "dropped": 0},
  "source_gaps": 0, "source_duplicates": 0, "source_first_index": 1223,
  "nodes": [
    {"id": "hp", "node": "meegxer.nodes.iir_filter", "role": "processor",
     "status": "ready", "params": {"low": 1.0}, "timing": {"blocks": 1180, "mean_ms": 0.15, "max_ms": 2.1}}
  ]
}
```

- Node `status`: `ready`, `uncalibrated`, `calibrating`, `needs-refit` (processors), or
  `stopped` before the first start. Nodes that need calibration also report
  `pass_uncalibrated`, and, while stopped, `calibrated` (whether a model is kept).
- Without a valid model a node blocks: nothing passes it (default, rule 7). With
  `pass_uncalibrated` the data passes unprocessed and every downstream block lists the
  node in `bypassed`; display frames carry the same list.
- `overruns`: blocks whose processing took longer than their duration.
- `latency`: clock at the sinks minus the LSL timestamp of the block's last sample.
- `source_*`: present when the source stream carries a `SAMPLE_INDEX` channel.
- `offline`: present after `run_offline`: `{state, error, skipped, elapsed_s, n_samples,
  sfreq, models, probes, features, recordings, review, stopping}` (0.2.19:
  `stopping` while a stopped run finishes its strip), where `state` is `running`,
  `review` (0.2.8: the run waits at a review strip for `submit_review` or
  `cancel_review`; `review` is then `{node, file, reason, marks, labels,
  duration_s, reject, viewed, bads, display}`: `reason` `not reviewed` or `input changed`, `marks` those made on
  another input, to confirm, `labels` the strip's and the input's event types,
  `reject` (0.2.9; 0.2.11: the labels the window's Reject key gives, cut by a
  Reject segments strip downstream), `viewed` (0.2.9) the
  spans looked at in the earlier review, `bads` (0.2.12) the channels
  marked bad then, `display` (0.2.12) the strip's display presets: any of
  `montage`, `highpass`, `lowpass`, `notch`, `sensitivity` (µV/mm), `page`
  (s), the display the window opens with (the reviewer can change it),
  `points` (0.2.13) the review points: the events the strip names, `{onset,
  duration, label}`; 0.2.17: the suggestions, each `{onset, duration, label,
  origin}` (`origin` the strip that proposed it, `input` the recording's own;
  `channel` too for a `bad_channel`), and `decision` (0.2.17) `free` or
  `verify`; a mark answering a suggestion carries it as `suggestion` (until
  0.2.17: its onset as `point`), in `marks` and in `submit_review`'s marks,
  and a mark from an earlier review its `verdict`, `kind` (0.2.15) `marks`, or `epochs`
  for a Review epochs strip, whose request also has `epochs`: one
  `{onset, duration, label, keep, event}` per epoch (its span in seconds, the
  label and verdict to start from, the label it was cut around), the data
  from `offline_review_epochs`;
  otherwise `null`), `done`, `error` or (0.2.19) `stopped` (`stop_offline`),
  `skipped` lists the saves a writer skipped because its file existed (without
  `overwrite`; the run still finished), `recordings` (0.2.7) is empty except for
  an all-subjects run (below), `n_samples` is the length of the input and
  `sfreq` its rate, `models` lists the
  nodes with a fitted model once the run is done (offline runs fit on the whole
  input, or reuse a matching model), `probes` maps each probe id that has data to
  `{ch_labels, sfreq, first_sample, n_filled, n_samples, bypassed, ch_classes,
  ch_scores, filters}` (`filters` as in the display protocol; `n_filled` grows while the run goes; the classes are empty except
  on a rail of components), and `features` maps each Features probe that has
  windows to `{dims, coords, n_windows}` and its units as in `offline_features` (read them with `offline_features`).

## Errors

| Code | Meaning |
|---|---|
| -32700 | not valid JSON |
| -32600 | not a JSON-RPC 2.0 request object |
| -32601 | unknown method |
| -32602 | invalid params (wrong names or types) |
| -32000 | the request was understood but failed; `message` says why |
