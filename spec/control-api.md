# MeegXer control API (0.2.3, unstable)

JSON-RPC 2.0 over a ZeroMQ REQ/REP socket (CLAUDE.md rule 3). The daemon binds
`tcp://127.0.0.1:5555` by default. Every request is one JSON object; every reply is
one JSON object with either `result` or `error`. The socket has no authentication:
bind it to a non-local interface only on a trusted network.

```json
{"jsonrpc": "2.0", "method": "set_params", "params": {"node": "hp", "params": {"low": 0.5}}, "id": 7}
{"jsonrpc": "2.0", "id": 7, "result": {"kind": "highpass", "low": 0.5, "high": 40.0, "order": 4}}
```

`params` is always an object (named parameters).

## Methods

| Method | Params | Result |
|---|---|---|
| `ping` | – | `{version, spec_version, display}`: `display` is the display endpoint |
| `list_nodes` | – | installed node manifests: `name, version, role, type, availability, requires_fit, description, live_params, inputs, outputs, params[]` (each param: `name, type, default, min, max, choices, unit, description, advanced, primary`) |
| `load_pipeline` | `pipeline` (see `pipeline.schema.json`) | the normalised pipeline. Refused while running, or if any node cannot run live (offline-only, or a band-pass with `phase` `zero`) |
| `get_pipeline` | – | current pipeline, including live parameter changes |
| `get_session` | `name` (optional) | the loaded pipeline as a session (`session.md`), with the models calibrated so far; paths absolute (the client writes the file, making paths relative to it) |
| `load_session` | `session`, `allow_version_mismatch` (default false) | `{pipeline, models, dropped_models, warnings}`: loads the pipeline with its fitted models. Paths must be absolute (the client resolves them against the session file). Refused like `load_pipeline`, and on a node version or plugin image mismatch unless allowed (that node's model is then dropped). A model is used live only if its fingerprint matches this input |
| `start` | – | status. A restart builds fresh sources and sinks from the current pipeline and keeps fitted models (drop them with `clear_calibration`) |
| `stop` | – | status |
| `status` | – | see below |
| `set_params` | `node`, `params` | the node's full parameter set after the change. Processors only while running. Parameters in the node's `live_params` keep streaming state; others restart it |
| `calibrate` | `node`, `seconds` | status. Buffers `seconds` of the node's input, fits, then the node runs |
| `clear_calibration` | `node` (optional) | `{cleared, status}`: drops the fitted model of one node, or of every node when `node` is omitted |
| `set_pass_uncalibrated` | `node`, `enabled` | status. Per-node opt-in: while the node has no valid model, pass data through unprocessed (marked as bypassed) instead of blocking |
| `set_strip_params` | `node`, `params` (up to 3 names) | the pipeline. Display only: which parameters the node's strip shows in the UI's advanced mode |
| `describe_source` | `node` (a source entry of a pipeline, with optional `recording`) | `{sfreq, n_samples, duration_s, stream_id, channels, events, event_labels, events_origin, recording}`. `channels`: per channel `{label, original, type, bad, position}` after the overrides (`original` is the label the source reports; `position` is `[x, y, z]` in metres or `null`). `events`: `{onset, duration, label}` (seconds) after renaming and dropping; `event_labels`: the event names before renaming; `events_origin`: `recording`, `file` or `none`. Opens the file or stream to read it |
| `list_montages` | – | names of the standard montages `recording.montage` accepts |
| `lookup_positions` | `labels`, `montage` (standard name or montage file, incl. EEGLAB `.ced`) | `{positions: {label: [x, y, z]}, unmatched: [label]}`: positions by channel name, like EEGLAB's channel-location lookup. Matching tries the label, then case-insensitively, then without a type prefix or reference suffix (`EEG Fp1-REF` → `Fp1`), then the 10-20 name of old labels (T3 → T7). Bipolar labels do not match |
| `run_offline` | `pipeline` | offline status (below). Runs the pipeline over its whole, finite input in the background with the batch scheduler (offline variants such as the zero-phase band-pass apply). Its sinks write files. Its probes write their samples and min/max zoom levels to memory-mapped files in a temporary folder (`MEEGXER_OFFLINE_DIR`) as the run goes, so they can be viewed before it ends; the previous run's files are deleted. Processor outputs go to a disk cache capped at `MEEGXER_OFFLINE_CACHE_GB` (default 5). Refused while the live pipeline runs or another offline run is in progress |
| `offline_window` | `probe`, `start`, `n_samples`, `max_bins` (default 2000, ≤ 20000) | `{probe, ch_labels, sfreq, bin_size, start_bin, mins, maxs}`: min/max of samples `[start, start + n_samples)` of one probe of the current offline run in at most `max_bins` bins (`bin_size` 1 when zoomed in to single samples), clipped to what has been written; `mins`/`maxs` are channel × bin lists in volts (`null` where empty). Bin `k` covers samples `[k * bin_size, (k + 1) * bin_size)`, as on the display stream |
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
    {"id": "hp", "node": "meegxer.sterile.iir_filter", "role": "processor",
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
- `offline`: present after `run_offline`: `{state, error, elapsed_s, n_samples,
  probes}`, where `state` is `running`, `done` or `error`, `n_samples` is the length
  of the input, and `probes` maps each probe id that has data to `{ch_labels, sfreq,
  first_sample, n_filled, n_samples, bypassed}`; `n_filled` grows while the run goes.

## Errors

| Code | Meaning |
|---|---|
| -32700 | not valid JSON |
| -32600 | not a JSON-RPC 2.0 request object |
| -32601 | unknown method |
| -32602 | invalid params (wrong names or types) |
| -32000 | the request was understood but failed; `message` says why |
