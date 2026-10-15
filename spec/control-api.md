# MeegXer control API (0.2.2, unstable)

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
| `start` | – | status. A restart builds fresh sources and sinks from the current pipeline and keeps fitted models (drop them with `clear_calibration`) |
| `stop` | – | status |
| `status` | – | see below |
| `set_params` | `node`, `params` | the node's full parameter set after the change. Processors only while running. Parameters in the node's `live_params` keep streaming state; others restart it |
| `calibrate` | `node`, `seconds` | status. Buffers `seconds` of the node's input, fits, then the node runs |
| `clear_calibration` | `node` (optional) | `{cleared, status}`: drops the fitted model of one node, or of every node when `node` is omitted |
| `set_pass_uncalibrated` | `node`, `enabled` | status. Per-node opt-in: while the node has no valid model, pass data through unprocessed (marked as bypassed) instead of blocking |
| `set_strip_params` | `node`, `params` (up to 3 names) | the pipeline. Display only: which parameters the node's strip shows in the UI's advanced mode |
| `run_offline` | `pipeline` | offline status (below). Runs the pipeline over its whole, finite input in the background with the batch scheduler (offline variants such as the zero-phase band-pass apply). Its sinks write files; its probes are kept as display envelopes for `offline_window`. Refused while the live pipeline runs or another offline run is in progress |
| `offline_window` | `probe`, `start_bin`, `n_bins` (≤ 20000) | `{probe, ch_labels, sfreq, bin_size, start_bin, mins, maxs}`: bins of one probe from the last offline run, clipped to its data; `mins`/`maxs` are channel × bin lists in volts (`null` where empty), aligned like display frames |
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
- `offline`: present after `run_offline`: `{state, error, elapsed_s, probes}`, where
  `state` is `running`, `done` or `error`, and `probes` maps each probe id to
  `{ch_labels, sfreq, bin_size, first_bin, n_bins, bypassed}` once the run is done.

## Errors

| Code | Meaning |
|---|---|
| -32700 | not valid JSON |
| -32600 | not a JSON-RPC 2.0 request object |
| -32601 | unknown method |
| -32602 | invalid params (wrong names or types) |
| -32000 | the request was understood but failed; `message` says why |
