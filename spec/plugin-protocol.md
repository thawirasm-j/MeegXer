# MeegXer plugin protocol (`meegxer.plugin/0.2`, spec 0.2.x, unstable)

How the MeegXer host talks to an external processing plugin (CLAUDE.md rules 9 and
11). The protocol is language-neutral: the Python and Octave SDKs implement it, they
do not define it. A plugin is a program that reads requests on **stdin** and writes
replies on **stdout**; anything it writes to **stderr** is treated as log text.

A plugin whose own libraries print to stdout (e.g. EEGLAB functions) may move the
protocol to another file descriptor inside its image: when the environment variable
`MEEGXER_PROTOCOL_FD` is set, the SDKs write replies to that descriptor instead. The
image's entrypoint then duplicates the real stdout onto it and points stdout at stderr,
e.g. `sh -c 'exec 3>&1 1>&2; exec PROGRAM'` with `MEEGXER_PROTOCOL_FD=3`. The host
sees no difference.

## Sandbox

The host runs a plugin image as

```
docker run --rm -i --network none --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 2g IMAGE
```

(or the same with Podman). The plugin has no network, no devices and a read-only
filesystem. All data arrives over stdin. Plugins never perform actions: they return
blocks and Events, and trusted host-side sinks act on them.

## Framing

Every message, in both directions, is one frame:

| Bytes | Content |
|---|---|
| 4 | magic `MXP1` (ASCII) |
| 4 | header length `H`, unsigned 32-bit little-endian |
| `H` | header: UTF-8 JSON object |
| 8 | payload length `P`, unsigned 64-bit little-endian |
| `P` | payload: the arrays listed in the header, concatenated |

The header's `arrays` list describes the payload:

```json
"arrays": [{"name": "data", "dtype": "float32", "shape": [8, 250], "offset": 0, "nbytes": 8000}]
```

- `dtype` is one of `float32`, `float64`, `int32`, `int64`, `uint8`, `bool`; all
  little-endian.
- Arrays are in C (row-major) order. Signal data is `(n_channels, n_samples)`.
- Signal data travels as `float32` in volts. Model and state arrays keep their own
  dtype (use `float64` for filter state so block-wise processing stays exact).

## Trees (model and state)

A model or state is a JSON value in which arrays are replaced by `{"mxarray": NAME}`,
naming an entry of `arrays`, and non-finite floats by `{"mxfloat": "nan" | "inf" |
"-inf"}`. Allowed values: numbers, strings, booleans, null, lists and objects with
string keys. An object whose only key is `mxarray` or `mxfloat` is always such a
reference.

## Signal info

```json
{"ch_labels": ["Fz", "Cz"], "sfreq": 250.0, "stream_id": "amp", "bads": []}
```

## Messages

Every request carries `type` and an integer `id`; every reply carries the same `id`.
There is one request in flight at a time.

### `hello` → `hello`

Request: `{"type": "hello", "id": 1, "spec_version": "0.2.1"}`.
Reply: `{"type": "hello", "id": 1, "spec_version": "0.2.1", "manifest": {...},
"sdk": "meegxer-python/0.1.0"}`. The manifest follows `manifest.schema.json` with
`role` `processor`. Names starting with `meegxer.` are reserved for built-ins.

### `fit` → `model`

Request fields: `params`, `info`, `sample_index`, `timestamp`; array `data` (the
calibration signal). Reply: `{"type": "model", "model": TREE}` plus the model's
arrays. Only sent when the manifest has `requires_fit: true`.

With `fit_events: true` in the manifest (0.2.16) the request also carries
`events`: the events inside the calibration stretch, each
`{"sample_index": int, "label": str, "duration": int, "payload": {...}}` at
absolute sample indices (the data's first sample is `sample_index`). The host
never sends such a `fit` without events. The SDKs hand them to the plugin's fit
as `info["events"]` (Python) or `info.events` (Octave).

### `load_model` → `ok`

Request fields: `model_id` (string chosen by the host), `model` (TREE) plus arrays.
The plugin keeps the model under that id until it exits. The host sends this once per
model and worker; models can be large, so `transform` refers to them by id.

### `transform` → `result`

Request fields: `params`, `info`, `sample_index`, `timestamp`, `model_id` (string or
null), `state` (TREE or null; null on the first block of a stream); arrays `data` plus
the state's arrays. Reply: `{"type": "result", "state": TREE, "events": [...]}` with
array `data` (the output block) plus the new state's arrays. Optional reply field
`info` changes the output signal info (e.g. dropped channels).

Reply field `sample_index` (0.2) is the sample index of the output's first sample.
Without it the output is aligned with the input. A plugin with look-ahead (output
lags its input) reports an earlier index and may return fewer samples than it
received, including none; the host keeps the output aligned by sample index (CLAUDE.md
rule 8), so the delay never shifts it against other channels or probes. The host
never hands such an empty result on (since 0.2.8): no node, plugin or built-in, is
sent a `transform` with zero samples. Such a
plugin should also declare `latency_samples` in its manifest.

Each event is `{"sample_index": int, "label": str, "duration": int, "payload": {...}}`.

**Feature results (0.2.14).** A plugin whose manifest has `type: "analysis"` (a
Features output) replies without a `data` array. Its `result` carries `features`,
the feature windows completed by this call (possibly none), in order:

```json
{"type": "result", "id": 7, "state": {...}, "events": [],
 "features": [{"window": [1000, 1500], "dims": ["channel", "feature"],
               "coords": {"channel": ["Fz", "Cz"], "feature": ["alpha", "beta"]},
               "data": "f0"}]}
```

`window` is the source sample range `[start, stop)` the values were computed from,
`dims` names each axis of the array named by `data` (sent like any other array),
`coords` optionally labels the entries of a dimension (`channel` must hold
channel labels), and `timestamp` optionally gives the LSL time of the window's
first sample. The host keeps the windows' order; without `timestamp` it times a
window from the block's timestamp, which can differ in the last bits with how the
stream is cut, so a plugin that wants exact times anchors them to its first block
(keeping that anchor in `state`). The result must not depend on how the stream is cut into blocks
(acceptance test 1), so a windowing plugin keeps its unfinished window in `state`.

**Feature units (0.2.19).** A feature window may state its unit: `unit` for every
value (`"V^2/Hz"`, `"dB"`, `"Hz"`; `"1"` for a dimensionless value such as
coherence), or, when the values along one dimension differ in unit, `unit_dim`
(one of `dims`) and `units`, one unit per entry of that dimension, in order:

```json
{"window": [1000, 1500], "dims": ["channel", "measure"],
 "coords": {"measure": ["exponent", "peak_frequency"]},
 "unit_dim": "measure", "units": ["1", "Hz"], "data": "f0"}
```

Units are plain ASCII (`^` for powers). Without them the unit is not stated;
hosts show the values unlabelled and write no unit.

**Rails (0.2.17).** A plugin whose manifest has extra ports (each naming a
`rail`, see `manifest.schema.json`) gets a `rails` object in `transform` and in
`fit`, by input port name, over the same samples as `data`:

```json
"rails": {"ics": {"type": "signal", "data": "r1", "info": {...}, "sample_index": 1000},
          "power": {"type": "features", "windows": [ ...as in features... ]},
          "cues": {"type": "events", "events": [{"sample_index": 1010, "label": "left", ...}]}}
```

A Signal rail covers exactly the block's samples; a Features rail holds the
windows that ended inside the block (`start < window end <= stop`); the Events
rail holds the events inside it. Its `result` may carry `rails` for its extra
outputs, by output port name: `{"type": "signal", "data": NAME, "sample_index":
int, "info": {...}}` (index and info optional: aligned with the input, same info)
or `{"type": "features", "windows": [...]}`. Events are always written through
`events`. The Python SDK hands the rails to the plugin as `info["rails"]` and takes
them from a dict result's `rails`.

The host owns model and state; the plugin keeps nothing between blocks except the
models it was given. A plugin that crashes or times out is restarted and continues
with the state the host already holds.

### `flush` → `result` (0.2)

Sent once at the end of a stream (offline runs, conformance), with the same fields as
`transform` but no `data`, `sample_index` or `timestamp`. A plugin with look-ahead
replies with the samples it still holds (a `result` with `sample_index`); any other
plugin replies with an empty `data` array, or with a `bad_request` error (0.1
plugins), which the host treats as "nothing held".

### `shutdown` → `ok`

The plugin replies and exits with status 0.

### Errors

Instead of the expected reply a plugin may answer
`{"type": "error", "id": N, "error": CODE, "message": "..."}`:

| Code | Meaning |
|---|---|
| `bad_request` | malformed frame or unknown message type; the plugin keeps running |
| `unknown_model` | `model_id` not loaded; the host sends `load_model` and retries |
| `plugin_error` | the plugin's own code failed on this request |

## Conformance

`meegxer-plugin check IMAGE` runs a plugin through the protocol and the processor
contract: handshake, manifest validation, an unknown message type, fit (if
required), acceptance test 1 (block-wise output with random block sizes equals one
pass, with state carried by the host), recovery after the worker is killed mid-stream,
and clean shutdown.
