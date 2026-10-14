# MeegXer plugin protocol (`meegxer.plugin/0.1`, spec 0.2.x, unstable)

How the MeegXer host talks to an external processing plugin (CLAUDE.md rules 9 and
11). The protocol is language-neutral: the Python and Octave SDKs implement it, they
do not define it. A plugin is a program that reads requests on **stdin** and writes
replies on **stdout**; anything it writes to **stderr** is treated as log text.

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

Each event is `{"sample_index": int, "label": str, "duration": int, "payload": {...}}`.

The host owns model and state; the plugin keeps nothing between blocks except the
models it was given. A plugin that crashes or times out is restarted and continues
with the state the host already holds.

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
