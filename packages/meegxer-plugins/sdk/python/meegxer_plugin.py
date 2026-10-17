"""MeegXer plugin SDK for Python (protocol meegxer.plugin/0.2, spec 0.2.x).

A single-file, numpy-only implementation of spec/plugin-protocol.md for use inside
plugin images. Copy this file into your image and write::

    import numpy as np
    from meegxer_plugin import Plugin, run

    class Gain(Plugin):
        manifest = {
            "spec_version": "0.2.1", "name": "lab.gain", "version": "0.1.0",
            "type": "stream", "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": [{"name": "k", "type": "float", "default": 2.0}],
        }

        def transform(self, data, info, model, state, params):
            return data * params["k"], state or {}, []

    if __name__ == "__main__":
        run(Gain())

``data`` is a float64 array (n_channels, n_samples) in volts; ``info`` holds
``ch_labels``, ``sfreq``, ``stream_id``, ``bads`` plus ``sample_index`` and
``timestamp`` of the block. ``state`` is None on the first block. Return
``(data, state, events)``; events are dicts with ``sample_index``, ``label`` and
optionally ``duration`` and ``payload``. A plugin whose output lags its input
(look-ahead) returns ``(data, state, events, first)`` with ``first`` the sample index
of its first output sample (output may be shorter than the input), and returns what
it still holds at the end of the stream from ``flush(info, model, state, params)``.
Keep everything that must survive between blocks in ``state``: the host owns it,
saves it and restores it after a restart.

An analysis plugin (manifest ``"type": "analysis"``, a Features output) returns
the feature windows it completed instead of ``data``: a list of dicts with
``window`` ([start, stop) sample range), ``dims`` (axis names), ``data`` (the
array) and optionally ``coords`` ({dim: labels}) and ``timestamp`` (LSL time of
the window's first sample; anchor it to the first block so it does not depend on
how the stream is cut).

Rails (spec 0.2.17): extra input ports (each naming a rail) arrive in
``info["rails"]``, by port name, over the same samples as ``data``: a Signal rail
as an array, a Features rail as a list of window dicts (as above), the Events
rail as a list of event dicts; ``fit`` gets them too. To write extra output
ports, return a dict instead of a tuple: ``{"data": ..., "state": ...,
"events": [...], "first": ..., "rails": {port: value}}``, where a Signal rail's
value is an array over the input's samples, or ``{"data": array,
"sample_index": int}`` when it lags, and a Features rail's value is a list of
window dicts.
Never print to stdout (it carries the protocol); log to stderr.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import traceback
from typing import Any, BinaryIO, ClassVar

import numpy as np

SDK = "meegxer-python/0.1.0"
SPEC_VERSION = "0.2.18"
MAGIC = b"MXP1"
_DTYPES = {"float32", "float64", "int32", "int64", "uint8", "bool"}


class Plugin:
    """Subclass and set ``manifest``; override ``transform`` (and ``fit``)."""

    manifest: ClassVar[dict[str, Any]] = {}

    def fit(
        self, data: np.ndarray, info: dict[str, Any], params: dict[str, Any]
    ) -> dict:
        raise NotImplementedError("this plugin does not implement fit")

    def transform(
        self,
        data: np.ndarray,
        info: dict[str, Any],
        model: dict | None,
        state: dict | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def flush(
        self,
        info: dict[str, Any],
        model: dict | None,
        state: dict | None,
        params: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        """End of stream: what a lagging plugin still holds, as transform's
        4-tuple; None (the default) when nothing is held."""
        return None


# -- wire format ----------------------------------------------------------------


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def read_frame(stream: BinaryIO) -> tuple[dict, dict[str, np.ndarray]]:
    magic = stream.read(4)
    if not magic:
        raise EOFError
    magic += _read_exact(stream, 4 - len(magic)) if len(magic) < 4 else b""
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    (hlen,) = struct.unpack("<I", _read_exact(stream, 4))
    header = json.loads(_read_exact(stream, hlen))
    (plen,) = struct.unpack("<Q", _read_exact(stream, 8))
    payload = _read_exact(stream, plen)
    arrays = {}
    for d in header.pop("arrays", []):
        if d["dtype"] not in _DTYPES:
            raise ValueError(f"unknown dtype {d['dtype']}")
        raw = payload[d["offset"] : d["offset"] + d["nbytes"]]
        dtype = np.dtype(d["dtype"]).newbyteorder("<")
        arrays[d["name"]] = np.frombuffer(raw, dtype=dtype).reshape(d["shape"]).copy()
    return header, arrays


def write_frame(stream: BinaryIO, header: dict, arrays: dict[str, np.ndarray]) -> None:
    descs, chunks, offset = [], [], 0
    for name, arr in arrays.items():
        arr = np.asarray(arr)
        dtype = "bool" if arr.dtype == np.bool_ else arr.dtype.name
        if dtype not in _DTYPES:
            raise TypeError(f"array {name!r} has unsupported dtype {arr.dtype}")
        raw = np.ascontiguousarray(
            arr, dtype=np.dtype(dtype).newbyteorder("<")
        ).tobytes()
        descs.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)
        offset += len(raw)
    head = json.dumps({**header, "arrays": descs}, allow_nan=False).encode()
    payload = b"".join(chunks)
    stream.write(MAGIC + struct.pack("<I", len(head)) + head)
    stream.write(struct.pack("<Q", len(payload)) + payload)
    stream.flush()


def to_wire(tree: Any, arrays: dict[str, np.ndarray], prefix: str) -> Any:
    if isinstance(tree, np.ndarray):
        name = f"{prefix}{len(arrays)}"
        arrays[name] = tree
        return {"mxarray": name}
    if isinstance(tree, np.generic):
        return to_wire(tree.item(), arrays, prefix)
    if isinstance(tree, float) and not math.isfinite(tree):
        return {
            "mxfloat": "nan" if math.isnan(tree) else ("inf" if tree > 0 else "-inf")
        }
    if isinstance(tree, (list, tuple)):
        return [to_wire(v, arrays, prefix) for v in tree]
    if isinstance(tree, dict):
        return {str(k): to_wire(v, arrays, prefix) for k, v in tree.items()}
    return tree


def from_wire(tree: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(tree, dict):
        if set(tree) == {"mxarray"}:
            return arrays[tree["mxarray"]]
        if set(tree) == {"mxfloat"}:
            return float(tree["mxfloat"])
        return {k: from_wire(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [from_wire(v, arrays) for v in tree]
    return tree


def _features_to_wire(
    windows: list, arrays: dict[str, np.ndarray], prefix: str = "f"
) -> list:
    """Feature windows (dicts with window, dims, data, coords) for a result."""
    wire = []
    for i, w in enumerate(windows):
        name = f"{prefix}{i}"
        arrays[name] = np.asarray(w["data"], dtype=np.float64)
        coords = {str(k): [v.item() if isinstance(v, np.generic) else v for v in vals]
                  for k, vals in (w.get("coords") or {}).items()}  # fmt: skip
        wire.append({"window": [int(w["window"][0]), int(w["window"][1])],
                     "dims": [str(d) for d in w["dims"]], "coords": coords,
                     "data": name})  # fmt: skip
        if w.get("timestamp") is not None:
            wire[-1]["timestamp"] = float(w["timestamp"])
    return wire


def _rails_from_wire(wire: dict, arrays: dict[str, np.ndarray]) -> dict:
    """The rails a request carries (spec 0.2.17), as plain Python values."""
    rails = {}
    for port, item in (wire or {}).items():
        kind = item.get("type")
        if kind == "signal":
            rails[port] = arrays[item["data"]].astype(np.float64)
        elif kind == "features":
            rails[port] = [dict(w, data=arrays[w["data"]].astype(np.float64))
                           for w in item.get("windows", [])]  # fmt: skip
        else:
            rails[port] = list(item.get("events", []))
    return rails


def _rails_to_wire(
    rails: dict, manifest: dict, first: int, arrays: dict[str, np.ndarray]
) -> dict:
    """What the extra output ports wrote, for a result (spec 0.2.17)."""
    types = {p["name"]: p["type"] for p in manifest.get("outputs", [])[1:]}
    wire = {}
    for port, value in (rails or {}).items():
        if types.get(port) == "features":
            windows = _features_to_wire(value or [], arrays, f"w{port}_")
            wire[port] = {"type": "features", "windows": windows}
            continue
        start = first
        if isinstance(value, dict):
            start = int(value.get("sample_index", first))
            value = value["data"]
        name = f"r{port}"
        arrays[name] = np.asarray(value, dtype=np.float32)
        wire[port] = {"type": "signal", "data": name, "sample_index": start}
    return wire


# -- main loop ------------------------------------------------------------------


def run(
    plugin: Plugin, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None
) -> None:
    """Serve requests until ``shutdown`` or end of input."""
    inp = stdin or sys.stdin.buffer
    out = stdout or sys.stdout.buffer
    sys.stdout = sys.stderr  # stray print() calls must not corrupt the protocol
    models: dict[str, dict] = {}

    while True:
        try:
            header, arrays = read_frame(inp)
        except EOFError:
            return
        except Exception as exc:  # unreadable frame: we cannot resynchronise
            print(f"meegxer_plugin: bad frame: {exc}", file=sys.stderr)
            return
        rid = header.get("id")
        kind = header.get("type")
        try:
            if kind == "hello":
                write_frame(
                    out,
                    {
                        "type": "hello",
                        "id": rid,
                        "spec_version": SPEC_VERSION,
                        "manifest": plugin.manifest,
                        "sdk": SDK,
                    },
                    {},
                )
            elif kind == "shutdown":
                write_frame(out, {"type": "ok", "id": rid}, {})
                return
            elif kind == "load_model":
                models[header["model_id"]] = from_wire(header["model"], arrays)
                write_frame(out, {"type": "ok", "id": rid}, {})
            elif kind == "fit":
                info = dict(
                    header["info"],
                    sample_index=header.get("sample_index", 0),
                    timestamp=header.get("timestamp", 0.0),
                    events=header.get("events", []),  # labelled trials (0.2.16)
                )
                if "rails" in header:  # rails the extra inputs read (0.2.17)
                    info["rails"] = _rails_from_wire(header["rails"], arrays)
                model = plugin.fit(
                    arrays["data"].astype(np.float64), info, header["params"]
                )
                model_arrays: dict[str, np.ndarray] = {}
                tree = to_wire(model, model_arrays, "m")
                write_frame(
                    out, {"type": "model", "id": rid, "model": tree}, model_arrays
                )
            elif kind in ("transform", "flush"):
                model_id = header.get("model_id")
                if model_id is not None and model_id not in models:
                    write_frame(
                        out,
                        {
                            "type": "error",
                            "id": rid,
                            "error": "unknown_model",
                            "message": f"model {model_id} not loaded",
                        },
                        {},
                    )
                    continue
                info = dict(header["info"])
                if "sample_index" in header:
                    info.update(
                        sample_index=header["sample_index"],
                        timestamp=header["timestamp"],
                    )
                if "rails" in header:  # rails the extra inputs read (0.2.17)
                    info["rails"] = _rails_from_wire(header["rails"], arrays)
                state = header.get("state")
                state = None if state is None else from_wire(state, arrays)
                model = models.get(model_id) if model_id else None
                if kind == "transform":
                    got = plugin.transform(
                        arrays["data"].astype(np.float64),
                        info,
                        model,
                        state,
                        header["params"],
                    )
                    first = header["sample_index"]
                else:
                    got = plugin.flush(info, model, state, header["params"])
                    if got is None:
                        empty = (
                            []
                            if plugin.manifest.get("type") == "analysis"
                            else np.zeros((len(info["ch_labels"]), 0))
                        )
                        got = (empty, state, [], 0)
                    first = 0
                rails = {}
                if isinstance(got, dict):  # a result that writes rails (0.2.17)
                    data, new_state = got.get("data"), got.get("state")
                    events = got.get("events", [])
                    first = int(got.get("first", first))
                    rails = got.get("rails", {})
                else:
                    data, new_state, events = got[0], got[1], got[2]
                    if len(got) > 3:
                        first = int(got[3])
                out_arrays: dict[str, np.ndarray] = {}
                reply = {"type": "result", "id": rid}
                if plugin.manifest.get("type") == "analysis":
                    reply["features"] = _features_to_wire(data or [], out_arrays)
                else:
                    out_arrays["data"] = np.asarray(data, dtype=np.float32)
                    reply["sample_index"] = first
                if rails:
                    reply["rails"] = _rails_to_wire(
                        rails, plugin.manifest, first, out_arrays
                    )
                reply["state"] = to_wire(new_state or {}, out_arrays, "s")
                reply["events"] = [{"duration": 0, "payload": {}, **e} for e in events]
                write_frame(out, reply, out_arrays)
            else:
                write_frame(
                    out,
                    {
                        "type": "error",
                        "id": rid,
                        "error": "bad_request",
                        "message": f"unknown message type {kind!r}",
                    },
                    {},
                )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            write_frame(
                out,
                {
                    "type": "error",
                    "id": rid,
                    "error": "plugin_error",
                    "message": f"{type(exc).__name__}: {exc}",
                },
                {},
            )
