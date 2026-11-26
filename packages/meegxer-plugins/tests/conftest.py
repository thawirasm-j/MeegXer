"""Fixtures: small SDK plugins run as local (unsandboxed) processes."""

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from meegxer.plugins.worker import LocalLauncher

SDK_DIR = Path(__file__).resolve().parents[1] / "sdk" / "python"
PLUGINS_DIR = Path(__file__).resolve().parents[1] / "plugins"

PREAMBLE = f"""
import os, sys, time
sys.path.insert(0, {str(SDK_DIR)!r})
import numpy as np
from meegxer_plugin import Plugin, run

def manifest(name, requires_fit=False, params=()):
    return {{
        "spec_version": "0.2.1", "name": name, "version": "0.1.0", "type": "stream",
        "requires_fit": requires_fit,
        "inputs": [{{"name": "in", "type": "signal"}}],
        "outputs": [{{"name": "out", "type": "signal"}}],
        "params": list(params),
    }}
"""

PLUGINS = {
    "railed": """
class Railed(Plugin):
    manifest = {
        "spec_version": "0.2.17", "name": "test.railed", "version": "0.1.0",
        "type": "stream", "params": [],
        "inputs": [{"name": "in", "type": "signal"},
                   {"name": "mix", "type": "signal", "rail": "mixed"},
                   {"name": "cues", "type": "events", "rail": "events"}],
        "outputs": [{"name": "out", "type": "signal"},
                    {"name": "twice", "type": "signal", "rail": "twice"},
                    {"name": "marks", "type": "features", "rail": "marks"}],
    }

    def transform(self, data, info, model, state, params):
        rails = info["rails"]
        marks = [{"window": [e["sample_index"], e["sample_index"] + 1],
                  "dims": ["feature"], "data": np.array([1.0])}
                 for e in rails["cues"]]
        return {"data": data - rails["mix"], "state": state or {},
                "rails": {"twice": data * 2, "marks": marks}}

run(Railed())
""",
    "ema": """
class Ema(Plugin):
    manifest = manifest("test.ema", params=[
        {"name": "alpha", "type": "float", "default": 0.1, "min": 0, "max": 1}])

    def transform(self, data, info, model, state, params):
        marker = os.environ.get("CRASH_MARKER")
        if marker and os.path.exists(marker):
            os.remove(marker)
            os._exit(3)  # one-shot crash
        if os.environ.get("HANG_MARKER") and os.path.exists(os.environ["HANG_MARKER"]):
            os.remove(os.environ["HANG_MARKER"])
            time.sleep(30)
        a = params["alpha"]
        y = data[:, 0].copy() if state is None else state["y"]
        out = np.empty_like(data)
        for i in range(data.shape[1]):
            y = (1 - a) * y + a * data[:, i]
            out[:, i] = y
        events = [{"sample_index": int(info["sample_index"]) + int(i), "label": "high"}
                  for i in np.flatnonzero(out[0] > 3e-5)[:1]]
        return out, {"y": y, "n": int(info["sample_index"]) + data.shape[1]}, events

run(Ema())
""",
    "center": """
class Center(Plugin):
    manifest = manifest("test.center", requires_fit=True)

    def fit(self, data, info, params):
        return {"mean": data.mean(axis=1), "label": "ok", "nan": float("nan")}

    def transform(self, data, info, model, state, params):
        assert model["label"] == "ok" and np.isnan(model["nan"])
        return data - model["mean"][:, None], {}, []

run(Center())
""",
    "forgetful": """
class Forgetful(Plugin):
    manifest = manifest("test.forgetful")

    def transform(self, data, info, model, state, params):
        y = data[:, 0].copy()  # bug: ignores the state it is given
        out = np.empty_like(data)
        for i in range(data.shape[1]):
            y = 0.9 * y + 0.1 * data[:, i]
            out[:, i] = y
        return out, {"y": y}, []

run(Forgetful())
""",
    "printer": """
class Printer(Plugin):
    manifest = manifest("test.printer")

    def transform(self, data, info, model, state, params):
        os.write(1, b"debug output on stdout\\n")  # corrupts the protocol
        return data, {}, []

run(Printer())
""",
    "hostile": """
# a hostile plugin (the red team, docs/SECURITY.md): MODE picks the attack;
# it speaks the wire format itself, so it can send what no SDK would
from meegxer.plugins.wire import read_message, write_message
MODE = os.environ.get("MODE", "")
inp, out = sys.stdin.buffer, sys.stdout.buffer
MANIFEST = manifest("test.hostile")
if MODE == "stderr_flood":  # 50 MB without a newline, then a line
    for _ in range(50):
        sys.stderr.buffer.write(b"A" * 2**20)
    sys.stderr.buffer.write(b"\\nafter the flood\\n")
    sys.stderr.flush()
while True:
    try:
        msg = read_message(inp)
    except EOFError:
        break
    h = msg.header
    rid = h.get("id")
    if h["type"] == "hello":
        write_message(out, {"type": "hello", "id": rid, "spec_version": "0.2.1",
                            "manifest": MANIFEST, "sdk": "hostile"})
        continue
    if h["type"] == "shutdown":
        write_message(out, {"type": "ok", "id": rid})
        break
    data = msg.arrays["data"]
    reply = {"type": "result", "id": rid, "state": {}, "events": []}
    arrays = {"data": data}
    if MODE == "longer":  # 100 times the samples it was given
        arrays["data"] = np.zeros((data.shape[0], data.shape[1] * 100))
    elif MODE == "future":  # its output far past its input
        reply["sample_index"] = 10**15
    elif MODE.startswith("event_"):  # one bad event (or too many)
        at = int(h["sample_index"])
        bad = {
            "event_dict": {"sample_index": at, "label": {"evil": [1, 2]}},
            "event_long": {"sample_index": at, "label": "A" * 300},
            "event_tab": {"sample_index": at, "label": "a\\tb\\nc"},
            "event_future": {"sample_index": 10**12, "label": "press"},
            "event_field": {"sample_index": at, "label": "x", "__class__": 1},
            "event_payload": {"sample_index": at, "label": "x",
                              "payload": {"deep": {"a": [1]}}},
            "event_duration": {"sample_index": at, "label": "x", "duration": -5},
        }
        reply["events"] = ([{"sample_index": at, "label": "x"}] * 300
                           if MODE == "event_flood" else [bad[MODE]])
    elif MODE == "honest_events":
        at = int(h["sample_index"])
        reply["events"] = [{"sample_index": at + 3, "label": "blink",
                            "duration": 25, "payload": {"amplitude_uv": 120.5,
                                                        "channel": "Fp1"}}]
    write_message(out, reply, arrays)
""",
    "reserved": """
class Reserved(Plugin):
    manifest = manifest("meegxer.sneaky")

    def transform(self, data, info, model, state, params):
        return data, {}, []

run(Reserved())
""",
}


@pytest.fixture
def plugin(tmp_path: Path) -> Callable[..., LocalLauncher]:
    def make(name: str, **env: str) -> LocalLauncher:
        script = tmp_path / f"{name}.py"
        script.write_text(PREAMBLE + textwrap.dedent(PLUGINS[name]))
        return LocalLauncher((sys.executable, "-u", str(script)), allow=True, env=env)

    return make


@pytest.fixture
def reference_plugin() -> LocalLauncher:
    script = PLUGINS_DIR / "average_reference" / "plugin.py"
    return LocalLauncher(
        (sys.executable, "-u", str(script)),
        allow=True,
        env={"PYTHONPATH": str(SDK_DIR)},
    )
