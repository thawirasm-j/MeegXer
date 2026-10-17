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
