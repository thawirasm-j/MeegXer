"""Acceptance tests 2 and 3 (docs/DESIGN.md §6): stability and latency.

Starts a synthetic LSL stream carrying a sample counter, runs a ``meegxer-daemon``
subprocess with ``LSL in -> high-pass -> notch -> LSL out`` (the output also
carries the sample index), tunes a parameter live every ``--tune-every`` seconds
through the control API, and receives the output. Pass criteria: no dropped or
duplicated samples anywhere, and end-to-end p95 latency under ``--max-p95-ms``.
LSL delivers samples from the moment an inlet connects, so samples sent before the
daemon (or the receiver) connected are reported but not counted as lost.

    uv run python tools/stability_test.py --minutes 30 --channels 64 --sfreq 1000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pylsl

from meegxer.core.control import ControlClient
from meegxer.core.graph import Graph
from meegxer.core.lsl import LslSink
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource


def free_endpoint() -> str:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


class Receiver:
    """Checks the output index channel incrementally and records latency."""

    def __init__(self, name: str) -> None:
        found = pylsl.resolve_byprop("name", name, timeout=30)
        if not found:
            raise TimeoutError(f"output stream {name!r} not found")
        self.inlet = pylsl.StreamInlet(
            found[0], max_buflen=60, processing_flags=pylsl.proc_clocksync
        )
        self.inlet.open_stream(timeout=30)
        self.first: int | None = None
        self.last: int | None = None
        self.received = 0
        self.gaps = 0
        self.duplicates = 0
        self.latencies: list[float] = []

    def poll(self) -> None:
        samples, stamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1_000_000)
        if not samples:
            return
        now = pylsl.local_clock()
        idx = np.asarray(samples)[:, -1].astype(np.int64)
        previous = idx[:1] - 1 if self.last is None else np.array([self.last])
        steps = np.diff(np.concatenate([previous, idx]))
        self.gaps += int(np.sum(steps[steps > 1] - 1))
        self.duplicates += int(np.sum(1 - steps[steps < 1]))
        if self.first is None:
            self.first = int(idx[0])
        self.last = int(idx[-1])
        self.received += idx.size
        self.latencies.append(now - stamps[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sfreq", type=float, default=1000.0)
    parser.add_argument("--tune-every", type=float, default=30.0)
    parser.add_argument("--max-p95-ms", type=float, default=150.0)
    parser.add_argument("--report", help="write the JSON report here")
    parser.add_argument(
        "--plugin-image", action="append", default=[],
        help="insert this plugin image after the notch (repeatable)",
    )  # fmt: skip
    args = parser.parse_args()

    tag = uuid.uuid4().hex[:6]
    gen_name, out_name = f"mx-gen-{tag}", f"mx-out-{tag}"
    seconds = args.minutes * 60

    generator = RealtimeScheduler(Graph.chain([
        SyntheticSource({"duration": seconds, "n_channels": args.channels,
                         "sfreq": args.sfreq}),
        LslSink({"stream_name": gen_name, "include_index": True}),
    ]))  # fmt: skip
    generator.start()

    workdir = Path(tempfile.mkdtemp(prefix="meegxer-stability-"))
    env = dict(os.environ)
    plugin_nodes = []
    if args.plugin_image:
        from meegxer.plugins import index

        env[index.INDEX_ENV] = str(workdir / "plugins.json")
        for i, image in enumerate(args.plugin_image):
            entry = index.add_image(image, path=workdir / "plugins.json")
            plugin_nodes.append(
                {"id": f"plugin{i + 1}", "node": entry["manifest"]["name"]}
            )

    control, display = free_endpoint(), free_endpoint()
    daemon_log = open(workdir / "daemon.log", "w")  # noqa: SIM115 - closed in finally
    daemon = subprocess.Popen(
        [sys.executable, "-m", "meegxer.core.daemon", "--control", control,
         "--display", display],
        stdout=daemon_log, stderr=subprocess.STDOUT, text=True, env=env,
    )  # fmt: skip
    client = ControlClient(control, timeout=30)
    report: dict[str, object] = {}
    try:
        client.call("ping")
        client.call("load_pipeline", pipeline={
            "spec_version": "0.2.0",
            "nodes": [
                {"id": "input", "node": "meegxer.core.lsl_in",
                 "params": {"stream_name": gen_name, "timeout": 30.0}},
                {"id": "hp", "node": "meegxer.nodes.iir_filter",
                 "params": {"kind": "highpass", "low": 1.0}},
                {"id": "notch", "node": "meegxer.nodes.notch"},
                *plugin_nodes,
                {"id": "out", "node": "meegxer.core.lsl_out",
                 "params": {"stream_name": out_name, "include_index": True}},
            ],
        })  # fmt: skip
        client.call("start")
        receiver = Receiver(out_name)

        started = time.monotonic()
        next_tune, tunes, low = started + args.tune_every, 0, 1.0
        next_print = started + 60
        while not generator.wait(0.05):
            receiver.poll()
            now = time.monotonic()
            if now >= next_tune:
                low = 0.5 if low == 1.0 else 1.0
                client.call("set_params", node="hp", params={"low": low})
                tunes += 1
                next_tune = now + args.tune_every
            if now >= next_print:
                status = client.call("status")
                p95 = status["latency"].get("p95_ms", float("nan"))
                print(f"[{(now - started) / 60:5.1f} min] received {receiver.received}"
                      f" gaps {receiver.gaps} dups {receiver.duplicates}"
                      f" overruns {status['overruns']} latency p95 {p95:.1f} ms",
                      flush=True)  # fmt: skip
                next_print = now + 60

        generated = generator.metrics.samples
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            receiver.poll()
            if receiver.last is not None and receiver.last >= generated - 1:
                break
            time.sleep(0.05)
        status = client.call("status")
        lat = np.asarray(receiver.latencies) * 1e3
        report = {
            "minutes": args.minutes,
            "plugins": args.plugin_image,
            "channels": args.channels,
            "sfreq": args.sfreq,
            "generated": generated,
            "daemon_processed": status["samples"],
            "daemon_input_gaps": status.get("source_gaps"),
            "daemon_input_duplicates": status.get("source_duplicates"),
            "sent_before_daemon_connected": status.get("source_first_index"),
            "received": receiver.received,
            "received_first_index": receiver.first,
            "received_last_index": receiver.last,
            "output_gaps": receiver.gaps,
            "output_duplicates": receiver.duplicates,
            "overruns": status["overruns"],
            "blocks": status["blocks"],
            "live_param_changes": tunes,
            "daemon_latency_ms": status["latency"],
            "end_to_end_latency_ms": {
                "mean": float(lat.mean()) if lat.size else None,
                "p95": float(np.percentile(lat, 95)) if lat.size else None,
                "max": float(lat.max()) if lat.size else None,
            },
            "stage_timing": {n["id"]: n["timing"] for n in status["nodes"]},
            "errors": status["errors"],
        }
        client.call("stop")
        client.call("shutdown")
    finally:
        generator.stop()
        client.close()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
        daemon_log.close()

    first = report["sent_before_daemon_connected"]
    assert isinstance(first, int) and isinstance(report["generated"], int)
    expected_processed = report["generated"] - first  # LSL delivers from connect on
    last = report["received_last_index"]
    lost_at_end = (expected_processed - 1) - (last if isinstance(last, int) else -1)
    e2e = report["end_to_end_latency_ms"]
    assert isinstance(e2e, dict)
    passed = (
        report["daemon_input_gaps"] == 0
        and report["daemon_input_duplicates"] == 0
        and report["output_gaps"] == 0
        and report["output_duplicates"] == 0
        and report["daemon_processed"] == expected_processed
        and lost_at_end == 0
        and not report["errors"]
        and e2e["p95"] is not None
        and e2e["p95"] < args.max_p95_ms
    )
    report["passed"] = passed
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
