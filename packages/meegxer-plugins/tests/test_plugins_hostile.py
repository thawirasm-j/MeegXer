"""The red team's attacks on the host, kept as regressions (docs/SECURITY.md):
a plugin is untrusted (rule 9), so what it sends back is bounded and checked
before the host keeps it. The ``hostile`` fixture plugin picks its attack by
``MODE``."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.stage import Stage
from meegxer.plugins.adapter import plugin_class, probe_manifest
from meegxer.plugins.worker import (
    MAX_STDERR_LINE,
    LocalLauncher,
    PluginError,
    PluginWorker,
)

INFO = SignalInfo(("C3", "C4", "Cz"), 250.0)


def block(start: int = 0, n: int = 250) -> Block:
    data = np.random.default_rng(start).standard_normal((3, n)) * 1e-5
    return Block(data, INFO, start, start / INFO.sfreq)


def hostile(plugin: Callable[..., LocalLauncher], mode: str) -> Stage:
    launcher = plugin("hostile", MODE=mode)
    return Stage(plugin_class(probe_manifest(launcher), launcher)(), {})


def transform(stage: Stage, b: Block) -> object:
    try:
        return stage.transform(b, "fp")
    finally:
        stage.processor.close()


def test_a_stderr_line_without_end_is_cut(plugin: Callable[..., LocalLauncher]) -> None:
    """50 MB on stderr with no newline: the host keeps the first
    ``MAX_STDERR_LINE`` bytes of it (marked cut), then the lines after it."""
    worker = PluginWorker(plugin("hostile", MODE="stderr_flood"))
    try:
        worker.start()
        deadline = time.time() + 30
        while "after the flood" not in worker.stderr_tail() and time.time() < deadline:
            time.sleep(0.05)
        tail = worker.stderr_tail()
    finally:
        worker.close()
    lines = tail.splitlines()
    assert lines[-1] == "after the flood"
    assert lines[-2].endswith("… (cut)")
    assert len(lines[-2]) <= MAX_STDERR_LINE + len(" … (cut)")
    assert len(tail) < 2 * MAX_STDERR_LINE


@pytest.mark.parametrize(
    ("mode", "message"),
    [("longer", r"covers samples 0 to 25000, but it was given samples up to 250"),
     ("future", r"covers samples 1000000000000000 to")],
)  # fmt: skip
def test_output_beyond_its_input_is_refused(
    plugin: Callable[..., LocalLauncher], mode: str, message: str
) -> None:
    """100 times the samples it was given, or its output at sample 10**15:
    refused, naming the plugin (it grew memory on every block, and broke
    alignment by sample index, rule 8)."""
    with pytest.raises(PluginError, match=r"test\.hostile: its output " + message):
        transform(hostile(plugin, mode), block())


def test_honest_plugins_still_pass(plugin: Callable[..., LocalLauncher]) -> None:
    """The same plugin, answering honestly, a block later in the stream."""
    result = transform(hostile(plugin, ""), block(start=500))
    assert result.block.sample_index == 500  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("event_dict", "label must be text"),
        ("event_long", "label must be text of 1 to 256"),
        ("event_tab", "control character"),
        ("event_future", r"an event at sample 1000000000000, but it was given"),
        ("event_field", r"unknown fields \['__class__'\]"),
        ("event_payload", "payload must be up to 32 plain values"),
        ("event_duration", "duration must be a whole number"),
        ("event_flood", "300 events for a block of 250 samples"),
    ],
)  # fmt: skip
def test_bad_events_are_refused(
    plugin: Callable[..., LocalLauncher], mode: str, message: str
) -> None:
    """Events reach files (a tab breaks events.tsv), the GUI and the action
    sinks: anything but a short text label within the input is refused."""
    with pytest.raises(PluginError, match=r"test\.hostile: .*" + message):
        transform(hostile(plugin, mode), block())


def test_honest_events_pass(plugin: Callable[..., LocalLauncher]) -> None:
    result = transform(hostile(plugin, "honest_events"), block(start=250))
    (event,) = result.events  # type: ignore[attr-defined]
    assert (event.sample_index, event.label, event.duration) == (253, "blink", 25)
    assert dict(event.payload) == {"amplitude_uv": 120.5, "channel": "Fp1"}
