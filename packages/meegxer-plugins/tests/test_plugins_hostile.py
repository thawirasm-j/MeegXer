"""The red team's attacks on the host, kept as regressions (docs/SECURITY.md):
a plugin is untrusted (rule 9), so what it sends back is bounded and checked
before the host keeps it. The ``hostile`` fixture plugin picks its attack by
``MODE``."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.block import Block, SignalInfo
from meegxer.core.stage import Stage
from meegxer.plugins.adapter import plugin_class, probe_manifest
from meegxer.plugins.script import ScriptError
from meegxer.plugins.sterile import script_files
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


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_a_script_folder_cannot_link_out(tmp_path: Path) -> None:
    """A shared folder holding ``notes.txt -> ~/.ssh/id_rsa``: the key was
    read and sent into the Sterile container. A link out is refused; a link
    to a file in the folder is fine."""
    secret = tmp_path / "home" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("PRIVATE KEY")
    folder = tmp_path / "lab_x"
    folder.mkdir()
    (folder / "lab_x.py").write_text("def process(raw):\n    return raw\n")
    (folder / "table.csv").write_text("1,2\n")
    os.symlink(folder / "table.csv", folder / "same.csv")  # inside: fine
    assert [n for n, _ in script_files(folder)] == ["lab_x.py", "same.csv",
                                                    "table.csv"]  # fmt: skip
    os.symlink(os.path.relpath(secret, folder), folder / "notes.txt")
    with pytest.raises(ScriptError, match=r"notes\.txt links to a file outside"):
        script_files(folder)


def test_a_plugin_that_hangs_live_is_passed_over(
    plugin: Callable[..., LocalLauncher],
) -> None:
    """Live, a plugin that stops answering froze the whole run ~40 s (10 s,
    retried with restarts). Now it gets ``LIVE_BUDGET`` per block, no
    restart: its input passes through it, marked bypassed, and the run goes
    on, saying so."""
    from meegxer.core.graph import Graph
    from meegxer.core.probes import MemoryProbe
    from meegxer.core.realtime import LIVE_BUDGET, RealtimeScheduler
    from meegxer.core.sources import SyntheticSource

    stage = hostile(plugin, "hang_later")
    probe = MemoryProbe()
    source = SyntheticSource({"duration": 4.0, "n_channels": 3, "sfreq": 250.0})
    graph = Graph.chain([source, stage, probe])
    scheduler = RealtimeScheduler(graph)
    started = time.monotonic()
    try:
        scheduler.start()
        assert scheduler.wait(30)
    finally:
        stage.processor.close()
    took = time.monotonic() - started
    status = scheduler.status()
    node = graph.nodes[1].id
    assert any(f"{node}: no answer within {LIVE_BUDGET:g} s" in e
               for e in status["errors"])  # fmt: skip
    assert next(n for n in status["nodes"] if n["id"] == node)["status"] == "error"
    out = probe.signal()
    assert out.data.shape[1] == 1000  # every sample reached the probe
    assert took < 4.0 + LIVE_BUDGET + 3.0  # not ~40 s


def test_live_containers_leave_the_run_a_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One script took every core (400%), and the host's live timing
    suffered. In a live run a plugin's container is pinned to the top
    ``general.live_cpus`` CPUs (0: all but one), leaving CPU 0 to the run;
    offline, any CPU."""
    from meegxer.plugins import worker
    from meegxer.plugins.adapter import PluginProcessor
    from meegxer.plugins.octave_script import OctaveScriptLauncher
    from meegxer.plugins.sterile import SterileScriptLauncher
    from meegxer.plugins.worker import ContainerLauncher, live_cpus

    monkeypatch.setattr(worker, "find_engine", lambda engine=None: "docker")
    monkeypatch.setattr(worker, "engine_cpus", lambda engine=None: 4)
    monkeypatch.delenv("MEEGXER_LIVE_CPUS", raising=False)
    assert live_cpus() == "1-3"  # all but CPU 0
    monkeypatch.setenv("MEEGXER_LIVE_CPUS", "1")
    assert live_cpus() == "3"
    monkeypatch.setenv("MEEGXER_LIVE_CPUS", "64")
    assert live_cpus() is None  # all of them: no pinning
    monkeypatch.setattr(worker, "engine_cpus", lambda engine=None: 1)
    monkeypatch.delenv("MEEGXER_LIVE_CPUS")
    assert live_cpus() is None  # one CPU: nothing to leave
    monkeypatch.setattr(worker, "engine_cpus", lambda engine=None: 4)

    def cpuset(command: list[str]) -> str | None:
        flag = "--cpuset-cpus"
        return command[command.index(flag) + 1] if flag in command else None

    for launcher in (ContainerLauncher("some/plugin:1"),
                     SterileScriptLauncher("x.py", (("x.py", b""),))):  # fmt: skip
        cls = type("P", (PluginProcessor,), {"launcher": launcher,
                                              "manifest": None})  # fmt: skip
        p = cls()
        assert cpuset(p.worker.launcher.command("n")) is None  # offline
        p.go_live(2.0)
        assert cpuset(p.worker.launcher.command("n")) == "1-3"
        p.go_live(None)
        assert cpuset(p.worker.launcher.command("n")) is None
    octave = OctaveScriptLauncher("x.m", image="meegxer-octave:t").with_cpuset("1-3")
    assert octave._container().cpuset == "1-3"


def test_trying_a_python_script_asks_once_per_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run``/``check --script`` started a downloaded script with the user's
    permissions, without the notice ``add`` shows. Now the first run of each
    version asks (here: refused without a terminal), ``--accept-risk``
    accepts it, and an edited script asks again."""
    import mne

    from meegxer.plugins.cli import main

    rec = tmp_path / "rec_raw.fif"
    info = mne.create_info(["C3", "C4"], 100.0, "eeg")
    mne.io.RawArray(np.zeros((2, 200)), info, verbose=False).save(rec, verbose=False)
    marker = tmp_path / "ran.txt"
    script = tmp_path / "lab_try.py"
    script.write_text(
        f"def process(raw):\n    open({str(marker)!r}, 'a').write('x')\n"
        "    return raw\n"
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    run = ["run", "--script", str(script), "-i", str(rec)]
    assert main(run) == 1 and not marker.exists()  # not accepted: not run
    assert main([*run, "--accept-risk"]) == 0 and marker.read_text() == "x"
    assert main(run) == 0 and marker.read_text() == "xx"  # this version: known
    script.write_text(script.read_text() + "# edited\n")
    assert main(run) == 1 and marker.read_text() == "xx"  # asks again
    assert main(["check", "--script", str(script)]) == 1  # check asks too
    # stdin from NUL on Windows: isatty() is True, but input() meets the end of
    # the stream; a refusal that says what to do, no EOFError traceback
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(EOFError))
    assert main(run) == 1 and marker.read_text() == "xx"
