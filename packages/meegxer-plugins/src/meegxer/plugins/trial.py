"""Try a custom script on a recording before adding it (``meegxer-plugin run
--script FILE -i REC``; ``docs/PORTING.md``).

The script runs as the one strip of a pipeline over the recording, offline,
as it would once added (the same conformance and the same runner, through
the plugin protocol), and what came out is summed up: its channels, rate,
length and events, its feature windows, or the epochs it kept (an epochs
script runs after an Epochs strip, with its settings from ``epochs``). ``out``
writes the result too: a ``.set`` or ``.edf`` for a signal, a ``.tsv`` for
features, a ``-epo.fif`` or ``.set`` for epochs. Running it is the
user's own command on their own file, as ``meegxer-plugin check --script``
is: nothing is added or recorded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meegxer.core.batch import BatchScheduler
from meegxer.core.block import Block, FeatureBlock
from meegxer.core.graph import Graph
from meegxer.core.manifest import SPEC_VERSION
from meegxer.core.probes import FeatureProbe, MemoryProbe
from meegxer.core.registry import Registry
from meegxer.plugins import index
from meegxer.plugins.adapter import plugin_class, probe_manifest
from meegxer.plugins.worker import LocalLauncher

SIGNAL_WRITERS = {".set": "meegxer.core.set_writer", ".edf": "meegxer.core.edf_writer"}
FEATURE_WRITERS = {
    ".tsv": "meegxer.core.feature_tsv",
    ".csv": "meegxer.core.feature_tsv",
}
EPOCHS_WRITERS = {".fif": "meegxer.core.epochs_writer",
                  ".set": "meegxer.core.epochs_writer"}  # fmt: skip
EPOCHS_STRIP = "meegxer.nodes.epochs"  # what cuts the epochs an epochs script gets
WHAT = {"signal": "a signal", "features": "features", "epochs": "epochs"}


@dataclass
class Trial:
    """What a script made of a recording."""

    name: str
    kind: str  # "signal", "features" or "epochs"
    channels: list[str] = field(default_factory=list)
    sfreq: float = 0.0
    n_samples: int = 0
    events: Counter[str] = field(default_factory=Counter)
    windows: int = 0
    features: list[str] = field(default_factory=list)
    written: str = ""
    epochs_in: int = 0  # epochs: how many the Epochs strip cut
    kept: Counter[str] = field(default_factory=Counter)  # epochs kept, by label

    def summary(self) -> str:
        if self.kind == "epochs":
            labels = ", ".join(f"{k} {v}" for k, v in sorted(self.kept.items()))
            text = (f"{self.name}: kept {sum(self.kept.values())} of "
                    f"{self.epochs_in} epochs"
                    + (f" ({labels})" if labels else "")
                    + f" x {len(self.channels)} channels")  # fmt: skip
        elif self.kind == "features":
            text = (f"{self.name}: {self.windows} windows of "
                    f"{len(self.features)} features ({', '.join(self.features)}) "
                    f"x {len(self.channels)} channels")  # fmt: skip
        else:
            seconds = self.n_samples / self.sfreq if self.sfreq else 0.0
            text = (f"{self.name}: {len(self.channels)} channels, "
                    f"{self.n_samples} samples at {self.sfreq:g} Hz "
                    f"({seconds:.1f} s)")  # fmt: skip
        if self.events:
            labels = ", ".join(f"{k} {v}" for k, v in sorted(self.events.items()))
            text += f"; events: {labels}"
        if self.written:
            text += f"; wrote {self.written}"
        return text


def try_script(
    script: Path,
    recording: Path,
    *,
    out: Path | None = None,
    params: dict[str, Any] | None = None,
    overwrite: bool = False,
    epochs: dict[str, Any] | None = None,
) -> Trial:
    """Run ``script`` over ``recording`` as one strip, offline, and say what
    came out (and write it to ``out``). An epochs script gets the epochs an
    Epochs strip with the settings ``epochs`` cuts (its defaults: every event,
    -0.2 to 0.8 s)."""
    command = index.script_command(script)
    launcher = LocalLauncher(tuple(command), allow=True)
    origin = index.origin_of({"command": command, "script": str(script)})
    cls = plugin_class(probe_manifest(launcher), launcher, origin=origin)
    name = cls.manifest.name
    # the file as it is now, in place of the copy added before (if any)
    installed = Registry.from_entry_points().as_mapping()
    registry = Registry([*(c for n, c in installed.items() if n != name), cls])
    kind = cls.manifest.outputs[0].type.value
    if any(p.rail == "epochs" for p in cls.manifest.rail_inputs):
        kind = "epochs"
    result_port = "script:epochs" if kind == "epochs" else "script"
    if kind == "signal":
        tap = {"node": "meegxer.core.probe", "params": {"keep_seconds": 0}}
    else:
        tap = {"node": "meegxer.core.feature_probe", "params": {"keep_windows": 0}}
    nodes: list[dict[str, Any]] = [
        {"id": "input", "node": "meegxer.core.file",
         "params": {"path": str(recording)}},
        *([{"id": "cut", "node": EPOCHS_STRIP, "params": dict(epochs or {})}]
          if kind == "epochs" else []),
        {"id": "script", "node": name, "params": dict(params or {})},
        {"id": "tap", "input": result_port, **tap},
    ]  # fmt: skip
    if kind == "epochs":
        nodes.append({"id": "cut_tap", "input": "cut:epochs", **tap})
    if out is not None:
        writers = {"signal": SIGNAL_WRITERS, "features": FEATURE_WRITERS,
                   "epochs": EPOCHS_WRITERS}[kind]  # fmt: skip
        writer = writers.get(out.suffix.lower())
        if writer is None:
            raise ValueError(
                f"{out.name}: write {WHAT[kind]} to {' or '.join(sorted(writers))}"
            )
        settings: dict[str, Any] = {"path": str(out), "overwrite": overwrite}
        if kind == "epochs":
            settings["erp"] = False  # only the file asked for
        nodes.append({"id": "out", "node": writer, "input": result_port,
                      "params": settings})  # fmt: skip
    graph = Graph.from_dict({"spec_version": SPEC_VERSION, "nodes": nodes}, registry)
    try:
        result = BatchScheduler().run(graph)
    finally:
        graph["script"].impl.processor.close()  # type: ignore[union-attr]
    trial = Trial(name, kind, written=str(out) if out else "")
    for event in result.events.get("script", ()):  # what the script added
        trial.events[event.label] += 1
    probe = graph["tap"].impl
    if kind == "epochs":
        assert isinstance(probe, FeatureProbe)
        cut = graph["cut_tap"].impl
        assert isinstance(cut, FeatureProbe)
        given = [w for w in cut.windows if isinstance(w, FeatureBlock)]
        kept = [w for w in probe.windows if isinstance(w, FeatureBlock)]
        trial.epochs_in = len(given)
        for w in kept:
            trial.kept[str(w.coords["epoch"][0])] += 1
        if kept or given:
            first = (kept or given)[0]
            trial.channels = [str(c) for c in first.coords.get("channel", ())]
    elif isinstance(probe, FeatureProbe):
        windows = [w for w in probe.windows if isinstance(w, FeatureBlock)]
        trial.windows = len(windows)
        if windows:
            first = windows[0]
            trial.channels = [str(c) for c in first.coords.get("channel", ())]
            trial.features = [str(f) for f in first.coords.get("feature", ())]
    elif isinstance(probe, MemoryProbe):
        blocks = [b for b in probe.blocks if isinstance(b, Block)]
        if blocks:
            trial.channels = list(blocks[0].info.ch_labels)
            trial.sfreq = float(blocks[0].info.sfreq)
            trial.n_samples = sum(b.n_samples for b in blocks)
    return trial
