"""Try a custom script on a recording before adding it (``meegxer-plugin run
--script FILE -i REC``; ``docs/PORTING.md``).

The script runs as the one strip of a pipeline over the recording, offline,
as it would once added (the same conformance and the same runner, through
the plugin protocol), and what came out is summed up: its channels, rate,
length and events, or its feature windows. ``out`` writes the result too: a
``.set`` or ``.edf`` for a signal, a ``.tsv`` for features. Running it is the
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


@dataclass
class Trial:
    """What a script made of a recording."""

    name: str
    kind: str  # "signal" or "features"
    channels: list[str] = field(default_factory=list)
    sfreq: float = 0.0
    n_samples: int = 0
    events: Counter[str] = field(default_factory=Counter)
    windows: int = 0
    features: list[str] = field(default_factory=list)
    written: str = ""

    def summary(self) -> str:
        if self.kind == "features":
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
) -> Trial:
    """Run ``script`` over ``recording`` as one strip, offline, and say what
    came out (and write it to ``out``)."""
    command = index.script_command(script)
    launcher = LocalLauncher(tuple(command), allow=True)
    origin = index.origin_of({"command": command, "script": str(script)})
    cls = plugin_class(probe_manifest(launcher), launcher, origin=origin)
    registry = Registry([*Registry.from_entry_points().as_mapping().values(), cls])
    name = cls.manifest.name
    kind = cls.manifest.outputs[0].type.value
    if kind == "features":
        tap = {"node": "meegxer.core.feature_probe", "params": {"keep_windows": 0}}
    else:
        tap = {"node": "meegxer.core.probe", "params": {"keep_seconds": 0}}
    nodes: list[dict[str, Any]] = [
        {"id": "input", "node": "meegxer.core.file",
         "params": {"path": str(recording)}},
        {"id": "script", "node": name, "params": dict(params or {})},
        {"id": "tap", "input": "script", **tap},
    ]  # fmt: skip
    if out is not None:
        writers = FEATURE_WRITERS if kind == "features" else SIGNAL_WRITERS
        writer = writers.get(out.suffix.lower())
        if writer is None:
            raise ValueError(
                f"{out.name}: write {'features' if kind == 'features' else 'a signal'} "
                f"to {' or '.join(sorted(writers))}"
            )
        settings = {"path": str(out), "overwrite": overwrite}
        nodes.append({"id": "out", "node": writer, "input": "script",
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
    if isinstance(probe, FeatureProbe):
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
