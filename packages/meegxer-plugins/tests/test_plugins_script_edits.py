"""An edited custom script is noticed while MeegXer runs (found running
docs/TESTING.md Y4): its code is part of the strip's fingerprint, so its cached
output is not reused and a model fitted with the old code needs a refit. A
Sterile run keeps its cached outputs apart from a local run's, so it computes
in its container."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.cache import MemoryCache
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage, StageStatus, signal_fingerprint
from meegxer.plugins import adapter, index
from meegxer.plugins.adapter import PluginProcessor, plugin_class, probe_manifest
from meegxer.plugins.worker import LocalLauncher

SCALE = '''"""Scale (an edited script's test)."""


def process(raw):
    return raw.get_data() * 2.0
'''

CALIBRATED = '''"""Remove the calibration's channel means."""


def fit(raw):
    return {"mean": raw.get_data().mean(axis=1, keepdims=True)}


def transform(data, state, model):
    return data - model["mean"], state
'''


def strip(path: Path) -> type[PluginProcessor]:
    """A script strip as `meegxer-plugin add --script` makes it."""
    command = index.script_command(path)
    launcher = LocalLauncher(tuple(command), allow=True)
    origin = index.origin_of({"command": command, "script": str(path)})
    return plugin_class(probe_manifest(launcher), launcher, origin=origin)


def run(
    cls: type, cache: MemoryCache, **options: object
) -> tuple[np.ndarray, list[str]]:
    probe = MemoryProbe()
    src = SyntheticSource({"duration": 1.0, "n_channels": 2, "sfreq": 100.0})
    graph = Graph.chain([src, Stage(cls(), {}), probe])
    try:
        result = BatchScheduler(cache=cache, **options).run(graph)  # type: ignore[arg-type]
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    return probe.signal().data, result.cached


def test_an_edited_script_is_not_served_from_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "scale.py"
    path.write_text(SCALE)
    cls = strip(path)  # made once, as the daemon does: the edit comes later
    cache = MemoryCache()
    first, cached = run(cls, cache)
    assert cached == []
    again, cached = run(cls, cache)
    assert len(cached) == 1 and np.array_equal(again, first)  # unchanged: reused
    path.write_text(SCALE.replace("2.0", "3.0"))
    edited, cached = run(cls, cache)
    assert cached == []  # the new code ran
    np.testing.assert_allclose(edited, first * 1.5, rtol=1e-6)
    # and the session records the code as it is now
    assert cls.provenance()["sha256"] == index.code_sha256(index.script_command(path))


def test_a_model_fitted_with_the_old_code_needs_a_refit(tmp_path: Path) -> None:
    path = tmp_path / "centre.py"
    path.write_text(CALIBRATED)
    stage = Stage(strip(path)(), {})
    block = SyntheticSource({"duration": 1.0, "n_channels": 2, "sfreq": 100.0}).read(
        10**6
    )
    assert block is not None
    fp = signal_fingerprint(block.info)
    try:
        stage.fit(block, fp)
        assert stage.status(fp) is StageStatus.READY
        path.write_text(CALIBRATED + "\n# edited\n")
        assert stage.status(fp) is StageStatus.NEEDS_REFIT  # never silently reused
    finally:
        stage.processor.close()


def test_built_in_strips_keep_their_fingerprints(tmp_path: Path) -> None:
    """Only plugins add their code: a built-in strip's fingerprint (and so
    its cached outputs and saved models) is what it was."""
    from meegxer.core.node import Processor
    from meegxer.core.registry import Registry
    from meegxer.core.serialization import canonical_json, fingerprint

    cls = Registry.from_entry_points().as_mapping()["meegxer.nodes.notch"]
    notch = cls()
    assert isinstance(notch, Processor)
    stage = Stage(notch, {})
    want = fingerprint("stage", stage.manifest.name, stage.manifest.version,
                       canonical_json(stage.params), ",".join(stage.channel_types),
                       "input")  # fmt: skip
    assert stage.code_identity() == "" and stage.fingerprint("input") == want


def test_a_sterile_run_keeps_its_cached_outputs_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local run's cached output is not a Sterile run's: switched on, the
    run computes in its container (the same output), then reuses its own."""
    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    monkeypatch.setenv("MEEGXER_STERILE_LOCAL", "1")  # the worker as a process
    path = tmp_path / "scale.py"
    path.write_text(SCALE)
    cls = strip(path)
    cache = MemoryCache()
    local, cached = run(cls, cache, sterile=False)
    assert cached == []
    isolated, cached = run(cls, cache, sterile=True)
    assert cached == [] and np.array_equal(isolated, local)  # computed, same output
    _, cached = run(cls, cache, sterile=True)
    assert len(cached) == 1  # its own, now
    _, cached = run(cls, cache, sterile=False)
    assert len(cached) == 1  # and the local run's is still there


def test_a_big_script_folder_is_looked_at_once_a_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live strip asks on every block: a big folder (data files) is looked
    at again at most once a second, then hashed only if a file changed."""
    folder = tmp_path / "big_method"
    folder.mkdir()
    (folder / "main.py").write_text(SCALE)
    command = tuple(index.script_command(folder))
    first = adapter.current_code(command, "recorded")
    monkeypatch.setattr(adapter, "HASH_ALWAYS_BYTES", 0)  # every folder is big
    monkeypatch.setattr(adapter, "RECHECK_SECONDS", 3600.0)
    adapter._CODE.pop(command, None)
    assert adapter.current_code(command, "recorded") == first
    (folder / "weights.txt").write_text("1 2 3\n")
    assert adapter.current_code(command, "recorded") == first  # not yet looked at
    monkeypatch.setattr(adapter, "RECHECK_SECONDS", 0.0)
    assert adapter.current_code(command, "recorded") == index.code_sha256(list(command))
    assert adapter.current_code(command, "recorded") != first
