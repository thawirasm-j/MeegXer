"""Live Octave scripts (M11.5; docs/OCTAVE.md; decided 2026-11-25):
``function [EEG, state] = name (EEG, state, varargin)``, called block by
block, live and offline alike, with the state the host keeps between blocks
([] on the first). The runs need the image (MEEGXER_TEST_OCTAVE_IMAGE)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.realtime import RealtimeScheduler
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.plugins.adapter import plugin_class
from meegxer.plugins.octave_script import (
    OctaveScriptLauncher,
    origin_of,
    script_manifest,
)
from meegxer.plugins.scaffold import new_script
from meegxer.plugins.script import ScriptError

IMAGE = os.environ.get("MEEGXER_TEST_OCTAVE_IMAGE", "")
needs_image = pytest.mark.skipif(not IMAGE, reason="set MEEGXER_TEST_OCTAVE_IMAGE")
SOURCE = {"duration": 2.0, "n_channels": 3, "sfreq": 200.0}

# counts the blocks it saw and the samples before each, in its state
COUNTER = """function [EEG, state] = counter (EEG, state)
% Count the blocks.
  if (isempty (state))
    state = struct ("blocks", 0, "seen", 0);
  end
  assert (abs (EEG.xmin - state.seen / EEG.srate) < 1e-9);  % where it starts
  state.blocks = state.blocks + 1;
  state.seen = state.seen + EEG.pnts;
  EEG.data(:) = state.blocks;  % each block numbered
end
"""


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / f"{name}.m"
    path.write_text(text)
    return path


def test_a_script_that_keeps_state_runs_live(tmp_path: Path) -> None:
    manifest = script_manifest(write(tmp_path, "counter", COUNTER))
    assert manifest["availability"] == "both"
    path = new_script("lab_highpass", "octave-live", tmp_path)
    assert path.name == "lab_highpass.m"
    manifest = script_manifest(path)
    assert manifest["availability"] == "both"
    assert [p["name"] for p in manifest["params"]] == ["cutoff", "order"]
    offline = script_manifest(new_script("lab_filter", "octave", tmp_path))
    assert offline["availability"] == "offline"


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("nostate", "function [EEG, state] = nostate (EEG)\n% S.\nend\n",
         "take it as the second argument"),
        ("other", "function [EEG, s] = other (EEG, state)\n% S.\nend\n",
         r"\[EEG, s\] = other \(EEG, s, varargin\)"),
        ("three", "function [EEG, a, b] = three (EEG, a, b)\n% S.\nend\n",
         "return EEG, or EEG and its state"),
    ],
)  # fmt: skip
def test_live_conventions_are_explained(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    with pytest.raises(ScriptError, match=message):
        script_manifest(write(tmp_path, name, text))


# -- in the image -------------------------------------------------------------------


def _cls(path: Path) -> type:
    return plugin_class(script_manifest(path), OctaveScriptLauncher(str(path), IMAGE),
                        origin=origin_of(path, IMAGE))  # fmt: skip


def _raw() -> np.ndarray:
    probe = MemoryProbe()
    BatchScheduler().run(Graph.chain([SyntheticSource(SOURCE), probe]))
    return probe.signal().data


@needs_image
def test_the_state_carries_from_block_to_block(tmp_path: Path) -> None:
    """[] on the first block; what the script returns comes back with the
    next; EEG.xmin is the block's start."""
    cls = _cls(write(tmp_path, "counter", COUNTER))
    probe = MemoryProbe()
    graph = Graph.chain([SyntheticSource(SOURCE), Stage(cls(), {}), probe])
    try:
        BatchScheduler(block_size=100).run(graph)
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    y = probe.signal().data
    assert y.shape == (3, 400)
    np.testing.assert_allclose(y[0, ::100], np.array([1, 2, 3, 4]) * 1e-6)


@needs_image
def test_the_live_template_filters_live_as_offline(tmp_path: Path) -> None:
    """The template (a Butterworth high-pass with filter's carried state)
    gives scipy's result, in a live run and in a batch run in other blocks."""
    cls = _cls(new_script("lab_highpass", "octave-live", tmp_path))

    def graph(probe: MemoryProbe) -> Graph:
        stage = Stage(cls(), {"cutoff": 2.0, "order": 2})
        return Graph.chain([SyntheticSource(SOURCE), stage, probe])

    batch_probe, live_probe = MemoryProbe(), MemoryProbe()
    batch = graph(batch_probe)
    live = graph(live_probe)
    try:
        BatchScheduler(block_size=37).run(batch)
        scheduler = RealtimeScheduler(live)
        scheduler.start()
        assert scheduler.wait(60)
        assert scheduler.status()["errors"] == []
    finally:
        for g in (batch, live):
            g.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    x = _raw().astype(np.float32).astype(np.float64)  # as the data travels
    b, a = signal.butter(2, 2.0, "highpass", fs=200.0)
    expected = signal.lfilter(b, a, x, axis=-1)
    for probe in (batch_probe, live_probe):
        np.testing.assert_allclose(probe.signal().data, expected, rtol=1e-6,
                                   atol=1e-11)  # fmt: skip


@needs_image
def test_the_live_template_passes_the_conformance_suite(tmp_path: Path) -> None:
    """Acceptance test 1 among them: block by block == one pass."""
    from meegxer.plugins.conformance import format_report, run_conformance

    path = new_script("lab_highpass", "octave-live", tmp_path)
    checks = run_conformance(OctaveScriptLauncher(str(path), IMAGE))
    assert all(c.passed for c in checks), format_report(checks)
    blockwise = next(c for c in checks if c.name.startswith("acceptance test 1"))
    assert "skipped" not in blockwise.detail


@needs_image
def test_a_script_that_forgets_its_state_fails_the_suite(tmp_path: Path) -> None:
    """Restarting the filter every block: conformance says block-wise differs."""
    from meegxer.plugins.conformance import run_conformance

    forgets = write(tmp_path, "forgets", """function [EEG, state] = forgets (EEG, state)
% Forget the filter state.
  [b, a] = butter (2, 2 / (EEG.srate / 2), "high");
  EEG.data = filter (b, a, double (EEG.data), [], 2);
end
""")  # fmt: skip
    checks = run_conformance(OctaveScriptLauncher(str(forgets), IMAGE))
    blockwise = next(c for c in checks if c.name.startswith("acceptance test 1"))
    assert not blockwise.passed
