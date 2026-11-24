"""Octave/EEGLAB scripts (M11.5; docs/OCTAVE.md): settings read from the help
block, the script sent into the Octave image (sandboxed), the recording given
as an EEGLAB EEG structure in microvolts. The runs need the image: they run
with MEEGXER_TEST_OCTAVE_IMAGE (CI's container job)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.plugins import index, octave_image
from meegxer.plugins.adapter import plugin_class
from meegxer.plugins.octave_script import (
    RUN,
    OctaveScriptLauncher,
    origin_of,
    script_manifest,
)
from meegxer.plugins.scaffold import new_script
from meegxer.plugins.script import ScriptError

IMAGE = os.environ.get("MEEGXER_TEST_OCTAVE_IMAGE", "")
needs_image = pytest.mark.skipif(not IMAGE, reason="set MEEGXER_TEST_OCTAVE_IMAGE")

BANDPASS = """function EEG = lab_bandpass (EEG, varargin)
% Band-pass with EEGLAB's firfilt (pop_eegfiltnew).
%
% Parameters
% ----------
% low : float = 1
%     Low cut-off (Hz).
% high : float = 40
%     High cut-off
%     (Hz).
% order : int = 0
%     Filter order (0: firfilt's own choice).
% method : {'fir', 'none'} = 'fir'
%     Filter or not.
% mark : bool = true
%     Add a marker at 1 s.
% label : str = "checked"
%     The marker's label.
  opt = struct (varargin{:});
  if (strcmp (opt.method, 'fir'))
    EEG = pop_eegfiltnew (EEG, opt.low, opt.high);
  end
  if (opt.mark)
    EEG.event(end + 1) = struct ('type', opt.label, 'latency', EEG.srate + 1, ...
                                 'duration', 0);
  end
end
"""


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / f"{name}.m"
    path.write_text(text)
    return path


def test_settings_come_from_the_help_block(tmp_path: Path) -> None:
    manifest = script_manifest(write(tmp_path, "lab_bandpass", BANDPASS))
    assert manifest["name"] == "script.lab_bandpass"
    assert (
        manifest["description"] == "Band-pass with EEGLAB's firfilt (pop_eegfiltnew)."
    )
    assert manifest["availability"] == "offline"
    params = {p["name"]: p for p in manifest["params"]}
    assert params["low"] == {"name": "low", "type": "float", "default": 1.0,
                             "unit": "Hz", "description": "Low cut-off"}  # fmt: skip
    assert params["high"]["unit"] == "Hz"  # the unit on its own line
    assert params["order"]["type"] == "int" and params["order"]["default"] == 0
    assert params["method"]["type"] == "enum"
    assert params["method"]["choices"] == ["fir", "none"]
    assert params["mark"]["default"] is True
    assert params["label"] == {"name": "label", "type": "string",
                               "default": "checked",
                               "description": "The marker's label"}  # fmt: skip
    # a script without settings needs no varargin
    plain = write(tmp_path, "times2", "function EEG = times2 (EEG)\n"
                  "% Double.\n  EEG.data = EEG.data * 2;\nend\n")  # fmt: skip
    assert script_manifest(plain)["params"] == []


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("other", "function EEG = mine (EEG)\nend\n", "named as its file"),
        ("nofunc", "x = 1;\n", "one function"),
        ("noeeg", "function EEG = noeeg ()\nend\n", "first argument"),
        ("noargs", "function EEG = noargs (EEG)\n% Parameters\n% ----------\n"
                   "% low : float = 1\n%     Low.\nend\n", "varargin"),
        ("nodefault", "function EEG = nodefault (EEG, varargin)\n% Parameters\n"
                      "% ----------\n% low : float\n%     Low.\nend\n", "default"),
        ("badtype", "function EEG = badtype (EEG, varargin)\n% Parameters\n"
                    "% ----------\n% n : int = 1.5\n%     N.\nend\n", "not a int"),
        ("badchoice", "function EEG = badchoice (EEG, varargin)\n% Parameters\n"
                      "% ----------\n% m : {'a', 'b'} = 'c'\n%     M.\nend\n",
         "not one of"),
    ],
)  # fmt: skip
def test_script_conventions_are_explained(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    with pytest.raises(ScriptError, match=message):
        script_manifest(write(tmp_path, name, text))


def test_the_launcher_sends_the_code_into_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is mounted: the code and its manifest go in as the first frame,
    read when the process starts (an edit needs no rebuild); the container is
    the plugin sandbox, its entry point the runner."""
    monkeypatch.setattr("meegxer.plugins.octave_script.check_image",
                        lambda tag, engine=None: tag)  # fmt: skip
    monkeypatch.setattr("meegxer.plugins.worker.find_engine",
                        lambda preferred=None: "docker")  # fmt: skip
    path = write(tmp_path, "lab_bandpass", BANDPASS)
    launcher = OctaveScriptLauncher(str(path), image="meegxer-octave:test")
    command = launcher.command("meegxer-plugin-x")
    for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL"):
        assert flag in command
    assert command[-3:] == ["meegxer-octave:test", "-c", RUN]
    assert command[command.index("--entrypoint") + 1] == "sh"
    ((header, arrays),) = launcher.preamble()
    assert header["name"] == "lab_bandpass.m"
    assert header["manifest"] == script_manifest(path)
    assert arrays["f0"].tobytes() == path.read_bytes()
    path.write_text(BANDPASS.replace("high : float = 40", "high : float = 30"))
    ((header, arrays),) = launcher.preamble()  # read again at the next start
    high = next(p for p in header["manifest"]["params"] if p["name"] == "high")
    assert high["default"] == 30.0 and arrays["f0"].tobytes() == path.read_bytes()
    monkeypatch.setenv("MEEGXER_OCTAVE_IMAGE", "meegxer-octave:ci")
    assert OctaveScriptLauncher(str(path)).tag == "meegxer-octave:ci"


def test_added_scripts_need_no_notice_and_record_their_code(tmp_path: Path) -> None:
    """An Octave script runs sandboxed, so it is added without the local-code
    notice; sessions record its SHA-256 (as it is now), its image and EEGLAB;
    shared, only its file name."""
    from meegxer.core.session import share_safe

    path = write(tmp_path, "lab_bandpass", BANDPASS)
    entry = index.add_octave_script(path)
    try:
        assert "consent" not in entry and "command" not in entry
        launcher = index.launcher_for_entry(entry)
        assert isinstance(launcher, OctaveScriptLauncher) and launcher.sandboxed
        origin = index.origin_of(entry)
        assert origin == origin_of(path)
        assert origin["eeglab"] == octave_image.EEGLAB_VERSION
        names = [c.manifest.name for c in index.node_classes()]
        assert "script.lab_bandpass" in names  # loaded without consent
        cls = next(c for c in index.node_classes()
                   if c.manifest.name == "script.lab_bandpass")  # fmt: skip
        before = cls.provenance()["sha256"]
        path.write_text(BANDPASS + "\n% edited\n")
        assert cls.provenance()["sha256"] != before  # the code as it is now
        shared = share_safe({"meegxer_session": "1.0", "pipeline": {},
                        "nodes": {"s": {"node": "script.lab_bandpass",
                                        "version": "0.1.0",
                                        "provenance": origin}}})  # fmt: skip
        assert shared["nodes"]["s"]["provenance"]["octave_script"] == "lab_bandpass.m"
    finally:
        index.remove("script.lab_bandpass")


def test_sterile_mode_leaves_an_octave_script_in_its_own_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meegxer.plugins.sterile import isolate

    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    path = write(tmp_path, "lab_bandpass", BANDPASS)
    cls = plugin_class(script_manifest(path), OctaveScriptLauncher(str(path)),
                       origin=origin_of(path))  # fmt: skip
    src = SyntheticSource({"duration": 1.0, "n_channels": 2, "sfreq": 100.0})
    graph = Graph.chain([src, Stage(cls(), {})])
    with isolate(graph, local=True) as isolation:
        assert isolation.proxied == []  # no Sterile proxy, no refusal
        processor = graph.nodes[1].impl.processor  # type: ignore[union-attr]
        assert isinstance(processor.launcher, OctaveScriptLauncher)  # type: ignore[union-attr]


def test_the_template_is_a_valid_octave_script(tmp_path: Path) -> None:
    path = new_script("lab_filter", "octave", tmp_path)
    assert path.name == "lab_filter.m"
    manifest = script_manifest(path)
    assert [p["name"] for p in manifest["params"]] == ["low", "high"]


# -- in the image -------------------------------------------------------------------


Run = tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]


def run(path: Path, params: dict[str, Any] | None = None, **source: Any) -> Run:
    """The script over a synthetic recording, in the image: (input, output,
    events it added)."""
    cls = plugin_class(script_manifest(path), OctaveScriptLauncher(str(path), IMAGE),
                       origin=origin_of(path, IMAGE))  # fmt: skip
    src = {"duration": 4.0, "n_channels": 3, "sfreq": 250.0, **source}
    probe, raw = MemoryProbe(), MemoryProbe()
    graph = Graph.chain([SyntheticSource(src), Stage(cls(), params or {}), probe])
    try:
        result = BatchScheduler().run(graph)
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    BatchScheduler().run(Graph.chain([SyntheticSource(src), raw]))
    events = [(e.sample_index, e.label)
              for e in result.events.get(graph.nodes[1].id, ())]  # fmt: skip
    return raw.signal().data, probe.signal().data, events


@needs_image
def test_eeg_is_the_recording_in_microvolts(tmp_path: Path) -> None:
    """The bridge is exact (to the float32 every plugin's data travels in):
    doubling EEG.data doubles the signal; adding 1 to it adds 1 µV."""
    times2 = write(tmp_path, "times2", "function EEG = times2 (EEG)\n% Double.\n"
                   "  EEG.data = EEG.data * 2;\nend\n")  # fmt: skip
    x, y, _ = run(times2)
    np.testing.assert_array_equal(y, (x.astype(np.float32) * 2).astype(np.float64))
    plus = write(tmp_path, "plus1uv", "function EEG = plus1uv (EEG)\n% Add 1 µV.\n"
                 "  EEG.data = EEG.data + 1;\nend\n")  # fmt: skip
    x, y, _ = run(plus)
    np.testing.assert_allclose(y - x, 1e-6, atol=1e-11)
    # the structure EEGLAB functions expect
    shape = write(tmp_path, "shape", "function EEG = shape (EEG)\n% Check.\n"
                  "  assert (EEG.nbchan == 3 && EEG.pnts == 1000);\n"
                  "  assert (EEG.srate == 250);\n"
                  "  assert (strcmp (EEG.chanlocs(2).labels, 'Fp2'));\n"
                  "  assert (EEG.trials == 1 && EEG.xmin == 0);\nend\n")  # fmt: skip
    run(shape)


@needs_image
def test_settings_go_in_by_name_and_events_come_out(tmp_path: Path) -> None:
    path = write(tmp_path, "lab_bandpass", BANDPASS)
    _, unfiltered, events = run(path, {"method": "none", "label": "here"})
    assert events == [(250, "here")]  # latency srate + 1: 1 s
    x, _, _ = run(path, {"mark": False, "method": "none"})
    np.testing.assert_allclose(unfiltered, x, atol=1e-11)  # nothing done
    _, filtered, events = run(path, {"mark": False})
    assert events == [] and not np.allclose(filtered, x)  # pop_eegfiltnew ran


@needs_image
def test_a_script_that_changes_the_channels_is_explained(tmp_path: Path) -> None:
    path = write(tmp_path, "drop", "function EEG = drop (EEG)\n% Drop one.\n"
                 "  EEG.data = EEG.data(1:2, :);\nend\n")  # fmt: skip
    with pytest.raises(Exception, match=r"must return 3 channels x 1000 samples"):
        run(path)
    broken = write(tmp_path, "broken", "function EEG = broken (EEG)\n% Fails.\n"
                   "  error ('my own words');\nend\n")  # fmt: skip
    with pytest.raises(Exception, match="my own words"):
        run(broken)


@needs_image
def test_the_template_passes_the_conformance_suite(tmp_path: Path) -> None:
    from meegxer.plugins.conformance import format_report, run_conformance

    path = new_script("lab_filter", "octave", tmp_path)
    checks = run_conformance(OctaveScriptLauncher(str(path), IMAGE))
    assert all(c.passed for c in checks), format_report(checks)
    assert json.dumps([c.name for c in checks])
