"""The Octave image (M11.5; docs/OCTAVE.md): GNU Octave with a pinned EEGLAB,
built from MeegXer's recipe and Octave files, labelled with their hash, checked
before use. A fake engine stands in for Docker; with MEEGXER_TEST_OCTAVE_IMAGE
(CI's container job) the real image runs its self-check."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from meegxer.plugins import octave_image
from meegxer.plugins.cli import main
from meegxer.plugins.octave_image import (
    EEGLAB_VERSION,
    PINS,
    build_image,
    check_image,
    code_hash,
    context,
    recipe,
)
from meegxer.plugins.sterile import SterileError

SDK = Path(__file__).resolve().parents[1] / "sdk" / "octave" / "meegxer_plugin_run.m"
REAL_IMAGE = os.environ.get("MEEGXER_TEST_OCTAVE_IMAGE", "")


def test_the_packaged_octave_sdk_is_the_published_one() -> None:
    packaged = octave_image.OCTAVE_DIR / "meegxer_plugin_run.m"
    assert packaged.read_bytes() == SDK.read_bytes()


def test_the_recipe_pins_eeglab_and_leaves_out_what_it_cannot_ship(
    tmp_path: Path,
) -> None:
    """EEGLAB's core with firfilt and ICLabel at the commits its release pins,
    each by its full commit; clean_rawdata (the UC Regents' ASR licence) and
    dipfit are left out; MeegXer's Octave files are copied in; the label is
    the code hash."""
    text = recipe()
    assert EEGLAB_VERSION == "2026.0.0"
    for url, commit in PINS.values():
        assert len(commit) == 40 and f"{url} {commit}" in text
    assert "clean_rawdata" not in text and "dipfit" not in text
    assert "rm -rf /opt/eeglab/sample_data" in text
    folder = context(tmp_path / "ctx")
    assert sorted(p.name for p in (folder / "octave").iterdir()) == [
        "meegxer_octave_check.m", "meegxer_octave_script.m",
        "meegxer_plugin_run.m", "yulewalk.m"]  # fmt: skip
    dockerfile = (folder / "Dockerfile").read_text()
    assert f'org.meegxer.octave.code="{code_hash()}"' in dockerfile
    assert "{code}" not in dockerfile and "{version}" not in dockerfile


def test_the_code_hash_covers_the_octave_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = code_hash()
    copy = tmp_path / "octave"
    copy.mkdir()
    for path in octave_image.octave_files():
        (copy / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(octave_image, "OCTAVE_DIR", copy)
    octave_image.code_hash.cache_clear()
    try:
        assert code_hash() == before  # the same files: the same hash
        (copy / "meegxer_octave_check.m").write_text("% edited\n")
        octave_image.code_hash.cache_clear()
        assert code_hash() != before
    finally:
        octave_image.code_hash.cache_clear()


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``docker`` on PATH: records its arguments; ``image inspect``
    answers with the label in ``label.txt`` (no file: no such image); ``run``
    prints the self-check's line from ``check.json``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, pathlib\n"
        f"root = pathlib.Path({str(tmp_path)!r})\n"
        "with (root / 'calls.jsonl').open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'version':\n"
        "    print('29.0.0'); sys.exit(0)\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    label = root / 'label.txt'\n"
        "    if not label.exists(): sys.exit(1)\n"
        "    print(label.read_text()); sys.exit(0)\n"
        "if sys.argv[1] == 'run':\n"
        "    print('EEGLAB says hello')  # EEGLAB's own prints come first\n"
        "    print((root / 'check.json').read_text()); sys.exit(0)\n"
        "print('built'); sys.exit(0)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="a fake engine as a script")
def test_build_and_check(engine: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lines: list[str] = []
    assert build_image(tag="meegxer-octave:test", engine="docker", log=lines.append)
    assert lines[0].startswith(f"Building meegxer-octave:test: GNU Octave and EEGLAB "
                               f"{EEGLAB_VERSION}")  # fmt: skip
    with pytest.raises(SterileError, match="not built yet: run `meegxer-plugin "
                                           "octave build`"):  # fmt: skip
        check_image(tag="meegxer-octave:test", engine="docker")
    (engine / "label.txt").write_text("0" * 64)
    with pytest.raises(SterileError, match="built from other code"):
        check_image(tag="meegxer-octave:test", engine="docker")
    (engine / "label.txt").write_text(code_hash())
    assert check_image(tag="meegxer-octave:test", engine="docker")
    report = {"octave": "8.4.0", "signal": "1.4.5", "eeglab": EEGLAB_VERSION,
              "filter_ok": True}  # fmt: skip
    (engine / "check.json").write_text(json.dumps(report))
    assert octave_image.self_check("meegxer-octave:test", "docker") == report
    run = next(json.loads(line) for line in
               (engine / "calls.jsonl").read_text().splitlines()
               if json.loads(line)[0] == "run")  # fmt: skip
    # sandboxed as every plugin container
    for flag in ("--network", "none", "--read-only", "--cap-drop", "ALL"):
        assert flag in run
    assert run[-2:] == ["--eval", "meegxer_octave_check"]


@pytest.mark.skipif(sys.platform == "win32", reason="a fake engine as a script")
def test_the_cli_says_what_the_image_holds(
    engine: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MEEGXER_OCTAVE_IMAGE", raising=False)  # the default tag
    assert main(["octave", "check", "--engine", "docker"]) == 1
    assert "not built yet" in capsys.readouterr().err
    (engine / "label.txt").write_text(code_hash())
    (engine / "check.json").write_text(json.dumps(
        {"octave": "8.4.0", "signal": "1.4.5", "eeglab": EEGLAB_VERSION,
         "filter_ok": True}))  # fmt: skip
    assert main(["octave", "check", "--engine", "docker"]) == 0
    said = capsys.readouterr().out
    assert said.startswith(f"ok: {octave_image.IMAGE} is up to date (GNU Octave 8.4.0")
    assert f"EEGLAB {EEGLAB_VERSION}" in said
    (engine / "check.json").write_text(json.dumps(
        {"octave": "8.4.0", "signal": "1.4.5", "eeglab": EEGLAB_VERSION,
         "filter_ok": False}))  # fmt: skip
    assert main(["octave", "check", "--engine", "docker"]) == 1
    assert "pop_eegfiltnew gave a wrong result" in capsys.readouterr().err
    assert main(["octave", "build", "--engine", "docker"]) == 0
    build = ["octave", "build", "--engine", "docker", "--with"]
    assert main([*build, "clean_rawdata"]) == 0
    assert main([*build, "nosuch"]) == 1
    assert "not an EEGLAB plugin MeegXer knows" in capsys.readouterr().err


@pytest.mark.skipif(not REAL_IMAGE, reason="set MEEGXER_TEST_OCTAVE_IMAGE")
def test_the_real_image_runs_eeglab() -> None:
    """The built image (CI's container job): GNU Octave with the signal
    package, the pinned EEGLAB, and pop_eegfiltnew passing 10 Hz while
    removing 60 Hz, in the sandbox."""
    found = octave_image.self_check(REAL_IMAGE)
    assert found["eeglab"] == EEGLAB_VERSION and found["filter_ok"] is True
    assert found["octave"] and found["signal"]
