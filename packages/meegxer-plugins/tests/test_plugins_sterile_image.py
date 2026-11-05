"""The Sterile image (``meegxer sterile build``): built from the installed
MeegXer, labelled with its code's hash, checked before a run. A fake engine
on PATH stands in for Docker (CI's container job builds the real image)."""

from __future__ import annotations

import json
import os
import stat
import sys
from importlib import metadata
from pathlib import Path

import pytest

from meegxer.core.sterile import build_image, check_image
from meegxer.plugins import sterile_image
from meegxer.plugins.sterile import CODE_PACKAGES, SterileError, code_hash
from meegxer.plugins.sterile_image import CODE_LABEL, context, requirements


def test_requirements_pin_what_is_installed() -> None:
    """Every third-party dependency of the worker's packages, at the version
    installed here; never MeegXer itself nor the keyboard sink's library."""
    pins = dict(line.split("==") for line in requirements())
    for name in ("numpy", "scipy", "mne", "scikit-learn", "python-picard"):
        assert pins[name] == metadata.version(name)
    assert not [p for p in pins if p.startswith("meegxer") or p == "pynput"]


def test_the_context_holds_the_installed_code(tmp_path: Path) -> None:
    folder = context(tmp_path)
    dockerfile = (folder / "Dockerfile").read_text()
    assert f'{CODE_LABEL}="{code_hash()}"' in dockerfile
    assert (
        f"python:{sys.version_info.major}.{sys.version_info.minor}-slim" in dockerfile
    )
    assert 'ENTRYPOINT ["python", "-m", "meegxer.plugins.sterile_worker"]' in dockerfile
    for name in CODE_PACKAGES:
        assert (folder / Path(*name.split(".")) / "__init__.py").is_file()
    assert (folder / "meegxer/plugins/sterile_worker.py").is_file()
    assert not list(folder.rglob("__pycache__"))
    assert (folder / "requirements.txt").read_text().splitlines() == requirements()


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``docker`` on PATH: records its arguments; ``image inspect``
    answers with the label in ``label.txt`` (no file: no such image)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    script = bin_dir / "docker"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, pathlib\n"
        f"log = pathlib.Path({str(log)!r})\n"
        "with log.open('a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "label = pathlib.Path(" + repr(str(tmp_path / "label.txt")) + ")\n"
        "down = pathlib.Path(" + repr(str(tmp_path / "down")) + ")\n"
        "if down.exists():\n"
        "    sys.stderr.write('failed to connect to the docker API\\n')\n"
        "    sys.exit(1)\n"
        "if sys.argv[1] == 'version':\n"
        "    print('29.0.0'); sys.exit(0)\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    if not label.exists(): sys.exit(1)\n"
        "    print(label.read_text())\n"
        "else:\n"
        "    out = sys.stdout.buffer\n"
        "    out.write('step 1/3 \\u2501\\u2501\\n'.encode('utf-8'))  # a bar\n"
        "    out.write(b'odd \\xff byte\\n'); out.write(b'done\\n'); out.flush()\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


@pytest.mark.skipif(sys.platform == "win32", reason="a fake engine as a script")
def test_build_runs_the_engine_and_says_what_it_does(engine: Path) -> None:
    lines: list[str] = []
    tag = build_image(tag="meegxer-sterile:test", engine="docker", log=lines.append)
    assert tag == "meegxer-sterile:test"
    calls = [json.loads(line) for line in
             (engine / "calls.jsonl").read_text().splitlines()]  # fmt: skip
    assert calls[0][0] == "version"  # the engine answers first
    call = next(c for c in calls if c[0] == "build")
    assert call[:3] == ["build", "-t", "meegxer-sterile:test"]
    assert lines[0].startswith("Building meegxer-sterile:test from MeegXer")
    assert lines[-1] == "done"
    # the build's output is UTF-8 whatever the console's code page (Windows:
    # cp1252), and a byte that is not is shown, never a crash
    assert "step 1/3 \u2501\u2501" in lines and "odd \ufffd byte" in lines


@pytest.mark.skipif(sys.platform == "win32", reason="a fake engine as a script")
def test_check_says_missing_stale_or_up_to_date(engine: Path) -> None:
    with pytest.raises(
        SterileError, match="not built yet: run `meegxer sterile build`"
    ):
        check_image(tag="meegxer-sterile:test", engine="docker")
    (engine / "label.txt").write_text("0" * 64)
    with pytest.raises(SterileError, match="built from other code"):
        check_image(tag="meegxer-sterile:test", engine="docker")
    (engine / "label.txt").write_text(code_hash())
    assert check_image(tag="meegxer-sterile:test", engine="docker") == (
        "meegxer-sterile:test"
    )


def test_no_engine_is_said_in_words(monkeypatch: pytest.MonkeyPatch) -> None:
    def none(preferred: str | None = None) -> str:
        raise FileNotFoundError("no container engine found (install Docker or Podman)")

    monkeypatch.setattr(sterile_image, "find_engine", none)
    with pytest.raises(SterileError, match="needs Docker or Podman"):
        sterile_image.check_image()
    with pytest.raises(SterileError, match="needs Docker or Podman"):
        sterile_image.build_image()


@pytest.mark.skipif(sys.platform == "win32", reason="a fake engine as a script")
def test_an_engine_not_running_is_said_so(engine: Path) -> None:
    """Docker installed but Docker Desktop not started: said as that (not
    "not built yet"), for the check and the build alike."""
    (engine / "down").touch()
    with pytest.raises(SterileError, match="installed but not running: start "
                                           "Docker Desktop"):  # fmt: skip
        check_image(tag="meegxer-sterile:test", engine="docker")
    with pytest.raises(SterileError, match="installed but not running"):
        build_image(tag="meegxer-sterile:test", engine="docker", log=print)
