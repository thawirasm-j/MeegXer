"""Extra packages a custom script declares (``REQUIRES``; docs/CUSTOM_SCRIPTS.md):
read without running the script, checked against what is installed, recorded
in sessions and installed into the Sterile image."""

from __future__ import annotations

import copy
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from meegxer.plugins import index, sterile_image
from meegxer.plugins.cli import main
from meegxer.plugins.requires import (
    MissingRequirements,
    declared,
    installed,
    versions,
)
from meegxer.plugins.script import ScriptError
from meegxer.plugins.sterile import SterileError

NUMPY = metadata.version("numpy")
PYTEST = metadata.version("pytest")  # installed here, not shipped by MeegXer
MISSING = "meegxer-no-such-package-xyz"


def script(tmp_path: Path, requires: str, name: str = "needs.py") -> Path:
    path = tmp_path / name
    path.write_text(
        '"""Needs extra packages."""\n\n'
        f"{requires}\n\n\n"
        "def process(raw, gain: float = 1.0):\n"
        "    return raw.get_data() * gain\n"
    )
    return path


def test_requires_is_read_without_running_the_script(tmp_path: Path) -> None:
    path = script(tmp_path, 'REQUIRES = ["numpy", "mne-connectivity>=0.7"]\n'
                  "raise SystemExit('never run')")  # fmt: skip
    assert [str(r) for r in declared(path)] == ["numpy", "mne-connectivity>=0.7"]
    annotated = script(tmp_path, 'REQUIRES: tuple[str, ...] = ("numpy",)', "a.py")
    assert [r.name for r in declared(annotated)] == ["numpy"]
    assert declared(script(tmp_path, "", "none.py")) == []
    for bad, message in [
        ('REQUIRES = ["num" + "py"]', "written out"),
        ("REQUIRES = PACKAGES", "written out"),
        ('REQUIRES = ["numpy @ https://example.org/numpy.whl"]', "not a URL"),
        ('REQUIRES = ["numpy>>1"]', "REQUIRES 'numpy>>1'"),
    ]:
        with pytest.raises(ScriptError, match=message):
            declared(script(tmp_path, bad, "bad.py"))


def test_missing_packages_are_named_with_how_to_install_them(tmp_path: Path) -> None:
    ok = script(tmp_path, 'REQUIRES = ["NumPy"]')
    assert installed(ok) == {"numpy": NUMPY}
    assert versions(ok) == {"numpy": NUMPY}
    path = script(tmp_path, f'REQUIRES = ["numpy<1", "{MISSING}"]', "missing.py")
    with pytest.raises(MissingRequirements) as caught:
        installed(path, also=["pyriemann"])
    text = str(caught.value)
    assert f"numpy<1 ({NUMPY} installed)" in text
    assert f"{MISSING} (not installed)" in text
    # every script's packages in one command: a new --with list replaces the last
    assert (f"uv tool install --prerelease allow --with 'numpy<1' --with {MISSING} "
            "--with pyriemann meegxer") in text  # fmt: skip
    assert versions(path) == {"numpy": NUMPY, MISSING: "not installed"}


def test_the_cli_refuses_a_script_whose_packages_are_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = script(tmp_path, f'REQUIRES = ["{MISSING}"]', "lab_missing.py")
    for args in (["check", "--script", str(path)],
                 ["add", "--script", str(path), "--accept-risk"],
                 ["run", "--script", str(path), "-i", str(path)]):  # fmt: skip
        assert main(args) == 1
        err = capsys.readouterr().err
        assert f"{MISSING} (not installed)" in err and "uv tool install" in err
    assert not [e for e in index.load_index()
                if e["manifest"]["name"] == "script.lab_missing"]  # fmt: skip


def test_added_scripts_record_what_they_need(tmp_path: Path) -> None:
    """The index keeps what a script declares; sessions record the versions
    installed; a session saved with other versions warns, as library versions
    do."""
    from meegxer.core.batch import BatchScheduler
    from meegxer.core.graph import Graph
    from meegxer.core.registry import Registry
    from meegxer.core.session import build_session, load_session
    from meegxer.core.stage import Stage

    path = script(tmp_path, f'REQUIRES = ["numpy", "pytest>={PYTEST}"]', "lab_ok.py")
    assert main(["add", "--script", str(path), "--accept-risk"]) == 0
    (entry,) = [e for e in index.load_index()
                if e["manifest"]["name"] == "script.lab_ok"]  # fmt: skip
    assert entry["requires"] == ["numpy", f"pytest>={PYTEST}"]
    assert index.origin_of(entry)["requires"] == {"numpy": NUMPY, "pytest": PYTEST}
    assert {"numpy", f"pytest>={PYTEST}"} <= set(index.declared_requires())
    assert f"pytest>={PYTEST}" not in index.declared_requires(but="script.lab_ok")

    registry = Registry.from_entry_points()
    source = {"id": "input", "node": "meegxer.core.synthetic",
              "params": {"duration": 1.0, "n_channels": 2, "sfreq": 100.0}}  # fmt: skip
    graph = Graph.from_dict(
        {
            "spec_version": "0.2.9",
            "nodes": [source, {"id": "s", "node": "script.lab_ok"}],
        },
        registry,
    )
    try:
        BatchScheduler().run(graph)
        session = build_session(graph)
        provenance = session["nodes"]["s"]["provenance"]
        assert provenance["requires"] == {"numpy": NUMPY, "pytest": PYTEST}
        older: dict[str, Any] = copy.deepcopy(session)
        older["nodes"]["s"]["provenance"]["requires"]["pytest"] = "0.1"
        loaded = load_session(older, registry)
        assert any("pytest" in w and "recorded 0.1" in w for w in loaded.warnings)
        loaded_same = load_session(session, registry)
        assert not any("packages differ" in w for w in loaded_same.warnings)
    finally:
        impl = graph["s"].impl
        assert isinstance(impl, Stage)
        impl.processor.close()
        index.remove("script.lab_ok")


def test_the_image_installs_and_names_the_scripts_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``meegxer sterile build`` installs the added scripts' packages (pinned
    as here) and labels the image with them; a run whose scripts need one the
    image lacks is refused, saying to build again. A package MeegXer ships
    anyway (numpy) is not asked of the label."""
    path = script(tmp_path, f'REQUIRES = ["numpy", "pytest>={PYTEST}"]', "lab_img.py")
    assert main(["add", "--script", str(path), "--accept-risk"]) == 0
    try:
        packages = sterile_image.script_packages()
        assert packages == {"numpy": NUMPY, "pytest": PYTEST}
        folder = sterile_image.context(tmp_path / "ctx")
        pins = (folder / "requirements.txt").read_text().split()
        assert f"pytest=={PYTEST}" in pins and f"numpy=={NUMPY}" in pins
        assert f'org.meegxer.requires="numpy=={NUMPY},pytest=={PYTEST}"' in (
            folder / "Dockerfile").read_text()  # fmt: skip

        from meegxer.plugins.sterile import code_hash

        monkeypatch.setattr(sterile_image, "image_code", lambda *a, **k: code_hash())
        labels: dict[str, dict[str, str]] = {"value": {}}
        monkeypatch.setattr(sterile_image, "image_requires",
                            lambda *a, **k: labels["value"])  # fmt: skip
        with pytest.raises(SterileError, match=rf"lacks pytest=={PYTEST},.*build"):
            sterile_image.check_image("meegxer-sterile:test")
        labels["value"] = {"pytest": PYTEST}  # numpy: MeegXer's own, not asked
        assert (
            sterile_image.check_image("meegxer-sterile:test") == "meegxer-sterile:test"
        )
        with pytest.raises(SterileError, match=r"lacks pytest==0\.1"):
            sterile_image.check_image(
                "meegxer-sterile:test", requires={"pytest": "0.1"}
            )
    finally:
        index.remove("script.lab_img")
    missing = script(tmp_path, f'REQUIRES = ["{MISSING}"]', "lab_gone.py")
    entry = {"command": index.script_command(missing), "script": str(missing),
             "consent": {"accepted": "now"}, "requires": [MISSING],
             "manifest": {"name": "script.lab_gone"}}  # fmt: skip
    index.save_index([*index.load_index(), entry])
    try:
        with pytest.raises(SterileError, match=f"{MISSING}.*not installed here"):
            sterile_image.script_packages()
        assert MISSING not in sterile_image.script_packages(strict=False)
    finally:
        index.remove("script.lab_gone")


def test_a_sterile_run_refuses_a_script_whose_packages_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from meegxer.core.graph import Graph
    from meegxer.core.sources import SyntheticSource
    from meegxer.core.stage import Stage
    from meegxer.plugins.adapter import plugin_class, probe_manifest
    from meegxer.plugins.sterile import isolate
    from meegxer.plugins.worker import LocalLauncher

    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    path = script(tmp_path, "", "lab_later.py")
    command = index.script_command(path)
    launcher = LocalLauncher(tuple(command), allow=True)
    origin = index.origin_of({"command": command, "script": str(path)})
    cls = plugin_class(probe_manifest(launcher), launcher, origin=origin)
    # REQUIRES added after the strip was made: the run reads it at its start
    path.write_text(f'REQUIRES = ["{MISSING}"]\n' + path.read_text())
    src = SyntheticSource({"duration": 1.0, "n_channels": 2, "sfreq": 100.0})
    graph = Graph.chain([src, Stage(cls(), {})])
    try:
        with pytest.raises(SterileError, match="not installed"):
            isolate(graph, local=True)
    finally:
        graph.nodes[1].impl.processor.close()  # type: ignore[union-attr]
