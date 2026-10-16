import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from meegxer.core.registry import Registry
from meegxer.plugins import index
from meegxer.plugins.cli import main as cli_main
from meegxer.plugins.conformance import Check, format_report, run_conformance
from meegxer.plugins.worker import ContainerLauncher, LocalLauncher

REPO_RUNTIME = Path(__file__).resolve().parents[1]


def passed(checks: list[Check]) -> dict[str, bool]:
    return {c.name: c.passed for c in checks}


def test_good_plugin_is_conformant(plugin: Callable[..., LocalLauncher]) -> None:
    checks = run_conformance(plugin("ema"))
    assert all(c.passed for c in checks), format_report(checks)
    assert "restart with host-held state" in passed(checks)


def test_fitted_plugin_is_conformant(plugin: Callable[..., LocalLauncher]) -> None:
    checks = run_conformance(plugin("center"))
    assert all(c.passed for c in checks), format_report(checks)


def test_buggy_plugin_is_reported(plugin: Callable[..., LocalLauncher]) -> None:
    checks = run_conformance(plugin("forgetful"))
    results = passed(checks)
    assert results["handshake"] and results["transform"]
    assert not results["acceptance test 1 (block-wise == one pass)"]
    assert "NOT conformant" in format_report(checks)


def test_reference_python_plugin_is_conformant(reference_plugin: LocalLauncher) -> None:
    checks = run_conformance(reference_plugin)
    assert all(c.passed for c in checks), format_report(checks)


OCTAVE = shutil.which("octave-cli")


@pytest.mark.skipif(OCTAVE is None, reason="GNU Octave not installed")
def test_reference_octave_plugin_is_conformant() -> None:
    launcher = LocalLauncher((
        "octave-cli", "--no-gui", "--quiet", "--norc", "--no-history",
        "--path", str(REPO_RUNTIME / "sdk" / "octave"),
        "--path", str(REPO_RUNTIME / "plugins" / "octave_iir_bandpass"),
        "--eval", "octave_iir_bandpass",
    ), allow=True)  # fmt: skip
    checks = run_conformance(launcher, worker_options={"start_timeout": 120})
    assert all(c.passed for c in checks), format_report(checks)


IMAGES = [i for i in os.environ.get("MEEGXER_TEST_IMAGES", "").split(",") if i]


@pytest.mark.skipif(not IMAGES, reason="set MEEGXER_TEST_IMAGES to test plugin images")
@pytest.mark.parametrize("image", IMAGES)
def test_plugin_image_is_conformant_in_the_sandbox(image: str) -> None:
    checks = run_conformance(
        ContainerLauncher(image), worker_options={"start_timeout": 300}
    )
    assert all(c.passed for c in checks), format_report(checks)


def test_index_and_node_provider(
    plugin: Callable[..., LocalLauncher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "plugins.json"
    monkeypatch.setenv(index.INDEX_ENV, str(path))
    launcher = plugin("ema")
    entry = index.add_local(list(launcher.argv))
    assert entry["manifest"]["name"] == "test.ema" and entry["unsandboxed"]
    assert json.loads(path.read_text())["plugins"][0]["manifest"]["name"] == "test.ema"

    monkeypatch.delenv("MEEGXER_ALLOW_UNSANDBOXED", raising=False)
    assert index.node_classes() == []  # unsandboxed entries need explicit permission
    monkeypatch.setenv("MEEGXER_ALLOW_UNSANDBOXED", "1")
    registry = Registry.from_entry_points()  # runtime's provider entry point
    assert "test.ema" in registry.names()
    assert "meegxer.nodes.iir_filter" in registry.names()

    path.write_text(
        json.dumps({"plugins": [{"image": "x", "manifest": {"name": "bad"}}]})
    )
    assert index.node_classes() == []  # broken entries are skipped, not fatal
    assert index.remove("bad") and not index.remove("bad")


def test_cli_check_and_list(
    plugin: Callable[..., LocalLauncher],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = plugin("ema")
    assert cli_main(["check", "--local", *launcher.argv]) == 0
    assert "conformant" in capsys.readouterr().out
    bad = plugin("forgetful")
    assert cli_main(["check", "--local", *bad.argv]) == 1
    monkeypatch.setenv(index.INDEX_ENV, str(tmp_path / "i.json"))
    index.add_local([sys.executable, "-u", launcher.argv[-1]])
    capsys.readouterr()
    assert cli_main(["list"]) == 0
    assert "test.ema" in capsys.readouterr().out and cli_main(["remove", "nope"]) == 1
