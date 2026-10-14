"""Self-tests for the architecture guards, so they cannot silently become vacuous.

Each test injects a deliberate violation into a temporary copy and asserts that the
guard (import-linter contracts, sterile wheel check) reports it.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ["core", "sterile", "runtime", "gui", "cli"]


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lint_imports(
    tmp_path: Path, inject: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run lint-imports on a copy of all src trees with extra modules injected."""
    paths = []
    for pkg in PACKAGES:
        dst = tmp_path / pkg
        shutil.copytree(REPO / "packages" / f"meegxer-{pkg}" / "src", dst)
        paths.append(str(dst))
    for relpath, source in inject.items():
        target = tmp_path / relpath
        target.write_text(source)
    shutil.copy(REPO / "pyproject.toml", tmp_path / "pyproject.toml")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    lint_imports = Path(sys.executable).with_name(
        "lint-imports.exe" if sys.platform == "win32" else "lint-imports"
    )
    return subprocess.run(
        [str(lint_imports), "--no-cache"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_contracts_pass_on_clean_tree(tmp_path: Path) -> None:
    result = _lint_imports(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("relpath", "source", "broken_contract"),
    [
        ("core/meegxer/core/_bad.py", "import PySide6\n", "Rule 1"),
        ("core/meegxer/core/_bad.py", "import meegxer.gui\n", "Rule 1"),
        ("cli/meegxer/cli/_bad.py", "import meegxer.runtime\n", "Rule 10"),
        ("sterile/meegxer/sterile/_bad.py", "import docker\n", "Rule 10"),
        (
            "sterile/meegxer/sterile/_bad.py",
            "import pyqtgraph\n",
            "Built-ins stay headless",
        ),
    ],
)
def test_contracts_catch_violation(
    tmp_path: Path, relpath: str, source: str, broken_contract: str
) -> None:
    result = _lint_imports(tmp_path, {relpath: source})
    assert result.returncode != 0
    broken = [line for line in result.stdout.splitlines() if line.endswith("BROKEN")]
    assert any(line.startswith(broken_contract) for line in broken), result.stdout


def _fake_wheel(path: Path, files: list[str], requires: list[str]) -> Path:
    metadata = "Metadata-Version: 2.4\nName: fake\nVersion: 0.0.0\n" + "".join(
        f"Requires-Dist: {r}\n" for r in requires
    )
    with zipfile.ZipFile(path, "w") as zf:
        for name in files:
            zf.writestr(name, "")
        zf.writestr("fake-0.0.0.dist-info/METADATA", metadata)
    return path


def test_sterile_check_accepts_clean_wheel(tmp_path: Path) -> None:
    check = _load_tool("check_sterile")
    wheel = _fake_wheel(
        tmp_path / "fake-0.0.0-py3-none-any.whl",
        ["meegxer/core/__init__.py"],
        ["numpy>=2.1", "meegxer-core"],
    )
    assert check.check_wheels([wheel]) == []


@pytest.mark.parametrize(
    ("files", "requires"),
    [
        (["meegxer/runtime/__init__.py"], []),
        ([], ["meegxer-runtime"]),
        ([], ["meegxer_runtime>=0.1"]),
        ([], ["docker>=7 ; python_version >= '3.13'"]),
    ],
)
def test_sterile_check_rejects_runtime(
    tmp_path: Path, files: list[str], requires: list[str]
) -> None:
    check = _load_tool("check_sterile")
    wheel = _fake_wheel(tmp_path / "fake-0.0.0-py3-none-any.whl", files, requires)
    assert check.check_wheels([wheel]) != []
