"""Self-tests for the architecture guards, so they cannot silently become vacuous.

Each test injects a deliberate violation into a temporary copy and asserts that the
guard (the import-linter contracts) reports it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ["core", "nodes", "plugins", "gui", "cli"]


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
        ("cli/meegxer/cli/_bad.py", "import meegxer.plugins\n", "Plugins load only"),
        ("nodes/meegxer/nodes/_bad.py", "import docker\n", "Plugins load only"),
        (
            "nodes/meegxer/nodes/_bad.py",
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
