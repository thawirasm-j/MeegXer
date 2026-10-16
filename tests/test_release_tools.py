"""tools/set_version.py: one version for every package, siblings pinned exactly."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "set_version", ROOT / "tools/set_version.py"
)
assert _spec and _spec.loader
set_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(set_version)


def test_every_package_is_found_and_stamped() -> None:
    found = set_version.packages()
    assert set(found) == {"meegxer", "meegxer-core", "meegxer-nodes",
                          "meegxer-plugins", "meegxer-gui", "meegxer-cli"}  # fmt: skip
    for name, path in found.items():
        text = set_version.stamp(path.read_text(encoding="utf-8"), "1.2.0rc1",
                                 set(found) - {name})  # fmt: skip
        project = tomllib.loads(text)["project"]
        assert project["version"] == "1.2.0rc1"
        for dep in project["dependencies"]:
            if dep.split("==")[0].split(">")[0] in found:
                assert dep.endswith("==1.2.0rc1"), (name, dep)
            else:
                assert "1.2.0rc1" not in dep  # third-party pins untouched


def test_sibling_specifiers_and_extras_are_replaced() -> None:
    text = (
        'version = "0.0.0"\ndependencies = [\n'
        '    "meegxer-core[lsl]>=0.0",\n    "numpy>=2",\n]\n'
    )
    out = set_version.stamp(text, "0.3.0", {"meegxer-core"})
    assert '"meegxer-core[lsl]==0.3.0",' in out and '"numpy>=2",' in out


@pytest.mark.parametrize("tag", ["v0.1.0", "0.1.0a1", "v1.0rc2", "v2.0.0.post1"])
def test_release_versions_are_accepted(tag: str) -> None:
    assert set_version.VERSION.match(tag.removeprefix("v"))


@pytest.mark.parametrize("tag", ["latest", "v0.1.0-beta", "v1.0+local", "v1..0"])
def test_bad_tags_are_refused(tag: str) -> None:
    assert not set_version.VERSION.match(tag.removeprefix("v"))
