"""The extra packages a custom script declares (``REQUIRES``; docs/CUSTOM_SCRIPTS.md).

A script that imports a package MeegXer does not ship names it at the top of
its entry file::

    REQUIRES = ["pyriemann", "mne-connectivity>=0.7"]

Each item is a PyPI requirement (a name, optionally extras and a version
range; no URLs or paths). The host reads it without running the script
(``declared``), checks that it is installed in MeegXer's own Python
(``installed``: ``meegxer-plugin add|check|run`` refuse a script whose
packages are missing, saying how to install them), sessions record the
versions used, and ``meegxer sterile build`` installs the same versions into
the Sterile image (``meegxer.plugins.sterile_image``).
"""

from __future__ import annotations

import ast
import shlex
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from meegxer.plugins.script import ScriptError, entry_file


class MissingRequirements(ScriptError):
    """A script needs packages that are not installed (or not in the range)."""


def declared(script: Path) -> list[Requirement]:
    """The ``REQUIRES`` of a script (file or folder), read from its entry file
    without running it: a list (or tuple) of requirement strings."""
    path = entry_file(script)
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError as exc:
        raise ScriptError(f"{path.name}: {exc}") from None
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if "REQUIRES" in names:
            value = node.value
    if value is None:
        return []
    try:
        items = ast.literal_eval(value)
    except ValueError:
        items = None
    if not isinstance(items, list | tuple) or not all(
        isinstance(i, str) for i in items
    ):
        raise ScriptError(
            f"{path.name}: REQUIRES must be a list of package names written out, "
            'e.g. REQUIRES = ["pyriemann", "mne-connectivity>=0.7"]'
        )
    found = []
    for text in items:
        try:
            req = Requirement(text)
        except InvalidRequirement as exc:
            raise ScriptError(f"{path.name}: REQUIRES {text!r}: {exc}") from None
        if req.url:
            raise ScriptError(
                f"{path.name}: REQUIRES {text!r}: name a package from PyPI, "
                "not a URL or a path"
            )
        found.append(req)
    return found


def installed(script: Path, also: Iterable[str] = ()) -> dict[str, str]:
    """The installed version of each package ``script`` declares (by its
    normalised name), or ``MissingRequirements`` saying which are missing or
    outside their range and how to install them. ``also``: what the scripts
    added before declare, named in that command too (a new ``uv tool install
    --with`` list replaces the last one)."""
    versions: dict[str, str] = {}
    wrong: list[str] = []
    wanted = declared(script)
    for req in wanted:
        name = canonicalize_name(req.name)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            wrong.append(f"{req} (not installed)")
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            wrong.append(f"{req} ({version} installed)")
            continue
        versions[name] = version
    if wrong:
        every = list(dict.fromkeys([*(str(r) for r in wanted), *also]))
        quoted = [shlex.quote(r) for r in every]
        raise MissingRequirements(
            f"{script.name} needs {', '.join(wrong)}. Install into MeegXer's "
            "Python, with every extra package your scripts use (a new list "
            "replaces the last): `uv tool install --prerelease allow "
            + " ".join(f"--with {q}" for q in quoted)
            + f" meegxer`, or `pip install {' '.join(quoted)}` where MeegXer runs"
        )
    return dict(sorted(versions.items()))


def versions(script: Path) -> dict[str, str]:
    """What sessions record: each declared package (normalised name) and its
    installed version, ``"not installed"`` when it is not; empty when the
    script declares nothing or cannot be read."""
    try:
        wanted = declared(script)
    except (OSError, ScriptError):
        return {}
    found = {}
    for req in wanted:
        name = canonicalize_name(req.name)
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not installed"
    return dict(sorted(found.items()))
