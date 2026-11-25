"""EEGLAB plugins an Octave script declares (docs/OCTAVE.md; decided
2026-11-25): fetched when the Octave image is built, never while a script runs.

A script names what it needs in its help block::

    % Requires: clean_rawdata
    % Requires: zapline_plus = https://github.com/MariusKlug/zapline-plus @ 0123...

A name from ``CATALOG`` (the plugins MeegXer knows, each pinned to a full
commit and with its licence) or any git repository at a full 40-character
commit. ``meegxer-plugin octave build`` fetches every plugin the added
scripts declare into the image and lists them in a label
(``org.meegxer.octave.plugins``); a script whose plugins the image lacks is
refused, saying to build again. Each script gets only its own plugins on its
path. Firfilt and ICLabel are in every image (``octave_image.PINS``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from meegxer.plugins.script import ScriptError


@dataclass(frozen=True)
class EeglabPlugin:
    """An EEGLAB plugin pinned to one commit of its repository."""

    name: str  # its folder under EEGLAB's plugins/
    url: str
    commit: str
    licence: str = ""

    def pin(self) -> str:
        return f"{self.name}@{self.commit}"


# The plugins EEGLAB 2026.0.0 itself pins (its .gitmodules), at those commits.
CATALOG: dict[str, EeglabPlugin] = {
    p.name.lower(): p
    for p in (
        EeglabPlugin(
            "clean_rawdata", "https://github.com/sccn/clean_rawdata.git",
            "d4b143f2a7719cf12d46c9b3e15aa827edb05614",
            "GPL-3.0; asr_calibrate.m and asr_process.m: UC Regents, "
            "non-commercial use only",
        ),
        EeglabPlugin(
            "dipfit", "https://github.com/sccn/dipfit.git",
            "b0b660e7efca0a9b68a4afb4e1749042e327b90c", "GPL-2.0-or-later",
        ),
        EeglabPlugin(
            "EEG-BIDS", "https://github.com/sccn/EEG-BIDS.git",
            "8486b1dde369079578452e632ee1aac29fe86db8", "GPL-2.0-or-later",
        ),
    )
}  # fmt: skip
BUILT_IN = {"firfilt", "iclabel"}  # in every image

_REQUIRES = re.compile(r"^\s*[%#]+\s*Requires\s*:\s*(.+?)\s*$", re.IGNORECASE)
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_GIT = re.compile(r"^(\S+)\s*=\s*(\S+?)\s*@\s*([0-9a-fA-F]{40})$")
# a repository URL the build's shell can take as is: https, no metacharacters
_URL = re.compile(r"^https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~%+-]*)*$")


def _one(text: str, where: str) -> EeglabPlugin | None:
    """One declared plugin: a catalog name, or ``name = URL @ commit``."""
    text = text.strip()
    found = _GIT.match(text)
    if found:
        name, url, commit = found.groups()
        if not _NAME.match(name):
            raise ScriptError(f"{where}: {name!r} is not a plugin folder name")
        if name.lower() in BUILT_IN:
            raise ScriptError(f"{where}: {name} is in every Octave image already")
        if not _URL.match(url):
            raise ScriptError(f"{where}: {url!r} is not a repository URL MeegXer "
                              "fetches (https://host/path, letters, digits and "
                              "._~%+- only)")  # fmt: skip
        return EeglabPlugin(name, url, commit.lower())
    if text.lower() in BUILT_IN:
        return None  # in every image already
    known = CATALOG.get(text.lower())
    if known is None:
        names = ", ".join(sorted(p.name for p in CATALOG.values()))
        raise ScriptError(
            f"{where}: {text!r} is not an EEGLAB plugin MeegXer knows ({names}; "
            "firfilt and ICLabel are always there); name another by its "
            "repository and full commit: name = https://github.com/... @ COMMIT"
        )
    return known


def named(name: str) -> EeglabPlugin:
    """A plugin MeegXer knows, by name (``meegxer-plugin octave build --with``)."""
    plugin = _one(name, "--with")
    if plugin is None:
        raise ScriptError(f"{name} is in every Octave image already")
    return plugin


def declared(path: Path) -> list[EeglabPlugin]:
    """The EEGLAB plugins an Octave script's help block declares
    (``% Requires: ...`` lines; several names per line, comma-separated)."""
    found: dict[str, EeglabPlugin] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _REQUIRES.match(line)
        if not match:
            continue
        items = [match.group(1)] if "=" in match.group(1) else match.group(1).split(",")
        for item in items:
            plugin = _one(item, path.name)
            if plugin is None:
                continue
            same = found.get(plugin.name.lower())
            if same is not None and same != plugin:
                raise ScriptError(f"{path.name}: {plugin.name} is declared twice, "
                                  "at different commits")  # fmt: skip
            found[plugin.name.lower()] = plugin
    return sorted(found.values(), key=lambda p: p.name.lower())


def merged(plugins: list[EeglabPlugin], where: str = "") -> list[EeglabPlugin]:
    """``plugins`` once each, refusing one name at two commits (two scripts
    asking for one plugin folder, differently)."""
    found: dict[str, EeglabPlugin] = {}
    for plugin in plugins:
        same = found.get(plugin.name.lower())
        if same is not None and (same.url, same.commit) != (plugin.url, plugin.commit):
            raise ScriptError(
                f"{where}the EEGLAB plugin {plugin.name} is asked for at two "
                f"commits ({same.commit[:8]} from {same.url}, {plugin.commit[:8]} "
                f"from {plugin.url}); one image holds one: make the scripts agree"
            )
        found[plugin.name.lower()] = plugin
    return sorted(found.values(), key=lambda p: p.name.lower())


def label_text(plugins: list[EeglabPlugin]) -> str:
    """The image label: ``name@commit,...`` (sorted)."""
    return ",".join(p.pin() for p in sorted(plugins, key=lambda p: p.name.lower()))


def from_label(text: str) -> dict[str, str]:
    """An image label back as name -> commit."""
    out = {}
    for item in text.split(","):
        name, _, commit = item.strip().partition("@")
        if name and commit and not name.startswith("<"):  # "<no value>": none
            out[name] = commit
    return out
