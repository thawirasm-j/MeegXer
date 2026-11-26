"""Octave/EEGLAB scripts (M11.5; docs/OCTAVE.md): a ``.m`` file with one
function that takes and returns an EEGLAB ``EEG`` structure::

    function EEG = lab_bandpass (EEG, varargin)
    % Band-pass with EEGLAB's firfilt (the strip's description).
    %
    % Parameters
    % ----------
    % low : float = 1
    %     Low cut-off (Hz).
    % high : float = 40
    %     High cut-off (Hz).
      opt = struct (varargin{:});
      EEG = pop_eegfiltnew (EEG, opt.low, opt.high);
    end

The help block right after the ``function`` line gives the description (its
first line) and the settings: ``name : type = default`` (``float``, ``int``,
``bool``, ``str``, or a choice ``{'fir', 'iir'}``) with a description below
it; a trailing ``(unit)`` becomes the unit. MeegXer reads it without running
Octave, and calls the script EEGLAB-style, with every setting as a name-value
pair: ``EEG = lab_bandpass (EEG, 'low', 1, 'high', 40)``.

The script runs in the Octave image (``meegxer.plugins.octave_image``),
sandboxed like any plugin container: its code is sent in when its process
starts (nothing is mounted), and MeegXer's runner (``octave/
meegxer_octave_script.m``) hands it the recording as ``EEG`` in microvolts,
as EEGLAB expects, and takes back ``EEG.data`` (the same channels and length)
and the events it added to ``EEG.event``. That script sees the whole
recording at once (offline). A live script also takes and returns its state,
and is called block by block, live and offline alike (decided 2026-11-25)::

    function [EEG, state] = lab_highpass (EEG, state, varargin)

``state`` is ``[]`` on the first block; whatever it returns comes back with
the next one. ``EEG`` is that block (``EEG.xmin``: its start, in seconds since
the first sample).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from meegxer.plugins.octave_image import EEGLAB_VERSION, check_image, default_image
from meegxer.plugins.octave_plugins import declared
from meegxer.plugins.script import ScriptError
from meegxer.plugins.worker import ContainerLauncher

SPEC_VERSION = "0.2.5"
_FUNCTION = re.compile(
    r"^\s*function\s+(?:(\[[^\]]*\]|\w+)\s*=\s*)?(\w+)\s*(?:\(([^)]*)\))?"
)
_SETTING = re.compile(
    r"^(\w+)\s*:\s*(float|int|bool|str|\{[^}]*\})\s*(?:=\s*(.+?))?\s*$"
)
_UNIT = re.compile(r"\(([^()\s]{1,8})\)\.?\s*$")
_TYPES = {"float": "float", "int": "int", "bool": "bool", "str": "string"}
# the runner's command in the image: its stdout keeps the protocol (descriptor
# 3) while everything EEGLAB prints goes to stderr
RUN = ("exec 3>&1 1>&2; exec octave-cli --no-gui --quiet --norc --no-history "
       "--path /app --eval meegxer_octave_script")  # fmt: skip


def is_octave_script(path: Path) -> bool:
    return path.suffix.lower() == ".m"


def _comment(line: str) -> str | None:
    """The text of a comment line (``%`` or ``#``), or None."""
    stripped = line.strip()
    if stripped.startswith(("%", "#")):
        return stripped.lstrip("%#")[1:] if len(stripped.lstrip("%#")) else ""
    return None


def _value(text: str, kind: str, choices: list[str], name: str) -> Any:
    """A default as written (``1``, ``true``, ``'fir'``) as its setting's type."""
    raw = text.strip()
    if raw[:1] in "'\"" and raw[-1:] == raw[:1] and len(raw) >= 2:
        raw = raw[1:-1]
        if kind in ("string", "enum"):
            if kind == "enum" and raw not in choices:
                raise ScriptError(f"setting {name!r}: default {raw!r} is not one "
                                  f"of {choices}")  # fmt: skip
            return raw
    try:
        if kind == "bool" and raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
    except ValueError:
        pass
    raise ScriptError(f"setting {name!r}: {text.strip()!r} is not a {kind} default")


def script_manifest(path: Path) -> dict[str, Any]:
    """The manifest of an Octave script, from its ``function`` line and the
    help block below it."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = next((i for i, line in enumerate(lines) if _FUNCTION.match(line)), None)
    if head is None:
        raise ScriptError(
            f"{path.name}: an Octave script is one function: "
            f"function EEG = {path.stem} (EEG, varargin)"
        )
    match = _FUNCTION.match(lines[head])
    assert match is not None
    outs = [o.strip() for o in (match.group(1) or "").strip("[]").replace(",", " ")
            .split() if o.strip()]  # fmt: skip
    name, args = match.group(2), [a.strip() for a in (match.group(3) or "").split(",")]
    if name != path.stem:
        raise ScriptError(f"{path.name}: the function is {name!r}; Octave needs it "
                          f"named as its file, {path.stem!r}")  # fmt: skip
    if not args or not args[0]:
        raise ScriptError(f"{name} needs the recording as its first argument: "
                          f"function EEG = {name} (EEG, varargin)")  # fmt: skip
    live = len(outs) == 2
    if live and (len(args) < 2 or args[1] != outs[1]):
        raise ScriptError(f"{name} returns its state: take it as the second "
                          f"argument, function [EEG, {outs[1]}] = {name} (EEG, "
                          f"{outs[1]}, varargin)")  # fmt: skip
    if len(outs) > 2:
        raise ScriptError(f"{name}: return EEG, or EEG and its state: "
                          f"function [EEG, state] = {name} (EEG, state, "
                          "varargin)")  # fmt: skip
    help_lines = []
    for line in lines[head + 1 :]:
        text = _comment(line)
        if text is None:
            if line.strip():
                break
            continue
        help_lines.append(text.rstrip())
    description = next((t.strip() for t in help_lines if t.strip()), "")
    params = _settings(help_lines)
    if params and "varargin" not in args:
        raise ScriptError(f"{name} has settings: take them as name-value pairs, "
                          f"function EEG = {name} (EEG, varargin)")  # fmt: skip
    stem = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "script"
    return {
        "spec_version": SPEC_VERSION,
        "name": f"script.{stem}",
        "version": "0.1.0",
        "type": "stream",
        "availability": "both" if live else "offline",
        "description": description.rstrip(".") + "."
        if description
        else f"Octave script {path.name}",
        "inputs": [{"name": "in", "type": "signal"}],
        "outputs": [{"name": "out", "type": "signal"}],
        "params": params,
    }


def _settings(help_lines: list[str]) -> list[dict[str, Any]]:
    """The ``Parameters`` section of a help block as manifest params."""
    try:
        start = next(i for i, t in enumerate(help_lines)
                     if t.strip() == "Parameters")  # fmt: skip
    except StopIteration:
        return []
    params: list[dict[str, Any]] = []
    for text in help_lines[start + 1 :]:
        stripped = text.strip()
        if not stripped or re.fullmatch(r"-{3,}", stripped):
            continue
        if re.fullmatch(r"[A-Z][A-Za-z ]+", stripped) and not text.startswith(" "):
            break  # the next section (Returns, Notes...)
        found = None if text.startswith((" ", "\t")) else _SETTING.match(stripped)
        if found:
            name, kind_text, default = found.groups()
            if default is None:
                raise ScriptError(f"setting {name!r} needs a default value "
                                  f"({name} : {kind_text} = ...)")  # fmt: skip
            spec: dict[str, Any] = {"name": name}
            if kind_text.startswith("{"):
                choices = [c.strip().strip("'\"") for c in kind_text[1:-1].split(",")
                           if c.strip()]  # fmt: skip
                spec.update(type="enum", choices=choices)
                spec["default"] = _value(default, "enum", choices, name)
            else:
                spec["type"] = _TYPES[kind_text]
                spec["default"] = _value(default, spec["type"], [], name)
            params.append(spec)
        elif params and text.startswith((" ", "\t")):
            spec = params[-1]
            line = stripped
            unit = _UNIT.search(line)
            if unit and spec["type"] in ("int", "float") and "unit" not in spec:
                spec["unit"] = unit.group(1)
                line = line[: unit.start()].rstrip()
            joined = (spec.get("description", "") + " " + line).strip()
            spec["description"] = joined.rstrip(".")
        elif not params:
            raise ScriptError(f"Parameters: write each setting as `name : type = "
                              f"default`, not {stripped!r}")  # fmt: skip
    return params


def code_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin_of(path: Path, image: str = "") -> dict[str, Any]:
    """What sessions record of an Octave script: its file, the SHA-256 of its
    code, the image (EEGLAB) it runs with and the EEGLAB plugins it declares
    (name -> commit)."""
    origin: dict[str, Any] = {"octave_script": str(path.resolve()),
              "script": path.name,
              "sha256": code_sha256(path), "image": image or default_image(),
              "eeglab": EEGLAB_VERSION}  # fmt: skip
    plugins = declared(path)
    if plugins:
        origin["eeglab_plugins"] = {p.name: p.commit for p in plugins}
    return origin


@dataclass(frozen=True)
class OctaveScriptLauncher:
    """An Octave script in a container of the Octave image, sandboxed as any
    plugin container; its code (and manifest) is the first frame on stdin,
    read when the process starts, so an edited script needs no rebuild."""

    path: str
    image: str = ""  # default_image(): MEEGXER_OCTAVE_IMAGE, else the version's
    engine: str | None = None
    memory: str = "4g"
    cpuset: str | None = None  # a live run's CPUs (red team L1)

    sandboxed = True

    @property
    def tag(self) -> str:
        return self.image or default_image()

    def with_cpuset(self, cpuset: str | None) -> OctaveScriptLauncher:
        return dataclasses.replace(self, cpuset=cpuset)

    def _container(self) -> ContainerLauncher:
        return ContainerLauncher(self.tag, self.engine, memory=self.memory,
                                 cpuset=self.cpuset)  # fmt: skip

    def command(self, name: str) -> list[str]:
        # missing, stale, or without a plugin the script declares: say so now
        check_image(self.tag, self.engine, declared(Path(self.path)))
        base = self._container().command(name)  # [..., image]
        return [*base[:-1], "--entrypoint", "sh", base[-1], "-c", RUN]

    def kill(self, name: str) -> None:
        self._container().kill(name)

    def environment(self) -> dict[str, str] | None:
        return None

    def describe(self) -> str:
        return f"Octave script {Path(self.path).name} (container {self.tag})"

    def preamble(self) -> list[tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]]:
        path = Path(self.path)
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        header = {"type": "script", "name": path.name,
                  "manifest": script_manifest(path),
                  "plugins": [p.name for p in declared(path)],
                  "files": [{"name": path.name, "data": "f0",
                             "sha256": digest}]}  # fmt: skip
        return [(header, {"f0": np.frombuffer(source, dtype=np.uint8)})]
