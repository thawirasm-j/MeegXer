"""The Octave image: GNU Octave with a pinned EEGLAB, where Octave/EEGLAB
scripts run (``meegxer-plugin octave build|check``; docs/OCTAVE.md).

Octave scripts (M11.5) always run in this image, sandboxed like any plugin
container (no network, read-only, no capabilities), so a script gives the same
result on every computer and a session can name the EEGLAB it ran with. The
image is built on the user's machine from Ubuntu's GNU Octave and
octave-signal and EEGLAB's own repositories at pinned commits: EEGLAB's core
(``eeglab.m`` and ``functions/``, BSD-2-Clause) with its firfilt and ICLabel
plugins (GPL-2.0-or-later), as EEGLAB ``EEGLAB_VERSION`` pins them. Its other
plugins are left out: clean_rawdata's ASR files are under the UC Regents
licence (the EEGLAB ASR plugin image fetches it for those who accept it),
dipfit bundles FieldTrip. MeegXer's own Octave files (``octave/``: the plugin
SDK and the runner) are copied in; the image is labelled with a hash of them
and of this recipe, so an image built from other code is refused.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from meegxer.core import __version__
from meegxer.plugins.octave_plugins import (
    EeglabPlugin,
    from_label,
    label_text,
    merged,
)
from meegxer.plugins.sterile import SterileError
from meegxer.plugins.sterile_image import ensure_running
from meegxer.plugins.worker import find_engine

IMAGE = f"meegxer-octave:{__version__}"
IMAGE_ENV = "MEEGXER_OCTAVE_IMAGE"  # another tag (CI's meegxer-octave:ci)
CODE_LABEL = "org.meegxer.octave.code"
PLUGINS_LABEL = "org.meegxer.octave.plugins"  # the scripts' EEGLAB plugins
VERSION_LABEL = "org.meegxer.version"
OCTAVE_DIR = Path(__file__).parent / "octave"  # MeegXer's .m files

# EEGLAB 2026.0.0 and the plugin commits its release pins (its .gitmodules)
EEGLAB_VERSION = "2026.0.0"
PINS: dict[str, tuple[str, str]] = {
    "eeglab": ("https://github.com/sccn/eeglab.git",
               "bcc710a8edd712738e48879b6846958a2be7be1d"),
    "firfilt": ("https://github.com/sccn/firfilt.git",
                "ff8227f85816364e5cd26d9d22665091be4ead17"),
    "ICLabel": ("https://github.com/sccn/ICLabel.git",
                "644578bd34c77262ab5563ae4af5fb7269ccc81c"),
}  # fmt: skip

DOCKERFILE = """\
# The MeegXer Octave image (docs/OCTAVE.md), made by `meegxer-plugin octave build`.
# EEGLAB {eeglab_version}: its core (BSD-2-Clause) with the firfilt and ICLabel
# plugins (GPL-2.0-or-later), fetched from their repositories at pinned commits.
FROM ubuntu:24.04
RUN apt-get update \\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      octave octave-signal git ca-certificates \\
 && fetch() {{ git init -q "$1" && git -C "$1" fetch -q --depth 1 "$2" "$3" \\
      && git -C "$1" checkout -q FETCH_HEAD && rm -rf "$1/.git"; }} \\
 && fetch /opt/eeglab {eeglab_url} {eeglab_commit} \\
 && rm -rf /opt/eeglab/sample_data /opt/eeglab/tutorial_scripts /opt/eeglab/plugins/* \\
 && fetch /opt/eeglab/plugins/firfilt {firfilt_url} {firfilt_commit} \\
 && fetch /opt/eeglab/plugins/ICLabel {iclabel_url} {iclabel_commit} \\
{extra} && apt-get purge -y git && apt-get autoremove -y \\
 && rm -rf /var/lib/apt/lists/* \\
 && useradd --create-home --uid 10001 octave
COPY octave /app
# Read-only filesystem: only /tmp is writable. EEGLAB prints to stdout, so the
# protocol keeps the original stdout as descriptor 3 and prints go to stderr.
ENV HOME=/tmp MEEGXER_PROTOCOL_FD=3 MEEGXER_EEGLAB=/opt/eeglab
LABEL {code_label}="{code}" {version_label}="{version}" \\
      {plugins_label}="{plugins}"
USER octave
WORKDIR /tmp
"""


def default_image() -> str:
    """The image Octave scripts run in: ``MEEGXER_OCTAVE_IMAGE``, else
    ``meegxer-octave:VERSION``."""
    return os.environ.get(IMAGE_ENV) or IMAGE


def recipe(plugins: Sequence[EeglabPlugin] = ()) -> str:
    """The Dockerfile, without its labels' values; with the EEGLAB plugins
    the scripts declare fetched too (the code hash covers it without them)."""
    eeglab, firfilt, iclabel = PINS["eeglab"], PINS["firfilt"], PINS["ICLabel"]
    extra = "".join(
        f" && fetch /opt/eeglab/plugins/{p.name} {p.url} {p.commit} \\\n"
        for p in sorted(plugins, key=lambda p: p.name.lower())
    )
    return DOCKERFILE.format(
        extra=extra, plugins_label=PLUGINS_LABEL, plugins="{plugins}",
        eeglab_version=EEGLAB_VERSION,
        eeglab_url=eeglab[0], eeglab_commit=eeglab[1],
        firfilt_url=firfilt[0], firfilt_commit=firfilt[1],
        iclabel_url=iclabel[0], iclabel_commit=iclabel[1],
        code_label=CODE_LABEL, code="{code}",
        version_label=VERSION_LABEL, version="{version}",
    )  # fmt: skip


def octave_files() -> list[Path]:
    """MeegXer's Octave files copied into the image."""
    return sorted(p for p in OCTAVE_DIR.iterdir() if p.suffix == ".m")


@functools.cache
def code_hash() -> str:
    """A SHA-256 over the recipe and MeegXer's Octave files: an image built
    from other ones is refused (an edited checkout, another version)."""
    digest = hashlib.sha256(recipe().encode())
    for path in octave_files():
        digest.update(f"\0{path.name}\0".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scripts_plugins() -> list[EeglabPlugin]:
    """Every EEGLAB plugin the added Octave scripts declare (the plugin index)."""
    from meegxer.plugins.index import load_index

    return merged([
        EeglabPlugin(item["name"], item["url"], item["commit"],
                     item.get("licence", ""))
        for entry in load_index() for item in entry.get("eeglab_plugins", [])
    ], "the added Octave scripts: ")  # fmt: skip


def context(folder: Path, plugins: Sequence[EeglabPlugin] | None = None) -> Path:
    """Write the build context into ``folder``: the Dockerfile and ``octave/``,
    with ``plugins`` (by default those the added scripts declare)."""
    plugins = scripts_plugins() if plugins is None else list(plugins)
    target = folder / "octave"
    target.mkdir(parents=True, exist_ok=True)
    for path in octave_files():
        (target / path.name).write_bytes(path.read_bytes())
    dockerfile = recipe(plugins).replace("{code}", code_hash())
    dockerfile = dockerfile.replace("{version}", __version__)
    dockerfile = dockerfile.replace("{plugins}", label_text(list(plugins)))
    (folder / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return folder


def build_image(
    *,
    tag: str = IMAGE,
    engine: str | None = None,
    log: Callable[[str], None] = print,
    plugins: Sequence[EeglabPlugin] | None = None,
) -> str:
    """Build the Octave image (``tag``); the build's output goes to ``log``
    line by line. Needs the network once (Ubuntu's packages, EEGLAB)."""
    try:
        exe = find_engine(engine)
    except FileNotFoundError as exc:
        raise SterileError(f"Octave scripts need Docker or Podman: {exc}") from None
    ensure_running(exe)
    with tempfile.TemporaryDirectory(prefix="meegxer-octave-") as tmp:
        plugins = scripts_plugins() if plugins is None else list(plugins)
        folder = context(Path(tmp), plugins)
        extra = f", with {', '.join(p.name for p in plugins)}" if plugins else ""
        log(f"Building {tag}: GNU Octave and EEGLAB {EEGLAB_VERSION}{extra} "
            f"({code_hash()[:12]})…")  # fmt: skip
        proc = subprocess.Popen(
            [exe, "build", "-t", tag, str(folder)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )  # fmt: skip
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip())
        if proc.wait() != 0:
            raise SterileError(f"building {tag} failed (see the lines above)")
    return tag


def image_code(tag: str = IMAGE, engine: str | None = None) -> str | None:
    """The code hash ``tag`` was built from, or None when there is no such
    image."""
    exe = find_engine(engine)
    ensure_running(exe)
    label = '{{index .Config.Labels "' + CODE_LABEL + '"}}'
    done = subprocess.run([exe, "image", "inspect", "--format", label, tag],
                          capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace",
                          timeout=60)  # fmt: skip
    if done.returncode != 0:
        return None
    return done.stdout.strip() or ""


def image_plugins(tag: str = IMAGE, engine: str | None = None) -> dict[str, str]:
    """The EEGLAB plugins ``tag`` holds beyond its base: name -> commit."""
    exe = find_engine(engine)
    label = '{{index .Config.Labels "' + PLUGINS_LABEL + '"}}'
    done = subprocess.run([exe, "image", "inspect", "--format", label, tag],
                          capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace",
                          timeout=60)  # fmt: skip
    return from_label(done.stdout.strip()) if done.returncode == 0 else {}


def check_image(
    tag: str = IMAGE,
    engine: str | None = None,
    plugins: Sequence[EeglabPlugin] = (),
) -> str:
    """Refuse, saying what to do, when the Octave image is missing, was
    built from other code, or lacks an EEGLAB plugin a script declares
    (``plugins``, at its commit)."""
    try:
        found = image_code(tag, engine)
    except FileNotFoundError as exc:
        raise SterileError(f"Octave scripts need Docker or Podman: {exc}") from None
    if found is None:
        raise SterileError(
            f"the Octave image {tag} is not built yet: run "
            "`meegxer-plugin octave build`"
        )
    if found != code_hash():
        raise SterileError(
            f"the Octave image {tag} was built from other code than this MeegXer: "
            "run `meegxer-plugin octave build` again"
        )
    if plugins:
        has = image_plugins(tag, engine)
        lacking = [p.name for p in plugins if has.get(p.name) != p.commit]
        if lacking:
            raise SterileError(
                f"the Octave image {tag} lacks the EEGLAB plugin "
                f"{', '.join(lacking)} a script declares (Requires): run "
                "`meegxer-plugin octave build` again"
            )
    return tag


def sandbox(tag: str, engine: str | None = None) -> list[str]:
    """The command that runs ``tag`` sandboxed as every plugin container is:
    no network, read-only (only /tmp writable), no capabilities."""
    exe = find_engine(engine)
    return [exe, "run", "--rm", "-i", "--network", "none", "--read-only",
            "--tmpfs", "/tmp", "--cap-drop", "ALL", "--security-opt",
            "no-new-privileges", "--memory", "4g", tag]  # fmt: skip


def self_check(tag: str = IMAGE, engine: str | None = None) -> dict[str, Any]:
    """Run the image's self-check in its sandbox: the versions it holds, and
    whether EEGLAB's pop_eegfiltnew filtered a test signal as it should."""
    check_image(tag, engine)
    command = [*sandbox(tag, engine), "octave-cli", "--no-gui", "--quiet",
               "--norc", "--no-history", "--path", "/app",
               "--eval", "meegxer_octave_check"]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True, check=False,
                          encoding="utf-8", errors="replace",
                          timeout=300)  # fmt: skip
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
        raise SterileError(f"the Octave image {tag} did not start: {tail}")
    found: dict[str, Any] = json.loads(lines[-1])
    return found
