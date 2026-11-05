"""The Sterile worker's image, built from the installed MeegXer
(``meegxer sterile build``; ``docs/STERILE.md``).

The image holds exactly the code the host runs: the installed ``meegxer.core``,
``meegxer.nodes`` and ``meegxer.plugins`` are copied into it (whatever they
are: a release, or an edited checkout), and their third-party dependencies are
installed at the versions installed here. The image is labelled with the
code's hash (``meegxer.plugins.sterile.code_hash``), so a run can tell at once
whether it is up to date. Building needs the network once (the Python base
image and the dependencies); running it never does.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from meegxer.core import __version__
from meegxer.plugins.sterile import (
    CODE_PACKAGES,
    IMAGE,
    IMAGE_ENV,
    LOCAL_ENV,
    SterileError,
    code_files,
    code_hash,
    package_dir,
)
from meegxer.plugins.worker import find_engine

CODE_LABEL = "org.meegxer.code"
VERSION_LABEL = "org.meegxer.version"
DISTRIBUTIONS = ("meegxer-core", "meegxer-nodes", "meegxer-plugins")
# what the worker never runs: the keyboard sink's library (a host sink; on
# Linux it builds evdev from source)
LEFT_OUT = {"pynput", "evdev", "python-xlib"}
# the environment the image's Python sees, for dependency markers
_MACHINES = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64",
             "aarch64": "aarch64"}  # fmt: skip
MACHINE = _MACHINES.get(platform.machine().lower(), platform.machine())
LINUX: dict[str, str] = {"sys_platform": "linux", "platform_system": "Linux",
                         "os_name": "posix", "platform_machine": MACHINE}  # fmt: skip

DOCKERFILE = """\
# The MeegXer Sterile worker (docs/STERILE.md), made by `meegxer sterile build`.
FROM python:{python}-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \\
    && useradd --create-home --uid 10001 worker
COPY meegxer /app/meegxer
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \\
    HOME=/tmp MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp
LABEL {code_label}="{code}" {version_label}="{version}"
USER worker
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "meegxer.plugins.sterile_worker"]
"""


def requirements() -> list[str]:
    """The third-party distributions the worker's packages need, with every
    dependency of theirs, pinned to the versions installed here
    (``name==version``), as Linux sees their markers."""
    env = {key: str(value) for key, value in default_environment().items()}
    env.update(LINUX, extra="")
    pinned: dict[str, str] = {}
    todo = list(DISTRIBUTIONS)
    seen: set[str] = set()
    while todo:
        name = canonicalize_name(todo.pop())
        if name in seen or name in LEFT_OUT:
            continue
        seen.add(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue  # needed only where it is not installed (a marker)
        if not name.startswith("meegxer"):
            pinned[name] = dist.version
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            todo.append(req.name)
    return [f"{name}=={pinned[name]}" for name in sorted(pinned)]


def context(folder: Path) -> Path:
    """Write the build context into ``folder``: the Dockerfile, the pinned
    requirements and a copy of the worker's MeegXer packages."""
    for name in CODE_PACKAGES:
        root = package_dir(name)
        target = folder / Path(*name.split("."))
        for path in code_files(name):
            out = target / path.relative_to(root)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    (folder / "requirements.txt").write_text("\n".join(requirements()) + "\n")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    (folder / "Dockerfile").write_text(DOCKERFILE.format(
        python=python, code_label=CODE_LABEL, code=code_hash(),
        version_label=VERSION_LABEL, version=__version__,
    ))  # fmt: skip
    return folder


def build_image(
    *,
    tag: str = IMAGE,
    engine: str | None = None,
    log: Callable[[str], None] = print,
) -> str:
    """Build the Sterile image (``tag``) from the installed MeegXer; the build's
    output goes to ``log`` line by line. Returns the tag."""
    try:
        exe = find_engine(engine)
    except FileNotFoundError as exc:
        raise SterileError(f"Sterile mode needs Docker or Podman: {exc}") from None
    with tempfile.TemporaryDirectory(prefix="meegxer-sterile-") as tmp:
        folder = context(Path(tmp))
        log(f"Building {tag} from MeegXer {__version__} ({code_hash()[:12]})…")
        proc = subprocess.Popen(
            [exe, "build", "-t", tag, str(folder)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
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
    label = '{{index .Config.Labels "' + CODE_LABEL + '"}}'
    done = subprocess.run([exe, "image", "inspect", "--format", label, tag],
                          capture_output=True, text=True, check=False,
                          timeout=60)  # fmt: skip
    if done.returncode != 0:
        return None
    return done.stdout.strip() or ""


def check_image(tag: str | None = None, engine: str | None = None) -> str:
    """Refuse at once (before a run starts) when the Sterile image (``tag``,
    else ``MEEGXER_STERILE_IMAGE``, else ``meegxer-sterile:VERSION``) is
    missing or was built from other code, saying what to do. With
    ``MEEGXER_STERILE_LOCAL=1`` (development) no image is needed."""
    if os.environ.get(LOCAL_ENV) == "1" and tag is None:
        return "local worker (development, no container)"
    tag = tag or os.environ.get(IMAGE_ENV) or IMAGE
    try:
        found = image_code(tag, engine)
    except FileNotFoundError as exc:
        raise SterileError(f"Sterile mode needs Docker or Podman: {exc}") from None
    if found is None:
        raise SterileError(
            f"the Sterile image {tag} is not built yet: run `meegxer sterile build`"
        )
    if found != code_hash():
        raise SterileError(
            f"the Sterile image {tag} was built from other code than this MeegXer: "
            "run `meegxer sterile build` again"
        )
    return tag
