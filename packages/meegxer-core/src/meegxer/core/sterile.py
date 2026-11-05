"""Sterile mode (rule 10): a pipeline's processing runs isolated.

The core does not hold the container code (rule 9: plugins and the container
engine are reached only through entry points): ``sterilize`` loads the
``meegxer.isolation`` entry point (``meegxer.plugins.sterile:isolate``), which
moves every built-in processor of the graph into one Sterile worker, a
container with no network and a read-only filesystem, and gives back a handle
to close when the run ends. Sources, probes and sinks stay on the host.
``docs/STERILE.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points
from typing import Any, Protocol

from meegxer.core import settings

ENTRY_GROUP = "meegxer.isolation"
ENV = "MEEGXER_STERILE"  # 1/0, true/false, on/off: over the setting
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class SterileUnavailable(RuntimeError):
    """Sterile mode is not installed (no ``meegxer.isolation`` entry point)."""


class Isolation(Protocol):
    proxied: list[str]  # the node ids that run in the worker

    def close(self) -> None: ...

    def __enter__(self) -> Isolation: ...

    def __exit__(self, *exc: object) -> None: ...


def _entry(name: str) -> Any:
    found = [e for e in entry_points(group=ENTRY_GROUP) if e.name == name]
    if not found:
        raise SterileUnavailable(
            "Sterile mode needs the meegxer-plugins package (it ships with meegxer)"
        )
    return found[0].load()


def build_image(**options: Any) -> str:
    """Build the Sterile worker's image from the installed MeegXer (``meegxer
    sterile build``); ``options``: ``tag``, ``engine``, ``log`` (a function
    given each line of the build). Returns the image's tag."""
    return _entry("build")(**options)  # type: ignore[no-any-return]


def check_image(**options: Any) -> str:
    """Refuse (``RuntimeError`` with what to do) when the Sterile image is
    missing or built from other code; else its tag. ``options``: ``tag``,
    ``engine``."""
    return _entry("check")(**options)  # type: ignore[no-any-return]


def sterilize(graph: Any, **options: Any) -> Isolation:
    """Run ``graph``'s built-in processors in a Sterile worker; close the
    returned handle when the run ends. ``options``: ``image`` (default
    ``meegxer-sterile:VERSION``), ``local`` (a local process, for tests:
    needs ``MEEGXER_ALLOW_UNSANDBOXED=1``), ``timeout`` (seconds per call)."""
    return _entry("sterile")(graph, **options)  # type: ignore[no-any-return]


def enabled(flag: bool | None = None) -> bool:
    """Whether runs are Sterile: ``flag`` (``--sterile``/``--no-sterile``), else
    ``MEEGXER_STERILE``, else the ``general.sterile`` setting (default off)."""
    if flag is not None:
        return flag
    value = settings.resolve("general.sterile", ENV)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE | _FALSE:
            return text in _TRUE
        raise ValueError(f"{ENV}={value!r}: use 1 or 0 (true/false, on/off)")
    return bool(value)


@contextmanager
def running(graph: Any, flag: bool | None = None, **options: Any) -> Iterator[Any]:
    """Around a run: when Sterile mode is on (``enabled(flag)``), the graph's
    built-in processors run in a Sterile worker until the block ends, then
    they are the local ones again. Yields the isolation, or None when off."""
    if not enabled(flag):
        yield None
        return
    isolation = sterilize(graph, **options)
    try:
        yield isolation
    finally:
        isolation.close()
