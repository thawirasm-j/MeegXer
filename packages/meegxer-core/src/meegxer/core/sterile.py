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

from importlib.metadata import entry_points
from typing import Any, Protocol

ENTRY_GROUP = "meegxer.isolation"


class SterileUnavailable(RuntimeError):
    """Sterile mode is not installed (no ``meegxer.isolation`` entry point)."""


class Isolation(Protocol):
    proxied: list[str]  # the node ids that run in the worker

    def close(self) -> None: ...

    def __enter__(self) -> Isolation: ...

    def __exit__(self, *exc: object) -> None: ...


def sterilize(graph: Any, **options: Any) -> Isolation:
    """Run ``graph``'s built-in processors in a Sterile worker; close the
    returned handle when the run ends. ``options``: ``image`` (default
    ``meegxer-sterile:VERSION``), ``local`` (a local process, for tests:
    needs ``MEEGXER_ALLOW_UNSANDBOXED=1``), ``timeout`` (seconds per call)."""
    found = [e for e in entry_points(group=ENTRY_GROUP) if e.name == "sterile"]
    if not found:
        raise SterileUnavailable(
            "Sterile mode needs the meegxer-plugins package (it ships with meegxer)"
        )
    isolate = found[0].load()
    return isolate(graph, **options)  # type: ignore[no-any-return]
