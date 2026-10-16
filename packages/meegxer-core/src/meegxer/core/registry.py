"""Node discovery through the ``meegxer.nodes`` entry-point group.

Packages register node classes under their manifest name, e.g. in pyproject.toml::

    [project.entry-points."meegxer.nodes"]
    "meegxer.nodes.iir_filter" = "meegxer.nodes.filters:IIRFilter"

The core never imports another meegxer package directly; whatever is installed is
what is available, and plugins arrive the same way (CLAUDE.md rule 9).

Built-in nodes were named ``meegxer.sterile.*`` before 0.2.1; those names still
resolve (``canonical_name``), so older pipelines and sessions load. Entry points
from the old ``meegxer-sterile``/``meegxer-runtime`` distributions, if still
installed next to the new ones, are ignored.

Packages whose node classes are only known at run time (the plugin runtime builds
one class per installed plugin image) register a provider instead: a callable in
the ``meegxer.node_providers`` group that returns node classes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points

from meegxer.core.node import Probe, Processor, Sink, Source

type NodeClass = type[Source] | type[Processor] | type[Probe] | type[Sink]

ENTRY_POINT_GROUP = "meegxer.nodes"
PROVIDER_GROUP = "meegxer.node_providers"


# node-name prefixes before the 0.2.1 package rename -> current prefixes
LEGACY_PREFIXES = {"meegxer.sterile.": "meegxer.nodes."}
# modules of the pre-0.2.1 distributions (meegxer-sterile, meegxer-runtime)
LEGACY_MODULES = ("meegxer.sterile", "meegxer.runtime")


def canonical_name(name: str) -> str:
    """The current name of a node, e.g. ``meegxer.sterile.notch`` ->
    ``meegxer.nodes.notch``."""
    for old, new in LEGACY_PREFIXES.items():
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def _legacy(ep: EntryPoint) -> bool:
    module = ep.value.split(":", 1)[0]
    return any(module == m or module.startswith(m + ".") for m in LEGACY_MODULES)


class UnknownNodeError(LookupError):
    """No installed package provides a node with this name."""


class Registry:
    """Maps manifest names to node classes."""

    def __init__(self, classes: Iterable[NodeClass] = ()) -> None:
        self._classes: dict[str, NodeClass] = {}
        for cls in classes:
            self.add(cls)

    @classmethod
    def from_entry_points(cls) -> Registry:
        registry = cls()
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if not _legacy(ep):
                registry._add_entry_point(ep)
        for ep in entry_points(group=PROVIDER_GROUP):
            if _legacy(ep):
                continue
            for node_cls in ep.load()():
                registry.add(node_cls)
        return registry

    def _add_entry_point(self, ep: EntryPoint) -> None:
        node_cls = ep.load()
        if not (isinstance(node_cls, type) and hasattr(node_cls, "manifest")):
            raise TypeError(f"entry point {ep.name!r} is not a node class")
        if node_cls.manifest.name != ep.name:
            raise ValueError(
                f"entry point {ep.name!r} points at node {node_cls.manifest.name!r}"
            )
        self.add(node_cls)

    def add(self, cls: NodeClass) -> None:
        name = cls.manifest.name
        existing = self._classes.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"two node classes registered as {name!r}")
        self._classes[name] = cls

    def get(self, name: str) -> NodeClass:
        try:
            return self._classes[canonical_name(name)]
        except KeyError:
            raise UnknownNodeError(
                f"no installed node named {name!r}; available: {sorted(self._classes)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._classes)

    def as_mapping(self) -> Mapping[str, NodeClass]:
        return dict(self._classes)
