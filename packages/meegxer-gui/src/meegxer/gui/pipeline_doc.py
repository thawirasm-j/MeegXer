"""The GUI's model of a pipeline: a chain of strips (docs/DESIGN.md §3.7).

Pure Python (no Qt), so it is easy to test. The chain is the pipeline's node list
without probes: the Input strip (the source) first, then processors and sinks in
order. Probe taps are generated, one after the source and after every processor,
so any point of the chain can be shown without reloading the pipeline.

The mode comes from the Input strip: an LSL source is live; a file, BIDS or
synthetic source is played live or processed offline (``input_mode``).
Compatibility is strict: a strip whose availability does not fit the mode is
incompatible, and the band-pass is the one adaptable node (``phase``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["live", "offline"]
Manifest = dict[str, Any]

PROBE = "meegxer.core.probe"
TAP_PREFIX = "tap_"
LIVE_ONLY_SOURCES = {"meegxer.core.lsl_in"}
DEFAULT_OFFLINE_SOURCES = {"meegxer.core.file", "meegxer.core.bids"}
SPEC_VERSION = "0.2.6"
MAX_STRIP_PARAMS = 3
FILTER_NODES = {"meegxer.sterile.iir_filter", "meegxer.sterile.iir_filter_zerophase"}
FILTER_UNUSED = {"highpass": {"high"}, "lowpass": {"low"}}  # edges a kind ignores


@dataclass
class StripNode:
    id: str
    node: str
    params: dict[str, Any] = field(default_factory=dict)
    pass_uncalibrated: bool = False
    strip_params: list[str] = field(default_factory=list)
    recording: dict[str, Any] = field(default_factory=dict)  # input strip only
    channel_types: list[str] = field(default_factory=list)  # processors; [] = EEG


def tap_id(strip_id: str) -> str:
    return TAP_PREFIX + strip_id


class PipelineDoc:
    """Chain of strips plus the Input strip's mode."""

    def __init__(self, manifests: list[Manifest]) -> None:
        self.manifests: dict[str, Manifest] = {m["name"]: m for m in manifests}
        self.strips: list[StripNode] = []
        self.input_mode: Mode = "live"

    # -- manifests ----------------------------------------------------------------

    def manifest(self, strip: StripNode | str) -> Manifest:
        name = strip if isinstance(strip, str) else strip.node
        return self.manifests[name]

    def role(self, strip: StripNode | str) -> str:
        return str(self.manifest(strip)["role"])

    def nodes_for_palette(self) -> list[Manifest]:
        """Manifests that can be added as strips (processors and sinks)."""
        return sorted(
            (m for m in self.manifests.values() if m["role"] in ("processor", "sink")),
            key=lambda m: (m["role"], m["name"]),
        )

    def sources(self) -> list[Manifest]:
        return sorted(
            (m for m in self.manifests.values() if m["role"] == "source"),
            key=lambda m: m["name"],
        )

    # -- mode and compatibility ---------------------------------------------------

    @property
    def source(self) -> StripNode | None:
        return self.strips[0] if self.strips else None

    @property
    def mode(self) -> Mode:
        if self.source is not None and self.source.node in LIVE_ONLY_SOURCES:
            return "live"
        return self.input_mode

    def mode_is_fixed(self) -> bool:
        """LSL input is always live; other sources can be played or processed."""
        return self.source is not None and self.source.node in LIVE_ONLY_SOURCES

    def incompatibility(
        self, strip: StripNode | str, params: dict[str, Any] | None = None
    ) -> str | None:
        """Why this node cannot run in the current mode, or None."""
        manifest = self.manifest(strip)
        if isinstance(strip, StripNode):
            params = strip.params
        availability = manifest.get("availability", "both")
        if self.mode == "live":
            if availability == "offline":
                return "Offline only"
            if (params or {}).get("phase") == "zero":
                return "Zero-phase needs offline"
        elif availability == "online":
            return "Live only"
        return None

    def problems(self) -> list[str]:
        """Reasons the pipeline cannot start (empty when it can)."""
        issues = [
            f"{s.id}: {reason}"
            for s in self.strips
            if (reason := self.incompatibility(s)) is not None
        ]
        if self.source is None:
            issues.insert(0, "no input")
        return issues

    # -- strips -------------------------------------------------------------------

    def _new_id(self, node: str) -> str:
        base = node.rsplit(".", 1)[-1].replace("_", "")[:10] or "node"
        taken = {s.id for s in self.strips}
        n = 1
        while f"{base}{n}" in taken:
            n += 1
        return f"{base}{n}"

    def _defaults(self, node: str) -> dict[str, Any]:
        return {p["name"]: p["default"] for p in self.manifest(node)["params"]}

    def set_source(self, node: str, params: dict[str, Any] | None = None) -> StripNode:
        """Replace the Input strip (keeps its id, so taps stay where they are)."""
        if self.role(node) != "source":
            raise ValueError(f"{node} is not a source")
        strip = StripNode("input", node, {**self._defaults(node), **(params or {})})
        if self.strips and self.role(self.strips[0]) == "source":
            self.strips[0] = strip
        else:
            self.strips.insert(0, strip)
        if node not in LIVE_ONLY_SOURCES:
            self.input_mode = "offline" if node in DEFAULT_OFFLINE_SOURCES else "live"
        return strip

    def insert(self, index: int, node: str) -> StripNode:
        """Add a processor or sink at chain position ``index`` (after the input)."""
        if self.role(node) not in ("processor", "sink"):
            raise ValueError(f"{node} cannot be added as a strip")
        strip = StripNode(self._new_id(node), node, self._defaults(node))
        self.strips.insert(max(1, min(index, len(self.strips))), strip)
        return strip

    def move(self, strip_id: str, index: int) -> None:
        """Drag a strip to chain position ``index`` (the input stays first)."""
        strip = self.get(strip_id)
        if strip is self.source:
            raise ValueError("the Input strip stays first")
        old = self.strips.index(strip)
        self.strips.pop(old)
        if index > old:
            index -= 1
        self.strips.insert(max(1, min(index, len(self.strips))), strip)

    def remove(self, strip_id: str) -> None:
        strip = self.get(strip_id)
        if strip is self.source:
            raise ValueError("the Input strip cannot be removed")
        self.strips.remove(strip)

    def get(self, strip_id: str) -> StripNode:
        for strip in self.strips:
            if strip.id == strip_id:
                return strip
        raise KeyError(strip_id)

    def source_entry(self, *, with_recording: bool = True) -> dict[str, Any]:
        """The input strip as a pipeline node (for describing the recording)."""
        src = self.source
        if src is None:
            raise ValueError("no input")
        entry: dict[str, Any] = {"node": src.node, "params": copy.deepcopy(src.params)}
        if with_recording and src.recording:
            entry["recording"] = copy.deepcopy(src.recording)
        return entry

    def tappable(self) -> list[StripNode]:
        """Strips followed by a probe tap (the input and every processor)."""
        return [s for s in self.strips if self.role(s) in ("source", "processor")]

    # -- what a strip shows -------------------------------------------------------

    def primary_params(self, strip: StripNode) -> list[str]:
        params = self.manifest(strip)["params"]
        names = [p["name"] for p in params if p.get("primary")]
        if not names:
            names = [p["name"] for p in params if not p.get("advanced")][:1]
        unused = FILTER_UNUSED.get(str(strip.params.get("kind")), set())
        if strip.node in FILTER_NODES:
            names = [n for n in names if n not in unused]
        return names

    def shown_params(self, strip: StripNode, advanced: bool) -> list[str]:
        """Friendly mode: the crucial parameters. Advanced: up to 3 picked ones
        (the crucial ones until the user picks)."""
        if advanced and strip.strip_params:
            return list(strip.strip_params)
        return self.primary_params(strip)[: MAX_STRIP_PARAMS if advanced else None]

    def format_param(self, strip: StripNode, name: str) -> str:
        spec = next(p for p in self.manifest(strip)["params"] if p["name"] == name)
        value = strip.params.get(name, spec["default"])
        if spec["type"] == "bool":
            text = "on" if value else "off"
        elif spec["type"] == "float":
            text = f"{float(value):g}"
        elif spec["type"] == "string":
            text = str(value) or "-"
            if len(text) > 14:
                text = "…" + text[-13:]
        else:
            text = str(value)
        unit = spec.get("unit") or ""
        return f"{text} {unit}".strip()

    # -- pipeline files -----------------------------------------------------------

    def to_pipeline(
        self, *, taps: bool = True, tap_keep_seconds: float = 1.0
    ) -> dict[str, Any]:
        """The pipeline document, with a probe tap after every tappable strip
        (``taps=False`` for saving: taps are regenerated when loading)."""
        nodes: list[dict[str, Any]] = []
        tappable = {s.id for s in self.tappable()} if taps else set()
        for strip in self.strips:
            entry: dict[str, Any] = {
                "id": strip.id,
                "node": strip.node,
                "params": copy.deepcopy(strip.params),
            }
            if strip.pass_uncalibrated:
                entry["pass_uncalibrated"] = True
            if strip.strip_params:
                entry["strip_params"] = list(strip.strip_params)
            if strip.channel_types and strip.channel_types != ["eeg"]:
                entry["channel_types"] = list(strip.channel_types)
            if strip.recording:
                entry["recording"] = copy.deepcopy(strip.recording)
            nodes.append(entry)
            if strip.id in tappable:
                nodes.append(
                    {
                        "id": tap_id(strip.id),
                        "node": PROBE,
                        "input": strip.id,
                        "params": {"label": strip.id, "keep_seconds": tap_keep_seconds},
                    }
                )
        return {"spec_version": SPEC_VERSION, "nodes": nodes}

    def load_pipeline(self, data: dict[str, Any]) -> None:
        """Replace the chain from a pipeline document; probes become taps."""
        strips: list[StripNode] = []
        for entry in data.get("nodes", []):
            if entry["node"] == PROBE:
                continue
            if entry["node"] not in self.manifests:
                raise KeyError(f"node {entry['node']!r} is not installed")
            strips.append(
                StripNode(
                    entry.get("id") or self._new_id(entry["node"]),
                    entry["node"],
                    {**self._defaults(entry["node"]), **entry.get("params", {})},
                    bool(entry.get("pass_uncalibrated", False)),
                    list(entry.get("strip_params", [])),
                    dict(entry.get("recording", {})),
                    list(entry.get("channel_types", [])),
                )
            )
        if not strips or self.role(strips[0]) != "source":
            raise ValueError("a pipeline starts with its input (a source)")
        self.strips = strips
        if strips[0].node not in LIVE_ONLY_SOURCES:
            self.input_mode = (
                "offline" if strips[0].node in DEFAULT_OFFLINE_SOURCES else "live"
            )
