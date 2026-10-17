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

from meegxer.core.registry import canonical_name
from meegxer.gui.labels import choice_label, unit_label

Mode = Literal["live", "offline"]
Group = Literal["input", "preprocess", "features", "output"]
Manifest = dict[str, Any]

PROBE = "meegxer.core.probe"
TAP_PREFIX = "tap_"
LIVE_ONLY_SOURCES = {"meegxer.core.lsl_in"}
DEFAULT_OFFLINE_SOURCES = {"meegxer.core.file", "meegxer.core.bids"}
SYNTHETIC = "meegxer.core.synthetic"
# the GUI's defaults where they differ from the node's: synthetic input played
# live runs until stopped (a live calibration never runs out of it)
SOURCE_DEFAULTS: dict[str, dict[str, Any]] = {SYNTHETIC: {"duration": 0.0}}
SPEC_VERSION = "0.2.16"
MAX_STRIP_PARAMS = 3
FILTER_NODES = {"meegxer.nodes.iir_filter", "meegxer.nodes.iir_filter_zerophase"}
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
    input: str = ""  # the strip it reads; "" = the nearest one of its input type


@dataclass
class Rail:
    """A parallel rail above the signal (docs/RAILS.md): what one strip (or the
    input) produces for later strips to read. ``readers`` read it; ``lane`` is
    its row above the signal (0 is the lowest)."""

    id: str  # "events", or the id of the strip that produces it
    kind: Literal["events", "features"]
    label: str
    producer: str  # strip id
    readers: list[str] = field(default_factory=list)
    lane: int = 0


def tap_id(strip_id: str) -> str:
    return TAP_PREFIX + strip_id


GROUP_NAMES = {
    "input": "Input",
    "preprocess": "Preprocessing",
    "features": "Feature extraction",
    "output": "Output",
}
# menu headings within a group (manifest ``category``, spec 0.2.15), in order
CATEGORY_NAMES = {
    "filters": "Filters",
    "reference": "Reference & channels",
    "artifacts": "Artifacts",
    "rate": "Resampling",
    "epochs": "Epochs",
    "spectral": "Spectral",
    "time-frequency": "Time–frequency",  # noqa: RUF001
    "connectivity": "Connectivity",
    "complexity": "Complexity",
    "decoding": "Decoding & BCI",
    "events": "Event detection",
    "files": "Files",
    "streams": "Streams",
    "actions": "Actions",
}
# marks in front of nodes the user added: a custom script or local plugin runs
# unsandboxed on this computer; a container plugin runs isolated
ORIGIN_MARKS = {"custom": "✎", "plugin": "⬡"}


def category_label(category: str) -> str:
    if not category:
        return "Other"
    return CATEGORY_NAMES.get(category) or category.replace("-", " ").capitalize()


class PipelineDoc:
    """Chain of strips plus the Input strip's mode."""

    def __init__(self, manifests: list[Manifest]) -> None:
        self.manifests: dict[str, Manifest] = {m["name"]: m for m in manifests}
        self.strips: list[StripNode] = []
        self.input_mode: Mode = "live"
        self.input_events = 0  # events the input carries (from its description)

    # -- manifests ----------------------------------------------------------------

    def set_manifests(self, manifests: list[Manifest]) -> None:
        """The installed nodes changed (a plugin was added)."""
        self.manifests = {m["name"]: m for m in manifests}

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
        elif (
            self.mode == "offline"
            and self.source.node == SYNTHETIC
            and not float(self.source.params.get("duration") or 0)
        ):
            issues.insert(0, "the synthetic input runs forever: set its duration "
                             "to process it offline")  # fmt: skip
        return issues

    def missing_positions(self, channels: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Strips that need electrode positions (manifest ``requires_positions``)
        and the good channels they process that have none, given the input's
        channels as ``describe_source`` reports them (after the Recording edits)."""
        found: dict[str, list[str]] = {}
        for strip in self.strips:
            if not self.manifest(strip).get("requires_positions"):
                continue
            types = self.processed_types(strip)
            missing = [
                c["label"]
                for c in channels
                if (c.get("type") or "eeg") in types
                and not c.get("bad")
                and c.get("position") is None
            ]
            if missing:
                found[strip.id] = missing
        return found

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
        defaults = {**self._defaults(node), **SOURCE_DEFAULTS.get(node, {})}
        strip = StripNode("input", node, {**defaults, **(params or {})})
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
        for other in self.strips:  # who read it goes back to the nearest
            if other.input == strip_id:
                other.input = ""

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
        """Strips followed by a probe tap: the input and every processor with a
        Signal output (Features strips branch off the chain untapped)."""
        return [
            s
            for s in self.strips
            if self.role(s) in ("source", "processor")
            and [p["type"] for p in self.manifest(s)["outputs"][:1]] == ["signal"]
        ]

    def group(self, strip: StripNode | str) -> Group:
        """Input, preprocessing (Signal in and out), feature extraction (Signal
        to Features) or output (sinks): from the node's role and ports, so
        plugins are grouped too."""
        manifest = self.manifest(strip)
        role = manifest["role"]
        if role == "source":
            return "input"
        if role in ("sink", "probe"):
            return "output"
        outputs = {p["type"] for p in manifest["outputs"]}
        return "preprocess" if outputs <= {"signal"} else "features"

    def is_custom(self, strip: StripNode | str) -> bool:
        """Added by the user (a custom script or a plugin), not built in."""
        return bool(self.manifest(strip).get("origin", "builtin") != "builtin")

    def default_types(self, strip: StripNode | str) -> list[str]:
        """Channel types the node processes unless the strip names others."""
        return list(self.manifest(strip).get("channel_types") or ["eeg"])

    def processed_types(self, strip: StripNode) -> list[str]:
        return strip.channel_types or self.default_types(strip)

    def category(self, strip: StripNode | str) -> str:
        return str(self.manifest(strip).get("category") or "")

    def mark(self, strip: StripNode | str) -> str:
        """``✎`` (custom script or local plugin), ``⬡`` (container plugin) or
        nothing (built in)."""
        return ORIGIN_MARKS.get(str(self.manifest(strip).get("origin", "builtin")), "")

    def palette_sections(self) -> list[tuple[str, list[tuple[str, list[str]]]]]:
        """The add-strip menu: per group, its categories (known ones in their
        order, then others, then Other), each listing built-ins first, then the
        nodes the user added."""
        out = []
        palette = self.nodes_for_palette()
        order = list(CATEGORY_NAMES)
        for group in ("preprocess", "features", "output"):
            nodes = [m["name"] for m in palette if self.group(m["name"]) == group]
            if not nodes:
                continue

            def rank(c: str) -> tuple[bool, bool, int, str]:
                return (c == "", c not in order,
                        order.index(c) if c in order else 0, c)  # fmt: skip

            cats = sorted({self.category(n) for n in nodes}, key=rank)
            out.append((GROUP_NAMES[group], [
                (category_label(c), sorted(
                    (n for n in nodes if self.category(n) == c),
                    key=lambda n: (self.is_custom(n), n)))
                for c in cats
            ]))  # fmt: skip
        return out

    def group_label(self, strip: StripNode | str) -> str:
        """``Preprocessing``, ``Custom feature extraction``, ...: the group by
        data type, marked custom for scripts and plugins."""
        name = GROUP_NAMES[self.group(strip)]
        return f"Custom {name.lower()}" if self.is_custom(strip) else name

    def input_of(self, strip: StripNode) -> str | None:
        """The strip this one reads from: the one it names (``input``) if that is
        an earlier strip with the right output type, otherwise the nearest such
        strip (spec 0.2.7), as the pipeline resolves it."""
        choices = self.input_choices(strip)
        if strip.input in choices:
            return strip.input
        return choices[0] if choices else None

    def input_choices(self, strip: StripNode) -> list[str]:
        """Earlier strips this one could read from, nearest first: those whose
        output has the type of its input."""
        inputs = self.manifest(strip)["inputs"]
        if not inputs:
            return []
        wanted = inputs[0]["type"]
        index = self.strips.index(strip)
        return [
            earlier.id
            for earlier in reversed(self.strips[:index])
            if (outputs := self.manifest(earlier)["outputs"])
            and outputs[0]["type"] == wanted
        ]

    def set_input(self, strip_id: str, source: str) -> None:
        """Make a strip read ``source`` ("" for the nearest of its type)."""
        strip = self.get(strip_id)
        if source and source not in self.input_choices(strip):
            raise ValueError(f"{strip_id} cannot read from {source}")
        strip.input = source

    # -- rails (docs/RAILS.md) ------------------------------------------------------

    def rail_name(self, strip: StripNode) -> str:
        """A feature rail's name: its strip's title, numbered when the chain has
        more than one (Band power, Band power·2)."""
        from meegxer.gui.names import rail_title  # names imports nothing from here

        name = rail_title(strip.node, strip.params)
        same = [s for s in self.strips if self.produces_features(s)
                and rail_title(s.node, s.params) == name]  # fmt: skip
        n = same.index(strip) + 1 if strip in same else 1
        return name if n == 1 else f"{name}·{n}"

    def produces_features(self, strip: StripNode) -> bool:
        outputs = self.manifest(strip)["outputs"]
        return bool(outputs) and outputs[0]["type"] == "features"

    def rails(self) -> tuple[list[Rail], dict[str, str]]:
        """The parallel rails of this chain and the strips that need a rail that
        is not there (strip id -> rail).

        Events: the input's events, read by strips that fit on labelled trials.
        Features: one rail per feature strip, read by the strips that take it
        as input. Every rail runs on to the end of the chain, so each has its
        own row: Events on top, then the feature rails, earliest highest (a
        strip pushing up into its rail crosses no earlier rail)."""
        needs: dict[str, str] = {}
        rails: list[Rail] = []
        for strip in self.strips[1:]:
            if self.produces_features(strip):
                readers = [s.id for s in self.strips if self.input_of(s) == strip.id]
                rails.append(Rail(strip.id, "features", self.rail_name(strip),
                                  strip.id, readers))  # fmt: skip
        for i, rail in enumerate(rails):  # in chain order
            rail.lane = len(rails) - 1 - i
        labelled = [s.id for s in self.strips[1:] if self.manifest(s).get("fit_events")]
        source = self.source
        if source is not None and self.carries_events():
            rails.append(Rail("events", "events", "Events", source.id, labelled,
                              lane=len(rails)))  # fmt: skip
        else:
            needs.update({sid: "Events" for sid in labelled})
        return rails, needs

    def carries_events(self) -> bool:
        """The input starts the Events rail: a recording with events (offline or
        played live), or an LSL input with a marker stream."""
        source = self.source
        if source is None:
            return False
        if source.node == "meegxer.core.lsl_in":
            return bool(source.params.get("marker_stream"))
        return self.input_events > 0

    def lowered(self, strip: StripNode) -> bool:
        """Feature and output strips sit below the signal line and draw from it
        (or from a rail); the signal passes over them."""
        return self.group(strip) in ("features", "output")

    def reads_signal(self, strip: StripNode) -> bool:
        inputs = self.manifest(strip)["inputs"]
        return bool(inputs) and inputs[0]["type"] == "signal"

    def rail_lanes(self) -> int:
        """Rows the rails take above the signal (a missing rail keeps one)."""
        rails, needs = self.rails()
        lanes = max((r.lane + 1 for r in rails), default=0)
        return lanes + (1 if needs and not any(r.kind == "events" for r in rails)
                        else 0)  # fmt: skip

    def tap_delays(self, timing_ms: dict[str, float], sfreq: float) -> dict[str, float]:
        """Estimated ms each tappable strip runs behind the input: the measured
        computing time (``timing_ms``, per strip id) of the strips on its path,
        plus the built-in delay their manifests declare (``latency_samples``)."""
        totals: dict[str, float] = {}
        for strip in self.tappable():
            total = 0.0
            sid: str | None = strip.id
            while sid is not None:
                on_path = self.get(sid)
                total += timing_ms.get(sid, 0.0)
                declared = int(self.manifest(on_path).get("latency_samples") or 0)
                if declared and sfreq:
                    total += 1000 * declared / sfreq
                sid = self.input_of(on_path)
            totals[strip.id] = total
        base = totals.get(self.source.id, 0.0) if self.source else 0.0
        return {sid: max(0.0, t - base) for sid, t in totals.items()}

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
        elif spec["type"] == "enum":
            text = choice_label(name, value)
        elif spec["type"] == "float":
            text = f"{float(value):g}"
        elif spec["type"] == "string":
            text = str(value) or "-"
            if len(text) > 14:
                text = "…" + text[-13:]
        else:
            text = str(value)
        unit = unit_label(spec.get("unit"))
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
            if strip.channel_types and strip.channel_types != self.default_types(strip):
                entry["channel_types"] = list(strip.channel_types)
            if strip.recording:
                entry["recording"] = copy.deepcopy(strip.recording)
            if strip.input and strip.input in self.input_choices(strip):
                entry["input"] = strip.input
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
        nodes = data.get("nodes", []) if isinstance(data, dict) else None
        if not isinstance(nodes, list) or not all(isinstance(e, dict) for e in nodes):
            raise ValueError("not a pipeline document (expected {'nodes': [...]})")
        for entry in nodes:
            if entry["node"] == PROBE:
                continue
            node = canonical_name(entry["node"])  # names before 0.2.1 still load
            if node not in self.manifests:
                raise KeyError(f"node {entry['node']!r} is not installed")
            strips.append(
                StripNode(
                    entry.get("id") or self._new_id(node),
                    node,
                    {**self._defaults(node), **entry.get("params", {})},
                    bool(entry.get("pass_uncalibrated", False)),
                    list(entry.get("strip_params", [])),
                    dict(entry.get("recording", {})),
                    list(entry.get("channel_types", [])),
                    str(entry.get("input") or ""),
                )
            )
        if not strips or self.role(strips[0]) != "source":
            raise ValueError("a pipeline starts with its input (a source)")
        self.strips = strips
        if strips[0].node not in LIVE_ONLY_SOURCES:
            self.input_mode = (
                "offline" if strips[0].node in DEFAULT_OFFLINE_SOURCES else "live"
            )
