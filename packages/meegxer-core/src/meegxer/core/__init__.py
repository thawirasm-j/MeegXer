"""Headless engine: graph, schedulers, plugin spec, I/O and control API (no GUI)."""

from meegxer.core.block import (
    Block,
    ChannelMismatchError,
    Event,
    EventBlock,
    FeatureBlock,
    SampleGapError,
    SampleOverlapError,
    SignalInfo,
    align_blocks,
    concat_blocks,
    match_channels,
)
from meegxer.core.manifest import SPEC_VERSION, Manifest, ManifestError, ParamError
from meegxer.core.node import NodeRole, Probe, Processor, Sink, Source, TransformResult
from meegxer.core.ports import Port, PortType, PortTypeError, check_connection
from meegxer.core.ringbuffer import RingBuffer
from meegxer.core.stage import Stage, StageStatus, StaleModelError, signal_fingerprint

__version__ = "0.0.0"

__all__ = [
    "SPEC_VERSION",
    "Block",
    "ChannelMismatchError",
    "Event",
    "EventBlock",
    "FeatureBlock",
    "Manifest",
    "ManifestError",
    "NodeRole",
    "ParamError",
    "Port",
    "PortType",
    "PortTypeError",
    "Probe",
    "Processor",
    "RingBuffer",
    "SampleGapError",
    "SampleOverlapError",
    "SignalInfo",
    "Sink",
    "Source",
    "Stage",
    "StageStatus",
    "StaleModelError",
    "TransformResult",
    "__version__",
    "align_blocks",
    "check_connection",
    "concat_blocks",
    "match_channels",
    "signal_fingerprint",
]
