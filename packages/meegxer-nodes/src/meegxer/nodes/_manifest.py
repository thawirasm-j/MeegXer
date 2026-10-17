"""Manifest helper shared by the built-in nodes."""

from __future__ import annotations

from meegxer.core.manifest import Manifest
from meegxer.nodes.references import credits


def stream_manifest(
    name: str,
    version: str,
    params: list[dict[str, object]],
    *,
    description: str = "",
    live_params: list[str] | None = None,
    availability: str = "both",
    requires_fit: bool = False,
    requires_positions: bool = False,
    channel_types: list[str] | None = None,
    rail_outputs: list[dict[str, str]] | None = None,
) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": version,
            "type": "stream",
            "description": description,
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}, *(rail_outputs or [])],
            "params": params,
            "live_params": live_params or [],
            "availability": availability,
            "requires_fit": requires_fit,
            "requires_positions": requires_positions,
            **({"channel_types": channel_types} if channel_types else {}),
            **credits(name),
        }
    )
