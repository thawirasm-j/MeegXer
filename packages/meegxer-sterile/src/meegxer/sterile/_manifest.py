"""Manifest helper shared by the sterile built-ins."""

from __future__ import annotations

from meegxer.core.manifest import Manifest


def stream_manifest(
    name: str,
    version: str,
    params: list[dict[str, object]],
    *,
    description: str = "",
    live_params: list[str] | None = None,
) -> Manifest:
    return Manifest.from_dict(
        {
            "spec_version": "0.2.0",
            "name": name,
            "version": version,
            "type": "stream",
            "description": description,
            "inputs": [{"name": "in", "type": "signal"}],
            "outputs": [{"name": "out", "type": "signal"}],
            "params": params,
            "live_params": live_params or [],
        }
    )
