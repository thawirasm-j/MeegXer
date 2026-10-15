"""Friendly strip titles for node names."""

from __future__ import annotations

from typing import Any

TITLES = {
    "meegxer.core.synthetic": "Synthetic",
    "meegxer.core.lsl_in": "LSL stream",
    "meegxer.core.file": "File",
    "meegxer.core.bids": "BIDS",
    "meegxer.core.lsl_out": "LSL out",
    "meegxer.core.set_writer": "EEGLAB .set",
    "meegxer.core.edf_writer": "EDF",
    "meegxer.core.feature_tsv": "Feature TSV",
    "meegxer.sterile.notch": "Notch",
    "meegxer.sterile.notch_zerophase": "Notch (zero-phase)",
    "meegxer.sterile.rereference": "Re-reference",
    "meegxer.sterile.iir_filter_zerophase": "Filter (zero-phase)",
    "meegxer.sterile.ica": "ICA + ICLabel",
    "meegxer.sterile.psd": "PSD",
    "meegxer.sterile.band_power": "Band power",
    "meegxer.sterile.connectivity": "Connectivity",
    "meegxer.sterile.complexity": "Complexity",
    "meegxer.sterile.hilbert_envelope": "Envelope",
    "meegxer.core.lsl_features_out": "LSL features",
    "eeglab.asr": "ASR (EEGLAB)",
}
FILTER_KINDS = {
    "highpass": "High-pass",
    "lowpass": "Low-pass",
    "bandpass": "Band-pass",
    "bandstop": "Band-stop",
}


def title(node: str, params: dict[str, Any] | None = None) -> str:
    if node == "meegxer.sterile.iir_filter":
        return FILTER_KINDS.get(str((params or {}).get("kind")), "Band-pass")
    if node in TITLES:
        return TITLES[node]
    return node.rsplit(".", 1)[-1].replace("_", " ").capitalize()


def badge(node: str, params: dict[str, Any], mode: str) -> str:
    """Small note under a strip title, e.g. which filter variant runs."""
    if node == "meegxer.sterile.iir_filter":
        phase = params.get("phase", "auto")
        if phase == "zero" or (phase == "auto" and mode == "offline"):
            return "zero-phase"
        return "causal"
    if node.endswith("_zerophase"):
        return "zero-phase"
    return ""
