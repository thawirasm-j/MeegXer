"""Friendly strip titles for node names."""

from __future__ import annotations

from typing import Any

TITLES = {
    "meegxer.core.synthetic": "Synthetic EEG",
    "meegxer.core.lsl_in": "LSL stream",
    "meegxer.core.file": "Recording file",
    "meegxer.core.bids": "BIDS recording",
    "meegxer.core.lsl_out": "LSL output",
    "meegxer.core.set_writer": "EEGLAB file (.set)",
    "meegxer.core.edf_writer": "EDF file",
    "meegxer.core.feature_tsv": "Features table",
    "meegxer.core.bids_writer": "BIDS dataset",
    "meegxer.core.bids_features": "BIDS features table",
    "meegxer.sterile.notch": "Notch",
    "meegxer.sterile.notch_zerophase": "Notch (zero-phase)",
    "meegxer.sterile.rereference": "Re-reference",
    "meegxer.sterile.iir_filter_zerophase": "Filter (zero-phase)",
    "meegxer.sterile.ica": "ICA + ICLabel",
    "meegxer.sterile.psd": "Power spectrum",
    "meegxer.sterile.band_power": "Band power",
    "meegxer.sterile.connectivity": "Connectivity",
    "meegxer.sterile.complexity": "Complexity",
    "meegxer.sterile.hilbert_envelope": "Amplitude envelope",
    "meegxer.core.lsl_features_out": "LSL features output",
    "eeglab.asr": "ASR cleaning (EEGLAB)",
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
