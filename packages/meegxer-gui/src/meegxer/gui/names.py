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
    "meegxer.nodes.notch": "Notch",
    "meegxer.nodes.notch_zerophase": "Notch (zero-phase)",
    "meegxer.nodes.rereference": "Re-reference",
    "meegxer.nodes.iir_filter_zerophase": "Filter (zero-phase)",
    "meegxer.nodes.ica": "ICA + ICLabel",
    "meegxer.nodes.psd": "Power spectrum",
    "meegxer.nodes.band_power": "Band power",
    "meegxer.nodes.connectivity": "Connectivity",
    "meegxer.nodes.complexity": "Complexity",
    "meegxer.nodes.hilbert_envelope": "Amplitude envelope",
    "meegxer.nodes.fir_filter": "FIR filter",
    "meegxer.nodes.fir_filter_zerophase": "FIR filter (zero-phase)",
    "meegxer.nodes.interpolate_bads": "Interpolate bad channels",
    "meegxer.nodes.csd": "Surface Laplacian (CSD)",
    "meegxer.nodes.eog_regression": "EOG regression",
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
    if node == "meegxer.nodes.iir_filter":
        return FILTER_KINDS.get(str((params or {}).get("kind")), "Band-pass")
    if node == "meegxer.nodes.fir_filter" and params and params.get("kind"):
        return f"FIR {FILTER_KINDS.get(str(params['kind']), 'filter').lower()}"
    if node in TITLES:
        return TITLES[node]
    return node.rsplit(".", 1)[-1].replace("_", " ").capitalize()


def badge(node: str, params: dict[str, Any], mode: str) -> str:
    """Small note under a strip title, e.g. which filter variant runs."""
    if node in ("meegxer.nodes.iir_filter", "meegxer.nodes.fir_filter"):
        phase = params.get("phase", "auto")
        if phase == "zero" or (phase == "auto" and mode == "offline"):
            return "zero-phase"
        return "causal" if node.endswith("iir_filter") else "minimum-phase"
    if node.endswith("_zerophase"):
        return "zero-phase"
    return ""
