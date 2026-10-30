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
    "meegxer.core.feature_probe": "Features tap",
    "meegxer.core.erp_probe": "ERP tap",
    "meegxer.core.bids_writer": "BIDS dataset",
    "meegxer.core.bids_features": "BIDS features table",
    "meegxer.core.epochs_writer": "Epochs and ERPs file",
    "meegxer.core.bids_epochs": "BIDS epochs and ERPs",
    "meegxer.nodes.notch": "Notch",
    "meegxer.nodes.notch_zerophase": "Notch (zero-phase)",
    "meegxer.nodes.rereference": "Re-reference",
    "meegxer.nodes.pick_channels": "Pick channels",
    "meegxer.nodes.resample": "Resample",
    "meegxer.nodes.epochs": "Epochs",
    "meegxer.core.review": "Review",
    "meegxer.core.epoch_review": "Review epochs",
    "meegxer.core.reject_segments": "Reject segments",
    "meegxer.nodes.iir_filter_zerophase": "IIR filter (zero-phase)",
    "meegxer.nodes.ica": "ICA + ICLabel",
    "meegxer.nodes.psd": "Power spectrum",
    "meegxer.nodes.band_power": "Band power",
    "meegxer.nodes.connectivity": "Connectivity",
    "meegxer.nodes.complexity": "Complexity",
    "meegxer.nodes.hilbert_envelope": "Amplitude envelope",
    "meegxer.nodes.hilbert_envelope_analytic": "Amplitude envelope (analytic)",
    "meegxer.nodes.fir_filter": "FIR filter",
    "meegxer.nodes.fir_filter_zerophase": "FIR filter (zero-phase)",
    "meegxer.nodes.interpolate_bads": "Interpolate bad channels",
    "meegxer.nodes.csd": "Surface Laplacian (CSD)",
    "meegxer.nodes.eog_regression": "EOG regression",
    "meegxer.nodes.detect_amplitude": "Detect amplitude & flat",
    "meegxer.nodes.detect_muscle": "Detect muscle",
    "meegxer.nodes.detect_blinks": "Detect blinks",
    "meegxer.nodes.specparam": "Spectral parameterization",
    "meegxer.nodes.morlet": "Wavelet power (Morlet)",
    "meegxer.nodes.ssvep_cca": "SSVEP detection",
    "meegxer.nodes.csp": "Common spatial patterns (CSP)",
    "meegxer.core.lsl_features_out": "LSL features output",
    "eeglab.asr": "ASR cleaning (EEGLAB)",
}
SPECTRAL = ("meegxer.nodes.psd", "meegxer.nodes.band_power", "meegxer.nodes.specparam")
# short names for the rails a feature strip starts (docs/RAILS.md)
RAIL_NAMES = {
    "meegxer.nodes.psd": "PSD",
    "meegxer.nodes.band_power": "Band power",
    "meegxer.nodes.specparam": "Specparam",
    "meegxer.nodes.morlet": "Morlet power",
    "meegxer.nodes.connectivity": "Connectivity",
    "meegxer.nodes.complexity": "Complexity",
    "meegxer.nodes.ssvep_cca": "SSVEP",
    "meegxer.nodes.csp": "CSP",
}
FILTER_KINDS = {
    "highpass": "High-pass",
    "lowpass": "Low-pass",
    "bandpass": "Band-pass",
    "bandstop": "Band-stop",
}


def title(node: str, params: dict[str, Any] | None = None) -> str:
    if node == "meegxer.nodes.iir_filter":
        if params is None:  # the node itself (menus), not a strip
            return "IIR filter"
        return FILTER_KINDS.get(str(params.get("kind")), "Band-pass")
    if node == "meegxer.nodes.fir_filter" and params and params.get("kind"):
        return f"FIR {FILTER_KINDS.get(str(params['kind']), 'filter').lower()}"
    if node in TITLES:
        return TITLES[node]
    text = node.rsplit(".", 1)[-1].replace("_", " ")
    return text[:1].upper() + text[1:]  # keep the node's own case (e.g. FOOOF)


def rail_title(node: str, params: dict[str, Any] | None = None) -> str:
    """The short name of the rail a feature strip starts (PSD, CSP)."""
    return RAIL_NAMES.get(node) or title(node, params)


def badge(node: str, params: dict[str, Any], mode: str) -> str:
    """Small note under a strip title, e.g. which filter variant runs."""
    if node in ("meegxer.nodes.iir_filter", "meegxer.nodes.fir_filter"):
        phase = params.get("phase", "auto")
        if phase == "zero" or (phase == "auto" and mode == "offline"):
            return "zero-phase"
        return "causal" if node.endswith("iir_filter") else "minimum-phase"
    if node.endswith("_zerophase"):
        return "zero-phase"
    if node in SPECTRAL and params.get("method") == "multitaper":
        return "multitaper"
    if node == "meegxer.nodes.ssvep_cca":
        return "CCA" if params.get("method") == "cca" else "filter-bank CCA"
    return ""
