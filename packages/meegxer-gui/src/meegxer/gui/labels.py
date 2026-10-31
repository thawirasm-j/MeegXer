"""Friendly words for parameter names, choices and units.

The manifests use short machine names (``sfreq``, ``n_channels``, ``imcoh``); the
GUI shows plain words instead. Commonly known abbreviations stay (EEG, ICA, LSL,
BIDS, Hz, µV). Names without an entry are tidied mechanically, so plugin
parameters read reasonably too.
"""

from __future__ import annotations

import re
from typing import Any

# parameter name -> label (the same name means the same thing on every node)
PARAMS: dict[str, str] = {
    # input and files
    "n_channels": "Channels",
    "sfreq": "Sampling rate",
    "duration": "Duration",
    "seed": "Random seed",
    "start_time": "Start time",
    "path": "File",
    "file": "Marks file",  # the review strip's sidecar
    "labels": "Labels offered",
    "reject": "Cut out labels starting with",
    "root": "Dataset folder",
    "subject": "Subject",
    "session": "Session",
    "task": "Task",
    "run": "Run",
    "datatype": "Data type",
    "desc": "Output label",
    "overwrite": "Replace existing files",
    "erp": "Also write the ERPs (-ave.fif)",
    "stream_name": "Stream name",
    "stream_type": "Stream type",
    "timeout": "Search time",
    "unit": "Unit on the stream",
    "marker_stream": "Marker stream",
    "marker_type": "Marker stream type",
    "marker_wait": "Wait for late markers",
    "include_index": "Add a sample counter channel",
    "label": "Name",
    "keep_seconds": "History kept",
    "keep_windows": "Windows kept",
    # synthetic parts
    "neural": "Brain-like signal",
    "line": "Line noise",
    "ssvep": "Steady-state visual response (SSVEP)",
    "sine": "Pure sine wave",
    "dc": "DC offset",
    "blinks": "Eye blinks",
    "saccades": "Eye movements (saccades)",
    "slow_eye": "Slow eye movements",
    "ecg": "Heartbeat (ECG)",
    "emg": "Muscle activity (EMG)",
    "pops": "Electrode pops",
    "drift": "Slow drift",
    "markers": "Mark artefacts as events",
    "noise_uv": "Background size",
    "alpha_uv": "Alpha size",
    "alpha_hz": "Alpha frequency",
    "line_uv": "Line noise size",
    "line_hz": "Line frequency",
    "ssvep_uv": "SSVEP size",
    "ssvep_hz": "SSVEP frequency",
    "sine_uv": "Sine size",
    "sine_hz": "Sine frequency",
    "dc_mv": "Largest DC offset",
    "blink_rate": "Blinks",
    "blink_uv": "Blink size",
    "blink_ms": "Blink length",
    "saccade_rate": "Eye movements",
    "saccade_deg": "Largest gaze angle",
    "saccade_uv_per_deg": "Signal per degree of gaze",
    "slow_eye_uv": "Slow eye movement size",
    "ecg_bpm": "Heart rate",
    "ecg_uv": "Heartbeat size",
    "emg_rate": "Muscle bursts",
    "emg_uv": "Muscle burst size",
    "emg_s": "Muscle burst length",
    "pop_rate": "Electrode pops",
    "pop_uv": "Electrode pop size",
    "drift_uv": "Drift size",
    # filters and processing
    "kind": "Type",
    "low": "Low cut-off",
    "high": "High cut-off",
    "order": "Filter order",
    "phase": "Phase",
    "freq": "Frequency",
    "quality": "Sharpness (Q)",
    "harmonics": "Number of harmonics",
    "mode": "Reference",
    "channels": "Reference channels",
    "exclude": "Left out of the average",
    "exclude_bads": "Leave bad channels out",
    "drop_bads": "Drop bad channels",
    "keep_labels": "Keep these channels",
    "keep_eeg": "Keep EEG",
    "keep_eog": "Keep eye (EOG)",
    "keep_emg": "Keep muscle (EMG)",
    "keep_ecg": "Keep heart (ECG)",
    "keep_resp": "Keep breathing",
    "keep_stim": "Keep stimulus",
    "keep_misc": "Keep other",
    "method": "Method",
    "threshold": "Removal threshold",
    "reject_eye": "Remove eye components",
    "reject_muscle": "Remove muscle components",
    "reject_heart": "Remove heart components",
    "reject_line_noise": "Remove line-noise components",
    "reject_channel_noise": "Remove channel-noise components",
    "n_components": "Number of components",
    "max_iter": "Maximum iterations",
    "lookahead": "Look-ahead",
    "eog": "EOG channels",
    # detectors
    "peak": "Jump above",
    "flat": "Flat below",
    "flat_duration": "Shortest flat span",
    "bad_percent": "Channel bad above",
    "min_duration": "Shortest jump span",
    "zscore": "z-score threshold",
    "min_length_good": "Shortest good stretch",
    "mark_label": "Label given",
    "half_width": "Span either side of the peak",
    "blink_threshold": "Blink threshold",
    "lambda2": "Spline regularization",
    "stiffness": "Spline stiffness",
    "n_legendre_terms": "Legendre terms",
    # features
    "window": "Window length",
    "step": "Step between windows",
    "segment": "Segment length",
    "overlap": "Segment overlap",
    "taper": "Taper",
    "bands": "Frequency bands",
    "relative": "Relative to total power",
    "db": "In decibels",
    "fmin": "Lowest frequency",
    "fmax": "Highest frequency",
    "perm_entropy": "Permutation entropy",
    "sample_entropy": "Sample entropy",
    "spectral_entropy": "Spectral entropy",
    "hjorth_mobility": "Hjorth mobility",
    "hjorth_complexity": "Hjorth complexity",
    "lziv": "Lempel–Ziv complexity",  # noqa: RUF001
    "bandwidth": "Multitaper bandwidth",
    "aperiodic": "Aperiodic mode",
    "fit_unfiltered": "Fit unfiltered frequencies only",
    "max_n_peaks": "Maximum number of peaks",
    "peak_width_min": "Narrowest peak",
    "peak_width_max": "Widest peak",
    "min_peak_height": "Minimum peak height",
    "peak_threshold": "Peak threshold",
    "fstep": "Frequency step",
    "n_cycles": "Cycles per wavelet",
    "frequencies": "Stimulus frequencies",
    "subbands": "Sub-bands",
    "subband_low": "First sub-band edge",
    "subband_high": "Sub-band upper edge",
    "classes": "Classes (event labels)",
    "tmin": "Trial start",
    "tmax": "Trial end",
    "reg": "Covariance shrinkage",
    "events": "Around events",
    "baseline": "Baseline",
    "baseline_start": "Baseline start",
    "xdf_stream": "XDF signal stream",
    "xdf_markers": "XDF marker streams",
    "xdf_unit": "XDF unit if unstated",
    "ica": "MNE ICA file",
    "baseline_end": "Baseline end",
    "import_review": "Use the review made at import",
    # the Review strip: review points, and the display it opens with
    "points": "Suggestions to review (event labels)",
    "decision": "Decision",
    "montage": "Display montage",
    "highpass": "Display high-pass",
    "lowpass": "Display low-pass",
    "notch": "Display notch",
    "sensitivity": "Display sensitivity",
    "page": "Page length",
}

# (parameter name, value) -> label for enum choices
CHOICES: dict[tuple[str, str], str] = {
    ("baseline", "pre-stimulus"): "Before the event (tmin..0)",
    ("baseline", "custom"): "Custom (baseline start..end)",
    ("baseline", "none"): "None",
    ("xdf_unit", "uV"): "µV",
    ("xdf_unit", "mV"): "mV",
    ("xdf_unit", "V"): "V",
    ("decision", "free"): "Free (suggestions as marks)",
    ("decision", "verify"): "Verify (yes or no for each)",
    ("montage", "reviewer"): "Reviewer's choice",
    ("montage", "as recorded"): "As recorded",
    ("montage", "average"): "Average reference",
    ("montage", "longitudinal bipolar"): "Longitudinal bipolar",
    ("montage", "transverse bipolar"): "Transverse bipolar",
    ("montage", "laplacian"): "Laplacian (Hjorth)",
    ("kind", "highpass"): "High-pass",
    ("kind", "lowpass"): "Low-pass",
    ("kind", "bandpass"): "Band-pass",
    ("kind", "bandstop"): "Band-stop",
    ("phase", "auto"): "Automatic",
    ("phase", "causal"): "Causal (live)",
    ("phase", "zero"): "Zero-phase (offline)",
    ("method", "coh"): "Coherence",
    ("method", "imcoh"): "Imaginary coherence",
    ("method", "plv"): "Phase-locking value",
    ("method", "wpli"): "Weighted phase lag index",
    ("method", "picard"): "Picard (fast)",
    ("method", "infomax"): "Infomax (EEGLAB runica)",
    ("mode", "average"): "Average of all channels",
    ("mode", "channels"): "Chosen channels",
    ("taper", "hamming"): "Hamming",
    ("taper", "hann"): "Hann",
    ("method", "welch"): "Welch",
    ("method", "auto"): "FIR live, analytic offline",
    ("method", "fir"): "FIR Hilbert transformer",
    ("method", "analytic"): "Analytic signal (offline)",
    ("method", "multitaper"): "Multitaper (DPSS)",
    ("method", "cca"): "CCA",
    ("method", "fbcca"): "Filter-bank CCA",
    ("aperiodic", "fixed"): "Fixed (no knee)",
    ("aperiodic", "knee"): "With a knee",
    ("unit", "uV"): "µV",
    ("unit", "mV"): "mV",
    ("unit", "V"): "V",
    ("reg", "none"): "None",
    ("reg", "ledoit_wolf"): "Ledoit–Wolf",  # noqa: RUF001
    ("reg", "oas"): "Oracle approximating (OAS)",
}

UNITS = {"uV": "µV", "/min": "per minute", "deg": "°", "s": "s", "ms": "ms"}

_WORDS = {"uv": "µV", "mv": "mV", "hz": "Hz", "n": "number of", "ms": "ms",
          "db": "dB", "bpm": "per minute", "deg": "°", "s": "s"}  # fmt: skip


def param_label(name: str, spec: dict[str, Any] | None = None) -> str:
    """The friendly label of a parameter (a manifest ``label``, if any, wins)."""
    if spec and spec.get("label"):
        return str(spec["label"])
    if name in PARAMS:
        return PARAMS[name]
    words = [_WORDS.get(w, w) for w in re.split(r"[_\s]+", name) if w]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def choice_label(name: str, value: Any) -> str:
    return CHOICES.get((name, str(value)), str(value))


# Parameters kept in seconds (as MNE takes them) but shown and edited in ms,
# 50 ms a step: (node, parameter) -> step in ms
# file names a writer insists on (MNE's conventions): the save dialog completes
# a name typed without it (clean, clean.fif -> clean-ica.fif)
SAVE_ENDINGS: dict[tuple[str, str], str] = {
    ("meegxer.core.ica_writer", "path"): "-ica.fif",
    ("meegxer.core.epochs_writer", "path"): "-epo.fif",
}

SHOWN_IN_MS: dict[tuple[str, str], float] = {
    ("meegxer.nodes.epochs", "tmin"): 50.0,
    ("meegxer.nodes.epochs", "tmax"): 50.0,
    ("meegxer.nodes.epochs", "baseline_start"): 50.0,
    ("meegxer.nodes.epochs", "baseline_end"): 50.0,
}


def unit_label(unit: str | None) -> str:
    return UNITS.get(unit or "", unit or "")


def _preset_choices() -> None:
    """The Review strip's display choices: the reviewer's, off, or a value."""
    from meegxer.core.review import PRESET_CHOICES, PRESET_UNITS, REVIEWER

    for key, values in PRESET_CHOICES.items():
        CHOICES[(key, REVIEWER)] = "Reviewer's choice"
        for value in values:
            CHOICES[(key, value)] = (
                "Off" if value == "off" else f"{value} {PRESET_UNITS[key]}"
            )


_preset_choices()
