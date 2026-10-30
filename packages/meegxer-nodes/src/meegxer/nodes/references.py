"""What to cite for each built-in node (manifest ``references``, spec 0.2.12).

Each node cites the method's original paper(s) and the software it runs on.
DOIs were checked against PubMed or the publisher's listing; a reference whose
DOI could not be confirmed is given without one.
"""

from __future__ import annotations

from meegxer.core.references import AUTHORS, LICENSE, MNE, Reference

LUCK: Reference = {
    "citation": "Luck SJ (2014). An Introduction to the Event-Related Potential "
    "Technique, 2nd ed. MIT Press.",
}
SCIPY: Reference = {
    "citation": "Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0: "
    "fundamental algorithms for scientific computing in Python. Nat Methods "
    "17(3):261-272.",
    "doi": "10.1038/s41592-019-0686-2",
}
PICARD: Reference = {
    "citation": "Ablin P, Cardoso JF, Gramfort A (2018). Faster independent "
    "component analysis by preconditioning with Hessian approximations. IEEE "
    "Trans Signal Process 66(15):4040-4049.",
    "doi": "10.1109/TSP.2018.2844203",
}
INFOMAX: Reference = {
    "citation": "Bell AJ, Sejnowski TJ (1995). An information-maximization approach "
    "to blind separation and blind deconvolution. Neural Comput 7(6):1129-1159.",
    "doi": "10.1162/neco.1995.7.6.1129",
}
EXTENDED_INFOMAX: Reference = {
    "citation": "Lee TW, Girolami M, Sejnowski TJ (1999). Independent component "
    "analysis using an extended infomax algorithm for mixed subgaussian and "
    "supergaussian sources. Neural Comput 11(2):417-441.",
    "doi": "10.1162/089976699300016719",
}
ICLABEL: Reference = {
    "citation": "Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: An "
    "automated electroencephalographic independent component classifier, "
    "dataset, and website. NeuroImage 198:181-197.",
    "doi": "10.1016/j.neuroimage.2019.05.026",
}
MNE_ICALABEL: Reference = {
    "citation": "Li A, Feitelberg J, Saini AP, Höchenberger R, Scheltienne M (2022). "
    "MNE-ICALabel: Automatically annotating ICA components with ICLabel in "
    "Python. J Open Source Softw 7(76):4484.",
    "doi": "10.21105/joss.04484",
}
MNE_CONNECTIVITY: Reference = {
    "citation": "MNE-Connectivity: connectivity analysis of M/EEG data with "
    "MNE-Python (software).",
    "url": "https://mne.tools/mne-connectivity/",
}
IMCOH: Reference = {
    "citation": "Nolte G, Bai O, Wheaton L, Mari Z, Vorbach S, Hallett M (2004). "
    "Identifying true brain interaction from EEG data using the imaginary part "
    "of coherency. Clin Neurophysiol 115(10):2292-2307.",
    "doi": "10.1016/j.clinph.2004.04.029",
}
PLV: Reference = {
    "citation": "Lachaux JP, Rodriguez E, Martinerie J, Varela FJ (1999). Measuring "
    "phase synchrony in brain signals. Hum Brain Mapp 8(4):194-208.",
    "doi": "10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C",
}
WPLI: Reference = {
    "citation": "Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA "
    "(2011). An improved index of phase-synchronization for electrophysiological "
    "data in the presence of volume-conduction, noise and sample-size bias. "
    "NeuroImage 55(4):1548-1565.",
    "doi": "10.1016/j.neuroimage.2011.01.055",
}
ANTROPY: Reference = {
    "citation": "Vallat R. AntroPy: entropy and complexity of (EEG) time-series in "
    "Python (software).",
    "url": "https://github.com/raphaelvallat/antropy",
}
PERMUTATION_ENTROPY: Reference = {
    "citation": "Bandt C, Pompe B (2002). Permutation entropy: a natural complexity "
    "measure for time series. Phys Rev Lett 88(17):174102.",
    "doi": "10.1103/PhysRevLett.88.174102",
}
SAMPLE_ENTROPY: Reference = {
    "citation": "Richman JS, Moorman JR (2000). Physiological time-series analysis "
    "using approximate entropy and sample entropy. Am J Physiol Heart Circ "
    "Physiol 278(6):H2039-H2049.",
    "doi": "10.1152/ajpheart.2000.278.6.H2039",
}
HJORTH: Reference = {
    "citation": "Hjorth B (1970). EEG analysis based on time domain properties. "
    "Electroencephalogr Clin Neurophysiol 29(3):306-310.",
    "doi": "10.1016/0013-4694(70)90143-4",
}
LEMPEL_ZIV: Reference = {
    "citation": "Lempel A, Ziv J (1976). On the complexity of finite sequences. IEEE "
    "Trans Inf Theory 22(1):75-81.",
}
WELCH: Reference = {
    "citation": "Welch PD (1967). The use of fast Fourier transform for the "
    "estimation of power spectra: a method based on time averaging over short, "
    "modified periodograms. IEEE Trans Audio Electroacoust 15(2):70-73.",
}

WIDMANN: Reference = {
    "citation": "Widmann A, Schröger E, Maess B (2015). Digital filter design for "
    "electrophysiological data - a practical approach. J Neurosci Methods "
    "250:34-46.",
    "doi": "10.1016/j.jneumeth.2014.08.002",
}
PERRIN: Reference = {
    "citation": "Perrin F, Pernier J, Bertrand O, Echallier JF (1989). Spherical "
    "splines for scalp potential and current density mapping. Electroencephalogr "
    "Clin Neurophysiol 72(2):184-187.",
    "doi": "10.1016/0013-4694(89)90180-6",
}
KAYSER: Reference = {
    "citation": "Kayser J, Tenke CE (2006). Principal components analysis of "
    "Laplacian waveforms as a generic method for identifying ERP generator "
    "patterns: I. Evaluation with auditory oddball tasks. Clin Neurophysiol "
    "117(2):348-368.",
    "doi": "10.1016/j.clinph.2005.08.034",
}
GRATTON: Reference = {
    "citation": "Gratton G, Coles MG, Donchin E (1983). A new method for off-line "
    "removal of ocular artifact. Electroencephalogr Clin Neurophysiol "
    "55(4):468-484.",
    "doi": "10.1016/0013-4694(83)90135-9",
}
CROFT: Reference = {
    "citation": "Croft RJ, Barry RJ (2000). Removal of ocular artifact from the "
    "EEG: a review. Neurophysiol Clin 30(1):5-19.",
    "doi": "10.1016/S0987-7053(00)00055-1",
}
THOMSON: Reference = {
    "citation": "Thomson DJ (1982). Spectrum estimation and harmonic analysis. "
    "Proc IEEE 70(9):1055-1096.",
}
BABADI: Reference = {
    "citation": "Babadi B, Brown EN (2014). A review of multitaper spectral "
    "analysis. IEEE Trans Biomed Eng 61(5):1555-1564.",
    "doi": "10.1109/TBME.2014.2311996",
}
SPECPARAM: Reference = {
    "citation": "Donoghue T, Haller M, Peterson EJ, et al. (2020). Parameterizing "
    "neural power spectra into periodic and aperiodic components. Nat Neurosci "
    "23(12):1655-1665.",
    "doi": "10.1038/s41593-020-00744-x",
}
TALLON_BAUDRY: Reference = {
    "citation": "Tallon-Baudry C, Bertrand O, Delpuech C, Pernier J (1997). "
    "Oscillatory gamma-band (30-70 Hz) activity induced by a visual search task "
    "in humans. J Neurosci 17(2):722-734.",
    "doi": "10.1523/JNEUROSCI.17-02-00722.1997",
}
LIN_CCA: Reference = {
    "citation": "Lin Z, Zhang C, Wu W, Gao X (2006). Frequency recognition based "
    "on canonical correlation analysis for SSVEP-based BCIs. IEEE Trans Biomed "
    "Eng 53(12 Pt 2):2610-2614.",
    "doi": "10.1109/TBME.2006.886577",
}
CHEN_FBCCA: Reference = {
    "citation": "Chen X, Wang Y, Gao S, Jung TP, Gao X (2015). Filter bank "
    "canonical correlation analysis for implementing a high-speed SSVEP-based "
    "brain-computer interface. J Neural Eng 12(4):046008.",
    "doi": "10.1088/1741-2560/12/4/046008",
}
RAMOSER: Reference = {
    "citation": "Ramoser H, Müller-Gerking J, Pfurtscheller G (2000). Optimal "
    "spatial filtering of single trial EEG during imagined hand movement. IEEE "
    "Trans Rehabil Eng 8(4):441-446.",
    "doi": "10.1109/86.895946",
}
KOLES: Reference = {
    "citation": "Koles ZJ (1991). The quantitative extraction and topographic "
    "mapping of the abnormal components in the clinical EEG. Electroencephalogr "
    "Clin Neurophysiol 79(6):440-447.",
    "doi": "10.1016/0013-4694(91)90163-X",
}

_FILTERS = [SCIPY]
_SPECTRA = [WELCH, THOMSON, BABADI, SCIPY, MNE]
BY_NODE: dict[str, list[Reference]] = {
    "meegxer.nodes.iir_filter": _FILTERS,
    "meegxer.nodes.iir_filter_zerophase": _FILTERS,
    "meegxer.nodes.notch": _FILTERS,
    "meegxer.nodes.notch_zerophase": _FILTERS,
    "meegxer.nodes.fir_filter": [WIDMANN, MNE, SCIPY],
    "meegxer.nodes.fir_filter_zerophase": [WIDMANN, MNE],
    "meegxer.nodes.interpolate_bads": [PERRIN, MNE],
    "meegxer.nodes.csd": [PERRIN, KAYSER, MNE],
    "meegxer.nodes.eog_regression": [GRATTON, CROFT, MNE],
    "meegxer.nodes.detect_amplitude": [MNE],
    "meegxer.nodes.detect_muscle": [MNE],
    "meegxer.nodes.detect_blinks": [MNE],
    "meegxer.nodes.rereference": [],
    "meegxer.nodes.pick_channels": [],
    "meegxer.nodes.resample": [SCIPY],
    "meegxer.nodes.epochs": [LUCK, MNE],
    "meegxer.nodes.ica": [MNE, PICARD, EXTENDED_INFOMAX, INFOMAX, ICLABEL,
                          MNE_ICALABEL],
    "meegxer.nodes.psd": _SPECTRA,
    "meegxer.nodes.band_power": _SPECTRA,
    "meegxer.nodes.specparam": [SPECPARAM, *_SPECTRA],
    "meegxer.nodes.morlet": [TALLON_BAUDRY, MNE, SCIPY],
    "meegxer.nodes.ssvep_cca": [LIN_CCA, CHEN_FBCCA, SCIPY],
    "meegxer.nodes.csp": [RAMOSER, KOLES, MNE],
    "meegxer.nodes.connectivity": [MNE_CONNECTIVITY, MNE, IMCOH, PLV, WPLI],
    "meegxer.nodes.complexity": [ANTROPY, PERMUTATION_ENTROPY, SAMPLE_ENTROPY,
                                 HJORTH, LEMPEL_ZIV],
    "meegxer.nodes.hilbert_envelope": [SCIPY],
}  # fmt: skip


# where each built-in node is listed in the GUI (manifest ``category``)
CATEGORIES = {
    "meegxer.nodes.iir_filter": "filters",
    "meegxer.nodes.iir_filter_zerophase": "filters",
    "meegxer.nodes.notch": "filters",
    "meegxer.nodes.notch_zerophase": "filters",
    "meegxer.nodes.fir_filter": "filters",
    "meegxer.nodes.fir_filter_zerophase": "filters",
    "meegxer.nodes.rereference": "reference",
    "meegxer.nodes.pick_channels": "reference",
    "meegxer.nodes.resample": "filters",
    "meegxer.nodes.epochs": "epochs",
    "meegxer.nodes.interpolate_bads": "reference",
    "meegxer.nodes.csd": "reference",
    "meegxer.nodes.ica": "artifacts",
    "meegxer.nodes.eog_regression": "artifacts",
    "meegxer.nodes.detect_amplitude": "events",
    "meegxer.nodes.detect_muscle": "events",
    "meegxer.nodes.detect_blinks": "events",
    "meegxer.nodes.psd": "spectral",
    "meegxer.nodes.band_power": "spectral",
    "meegxer.nodes.connectivity": "connectivity",
    "meegxer.nodes.complexity": "complexity",
    "meegxer.nodes.hilbert_envelope": "time-frequency",
    "meegxer.nodes.specparam": "spectral",
    "meegxer.nodes.morlet": "time-frequency",
    "meegxer.nodes.ssvep_cca": "decoding",
    "meegxer.nodes.csp": "decoding",
}


def credits(name: str) -> dict[str, object]:
    """Manifest fields giving credit for a built-in node, and its category."""
    out: dict[str, object] = {
        "authors": AUTHORS,
        "references": BY_NODE.get(name, []),
        "license": LICENSE,
    }
    if name in CATEGORIES:
        out["category"] = CATEGORIES[name]
    return out
