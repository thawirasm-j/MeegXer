"""What to cite for each built-in node (manifest ``references``, spec 0.2.12).

Each node cites the method's original paper(s) and the software it runs on.
DOIs were checked against PubMed or the publisher's listing; a reference whose
DOI could not be confirmed is given without one.
"""

from __future__ import annotations

from meegxer.core.references import AUTHORS, LICENSE, MNE, Reference

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

_FILTERS = [SCIPY]
BY_NODE: dict[str, list[Reference]] = {
    "meegxer.nodes.iir_filter": _FILTERS,
    "meegxer.nodes.iir_filter_zerophase": _FILTERS,
    "meegxer.nodes.notch": _FILTERS,
    "meegxer.nodes.notch_zerophase": _FILTERS,
    "meegxer.nodes.rereference": [],
    "meegxer.nodes.ica": [MNE, PICARD, EXTENDED_INFOMAX, INFOMAX, ICLABEL,
                          MNE_ICALABEL],
    "meegxer.nodes.psd": [WELCH, SCIPY],
    "meegxer.nodes.band_power": [WELCH, SCIPY],
    "meegxer.nodes.connectivity": [MNE_CONNECTIVITY, MNE, IMCOH, PLV, WPLI],
    "meegxer.nodes.complexity": [ANTROPY, PERMUTATION_ENTROPY, SAMPLE_ENTROPY,
                                 HJORTH, LEMPEL_ZIV],
    "meegxer.nodes.hilbert_envelope": [SCIPY],
}  # fmt: skip


def credits(name: str) -> dict[str, object]:
    """Manifest fields giving credit for a built-in node."""
    return {
        "authors": AUTHORS,
        "references": BY_NODE.get(name, []),
        "license": LICENSE,
    }
