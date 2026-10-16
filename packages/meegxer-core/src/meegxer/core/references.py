"""References shared by built-in nodes (manifest ``references``, spec 0.2.12).

The core's own I/O nodes cite the software they read and write with; the built-in
processing nodes (meegxer-nodes) reuse these entries. DOIs were checked against
PubMed or the publisher's listing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

Reference = dict[str, str]

MNE: Reference = {
    "citation": "Gramfort A, Luessi M, Larson E, et al. (2013). MEG and EEG data "
    "analysis with MNE-Python. Front Neurosci 7:267.",
    "doi": "10.3389/fnins.2013.00267",
}
MNE_BIDS: Reference = {
    "citation": "Appelhoff S, Sanderson M, Brooks TL, et al. (2019). MNE-BIDS: "
    "Organizing electrophysiological data into the BIDS format and facilitating "
    "their analysis. J Open Source Softw 4(44):1896.",
    "doi": "10.21105/joss.01896",
}
EEG_BIDS: Reference = {
    "citation": "Pernet CR, Appelhoff S, Gorgolewski KJ, et al. (2019). EEG-BIDS, an "
    "extension to the brain imaging data structure for electroencephalography. "
    "Sci Data 6(1):103.",
    "doi": "10.1038/s41597-019-0104-8",
}

_BIDS = [MNE_BIDS, EEG_BIDS, MNE]
BY_NODE: dict[str, list[Reference]] = {
    "meegxer.core.file": [MNE],
    "meegxer.core.bids": _BIDS,
    "meegxer.core.set_writer": [MNE],
    "meegxer.core.edf_writer": [MNE],
    "meegxer.core.bids_writer": _BIDS,
    "meegxer.core.bids_features": _BIDS,
}

AUTHORS = ["MeegXer contributors"]
LICENSE = "GPL-3.0-or-later"


def as_text(reference: Mapping[str, str]) -> str:
    """One line: the citation, then its DOI (as a link) or URL."""
    text = reference["citation"]
    if reference.get("doi"):
        return f"{text} https://doi.org/{reference['doi']}"
    if reference.get("url"):
        return f"{text} {reference['url']}"
    return text


def collect(references: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """References in first-seen order, each once (same DOI, URL or text)."""
    seen: set[str] = set()
    out = []
    for ref in references:
        key = ref.get("doi") or ref.get("url") or ref["citation"]
        if key not in seen:
            seen.add(key)
            out.append(dict(ref))
    return out


def credits(name: str, references: list[Reference] | None = None) -> dict[str, object]:
    """Manifest fields giving credit for a built-in node."""
    return {
        "authors": AUTHORS,
        "references": BY_NODE.get(name, []) if references is None else references,
        "license": LICENSE,
    }
