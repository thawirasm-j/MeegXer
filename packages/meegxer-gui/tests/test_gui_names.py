import re

from meegxer.core.registry import Registry
from meegxer.gui.labels import CHOICES, PARAMS
from meegxer.gui.names import TITLES, badge, title

# acronyms the fallback title would spell in lower case ("Bids writer")
ACRONYMS = re.compile(r"\b(Bids|Lsl|Edf|Tsv|Ica|Psd|Asr|Eeglab)\b")


def test_installed_nodes_keep_acronyms_in_their_titles() -> None:
    registry = Registry.from_entry_points()
    bad = {n: title(n) for n in registry.names() if ACRONYMS.search(title(n))}
    assert not bad, bad


def test_built_in_nodes_have_proper_titles_and_labels() -> None:
    """Every built-in node, parameter and choice has a written label, so none is
    left to the mechanical fallback (which cannot know terms or case)."""
    registry = Registry.from_entry_points()
    built_in = [n for n in registry.names() if n.startswith("meegxer.")]
    special = {"meegxer.nodes.iir_filter", "meegxer.core.probe"}  # titled by params
    assert not [n for n in built_in if n not in TITLES and n not in special]
    missing = [
        (n, p.name, c)
        for n in built_in
        for p in registry.get(n).manifest.params
        for c in (None, *(p.choices or ()))
        if (c is None and p.name not in PARAMS and not getattr(p, "label", ""))
        or (c is not None and (p.name, str(c)) not in CHOICES)
    ]
    assert not missing, missing


def test_badges_name_the_variant() -> None:
    assert badge("meegxer.nodes.psd", {"method": "multitaper"}, "live") == "multitaper"
    assert badge("meegxer.nodes.psd", {"method": "welch"}, "live") == ""
    assert badge("meegxer.nodes.ssvep_cca", {"method": "cca"}, "live") == "CCA"
    assert badge("meegxer.nodes.ssvep_cca", {"method": "fbcca"}, "live") == (
        "filter-bank CCA"
    )
