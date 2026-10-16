import re

from meegxer.core.registry import Registry
from meegxer.gui.names import title

# acronyms the fallback title would spell in lower case ("Bids writer")
ACRONYMS = re.compile(r"\b(Bids|Lsl|Edf|Tsv|Ica|Psd|Asr|Eeglab)\b")


def test_installed_nodes_keep_acronyms_in_their_titles() -> None:
    registry = Registry.from_entry_points()
    bad = {n: title(n) for n in registry.names() if ACRONYMS.search(title(n))}
    assert not bad, bad
