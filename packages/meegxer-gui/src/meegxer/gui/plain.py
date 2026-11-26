"""Text from plugins, pipelines and recordings, shown as written.

Qt shows a label or a tooltip whose text looks like HTML as HTML, so a
plugin's description or a shared pipeline's value could draw its own
content, or a link (red team L2, docs/SECURITY.md). Such text goes through
``tip`` (tooltips) or a label set to plain text, and only web links open.
"""

from __future__ import annotations

import html

from PySide6.QtGui import Qt


def tip(text: str) -> str:
    """``text`` for a tooltip, shown as written (escaped when Qt would take
    it for HTML; otherwise unchanged)."""
    if not Qt.mightBeRichText(text):
        return text
    return "<qt>" + html.escape(text).replace("\n", "<br>") + "</qt>"


def web_link(url: str) -> bool:
    """Whether a link may open: ``https:`` or ``http:`` only (not ``file:``
    or a custom protocol, which the system would hand to a program)."""
    return url.strip().lower().startswith(("https://", "http://"))
