"""The review part of a recording's QC report (``docs/REVIEW.md`` step 6).

For each review of the recording, the run's QC row and report say who reviewed
and when, how much of the recording they looked at, what they marked (per
label: how many, how long), the channels they marked bad, how they answered a
detector's suggestions (accepted, edited, rejected, per detector), and for an
epoch review the epochs kept and rejected per label. The review at import
(the recording's own sidecar) comes first, then each Review and Review epochs
strip in the chain. Only what the sidecars hold is reported: nothing is
recomputed.
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from meegxer.core.review import Mark, Review, merge_spans


@dataclass
class ReviewSection:
    """One review in the report: the QC row's fields (``<key>_…``) and the
    rows of its table."""

    key: str  # "import", or the strip's id
    title: str
    row: dict[str, Any] = field(default_factory=dict)
    marks: list[tuple[str, int, float]] = field(default_factory=list)


def _marks_text(marks: Sequence[tuple[str, int, float]]) -> str:
    return "; ".join(f"{label}: {n} ({seconds:.1f} s)" for label, n, seconds in marks)


def _per_label(marks: Sequence[Mark]) -> list[tuple[str, int, float]]:
    count: Counter[str] = Counter()
    seconds: dict[str, float] = defaultdict(float)
    for m in marks:
        if m.verdict == "rejected":  # a suggestion said no to: not a mark
            continue
        count[m.label] += 1
        seconds[m.label] += m.duration
    return [(label, count[label], seconds[label]) for label in sorted(count)]


def _verdicts(marks: Sequence[Mark]) -> str:
    """How the suggestions were answered, per detector (strip that proposed
    them; ``input``: the recording's own events)."""
    by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    for m in marks:
        if m.suggestion is not None and m.verdict:
            by_origin[m.suggestion.origin or "input"][m.verdict] += 1
    parts = []
    for origin in sorted(by_origin):
        c = by_origin[origin]
        parts.append(f"{origin}: " + ", ".join(f"{v} {c[v]}" for v in
                     ("accepted", "edited", "rejected") if c[v]))  # fmt: skip
    return "; ".join(parts)


def review_section(
    key: str, title: str, review: Review, duration: float
) -> ReviewSection:
    """The report's part for one marks review (at import, or a Review strip)."""
    viewed = sum(b - a for a, b in merge_spans(review.viewed))
    section = ReviewSection(key, title, marks=_per_label(review.marks))
    row = section.row
    row[f"{key}_reviewer"] = review.reviewer
    row[f"{key}_reviewed"] = review.reviewed
    if duration > 0 and review.viewed:
        row[f"{key}_viewed_percent"] = round(100 * min(viewed, duration) / duration, 1)
    row[f"{key}_marks"] = _marks_text(section.marks)
    if review.bads:
        row[f"{key}_bad_channels"] = ",".join(review.bads)
    if verdicts := _verdicts(review.marks):
        row[f"{key}_verdicts"] = verdicts
    if review.raters:  # resolved from two reviewers (step 6b)
        from meegxer.core.two_reviewers import agreement_text

        row[f"{key}_raters"] = ", ".join(r or "?" for r in review.raters)
        row[f"{key}_kappa"] = agreement_text(review.agreement or {})
    if review.crop is not None:
        start, stop = review.crop
        row[f"{key}_crop_s"] = f"{start:g} to {'end' if stop is None else f'{stop:g}'}"
    return section


def epoch_review_section(key: str, title: str, record: Any) -> ReviewSection:
    """The report's part for an epoch review (Review epochs strip)."""
    kept: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for v in record.epochs:
        (kept if v.keep else rejected)[v.label] += 1
    section = ReviewSection(key, title)
    labels = sorted(set(kept) | set(rejected))
    section.row = {
        f"{key}_reviewer": record.reviewer,
        f"{key}_reviewed": record.reviewed,
        f"{key}_epochs": "; ".join(
            f"{lb}: {kept[lb]} kept, {rejected[lb]} rejected" for lb in labels
        ),
    }
    return section


def review_sections(graph: Any) -> list[ReviewSection]:
    """The reviews of the graph's recording: at import, then each review strip
    in the chain whose sidecar exists."""
    from meegxer.core.epoch_review import EPOCH_REVIEW_NODE, load_epoch_review
    from meegxer.core.import_review import find_import_review
    from meegxer.core.node import Source
    from meegxer.core.review import REVIEW_NODE, load_review, review_path

    source = graph.source.impl
    assert isinstance(source, Source)
    sfreq = float(source.info.sfreq)
    duration = float(source.n_samples or 0) / sfreq if sfreq else 0.0
    sections: list[ReviewSection] = []
    found = find_import_review(source)
    if found is not None:
        sections.append(
            review_section("import", "Review at import", found[1], duration)
        )
    for node in graph.nodes:
        name = node.manifest.name
        if name not in (REVIEW_NODE, EPOCH_REVIEW_NODE):
            continue
        try:
            path = review_path(graph, node.id, node.impl.params)
        except ValueError:
            continue
        if name == REVIEW_NODE:
            review = load_review(path)
            if review is not None:
                sections.append(review_section(node.id, f"Review ({node.id})",
                                               review, duration))  # fmt: skip
        else:
            record = load_epoch_review(path)
            if record is not None:
                sections.append(epoch_review_section(
                    node.id, f"Review epochs ({node.id})", record))  # fmt: skip
    return sections


def section_html(section: ReviewSection) -> str:
    """The section's table, and its marks per label."""
    names = {"reviewer": "Reviewer", "reviewed": "Reviewed (UTC)",
             "viewed_percent": "Viewed (% of the recording)", "marks": "Marks",
             "bad_channels": "Bad channels", "verdicts": "Suggestions answered",
             "crop_s": "Kept (s)", "epochs": "Epochs",
             "raters": "Reviewers (resolved by the reviewer above)",
             "kappa": "Agreement (Cohen's kappa, 1 s windows)"}  # fmt: skip
    rows = "".join(
        f"<tr><th>{html.escape(names.get(k.split('_', 1)[1], k))}</th>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in section.row.items() if v not in ("", None)
    )  # fmt: skip
    table = f"<table class='table table-sm'>{rows}</table>"
    if section.marks:
        lines = "".join(
            f"<tr><td>{html.escape(lb)}</td><td>{n}</td><td>{s:.1f}</td></tr>"
            for lb, n, s in section.marks
        )
        table += ("<table class='table table-sm'><tr><th>Label</th><th>Marks</th>"
                  f"<th>Seconds</th></tr>{lines}</table>")  # fmt: skip
    return table


def review_row(sections: Sequence[ReviewSection]) -> Mapping[str, Any]:
    """Every section's QC fields, in one row."""
    out: dict[str, Any] = {}
    for s in sections:
        out.update({k: v for k, v in s.row.items() if v not in ("", None)})
    return out
