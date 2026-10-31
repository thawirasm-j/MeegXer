"""Two reviewers, blind, then resolved (``docs/REVIEW.md`` step 6b).

With a Review strip's ``reviewers`` at 2, each reviewer marks the recording
without seeing the other's marks, in a sidecar of their own (``rater_path``).
Then one person resolves: the marks both made alike go on as they are; every
other mark, and every bad channel only one of them marked, is a suggestion
(its origin names the reviewer) to accept, edit or reject in verify mode.
The resolved marks are the strip's sidecar, as with one reviewer.

Agreement is Cohen's kappa (Cohen 1960), per label, over 1-second windows,
each marked or not by each reviewer: scikit-learn's ``cohen_kappa_score``.
The 1 s window is MeegXer's choice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from meegxer.core.review import BAD_CHANNEL, Mark, Review, Suggestion

WINDOW_S = 1.0  # MeegXer's choice: agreement is counted per second
SAME_S = 0.05  # marks this close (onset and duration) are the same mark


def rater_path(path: Path, k: int) -> Path:
    """Reviewer ``k``'s sidecar beside the strip's (resolved) one:
    ``…_desc-review1_events.tsv`` → ``…_desc-review1rater1_events.tsv``."""
    name = path.name
    if name.endswith("_events.tsv"):
        return path.with_name(name[: -len("_events.tsv")] + f"rater{k}_events.tsv")
    return path.with_name(f"{path.stem}_rater{k}{path.suffix}")


def _kept(review: Review) -> list[Mark]:
    """The reviewer's marks (a suggestion they rejected is none)."""
    return [m for m in review.marks if m.verdict != "rejected"]


def same(a: Mark, b: Mark) -> bool:
    """The same mark: label, and onset and duration within SAME_S."""
    return (a.label == b.label and abs(a.onset - b.onset) <= SAME_S
            and abs(a.duration - b.duration) <= SAME_S)  # fmt: skip


def origin(k: int, review: Review) -> str:
    return f"reviewer {k}" + (f" ({review.reviewer})" if review.reviewer else "")


def disagreements(
    first: Review, second: Review
) -> tuple[tuple[Mark, ...], tuple[Suggestion, ...], tuple[str, ...]]:
    """What resolving starts from: the marks both made alike (the first's),
    the suggestions to decide (every other mark of either, and each bad
    channel only one marked), and the bad channels both marked."""
    a, b = _kept(first), _kept(second)
    agreed = tuple(m for m in a if any(same(m, n) for n in b))
    points: list[Suggestion] = []
    for k, (own, other, review) in enumerate(((a, b, first), (b, a, second)), 1):
        for m in own:
            if not any(same(m, n) for n in other):
                points.append(Suggestion(m.onset, m.duration, m.label,
                                         origin(k, review)))  # fmt: skip
    both = tuple(ch for ch in first.bads if ch in second.bads)
    for k, (mine, theirs) in enumerate(((first, second), (second, first)), 1):
        for ch in mine.bads:
            if ch not in theirs.bads:
                points.append(Suggestion(0.0, 0.0, BAD_CHANNEL, origin(k, mine), ch))
    points.sort(key=lambda s: (s.onset, s.label, s.origin))
    return agreed, tuple(points), both


def _windows(marks: Sequence[Mark], label: str, n: int) -> np.ndarray:
    marked = np.zeros(n, dtype=int)
    for m in marks:
        if m.label != label:
            continue
        start = math.floor(m.onset / WINDOW_S)
        stop = math.ceil((m.onset + max(m.duration, 1e-9)) / WINDOW_S)
        marked[max(start, 0) : min(stop, n)] = 1
    return marked


def agreement(
    first: Review, second: Review, duration: float
) -> dict[str, float | None]:
    """Cohen's kappa per label over 1 s windows; None where it is undefined
    (both reviewers marked every window alike, e.g. neither used the label)."""
    from sklearn.metrics import cohen_kappa_score

    a, b = _kept(first), _kept(second)
    n = max(1, math.ceil(duration / WINDOW_S))
    out: dict[str, float | None] = {}
    for label in sorted({m.label for m in (*a, *b)}):
        x, y = _windows(a, label, n), _windows(b, label, n)
        if len({*x.tolist(), *y.tolist()}) < 2:  # one value only: undefined
            out[label] = None
            continue
        kappa = float(cohen_kappa_score(x, y))
        out[label] = None if math.isnan(kappa) else round(kappa, 3)
    return out


def agreement_text(kappa: dict[str, float | None]) -> str:
    return "; ".join(f"{label}: {'n/a' if k is None else f'{k:.2f}'}"
                     for label, k in kappa.items())  # fmt: skip
