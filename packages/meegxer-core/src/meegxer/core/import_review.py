"""Review at import (docs/REVIEW.md, Design v2: mode 1).

A recording is reviewed once, before any pipeline: its events (added, moved,
renamed, deleted), labelled spans, bad channels, and a crop at the start and
end. The review is saved per recording, independent of any pipeline:

- an input in a BIDS dataset: in its BIDS derivatives, as
  ``derivatives/meegxer/sub-01/eeg/sub-01_task-rest_desc-import_events.tsv``;
- any other file: next to it, as ``<file>_desc-import_events.tsv``.

It is a BIDS events file (onset and duration in seconds from the raw file's
first sample, ``trial_type``) with a JSON sidecar recording, under
``MeegXer``, the bad channels, the crop, who reviewed and when. The raw file is
never touched.

File and BIDS inputs apply it when they load (their ``import_review``
setting, on by default): the recording is cropped to the kept span (its first
sample is the crop's start), its events are the reviewed ones (shifted onto the
cropped time, those outside left out) and its bad channels include the
reviewed ones. The review is part of the input's identity, so a change asks for
a refit and a new review downstream (rule 7).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from meegxer.core.node import Source
from meegxer.core.review import (
    Mark,
    Review,
    load_review,
    merge_spans,
    now_iso,
    save_review,
)

IMPORT_DESC = "import"
IMPORT_NODE = "import"  # the review's "strip" in its sidecar


def import_review_path(source: Source) -> Path | None:
    """Where the review at import of ``source``'s recording is kept (None: the
    input has no file)."""
    from meegxer.core.bids_out import target_for

    try:
        graph = SimpleNamespace(source=SimpleNamespace(impl=source))
        return target_for(graph, {"desc": IMPORT_DESC}).path("events.tsv")
    except ValueError:
        pass
    files = source.input_files()
    if not files:
        return None
    raw = Path(files[0]).resolve()
    stem = raw.name.split(".")[0]
    return raw.with_name(f"{stem}_desc-{IMPORT_DESC}_events.tsv")


def find_import_review(source: Source) -> tuple[Path, Review] | None:
    """The review at import of ``source``'s recording, when there is one."""
    path = import_review_path(source)
    if path is None:
        return None
    review = load_review(path)
    return None if review is None else (path, review)


def review_files(source: Source) -> list[Path]:
    """The files of the review at import (part of the input's identity)."""
    path = import_review_path(source)
    if path is None or not path.is_file():
        return []
    sidecar = path.with_suffix(".json")
    return [path, *([sidecar] if sidecar.is_file() else [])]


def open_import_review(source: Source) -> dict[str, Any]:
    """What the review window needs to review ``source``'s recording at
    import: the raw recording's events and bad channels (``source`` must not
    apply the review), or those of the review saved before, with its crop."""
    path = import_review_path(source)
    if path is None:
        raise ValueError("this input has no file to review at import")
    raw_events = list(getattr(source, "annotations", list)())
    found = find_import_review(source)
    review = found[1] if found is not None else None
    marks = ([m.to_wire() for m in review.marks] if review is not None
             else [{"onset": float(e["onset"]), "duration": float(e["duration"]),
                    "label": str(e["label"])} for e in raw_events])  # fmt: skip
    labels = sorted({str(m["label"]) for m in marks}
                    | {str(e["label"]) for e in raw_events})  # fmt: skip
    n = source.n_samples
    return {
        "node": IMPORT_NODE, "file": str(path), "reason": "import",
        "marks": marks, "labels": labels or ["BAD"],
        "duration_s": 0.0 if n is None else n / source.info.sfreq,
        "reject": ["BAD"], "display": {}, "points": [],
        "viewed": [list(v) for v in review.viewed] if review is not None else [],
        "bads": list(review.bads) if review is not None else list(source.info.bads),
        "crop": None if review is None or review.crop is None else list(review.crop),
        "reviewer": review.reviewer if review is not None else "",
        "reviewed": review.reviewed if review is not None else "",
    }  # fmt: skip


def save_import_review(
    source: Source,
    marks: list[Mark],
    bads: list[str],
    crop: tuple[float, float | None] | None,
    viewed: list[tuple[float, float]],
    reviewer: str,
) -> Path:
    """Save the review at import of ``source``'s recording."""
    path = import_review_path(source)
    if path is None:
        raise ValueError("this input has no file to review at import")
    labels = set(source.info.ch_labels)
    unknown = sorted(set(bads) - labels)
    if unknown:
        raise ValueError(f"no channels {unknown} to mark bad")
    if crop is not None and (crop[0] < 0 or (crop[1] is not None
                                             and crop[1] <= crop[0])):  # fmt: skip
        raise ValueError("the crop must start at 0 s or later and end after it")
    review = Review(tuple(marks), "", now_iso(), merge_spans(viewed), reviewer,
                    tuple(dict.fromkeys(bads)), crop)  # fmt: skip
    save_review(path, review, IMPORT_NODE)
    return path


def crop_events(
    events: list[dict[str, Any]], crop: tuple[float, float | None] | None
) -> list[dict[str, Any]]:
    """Events (onset s from the raw file's start) on the cropped recording's
    time: shifted by the crop's start, those outside it left out."""
    if crop is None:
        return events
    t0, t1 = crop
    return [
        {**e, "onset": float(e["onset"]) - t0}
        for e in events
        if float(e["onset"]) >= t0 and (t1 is None or float(e["onset"]) < t1)
    ]
