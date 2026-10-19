"""One pipeline over many recordings of a BIDS dataset (``meegxer run --bids`` and
the GUI's all-subjects offline run share this).

Each recording gets a fresh graph: the pipeline's input is replaced by that
recording (its Recording settings carried over), its outputs go to the dataset's
BIDS derivatives (a BIDS writer is added when the chain has none), and a recording
whose session file is already there counts as done (``spec/bids-derivatives.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from meegxer.core.bids_out import (
    Target,
    bids_sinks,
    ensure_writer,
    find_recordings,
    target_for,
)
from meegxer.core.graph import Graph
from meegxer.core.node import Sink, Source
from meegxer.core.recording import RecordingOverrides, recording_of, with_recording
from meegxer.core.session import drop_shared_sinks
from meegxer.core.sources import BidsSource

_SHORT = {"subject": "sub", "session": "ses", "task": "task", "run": "run"}


def recording_label(params: dict[str, str]) -> str:
    """``sub-01_ses-a_task-rest``: a recording's BIDS entities."""
    return "_".join(f"{_SHORT[k]}-{v}" for k, v in params.items() if v and k in _SHORT)


def is_all_subjects(source: object) -> bool:
    """A BIDS input naming no subject: every subject of its dataset."""
    return isinstance(source, BidsSource) and not str(source.params["subject"])


def recordings_of(source: BidsSource) -> list[dict[str, str]]:
    """The recordings an all-subjects BIDS input stands for: every subject's,
    narrowed by its task, session and run when given."""
    p = source.params

    def only(name: str) -> list[str]:
        return [str(p[name])] if str(p[name]) else []

    return find_recordings(Path(str(p["root"])).resolve(), tasks=only("task"),
                           sessions=only("session"), runs=only("run"))  # fmt: skip


@dataclass
class PreparedRecording:
    label: str
    target: Target
    done: bool  # its session file exists (and overwrite is off): skip it
    notes: list[str] = field(default_factory=list)


def prepare_recording(
    graph: Graph, params: dict[str, str], root: str | Path | None, overwrite: bool
) -> PreparedRecording:
    """Point ``graph`` at one recording and its derivatives (in place)."""
    first = graph.source.impl
    assert isinstance(first, Source)
    settings = recording_of(first)
    source = (with_recording(BidsSource, params, RecordingOverrides.from_dict(settings))
              if settings else BidsSource(params))  # fmt: skip
    graph.replace(graph.source.id, source)
    drop_shared_sinks(graph)  # a shared session's file writers
    notes = []
    bids = set(map(id, bids_sinks(graph)))
    for node in list(graph.nodes):  # one fixed path cannot hold every recording
        if (isinstance(node.impl, Sink) and id(node.impl) not in bids
                and "path" in node.impl.params):  # fmt: skip
            graph.remove(node.id)
            notes.append(f"{node.id}: not written (one file cannot hold every "
                         "recording; the BIDS writer saves each)")  # fmt: skip
    writer = ensure_writer(graph, root, overwrite)
    target = target_for(graph, writer.params)
    # the session file is written last: without it the recording is unfinished
    # (e.g. interrupted while writing) and is done again
    done = target.path("meegxer.json").exists() and not overwrite
    if not done and target.path("eeg.set").exists():
        ensure_writer(graph, root, True)  # replace the partial files
    return PreparedRecording(recording_label(params), target, done, notes)
