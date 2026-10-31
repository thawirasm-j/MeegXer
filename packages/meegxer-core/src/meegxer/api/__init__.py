"""MeegXer from Python, for MNE users (M10.7, ``docs/PYTHON_API.md``).

>>> from meegxer.api import run
>>> clean = run("pipeline.json", raw)                 # mne Raw in, Raw out
>>> epochs = run("erp.json", "sub-01.set", output="ep1")  # an Epochs strip's

``run`` is a thin layer over the batch scheduler (rule 4: the same node code
as the GUI and ``meegxer run``): the pipeline's input becomes ``raw``, an
output node is added where the result is read, and the run goes as
``meegxer run`` goes (the strips fit on the whole recording, a session's
calibrated models are used, the pipeline's own writers write their files).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

__all__ = ["run"]

logger = logging.getLogger("meegxer")


def run(
    pipeline: str | PathLike[str] | Mapping[str, Any],
    raw: Any,
    *,
    output: str | None = None,
    ica: Any = None,
    block_size: int | None = None,
    allow_version_mismatch: bool = False,
) -> Any:
    """Run ``pipeline`` over ``raw`` and return what ``output`` gives.

    - ``pipeline``: a pipeline or session file (``.json``,
      ``.meegxer.json``), or the same as a dict. A session's calibrated
      models are used where they fit this input (else refitted).
    - ``raw``: an ``mne.io.Raw`` (left as it is: the run reads a copy; its
      annotations are the input's events), or a recording file the File
      input reads (``.set``, ``.edf``, ``.xdf``, …, with what it carries,
      e.g. a ``.set``'s ICA).
    - ``output``: where the result is read. None: the signal at the chain's
      end, as ``mne.io.Raw``. A strip's id: its signal (``mne.io.Raw``), or
      for an Epochs strip (or Review epochs) its epochs (``mne.Epochs``;
      None when no event was cut). ``"strip:port"``: that rail output, e.g.
      ``"ica1:ics"`` for the ICA strip's components (``misc`` channels).
    - ``ica``: an ``mne.preprocessing.ICA`` of ``raw``, or its file
      (``-ica.fif``): its components start the ICs rail at the input
      (``"input:ics"``), instead of a ``.set``'s own.

    The returned Raw has the events on the Events rail there as
    annotations (a ``boundary`` at each cut), the channels' types and
    positions, and no measurement date or subject information. What the run
    noted goes to the ``meegxer`` logger; saves it skipped and models it
    could not reuse are warnings. A Review strip not yet reviewed stops it
    (``meegxer.core.review.ReviewNeeded``).
    """
    from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
    from meegxer.core.memory import EpochsOutput, RawInput, RawOutput
    from meegxer.core.registry import Registry
    from meegxer.core.session import adopt_models, drop_shared_sinks, load_session
    from meegxer.core.sources import FileSource

    loaded = load_session(_document(pipeline), Registry.from_entry_points(),
                          allow_version_mismatch=allow_version_mismatch)  # fmt: skip
    graph = loaded.graph
    if isinstance(raw, str | PathLike):
        source: FileSource = FileSource({"path": str(raw)})
    else:
        source = RawInput(raw)
    if isinstance(ica, str | PathLike):
        import mne

        ica = mne.preprocessing.read_ica(str(ica), verbose="error")
    source.mne_ica = ica
    graph.replace(graph.source.id, source)
    for node_id in drop_shared_sinks(graph):  # a shared session's writers
        logger.info("%s writes nowhere (shared without its file name)", node_id)
    for node_id, why in _drop_unplaced_bids_writers(graph):
        warnings.warn(f"{node_id} left out: {why}", stacklevel=2)
    read_from, kind = _output(graph, output)
    collector = EpochsOutput() if kind == "epochs" else RawOutput()
    if isinstance(collector, RawOutput):
        collector.components = kind == "components"
    taken = {n.id for n in graph.nodes}
    name = next(f"api_output{k or ''}" for k in range(len(taken) + 1)
                if f"api_output{k or ''}" not in taken)  # fmt: skip
    graph.add(collector, id=name, input=read_from)
    adopt_models(loaded, offline_input_fingerprints(graph))
    for warning in loaded.warnings:
        warnings.warn(warning, stacklevel=2)
    result = BatchScheduler(block_size=block_size).run(graph)
    for skipped in result.skipped:
        warnings.warn(f"skipped: {skipped}", stacklevel=2)
    for note in result.notes:
        logger.info(note)
    if isinstance(collector, EpochsOutput):
        return collector.epochs
    return collector.raw


def _drop_unplaced_bids_writers(graph: Any) -> list[tuple[str, str]]:
    """Remove the BIDS writers that cannot name where they write (a template
    made for a BIDS input, run here on a Raw in memory or a file named
    without subject and task); each one's id and why."""
    from meegxer.core.bids_out import target_for
    from meegxer.core.node import Sink

    dropped = []
    for node in list(graph.nodes):
        if isinstance(node.impl, Sink) and node.manifest.name.startswith(
            "meegxer.core.bids_"
        ):
            try:
                target_for(graph, node.impl.params)
            except ValueError as exc:
                graph.remove(node.id)
                dropped.append((node.id, str(exc)))
    return dropped


def _document(pipeline: str | PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """The pipeline as a session document (a plain pipeline holds no models)."""
    from meegxer.core.session import SESSION_FORMAT, is_session, read_session

    if isinstance(pipeline, Mapping):
        data = dict(pipeline)
        return data if is_session(data) else {"meegxer_session": SESSION_FORMAT,
                                               "pipeline": data}  # fmt: skip
    return read_session(Path(pipeline))


def _output(graph: Any, output: str | None) -> tuple[str, str]:
    """Where the output node reads (``strip`` or ``strip:port``) and what it
    gets there: ``signal``, ``components`` (a component rail) or ``epochs``."""
    from meegxer.core.graph import GraphError
    from meegxer.core.node import NodeRole
    from meegxer.core.ports import PortType

    if output is None:  # the signal at the chain's end
        main = [n for n in graph.nodes
                if n.role in (NodeRole.SOURCE, NodeRole.PROCESSOR)
                and n.manifest.outputs[0].type is PortType.SIGNAL]  # fmt: skip
        return main[-1].id, "signal"
    node_id, _, port_name = output.partition(":")
    try:
        node = graph[node_id]
    except KeyError:
        known = ", ".join(n.id for n in graph.nodes)
        raise GraphError(f"no strip {node_id!r} in the pipeline ({known})") from None
    outputs = node.manifest.outputs
    if port_name:
        port = next((p for p in outputs if p.name == port_name), None)
        if port is None:
            names = ", ".join(p.name for p in outputs) or "none"
            raise GraphError(f"{node_id!r} has no output {port_name!r} ({names})")
    elif (
        outputs
        and outputs[0].type is PortType.SIGNAL
        and not any(p.rail == "epochs" for p in outputs[1:])
    ):
        return node_id, "signal"
    else:  # an Epochs strip: its epochs rail
        port = next((p for p in outputs[1:] if p.rail == "epochs"), None)
        if port is None:
            raise GraphError(
                f"{node_id!r} gives no signal or epochs to return "
                "(features: write them with a feature writer)"
            )
    where = f"{node_id}:{port.name}" if port is not outputs[0] else node_id
    if port.type is PortType.SIGNAL:
        return where, "components" if port.space == "component" else "signal"
    if port.rail == "epochs":
        return where, "epochs"
    raise GraphError(f"{output!r} gives features, not a signal or epochs "
                     "(write them with a feature writer)")  # fmt: skip
