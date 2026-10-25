"""BIDS-derivatives writers (``spec/bids-derivatives.md``).

``meegxer.core.bids_writer`` writes the processed recording with its sidecars
into a derivatives dataset; ``meegxer.core.bids_features`` writes a feature table
next to it. Both take the recording's entities (subject, session, task, run) from
the pipeline's input, so the batch scheduler binds them to the graph before a
run, and the signal writer saves the run's session file once every node has run.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from meegxer.core.block import (
    Block,
    Event,
    FeatureBlock,
    SignalInfo,
    contiguous_runs,
)
from meegxer.core.manifest import Manifest
from meegxer.core.node import AnyBlock, Sink, Source
from meegxer.core.ports import PortTypeError
from meegxer.core.recording import describe
from meegxer.core.references import as_text, collect, credits
from meegxer.core.registry import node_origin
from meegxer.core.sinks import SetSink, feature_columns, feature_units, tsv_value

BIDS_VERSION = "1.10.0"
CODE_URL = "https://github.com/thawirasm-j/MeegXer"
ENTITY_ORDER = (("subject", "sub"), ("session", "ses"), ("task", "task"),
                ("acquisition", "acq"), ("run", "run"))  # fmt: skip
BIDS_TYPES = {"eeg": "EEG", "eog": "EOG", "ecg": "ECG", "emg": "EMG",
              "resp": "RESP", "stim": "TRIG", "misc": "MISC", "": "EEG"}  # fmt: skip
COUNT_KEYS = {"EEG": "EEGChannelCount", "EOG": "EOGChannelCount",
              "ECG": "ECGChannelCount", "EMG": "EMGChannelCount",
              "MISC": "MiscChannelCount", "TRIG": "TriggerChannelCount"}  # fmt: skip
_LABEL = re.compile(r"^[A-Za-z0-9]+$")
# not (yet) covered by BIDS: the run's session file and feature tables
IGNORED = ("*_meegxer.json", "*_features.tsv", "*_features.json")
README = """# MeegXer derivatives

EEG processed with MeegXer (https://github.com/thawirasm-j/MeegXer). Each
recording's `*_eeg.json` names its raw source (`Sources`) and the processing
chain (`Description`); `*_meegxer.json` is the run's session file (pipeline,
node versions, fitted models, file hashes), which reruns it exactly. Feature
tables (`*_features.tsv`) and session files are not yet covered by BIDS and are
listed in `.bidsignore`.

`Authors`, `License` and the other credits in `dataset_description.json` come
from the raw dataset's; add the people who made this dataset if it lists none.
"""
# credits a derivative keeps from its raw dataset (BIDS: dataset_description)
INHERITED = ("Authors", "License", "Acknowledgements", "HowToAcknowledge",
             "Funding", "EthicsApprovals")  # fmt: skip


def _manifest(name: str, input_type: str, description: str) -> Manifest:
    return Manifest.from_dict({
        "spec_version": "0.2.0", "name": name, "version": "0.1.0",
        "type": "stream", "role": "sink", "availability": "offline",
        "description": description,
        "inputs": [{"name": "in", "type": input_type}], "outputs": [],
        "params": [
            {"name": "root", "type": "string", "default": "", "path": "folder",
             "description": "Derivatives folder "
                            "(default: <raw dataset>/derivatives/meegxer)"},
            {"name": "desc", "type": "string", "default": "clean", "primary": True,
             "description": "The desc- label of the outputs (letters and digits)"},
            {"name": "subject", "type": "string", "default": "", "advanced": True},
            {"name": "session", "type": "string", "default": "", "advanced": True},
            {"name": "task", "type": "string", "default": "", "advanced": True},
            {"name": "run", "type": "string", "default": "", "advanced": True},
            {"name": "overwrite", "type": "bool", "default": False},
        ],
        **credits(name),
    })  # fmt: skip


@dataclass(frozen=True)
class Target:
    """Where a recording's derivative files go."""

    root: Path
    folder: Path  # root/sub-XX[/ses-YY]/eeg
    stem: str  # sub-XX[_ses-YY]_task-ZZ[_acq-…][_run-N]_desc-LABEL
    raw_root: Path | None
    raw_file: Path | None

    def path(self, suffix: str) -> Path:
        return self.folder / f"{self.stem}_{suffix}"

    def subject_path(self, suffix: str) -> Path:
        """A file shared by the subject's (session's) recordings: electrodes and
        coordinate system carry only ``sub-``/``ses-`` (BIDS 1.10)."""
        parts = [p for p in self.stem.split("_") if p.startswith(("sub-", "ses-"))]
        return self.folder / f"{'_'.join(parts)}_{suffix}"


def find_dataset_root(path: Path) -> Path | None:
    """The BIDS dataset a file lies in (the folder with dataset_description.json)."""
    for parent in path.resolve().parents:
        if (parent / "dataset_description.json").is_file():
            return parent
    return None


def _entities_of(source: Source) -> tuple[dict[str, str], Path | None, Path | None]:
    """Entities, raw dataset root and raw file of the pipeline's input."""
    files = source.input_files()
    raw_file = Path(files[0]).resolve() if files else None
    params = source.params
    if source.manifest.name == "meegxer.core.bids":
        entities = {k: str(params.get(k, "")) for k, _ in ENTITY_ORDER}
        root = Path(str(params["root"])).resolve() if params.get("root") else None
        return {k: v for k, v in entities.items() if v}, root, raw_file
    if raw_file is None:
        return {}, None, None
    from mne_bids import get_entities_from_fname

    found = get_entities_from_fname(raw_file.name, on_error="ignore")
    entities = {k: str(v) for k, v in found.items() if v and k in dict(ENTITY_ORDER)}
    return entities, find_dataset_root(raw_file), raw_file


def target_for(graph: Any, params: Mapping[str, object]) -> Target:
    """The derivative location of the graph's input under ``params``."""
    source = graph.source.impl
    assert isinstance(source, Source)
    entities, raw_root, raw_file = _entities_of(source)
    for key in ("subject", "session", "task", "run"):
        if str(params.get(key, "")):
            entities[key] = str(params[key])
    missing = [k for k in ("subject", "task") if not entities.get(k)]
    if missing:
        raise ValueError(
            f"BIDS output needs {' and '.join(missing)}: name the input like "
            "sub-01_task-rest_eeg.set, use a BIDS input, or set them on the writer"
        )
    desc = str(params.get("desc") or "clean")
    for key, value in [*entities.items(), ("desc", desc)]:
        if not _LABEL.match(value):
            raise ValueError(f"BIDS {key} label {value!r} must be letters and digits")
    if str(params.get("root", "")):
        root = Path(str(params["root"])).resolve()
    elif raw_root is not None:
        root = raw_root / "derivatives" / "meegxer"
    else:
        raise ValueError("BIDS output: set 'root' (the input is not in a BIDS dataset)")
    folder = root / f"sub-{entities['subject']}"
    if entities.get("session"):
        folder = folder / f"ses-{entities['session']}"
    parts = [
        f"{short}-{entities[key]}" for key, short in ENTITY_ORDER if key in entities
    ]
    return Target(root, folder / "eeg", "_".join([*parts, f"desc-{desc}"]),
                  raw_root, raw_file)  # fmt: skip


# -- dataset level -------------------------------------------------------------------


def _version() -> str:
    try:
        return metadata.version("meegxer-core")
    except metadata.PackageNotFoundError:
        return "unknown"


def _relative(path: Path, start: Path) -> str:
    import os

    return Path(os.path.relpath(path, start)).as_posix()


def _pipeline_credits(graph: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """What a pipeline's nodes ask to be cited (``ReferencesAndLinks``) and
    ``GeneratedBy`` entries for its plugins and custom scripts."""
    references = collect(
        r.as_dict() for n in graph.nodes for r in n.manifest.references
    )
    generated = []
    for node in graph.nodes:
        impl = node.impl.processor if hasattr(node.impl, "processor") else node.impl
        origin = node_origin(type(impl))
        if origin == "builtin":
            continue
        entry: dict[str, Any] = {"Name": node.manifest.name,
                                 "Version": node.manifest.version}  # fmt: skip
        what = "Custom script (unsandboxed)" if origin == "custom" else "Plugin"
        entry["Description"] = f"{what}: {node.manifest.description}".rstrip(": ")
        image = type(impl).provenance().get("image")
        if image:
            entry["Container"] = {"Type": "docker", "Tag": image}
        generated.append(entry)
    return [as_text(r) for r in references], generated


def write_dataset_files(target: Target, graph: Any = None) -> None:
    """Create or update dataset_description.json and .bidsignore; with the
    pipeline's ``graph``, also what to cite and which plugins made it."""
    target.root.mkdir(parents=True, exist_ok=True)
    desc_path = target.root / "dataset_description.json"
    data: dict[str, Any] = {}
    if desc_path.is_file():
        data = json.loads(desc_path.read_text(encoding="utf-8"))
    raw = _raw_description(target.raw_root)
    raw_name = raw.get("Name")
    data.setdefault("Name", f"{raw_name} (MeegXer derivatives)"
                    if _given(raw_name) else "MeegXer derivatives")  # fmt: skip
    for key in INHERITED:
        if key not in data and _given(raw.get(key)):
            data[key] = raw[key]
    data.setdefault("BIDSVersion", BIDS_VERSION)
    data["DatasetType"] = "derivative"
    cite, extra = _pipeline_credits(graph) if graph is not None else ([], [])
    ours = {"MeegXer", *(g["Name"] for g in extra)}
    others = [g for g in data.get("GeneratedBy", []) if g.get("Name") not in ours]
    data["GeneratedBy"] = [{"Name": "MeegXer", "Version": _version(),
                            "CodeURL": CODE_URL}, *extra, *others]  # fmt: skip
    links = list(data.get("ReferencesAndLinks", []))
    links += [c for c in [*raw.get("ReferencesAndLinks", []), *cite]
              if c not in links]  # fmt: skip
    if links:
        data["ReferencesAndLinks"] = links
    if target.raw_root is not None:
        data.setdefault("DatasetLinks", {})["raw"] = _relative(target.raw_root,
                                                               target.root)  # fmt: skip
        sources = data.setdefault("SourceDatasets", [])
        entry = next((e for e in sources if e.get("URL") == "bids:raw:"), None)
        if entry is None:
            entry = {"URL": "bids:raw:"}
            sources.append(entry)
        if _given(raw.get("DatasetDOI")):
            entry.setdefault("DOI", raw["DatasetDOI"])
    desc_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    ignore = target.root / ".bidsignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.is_file() else []
    missing = [p for p in IGNORED if p not in lines]
    if missing:
        ignore.write_text("\n".join([*lines, *missing]) + "\n", encoding="utf-8")
    readme = target.root / "README.md"
    if not readme.is_file() and not (target.root / "README").is_file():
        readme.write_text(README, encoding="utf-8")


def _raw_description(raw_root: Path | None) -> dict[str, Any]:
    """The raw dataset's dataset_description.json, or {} without one."""
    path = None if raw_root is None else raw_root / "dataset_description.json"
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _given(value: Any) -> bool:
    """Not empty and not a placeholder such as MNE-BIDS's "[Unspecified]"."""
    values = value if isinstance(value, list) else [value]
    return any(
        isinstance(v, (str, dict))
        and v
        and not (isinstance(v, str) and v.strip().startswith("[Unspecified"))
        for v in values
    )


def _write_description(path: Path, description: str) -> Path:
    """A table's JSON sidecar with the Description that BIDS derivatives
    recommend for every derivative file."""
    path.write_text(json.dumps({"Description": description}, indent=2) + "\n",
                    encoding="utf-8")  # fmt: skip
    return path


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _chain(graph: Any, node_id: str) -> list[Any]:
    """The nodes from the source to ``node_id`` (inclusive), in order."""
    nodes = []
    current = graph[node_id]
    while current.input_node is not None:
        current = graph[current.input_node]
        nodes.append(current)
    return list(reversed(nodes))


def _describe_chain(graph: Any, node_id: str) -> str:
    steps = []
    for node in _chain(graph, node_id):
        params = ", ".join(f"{k}={v}" for k, v in sorted(node.impl.params.items())
                           if k not in ("path", "root"))  # fmt: skip
        steps.append(f"{node.manifest.name} {node.manifest.version}"
                     + (f" ({params})" if params else ""))  # fmt: skip
    return "MeegXer: " + " -> ".join(steps)


# -- the writers ---------------------------------------------------------------------


def _events_rows(events: list[dict[str, Any]], pieces: list[Block]) -> list[list[str]]:
    """The raw recording's events on the processed recording's time: shifted
    past the spans cut before them, left out inside a cut, with a ``boundary``
    row at each cut (docs/REVIEW.md)."""
    sfreq = pieces[0].info.sfreq
    spans = [(b.sample_index / sfreq, b.stop_index / sfreq) for b in pieces]
    rows: list[tuple[float, float, str]] = []
    for e in events:
        onset = float(e["onset"])
        removed = spans[0][0]  # the time before the first piece is not written
        for k, (start, stop) in enumerate(spans):
            if k:
                removed += start - spans[k - 1][1]  # the cut before this piece
            if onset < start:  # in a cut (or before the first piece)
                break
            if onset < stop:
                rows.append((onset - removed, float(e.get("duration", 0.0)),
                             str(e["label"])))  # fmt: skip
                break
    kept = 0.0
    for start, stop in spans[:-1]:  # a boundary at the end of each piece but the last
        kept += stop - start
        rows.append((kept, 0.0, "boundary"))
    rows.sort(key=lambda r: (r[0], r[2]))
    return [[repr(o), repr(d), label] for o, d, label in rows]


class _BidsSink(Sink):
    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._graph: Any = None
        self._node_id = ""
        self.written: list[Path] = []

    def bind(self, graph: Any, node_id: str) -> None:
        self._graph, self._node_id = graph, node_id

    def _target(self) -> Target:
        if self._graph is None:
            raise RuntimeError(f"{self.manifest.name} runs in batch runs only")
        return target_for(self._graph, self.params)


class BidsWriter(_BidsSink):
    """The processed recording, its sidecars and the run's session file in a
    BIDS derivatives dataset."""

    manifest: ClassVar[Manifest] = _manifest(
        "meegxer.core.bids_writer", "signal",
        "Processed recording into a BIDS derivatives dataset (with its session file)",
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._blocks: list[Block] = []
        self._events: tuple[Event, ...] | None = None  # the rail's, at the writer
        self.target: Target | None = None

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, Block):
            raise PortTypeError(f"{self.manifest.name} accepts Signal blocks only")
        self._blocks.append(block)

    def write_events(self, events: Sequence[Event]) -> None:
        self._events = tuple(events)

    def close(self) -> None:
        if not self._blocks:
            return
        target = self.target = self._target()
        signal_path = target.path("eeg.set")
        if self.keep_existing(signal_path):  # its sidecars and session stay too
            self._blocks.clear()
            return
        target.folder.mkdir(parents=True, exist_ok=True)
        write_dataset_files(target, self._graph)
        pieces = contiguous_runs(self._blocks)  # more than one: a review cut it
        sink = SetSink({"path": str(signal_path), "overwrite": True})
        for piece in pieces:
            sink.write(piece)  # a boundary event at each cut
        if self._events is not None:
            sink.write_events(self._events)
        sink.close()
        first = pieces[0]
        data = np.concatenate([b.data for b in pieces], axis=1)
        whole = Block(data, first.info, first.sample_index, first.timestamp)
        self.written = [signal_path]
        self.written.append(self._sidecar(target, whole))
        self.written.append(self._channels(target, whole.info))
        self.written.append(_write_description(
            target.path("channels.json"),
            "Channels of the processed recording: types, units and status "
            "after the recording's edits"))  # fmt: skip
        if self._events is not None:  # the input's and the strips' (a review's)
            rate = first.info.sfreq
            events = [{"onset": e.sample_index / rate, "duration": e.duration / rate,
                       "label": e.label}
                      for e in self._events if e.label != "boundary"]  # fmt: skip
        else:
            events = describe(self._graph.source.impl)["events"]
        rows = _events_rows(events, pieces)
        if rows:
            path = target.path("events.tsv")
            _write_tsv(path, ["onset", "duration", "trial_type"], rows)
            self.written.append(path)
            note = (
                "Events of the raw recording, after renaming and dropping, "
                "and those the pipeline's strips added (a review's marks)"
            )
            if len(pieces) > 1:
                note += ("; spans a review rejected are cut out: onsets count the "
                         "processed recording's time, events inside a cut are "
                         "left out, and a 'boundary' event marks each cut")  # fmt: skip
            self.written.append(_write_description(target.path("events.json"), note))
        if whole.info.positions and any(p is not None for p in whole.info.positions):
            self.written.extend(self._electrodes(target, whole.info))
        self._blocks.clear()

    def _sidecar(self, target: Target, whole: Block) -> Path:
        sidecar: dict[str, Any] = {}
        if target.raw_file is not None:
            raw_json = target.raw_file.with_suffix(".json")
            if raw_json.is_file():
                sidecar = json.loads(raw_json.read_text(encoding="utf-8"))
        info = whole.info
        types = [
            BIDS_TYPES.get(t, "MISC") for t in (info.ch_types or [""] * info.n_channels)
        ]
        for key in COUNT_KEYS.values():
            sidecar.pop(key, None)
        for bids_type, key in COUNT_KEYS.items():
            sidecar[key] = types.count(bids_type)
        sidecar["SamplingFrequency"] = info.sfreq
        sidecar["RecordingDuration"] = whole.n_samples / info.sfreq
        sidecar.setdefault("EEGReference", "n/a")
        sidecar.setdefault("PowerLineFrequency", "n/a")
        sidecar.setdefault("SoftwareFilters", "n/a")
        if target.raw_file is not None and target.raw_root is not None:
            relative = _relative(target.raw_file, target.raw_root)
            sidecar["Sources"] = [f"bids:raw:{relative}"]
        sidecar["Description"] = _describe_chain(self._graph, self._node_id)
        sidecar["MeegXerSession"] = target.path("meegxer.json").name
        path = target.path("eeg.json")
        path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _channels(target: Target, info: SignalInfo) -> Path:
        types = info.ch_types or ("",) * info.n_channels
        rows = [[label, BIDS_TYPES.get(t, "MISC"), "V",
                 "bad" if label in info.bads else "good"]
                for label, t in zip(info.ch_labels, types, strict=True)]  # fmt: skip
        path = target.path("channels.tsv")
        _write_tsv(path, ["name", "type", "units", "status"], rows)
        return path

    @staticmethod
    def _electrodes(target: Target, info: SignalInfo) -> list[Path]:
        rows = []
        for label, pos in zip(info.ch_labels, info.positions, strict=True):
            xyz = [repr(float(v)) for v in pos] if pos is not None else ["n/a"] * 3
            rows.append([label, *xyz])
        electrodes = target.subject_path("electrodes.tsv")
        _write_tsv(electrodes, ["name", "x", "y", "z"], rows)
        note = "Electrode positions used for processing"
        sidecar = _write_description(target.subject_path("electrodes.json"), note)
        coords = target.subject_path("coordsystem.json")
        coords.write_text(json.dumps({"EEGCoordinateSystem": "CapTrak",
                                      "EEGCoordinateUnits": "m"}, indent=2) + "\n",
                          encoding="utf-8")  # fmt: skip
        return [electrodes, sidecar, coords]

    def finish(self) -> None:
        """The run's session file next to the outputs (models, hashes, versions)."""
        if self.target is None or not self.written:
            return
        from meegxer.core.session import build_session, sink_outputs, write_session

        outputs: list[Path] = []
        for node in self._graph.nodes:
            if isinstance(node.impl, _BidsSink):
                outputs.extend(node.impl.written)
        outputs.extend(sink_outputs(self._graph))
        unique = list(dict.fromkeys(p.resolve() for p in outputs))
        session = build_session(self._graph, name=self.target.stem, outputs=unique)
        path = self.target.path("meegxer.json")
        write_session(path, session)


class BidsFeatures(_BidsSink):
    """A feature table (one row per window) with its sidecar, next to the
    recording's derivative."""

    manifest: ClassVar[Manifest] = _manifest(
        "meegxer.core.bids_features", "features",
        "Feature table into a BIDS derivatives dataset",
    )  # fmt: skip

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        super().__init__(params)
        self._blocks: list[FeatureBlock] = []

    def write(self, block: AnyBlock) -> None:
        if not isinstance(block, FeatureBlock):
            raise PortTypeError(f"{self.manifest.name} accepts Features blocks only")
        self._blocks.append(block)

    def close(self) -> None:
        if not self._blocks:
            return
        target = self._target()
        upstream = str(self._graph[self._node_id].input).replace(":", "-")
        table = target.path(f"{upstream}_features.tsv")
        if self.keep_existing(table):
            self._blocks.clear()
            return
        target.folder.mkdir(parents=True, exist_ok=True)
        write_dataset_files(target, self._graph)
        sfreq = self._graph.source.impl.info.sfreq
        columns = feature_columns(self._blocks[0])
        rows = []
        for block in self._blocks:
            if feature_columns(block) != columns:
                raise ValueError("feature layout changed between windows")
            start, stop = block.window
            rows.append([str(start), str(stop), repr(start / sfreq),
                         *(tsv_value(v) for v in block.data.ravel())])  # fmt: skip
        _write_tsv(table, ["window_start", "window_stop", "onset", *columns], rows)
        node = self._graph[upstream]
        first = self._blocks[0]
        sidecar = {
            "Description": f"{node.manifest.name} {node.manifest.version} "
                           f"features, one row per window",
            "Parameters": dict(node.impl.params),
            "Dimensions": list(first.dims),
            "Coordinates": {k: [str(v) for v in vals]
                            for k, vals in first.coords.items()},
            "window_start": {"Description": "First sample of the window",
                             "Units": "sample"},
            "window_stop": {"Description": "Sample after the window",
                            "Units": "sample"},
            "onset": {"Description": "Window start", "Units": "s"},
        }  # fmt: skip
        for column, unit in zip(columns, feature_units(first), strict=True):
            if unit:  # BIDS: a column's units; "1" (dimensionless) stays "1"
                sidecar[column] = {"Units": unit}
        sidecar_path = target.path(f"{upstream}_features.json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        self.written = [table, sidecar_path]
        self._blocks.clear()


# -- many recordings -----------------------------------------------------------------

RECORDING_EXTENSIONS = (".set", ".edf", ".bdf", ".vhdr", ".fif")


def find_recordings(
    root: Path,
    *,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    tasks: list[str] | None = None,
    runs: list[str] | None = None,
) -> list[dict[str, str]]:
    """The EEG recordings of a raw BIDS dataset that match, as BidsSource
    parameters (one per recording, in a stable order)."""
    from mne_bids import find_matching_paths

    paths = find_matching_paths(
        root, subjects=subjects or None, sessions=sessions or None,
        tasks=tasks or None, runs=runs or None, datatypes="eeg", suffixes="eeg",
        extensions=list(RECORDING_EXTENSIONS),
    )  # fmt: skip
    found: dict[str, dict[str, str]] = {}
    for path in paths:
        if "derivatives" in Path(str(path.fpath)).relative_to(root).parts:
            continue
        params = {"root": str(root), "subject": path.subject or "",
                  "session": path.session or "", "task": path.task or "",
                  "run": str(path.run or ""), "datatype": "eeg"}  # fmt: skip
        found.setdefault(path.basename.rsplit(".", 1)[0], params)
    return [found[key] for key in sorted(found)]


def bids_sinks(graph: Any) -> list[_BidsSink]:
    return [n.impl for n in graph.nodes if isinstance(n.impl, _BidsSink)]


def ensure_writer(graph: Any, root: str | Path | None, overwrite: bool) -> BidsWriter:
    """The graph's BIDS signal writer (one is added if it has none), with
    ``root`` and ``overwrite`` applied to all its BIDS writers."""
    writers = [s for s in bids_sinks(graph) if isinstance(s, BidsWriter)]
    if not writers:
        writer = BidsWriter({})
        graph.add(writer)
        writers = [writer]
    for sink in bids_sinks(graph):
        if root:
            sink.params["root"] = str(root)
        sink.params["overwrite"] = bool(overwrite or sink.params["overwrite"])
    return writers[0]
