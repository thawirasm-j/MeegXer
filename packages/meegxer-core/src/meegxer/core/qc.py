"""Quality-control reports (M10.8) for the recordings of a BIDS run.

Each recording gets an HTML report (``mne.Report``) next to its derivatives,
``…_desc-<label>_report.html``: a MeegXer table (the chain, how much of the
recording was kept, bad channels, what each fitted model does, epochs kept per
label), the power spectra of the input and of the processed recording, and the
epochs and ERPs when the run wrote them. The same table goes to ``…_qc.json``;
the dataset's ``qc.tsv`` and ``qc.html`` collect them, one row per recording.
"""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SUMMARY_TSV = "qc.tsv"
SUMMARY_HTML = "qc.html"
FIRST_COLUMNS = ("recording", "status", "message", "duration_s", "kept_percent",
                 "bad_channels", "epochs", "report")  # fmt: skip


def _bad_channels(channels_tsv: Path) -> list[str]:
    if not channels_tsv.is_file():
        return []
    with channels_tsv.open(newline="", encoding="utf-8") as fh:
        return [r["name"] for r in csv.DictReader(fh, delimiter="\t")
                if r.get("status") == "bad"]  # fmt: skip


def _table(rows: Mapping[str, Any]) -> str:
    cells = "".join(f"<tr><th>{html.escape(str(k))}</th>"
                    f"<td>{html.escape(str(v))}</td></tr>"
                    for k, v in rows.items())  # fmt: skip
    return f"<table class='table table-sm'>{cells}</table>"


def recording_qc(graph: Any, target: Any, label: str) -> dict[str, Any]:
    """The QC row of one finished recording (``graph`` after its run, ``target``
    its derivatives), with its report and ``_qc.json`` written."""
    import mne

    with mne.utils.use_log_level("error"):  # MNE's progress lines stay out
        return _recording_qc(graph, target, label)


def _recording_qc(graph: Any, target: Any, label: str) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")  # no screen: the daemon, a worker process or the CLI
    import mne

    from meegxer.core.bids_out import BidsWriter, _describe_chain
    from meegxer.core.stage import Stage

    source = graph.source.impl
    sfreq = float(source.info.sfreq)
    duration = float(source.n_samples or 0) / sfreq
    row: dict[str, Any] = {"recording": label, "status": "done",
                           "duration_s": round(duration, 3)}  # fmt: skip
    cleaned = None
    signal = target.path("eeg.set")
    if signal.is_file():
        cleaned = mne.io.read_raw_eeglab(signal, preload=True, verbose="error")
        kept = cleaned.n_times / cleaned.info["sfreq"]
        row["kept_percent"] = round(100 * kept / duration, 1) if duration else ""
    row["bad_channels"] = ",".join(_bad_channels(target.path("channels.tsv")))
    for node in graph.nodes:
        stage = node.impl
        if isinstance(stage, Stage) and stage.model is not None:
            described = stage.processor.describe_model(stage.model.data, stage.params)
            for key, value in described.items():
                row[f"{node.id}_{key}"] = value
    epochs = evokeds = None
    if target.path("epo.fif").is_file():
        epochs = mne.read_epochs(target.path("epo.fif"), verbose="error")
        row["epochs"] = "; ".join(f"{name}: {len(epochs[name])}"
                                  for name in epochs.event_id)  # fmt: skip
        if target.path("ave.fif").is_file():
            evokeds = mne.read_evokeds(target.path("ave.fif"), verbose="error")
    report_path = target.path("report.html")
    row["report"] = report_path.relative_to(target.root).as_posix()

    report = mne.Report(title=label, verbose="error")
    writer = next(n.id for n in graph.nodes if isinstance(n.impl, BidsWriter))
    chain = _describe_chain(graph, writer)
    report.add_html(_table({**row, "chain": chain}), title="MeegXer")
    raw = getattr(source, "_load", None)
    if raw is not None:
        report.add_raw(raw(), title="Input", psd=True, butterfly=False)
    if cleaned is not None:
        report.add_raw(cleaned, title="Processed", psd=True, butterfly=False)
    if epochs is not None:
        report.add_epochs(epochs, title="Epochs kept", psd=False)
    if evokeds:
        report.add_evokeds(evokeds)
    report.save(report_path, open_browser=False, overwrite=True, verbose="error")
    target.path("qc.json").write_text(json.dumps(row, indent=2) + "\n",
                                      encoding="utf-8")  # fmt: skip
    return row


def write_summary(root: Path, this_run: Sequence[Mapping[str, Any]]) -> list[Path]:
    """``qc.tsv`` and ``qc.html`` in the derivatives ``root``: the rows it had,
    replaced by every recording's ``_qc.json`` there and then by this run's
    rows (a failure or a recording not reviewed)."""
    import mne

    rows: dict[str, dict[str, Any]] = {}
    tsv = root / SUMMARY_TSV
    if tsv.is_file():  # recordings this run did not touch keep their rows
        with tsv.open(newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh, delimiter="\t"):
                rows[old["recording"]] = {k: v for k, v in old.items() if v != "n/a"}
    for path in sorted(root.glob("sub-*/**/*_qc.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(row["recording"])] = row
    for row in this_run:
        rows[str(row["recording"])] = dict(row)
    ordered = [rows[k] for k in sorted(rows)]
    columns = [c for c in FIRST_COLUMNS if any(c in r for r in ordered)]
    columns += sorted({k for r in ordered for k in r} - set(columns))
    with tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for r in ordered:
            writer.writerow([r.get(c, "") if r.get(c, "") != "" else "n/a"
                             for c in columns])  # fmt: skip

    def cell(r: Mapping[str, Any], c: str) -> str:
        value = html.escape(str(r.get(c, "")))
        if c == "report" and value:
            return f"<a href='{value}'>{value.rsplit('/', 1)[-1]}</a>"
        return value

    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(r, c)}</td>" for c in columns)
                   + "</tr>" for r in ordered)  # fmt: skip
    report = mne.Report(title="MeegXer QC summary", verbose="error")
    report.add_html(f"<table class='table table-sm'><tr>{head}</tr>{body}</table>",
                    title=f"{len(ordered)} recordings")  # fmt: skip
    page = root / SUMMARY_HTML
    report.save(page, open_browser=False, overwrite=True, verbose="error")
    return [tsv, page]
