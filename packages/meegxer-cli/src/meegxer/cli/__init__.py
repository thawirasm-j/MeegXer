"""Command-line interface: headless runs, batch jobs, Pi deployment."""

from pathlib import Path
from typing import Annotated, Any

import typer

__version__ = "0.0.0"

app = typer.Typer(no_args_is_help=True, help="MeegXer command-line interface.")

_SINKS = {".set": "meegxer.core.set_writer", ".edf": "meegxer.core.edf_writer"}


@app.command()
def version() -> None:
    """Print the installed MeegXer CLI version."""
    typer.echo(__version__)


@app.command()
def nodes() -> None:
    """List the installed processing nodes."""
    from meegxer.core.registry import Registry

    registry = Registry.from_entry_points()
    for name in registry.names():
        manifest = registry.get(name).manifest
        typer.echo(f"{name:36} {manifest.role:9} {manifest.version}")


@app.command()
def templates() -> None:
    """List the pipeline templates and the sample datasets."""
    from meegxer.core.samples import SAMPLES, fetched, sample_dir
    from meegxer.core.templates import TEMPLATES

    for t in TEMPLATES.values():
        typer.echo(f"{t.name:14} {t.title} (sample: {t.sample})\n{'':14} "
                   f"{t.description}")  # fmt: skip
    typer.echo("")
    for s in SAMPLES.values():
        root = sample_dir() / s.name
        have = sum(fetched(root, subject) for subject in s.subjects)
        size = f", {s.size} each" if s.size else ""
        typer.echo(f"{s.name:14} {s.title}: {have} of {len(s.subjects)} subjects "
                   f"fetched{size}, in {root}")  # fmt: skip


@app.command()
def new(
    template: Annotated[str, typer.Argument(help="Template name (meegxer templates).")],
    output: Annotated[Path, typer.Argument(help="Pipeline file to write.")],
    sample: Annotated[
        bool,
        typer.Option(
            help="Read the template's sample dataset (each subject is "
            "downloaded when a run first reads it)."
        ),
    ] = False,
    force: Annotated[bool, typer.Option(help="Replace the file if it exists.")] = False,
) -> None:
    """Write a new pipeline from a template."""
    import json

    from meegxer.core.templates import TEMPLATES, template_pipeline

    if template not in TEMPLATES:
        typer.echo(f"error: no template {template!r} (one of: "
                   f"{', '.join(TEMPLATES)})", err=True)  # fmt: skip
        raise typer.Exit(code=1)
    if output.exists() and not force:
        typer.echo(f"error: {output} exists (--force replaces it)", err=True)
        raise typer.Exit(code=1)
    pipeline = template_pipeline(template, sample=sample)
    output.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {output}")
    root = next(n for n in pipeline["nodes"] if n["id"] == "input").get("params", {})
    dataset = root.get("root") or "DATASET"
    typer.echo(f"run it: meegxer run {output} --bids {dataset}"
               + (" --subjects 001" if sample else ""))  # fmt: skip


@app.command()
def run(
    pipeline: Annotated[
        Path, typer.Argument(help="Pipeline or session (.meegxer.json) file.")
    ],
    input: Annotated[
        Path | None, typer.Option("--input", "-i", help="Recording to process.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output .set or .edf.")
    ] = None,
    session: Annotated[
        Path | None,
        typer.Option(help="Write a session file of this run (e.g. out.meegxer.json)."),
    ] = None,
    block_size: Annotated[
        int | None, typer.Option(help="Feed nodes in blocks of this many samples.")
    ] = None,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Reuse unchanged node outputs from here.")
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Replace existing output (else an existing file is kept, "
            "its save skipped)."
        ),
    ] = False,
    allow_version_mismatch: Annotated[
        bool,
        typer.Option(
            help="Run a session whose node versions or plugin images "
            "differ (their models are refitted)."
        ),
    ] = False,
    strict: Annotated[
        bool, typer.Option(help="Refuse a session whose input files changed.")
    ] = False,
    bids_out: Annotated[
        Path | None,
        typer.Option(help="Write BIDS derivatives here (adds a BIDS writer)."),
    ] = None,
    bids: Annotated[
        Path | None,
        typer.Option(help="Run every matching recording of this raw BIDS dataset."),
    ] = None,
    subjects: Annotated[
        list[str] | None,
        typer.Option("--subjects", help="With --bids: subjects (repeat or 01,02)."),
    ] = None,
    task: Annotated[list[str] | None, typer.Option(help="With --bids: tasks.")] = None,
    ses: Annotated[
        list[str] | None, typer.Option(help="With --bids: sessions.")
    ] = None,
    run_label: Annotated[
        list[str] | None, typer.Option("--run-label", help="With --bids: runs.")
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            help="With --bids: recordings run at once, each in its own "
            "process (0: one per CPU)."
        ),
    ] = 1,
    report: Annotated[
        bool,
        typer.Option(
            help="With --bids: a QC report per recording and the "
            "dataset's qc.tsv and qc.html."
        ),
    ] = True,
) -> None:
    """Run a pipeline or session offline over one recording, or with --bids over
    many (BIDS derivatives, one after another; done ones are skipped)."""
    if bids is not None:
        if input is not None or output is not None or session is not None:
            typer.echo("error: --bids runs many recordings; it writes BIDS "
                       "derivatives (no -i, -o or --session)", err=True)  # fmt: skip
            raise typer.Exit(code=1)
        _run_bids(pipeline, bids, bids_out, overwrite, allow_version_mismatch,
                  block_size, cache_dir, subjects, task, ses, run_label,
                  jobs, report)  # fmt: skip
        return
    from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
    from meegxer.core.cache import DirectoryCache
    from meegxer.core.graph import GraphError
    from meegxer.core.node import Sink
    from meegxer.core.registry import Registry
    from meegxer.core.review import ReviewNeeded
    from meegxer.core.session import (
        SessionError,
        adopt_models,
        build_session,
        drop_shared_sinks,
        load_session,
        model_status,
        read_session,
        shared_input_hint,
        sink_outputs,
        write_session,
    )
    from meegxer.core.sources import FileSource

    registry = Registry.from_entry_points()
    try:
        document = read_session(pipeline)
        loaded = load_session(
            document,
            registry,
            allow_version_mismatch=allow_version_mismatch,
            strict=strict,
        )
        graph = loaded.graph
        if input is not None:
            source = FileSource({"path": str(input)})
            graph.replace(graph.source.id, source)
        elif hint := shared_input_hint(graph):
            raise SessionError(hint)
        for node_id in drop_shared_sinks(graph):
            typer.echo(f"note: {node_id} writes nowhere (shared without its file "
                       "name); -o gives the output")  # fmt: skip
        if output is not None:
            sink_name = _SINKS.get(output.suffix.lower())
            if sink_name is None:
                raise GraphError(f"unsupported output type {output.suffix!r}")
            sink_cls = registry.get(sink_name)
            assert issubclass(sink_cls, Sink)
            graph.add(sink_cls({"path": str(output), "overwrite": overwrite}))
        if bids_out is not None:
            from meegxer.core.bids_out import ensure_writer

            ensure_writer(graph, bids_out, overwrite)
        if (
            output is None
            and bids_out is None
            and input is None
            and not any(n.role.value == "sink" for n in graph.nodes)
        ):
            typer.echo("warning: pipeline has no sink; nothing will be written")
        fps = offline_input_fingerprints(graph)
        adopt_models(loaded, fps)
        for warning in loaded.warnings:
            typer.echo(f"warning: {warning}")
        for node_id, state in model_status(graph, fps).items():
            typer.echo(f"model {node_id}: {state}")
        scheduler = BatchScheduler(
            block_size=block_size,
            cache=DirectoryCache(cache_dir) if cache_dir else None,
        )
        result = scheduler.run(graph)
        if session is not None:
            write_session(
                session,
                build_session(
                    graph,
                    name=document.get("name"),
                    description=document.get("description"),
                    outputs=sink_outputs(graph),
                ),
            )
    except ReviewNeeded as exc:  # nothing past the strip runs unreviewed
        typer.echo(f"warning: not reviewed, stopped: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (
        GraphError,
        SessionError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"computed: {', '.join(result.computed) or '-'}")
    if result.cached:
        typer.echo(f"from cache: {', '.join(result.cached)}")
    if result.fitted:
        typer.echo(f"fitted: {', '.join(result.fitted)}")
    for note in result.skipped:
        typer.echo(f"skipped: {note}")
    for note in result.notes:
        typer.echo(f"note: {note}")
    if output is not None and not any(str(output) in n for n in result.skipped):
        typer.echo(f"wrote {output}")
    if session is not None:
        typer.echo(f"wrote {session}")


def _split(values: list[str] | None) -> list[str]:
    return [v for item in values or [] for v in item.split(",") if v]


def _run_bids(
    pipeline: Path,
    root: Path,
    bids_out: Path | None,
    overwrite: bool,
    allow_version_mismatch: bool,
    block_size: int | None,
    cache_dir: Path | None,
    subjects: list[str] | None,
    tasks: list[str] | None,
    sessions: list[str] | None,
    runs: list[str] | None,
    jobs: int = 1,
    report: bool = True,
) -> None:
    """One pipeline over many recordings of a BIDS dataset, each with a fresh
    graph (models fitted per recording), ``jobs`` at a time."""
    from meegxer.core.bids_batch import (
        Outcome,
        RunOptions,
        recording_label,
        run_dataset,
    )
    from meegxer.core.bids_out import find_recordings
    from meegxer.core.qc import SUMMARY_HTML
    from meegxer.core.session import read_session

    try:
        document = read_session(pipeline)  # read once: a bad file stops here
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if jobs < 0:
        typer.echo("error: --jobs must be 0 (one per CPU) or more", err=True)
        raise typer.Exit(code=1)
    root = root.resolve()
    recordings = find_recordings(
        root, subjects=_split(subjects), tasks=_split(tasks),
        sessions=_split(sessions), runs=_split(runs),
    )  # fmt: skip
    if not recordings:
        typer.echo(f"error: no matching EEG recordings in {root}", err=True)
        raise typer.Exit(code=1)
    n = len(recordings)
    labels = [recording_label(p) for p in recordings]

    def note(i: int, text: str) -> None:
        typer.echo(f"[{i}/{n}] {labels[i - 1]}: {text}")

    def outcome(i: int, o: Outcome) -> None:
        prefix = f"[{i}/{n}] {o.label}"
        for text in o.notes:
            typer.echo(f"{prefix}: note: {text}")
        if o.status == "skipped":
            typer.echo(f"{prefix}: already done, skipped")
        elif o.status == "unreviewed":
            typer.echo(f"{prefix}: NOT REVIEWED, stopped: {o.message}", err=True)
        elif o.status == "failed":
            typer.echo(f"{prefix}: FAILED: {o.message}", err=True)
        else:
            fitted = f" (fitted: {', '.join(o.fitted)})" if o.fitted else ""
            typer.echo(f"{prefix}: wrote {o.output}{fitted}")

    options = RunOptions(
        bids_out=str(bids_out) if bids_out else None, overwrite=overwrite,
        allow_version_mismatch=allow_version_mismatch, block_size=block_size,
        cache_dir=str(cache_dir) if cache_dir else None, report=report,
    )  # fmt: skip
    outcomes = run_dataset(document, recordings, options, jobs=jobs,
                           on_note=note, on_outcome=outcome)  # fmt: skip
    by = {s: [o.label for o in outcomes if o.status == s]
          for s in ("done", "skipped", "failed", "unreviewed")}  # fmt: skip
    failed, unreviewed = by["failed"], by["unreviewed"]
    typer.echo(f"done {len(by['done'])}, skipped {len(by['skipped'])}, "
               f"failed {len(failed)}"
               + (f": {', '.join(failed)}" if failed else ""))  # fmt: skip
    if report:
        for summary in sorted({o.root for o in outcomes if o.root}):
            typer.echo(f"QC summary: {Path(summary) / SUMMARY_HTML}")
    if unreviewed:  # said last, so it is not missed
        typer.echo(f"warning: {len(unreviewed)} not reviewed, so not processed "
                   f"past the review strip: {', '.join(unreviewed)}; review them "
                   "in the GUI (an offline run pauses for each)", err=True)  # fmt: skip
    if failed:
        raise typer.Exit(code=1)
    if unreviewed:
        raise typer.Exit(code=2)


session_app = typer.Typer(no_args_is_help=True, help="Session files (reproducibility).")
app.add_typer(session_app, name="session")


@session_app.command("check")
def session_check(
    file: Annotated[Path, typer.Argument(help="Session file to check.")],
    allow_version_mismatch: Annotated[bool, typer.Option()] = False,
) -> None:
    """Report what rerunning a session would reuse, refit or warn about."""
    from meegxer.core.batch import offline_input_fingerprints
    from meegxer.core.registry import Registry
    from meegxer.core.session import (
        SessionError,
        adopt_models,
        load_session,
        model_status,
        read_session,
        shared_input_hint,
    )

    try:
        loaded = load_session(
            read_session(file),
            Registry.from_entry_points(),
            allow_version_mismatch=allow_version_mismatch,
        )
        shared = shared_input_hint(loaded.graph)
        if shared:  # no recording to fingerprint until one is given
            loaded.warnings.append(shared)
        finite = not shared and loaded.graph.source.manifest.availability != "online"
        fps = offline_input_fingerprints(loaded.graph) if finite else {}
        if finite:
            adopt_models(loaded, fps)
    except (SessionError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for warning in loaded.warnings:
        typer.echo(f"warning: {warning}")
    for node_id in loaded.dropped_models:
        typer.echo(f"model {node_id}: dropped (refitted)")
    if finite:
        for node_id, state in model_status(loaded.graph, fps).items():
            typer.echo(f"model {node_id}: {state}")
    typer.echo("ok" if not loaded.warnings else "ok, with warnings")


@session_app.command("save")
def session_save(
    file: Annotated[Path, typer.Argument(help="Where to write (NAME.meegxer.json).")],
    name: Annotated[str | None, typer.Option(help="Session name.")] = None,
    share_safe: Annotated[
        bool, typer.Option(help="Name no files (see `meegxer session share`).")
    ] = False,
    drop_text: Annotated[
        bool, typer.Option(help="With --share-safe: remove the name and description.")
    ] = False,
    control: Annotated[
        str, typer.Option(help="Daemon control endpoint.")
    ] = "tcp://127.0.0.1:5555",
) -> None:
    """Save the daemon's pipeline with its calibrated models as a session."""
    from meegxer.core.session import write_session

    session = _call(control, "get_session", name=name)
    assert isinstance(session, dict)
    if share_safe:
        session = _share(session, drop_text)
    write_session(file, session)
    typer.echo(f"wrote {file}")


@app.command()
def recover(
    file: Annotated[
        Path, typer.Argument(help="The file a crashed run was writing, or its .part.")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead.")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing file.")] = False,
) -> None:
    """Write the recording a crashed run left in its spool (FILE.part, next to
    the .set or .edf it was writing), then remove the spool."""
    from meegxer.core.sinks import recover as recover_spool

    try:
        written = recover_spool(file, output, overwrite)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {written}")


@session_app.command("share")
def session_share(
    file: Annotated[Path, typer.Argument(help="Session to share.")],
    out: Annotated[Path, typer.Argument(help="Where to write the shareable copy.")],
    drop_text: Annotated[
        bool, typer.Option(help="Remove the name and description without asking.")
    ] = False,
) -> None:
    """Write a copy of a session that names no files or folders (paths become
    placeholders such as <input>.set; hashes and models are kept). Rerun it
    with `meegxer run OUT -i RECORDING -o RESULT`."""
    from meegxer.core.session import read_session, write_session

    try:
        session = read_session(file)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    write_session(out, _share(session, drop_text))
    typer.echo(f"wrote {out} (no file or folder names)")


def _share(session: dict[str, object], drop_text: bool) -> dict[str, object]:
    """The share-safe copy, after the reminder about the free text it keeps."""
    import sys

    from meegxer.core.session import free_text, share_safe

    shared = share_safe(session)
    text = free_text(shared)
    if text and not drop_text:
        typer.echo("The session keeps the text you wrote; check that it names no "
                   "one:")  # fmt: skip
        for key, value in text.items():
            typer.echo(f"  {key}: {value}")
        if sys.stdin.isatty():
            drop_text = typer.confirm("Remove it?", default=False)
        else:
            typer.echo("(--drop-text removes it)")
    if drop_text:
        for key in text:
            shared.pop(key, None)
    return shared


ControlOpt = Annotated[str, typer.Option(help="Daemon control endpoint.")]
_DEFAULT_CONTROL = "tcp://127.0.0.1:5555"


def _client(control: str):  # type: ignore[no-untyped-def]
    from meegxer.core.control import ControlClient

    return ControlClient(control)


def _call(control: str, method: str, **params: object) -> object:
    from meegxer.core.control import RpcError

    client = _client(control)
    try:
        return client.call(method, **params)
    except (RpcError, TimeoutError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        client.close()


def _parse_value(text: str) -> object:
    import json

    try:
        return json.loads(text)
    except ValueError:
        return text


@app.command()
def daemon(
    pipeline: Annotated[
        Path | None, typer.Option(help="Pipeline JSON to load at startup.")
    ] = None,
    start: Annotated[
        bool, typer.Option(help="Start the pipeline immediately.")
    ] = False,
    control: ControlOpt = _DEFAULT_CONTROL,
    display: Annotated[
        str, typer.Option(help="Display endpoint.")
    ] = "tcp://127.0.0.1:5556",
) -> None:
    """Run the headless core daemon in the foreground."""
    from meegxer.core.daemon import main as daemon_main

    args = ["--control", control, "--display", display]
    if pipeline is not None:
        args += ["--pipeline", str(pipeline)]
    if start:
        args.append("--start")
    raise typer.Exit(code=daemon_main(args))


@app.command()
def load(pipeline: Path, control: ControlOpt = _DEFAULT_CONTROL) -> None:
    """Load a pipeline into the running daemon."""
    import json

    _call(control, "load_pipeline", pipeline=json.loads(pipeline.read_text()))
    typer.echo(f"loaded {pipeline}")


@app.command()
def start(control: ControlOpt = _DEFAULT_CONTROL) -> None:
    """Start the loaded pipeline."""
    _call(control, "start")
    typer.echo("started")


@app.command()
def stop(control: ControlOpt = _DEFAULT_CONTROL) -> None:
    """Stop the running pipeline."""
    _call(control, "stop")
    typer.echo("stopped")


@app.command()
def status(
    control: ControlOpt = _DEFAULT_CONTROL,
    as_json: Annotated[bool, typer.Option("--json", help="Raw JSON output.")] = False,
) -> None:
    """Show pipeline status, timing and latency."""
    import json

    result = _call(control, "status")
    assert isinstance(result, dict)
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return
    typer.echo(f"running: {result.get('running')}  samples: {result.get('samples', 0)}"
               f"  overruns: {result.get('overruns', 0)}")  # fmt: skip
    if result.get("latency"):
        lat = result["latency"]
        typer.echo(f"latency ms: mean {lat['mean_ms']:.1f}  p95 {lat['p95_ms']:.1f}"
                   f"  max {lat['max_ms']:.1f}")  # fmt: skip
    if result.get("input_events") or "source_markers" in result:
        typer.echo(f"events: {result.get('input_events', 0)} from the input"
                   f"  late: {result.get('events_late', 0)}"
                   + (f"  markers dropped: {result['source_markers_dropped']}"
                      if result.get("source_markers_dropped") else ""))  # fmt: skip
    for node in result.get("nodes", []):
        timing = node.get("timing") or {}
        mean = f"{timing['mean_ms']:.3f} ms" if timing else "-"
        action = ""
        if "armed" in node:  # an action sink (the keyboard)
            action = (f"  {'ARMED' if node['armed'] else 'disarmed'}"
                      f"  pressed {node.get('pressed', 0)}"
                      f"  dropped {node.get('dropped', 0)}")  # fmt: skip
        typer.echo(f"  {node['id']:16} {node['status']:13} {mean}{action}")
    for err in result.get("errors", []):
        typer.echo(f"error: {err}", err=True)


@app.command("set")
def set_param(
    node: str,
    assignments: Annotated[list[str], typer.Argument(help="name=value pairs")],
    control: ControlOpt = _DEFAULT_CONTROL,
) -> None:
    """Change processor parameters while running, e.g. `meegxer set hp low=0.5`."""
    params = {}
    for item in assignments:
        name, sep, value = item.partition("=")
        if not sep:
            typer.echo(f"error: expected name=value, got {item!r}", err=True)
            raise typer.Exit(code=1)
        params[name] = _parse_value(value)
    result = _call(control, "set_params", node=node, params=params)
    typer.echo(f"{node}: {result}")


@app.command()
def calibrate(
    node: str,
    seconds: Annotated[float, typer.Option(help="Calibration length.")] = 60.0,
    control: ControlOpt = _DEFAULT_CONTROL,
) -> None:
    """Record calibration data for a node, then fit it."""
    _call(control, "calibrate", node=node, seconds=seconds)
    typer.echo(f"calibrating {node} for {seconds} s")


@app.command("clear-calibration")
def clear_calibration(
    node: Annotated[
        str | None, typer.Argument(help="Node to clear; all nodes if omitted.")
    ] = None,
    control: ControlOpt = _DEFAULT_CONTROL,
) -> None:
    """Drop fitted models (they are kept across restarts otherwise)."""
    params = {"node": node} if node else {}
    result = _call(control, "clear_calibration", **params)
    assert isinstance(result, dict)
    typer.echo(f"cleared: {', '.join(result['cleared']) or '-'}")


@app.command()
def passthrough(
    node: str,
    enabled: Annotated[
        bool, typer.Option("--on/--off", help="Pass data through while uncalibrated.")
    ],
    control: ControlOpt = _DEFAULT_CONTROL,
) -> None:
    """Per node: pass data through unprocessed while it has no valid model."""
    _call(control, "set_pass_uncalibrated", node=node, enabled=enabled)
    typer.echo(f"{node}: pass through while uncalibrated = {enabled}")


@app.command()
def arm(
    node: str,
    on: Annotated[
        bool, typer.Option("--on/--off", help="Arm (default) or disarm it.")
    ] = True,
    control: ControlOpt = _DEFAULT_CONTROL,
) -> None:
    """Arm an action sink (e.g. the keyboard) for the running live run: it acts
    on its events until --off, Esc, or the run's end. MeegXer is not a medical
    device."""
    result = _call(control, "arm", node=node, on=on)
    assert isinstance(result, dict)
    nodes: list[dict[str, Any]] = result.get("nodes", [])
    state = next((n for n in nodes if n.get("id") == node), {})
    if state.get("armed"):
        typer.echo(f"{node}: ARMED (Esc, `meegxer arm {node} --off` or stop disarms)")
    else:
        typer.echo(f"{node}: disarmed")


@app.command()
def shutdown(control: ControlOpt = _DEFAULT_CONTROL) -> None:
    """Stop the daemon process."""
    _call(control, "shutdown")
    typer.echo("daemon shutting down")


@app.callback()
def main() -> None:
    """MeegXer command-line interface."""
