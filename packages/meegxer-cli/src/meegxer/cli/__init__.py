"""Command-line interface: headless runs, batch jobs, Pi deployment."""

from pathlib import Path
from typing import Annotated

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
    overwrite: Annotated[bool, typer.Option(help="Replace existing output.")] = False,
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
) -> None:
    """Run a pipeline or session offline over one recording, or with --bids over
    many (BIDS derivatives, one after another; done ones are skipped)."""
    if bids is not None:
        if input is not None or output is not None or session is not None:
            typer.echo("error: --bids runs many recordings; it writes BIDS "
                       "derivatives (no -i, -o or --session)", err=True)  # fmt: skip
            raise typer.Exit(code=1)
        _run_bids(pipeline, bids, bids_out, overwrite, allow_version_mismatch,
                  block_size, cache_dir, subjects, task, ses, run_label)  # fmt: skip
        return
    from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
    from meegxer.core.cache import DirectoryCache
    from meegxer.core.graph import GraphError
    from meegxer.core.node import Sink
    from meegxer.core.registry import Registry
    from meegxer.core.session import (
        SessionError,
        adopt_models,
        build_session,
        load_session,
        model_status,
        read_session,
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
    if output is not None:
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
) -> None:
    """One pipeline over many recordings of a BIDS dataset, one after another,
    each with a fresh graph (models fitted per recording)."""
    from meegxer.core.batch import BatchScheduler, offline_input_fingerprints
    from meegxer.core.bids_out import ensure_writer, find_recordings, target_for
    from meegxer.core.cache import DirectoryCache
    from meegxer.core.node import Source
    from meegxer.core.recording import (
        RecordingOverrides,
        recording_of,
        with_recording,
    )
    from meegxer.core.registry import Registry
    from meegxer.core.session import adopt_models, load_session, read_session
    from meegxer.core.sources import BidsSource

    try:
        document = read_session(pipeline)  # read once: a bad file stops here
    except (OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    root = root.resolve()
    recordings = find_recordings(
        root, subjects=_split(subjects), tasks=_split(tasks),
        sessions=_split(sessions), runs=_split(runs),
    )  # fmt: skip
    if not recordings:
        typer.echo(f"error: no matching EEG recordings in {root}", err=True)
        raise typer.Exit(code=1)
    registry = Registry.from_entry_points()
    cache = DirectoryCache(cache_dir) if cache_dir else None
    done, skipped, failed = [], [], []
    for i, params in enumerate(recordings, 1):
        short = {"subject": "sub", "session": "ses", "task": "task", "run": "run"}
        label = "_".join(f"{short[k]}-{v}" for k, v in params.items()
                         if v and k in short)  # fmt: skip
        prefix = f"[{i}/{len(recordings)}] {label}"
        try:
            loaded = load_session(
                document,
                registry,
                allow_version_mismatch=allow_version_mismatch,
            )
            graph = loaded.graph
            # the input's Recording settings (renames, types, montage, bad
            # channels) apply to every recording
            first = graph.source.impl
            assert isinstance(first, Source)
            settings = recording_of(first)
            source = (with_recording(BidsSource, params,
                                     RecordingOverrides.from_dict(settings))
                      if settings else BidsSource(params))  # fmt: skip
            graph.replace(graph.source.id, source)
            writer = ensure_writer(graph, bids_out, overwrite)
            target = target_for(graph, writer.params)
            # the session file is written last: without it the recording is
            # unfinished (e.g. interrupted while writing) and is done again
            if target.path("meegxer.json").exists() and not overwrite:
                skipped.append(label)
                typer.echo(f"{prefix}: already done, skipped")
                continue
            if target.path("eeg.set").exists():
                ensure_writer(graph, bids_out, True)  # replace the partial files
            adopt_models(loaded, offline_input_fingerprints(graph))
            result = BatchScheduler(block_size=block_size, cache=cache).run(graph)
        except Exception as exc:  # one bad recording does not stop the others
            failed.append(label)
            typer.echo(f"{prefix}: FAILED: {exc}", err=True)
            continue
        done.append(label)
        fitted = f" (fitted: {', '.join(result.fitted)})" if result.fitted else ""
        typer.echo(f"{prefix}: wrote {target.path('eeg.set')}{fitted}")
    typer.echo(f"done {len(done)}, skipped {len(skipped)}, failed {len(failed)}"
               + (f": {', '.join(failed)}" if failed else ""))  # fmt: skip
    if failed:
        raise typer.Exit(code=1)


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
    )

    try:
        loaded = load_session(
            read_session(file),
            Registry.from_entry_points(),
            allow_version_mismatch=allow_version_mismatch,
        )
        finite = loaded.graph.source.manifest.availability != "online"
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
    control: Annotated[
        str, typer.Option(help="Daemon control endpoint.")
    ] = "tcp://127.0.0.1:5555",
) -> None:
    """Save the daemon's pipeline with its calibrated models as a session."""
    from meegxer.core.session import write_session

    session = _call(control, "get_session", name=name)
    assert isinstance(session, dict)
    write_session(file, session)
    typer.echo(f"wrote {file}")


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
    for node in result.get("nodes", []):
        timing = node.get("timing") or {}
        mean = f"{timing['mean_ms']:.3f} ms" if timing else "-"
        typer.echo(f"  {node['id']:16} {node['status']:13} {mean}")
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
def shutdown(control: ControlOpt = _DEFAULT_CONTROL) -> None:
    """Stop the daemon process."""
    _call(control, "shutdown")
    typer.echo("daemon shutting down")


@app.callback()
def main() -> None:
    """MeegXer command-line interface."""
