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
    pipeline: Annotated[Path, typer.Argument(help="Pipeline JSON file.")],
    input: Annotated[
        Path | None, typer.Option("--input", "-i", help="Recording to process.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output .set or .edf.")
    ] = None,
    block_size: Annotated[
        int | None, typer.Option(help="Feed nodes in blocks of this many samples.")
    ] = None,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Reuse unchanged node outputs from here.")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace existing output.")] = False,
) -> None:
    """Run a pipeline offline over one recording."""
    from meegxer.core.batch import BatchScheduler
    from meegxer.core.cache import DirectoryCache
    from meegxer.core.graph import Graph, GraphError
    from meegxer.core.node import Sink
    from meegxer.core.registry import Registry
    from meegxer.core.sources import FileSource

    registry = Registry.from_entry_points()
    try:
        graph = Graph.load(pipeline, registry)
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
        if (
            output is None
            and input is None
            and not any(n.role.value == "sink" for n in graph.nodes)
        ):
            typer.echo("warning: pipeline has no sink; nothing will be written")
        scheduler = BatchScheduler(
            block_size=block_size,
            cache=DirectoryCache(cache_dir) if cache_dir else None,
        )
        result = scheduler.run(graph)
    except (GraphError, FileNotFoundError, FileExistsError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"computed: {', '.join(result.computed) or '-'}")
    if result.cached:
        typer.echo(f"from cache: {', '.join(result.cached)}")
    if output is not None:
        typer.echo(f"wrote {output}")


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
