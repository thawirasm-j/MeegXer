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


@app.callback()
def main() -> None:
    """MeegXer command-line interface."""
