"""Command-line interface: headless runs, batch jobs, Pi deployment."""

import typer

__version__ = "0.0.0"

app = typer.Typer(no_args_is_help=True, help="MeegXer command-line interface.")


@app.command()
def version() -> None:
    """Print the installed MeegXer CLI version."""
    typer.echo(__version__)


@app.callback()
def main() -> None:
    """MeegXer command-line interface."""
