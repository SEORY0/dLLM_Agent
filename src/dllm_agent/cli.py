"""Installed command-line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dllm_agent.errors import BoundaryError
from dllm_agent.io import write_model
from dllm_agent.pipeline import expand_plan
from dllm_agent.pipeline import scan as run_scan

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
_ERROR_CONSOLE = Console(stderr=True)


@app.command()
def scan(
    target: Annotated[Path, typer.Argument()],
    fixture: Annotated[Path, typer.Option("--fixture")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Scan a target with a versioned replay fixture."""
    try:
        report = run_scan(target, fixture)
        write_model(output, report)
    except BoundaryError as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error


@app.command()
def plan(
    config: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Validate and expand an experiment configuration."""
    try:
        expanded = expand_plan(config)
        write_model(output, expanded)
    except BoundaryError as error:
        _ERROR_CONSOLE.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error
