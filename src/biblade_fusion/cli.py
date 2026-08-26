"""BiBladeFusion command-line interface."""

from typing import Annotated

import typer

from biblade_fusion import __version__

app = typer.Typer(
    name="bbf",
    help="BiBladeFusion development and acquisition tools.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """BiBladeFusion command group."""


@app.command()
def version(
    short: Annotated[
        bool,
        typer.Option("--short", help="Print only the semantic version."),
    ] = False,
) -> None:
    """Show the BiBladeFusion version."""

    if short:
        typer.echo(__version__)
        return
    typer.echo(f"BiBladeFusion {__version__}")


if __name__ == "__main__":
    app()
