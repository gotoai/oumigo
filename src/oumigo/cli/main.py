"""Console entrypoint: ``oumigo <subcommand>``.

One CLI with subcommands grouped by layer:
  * ``oumigo agent ...``   — worker-node coordinator (L1)
  * ``oumigo manager ...`` — manager: control plane + router (L3)
  * ``oumigo cluster ...`` — cluster-level operations
"""

from __future__ import annotations

import typer

from oumigo import __version__

app = typer.Typer(help="oumigo — run and manage vLLM replica fleets.", no_args_is_help=True)

agent_app = typer.Typer(help="Worker-node coordinator (L1).", no_args_is_help=True)
manager_app = typer.Typer(help="Manager node: control plane + router (L3).", no_args_is_help=True)
cluster_app = typer.Typer(help="Cluster-level operations.", no_args_is_help=True)

app.add_typer(agent_app, name="agent")
app.add_typer(manager_app, name="manager")
app.add_typer(cluster_app, name="cluster")


@app.command("version")
def version() -> None:
    """Print the oumigo version."""
    typer.echo(__version__)


@agent_app.command("run")
def agent_run() -> None:
    """Start the worker coordinator (L1). [not implemented]"""
    typer.echo("oumigo agent run: not implemented yet")


@manager_app.command("run")
def manager_run() -> None:
    """Start the manager: control plane + router (L3). [not implemented]"""
    typer.echo("oumigo manager run: not implemented yet")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
