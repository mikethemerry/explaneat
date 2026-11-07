"""Main CLI entry point for ExplaNEAT"""
import os

import click
from .db.cli import database


@click.group()
def cli():
    """ExplaNEAT - Explainable NEAT with database support"""
    pass


# Add database commands
cli.add_command(database, name='db')


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (development).")
@click.option(
    "--mark-orphans/--no-mark-orphans",
    default=False,
    show_default=True,
    help="On startup, mark every experiment with status 'running' as 'interrupted'. "
    "Off by default: experiments run by other processes (scripts) are also 'running'.",
)
def api(host, port, reload, mark_orphans):
    """Start the REST API and the React Explorer (http://HOST:PORT/)."""
    import uvicorn

    os.environ["EXPLANEAT_MARK_ORPHANS"] = "1" if mark_orphans else "0"
    uvicorn.run("explaneat.api.app:app", host=host, port=port, reload=reload)


if __name__ == '__main__':
    cli()
