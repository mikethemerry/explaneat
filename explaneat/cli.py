"""Main CLI entry point for ExplaNEAT"""
import os
import sys

import click
from .db.cli import database


@click.group()
def cli():
    """ExplaNEAT - Explainable NEAT with database support"""
    pass


# Add database commands
cli.add_command(database, name='db')

# Export packs (explaneat/export/cli.py)
from .export.cli import export as _export  # noqa: E402
cli.add_command(_export, name='export')


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

    from . import settings

    os.environ["EXPLANEAT_MARK_ORPHANS"] = "1" if mark_orphans else "0"
    if host not in ("127.0.0.1", "localhost", "::1"):
        click.echo(f"WARNING: binding to {host} exposes the API (and any data it holds) "
                   "beyond this machine.", err=True)
    url = settings.resolve_database_url()
    click.echo(f"Database: {settings.redact(url)} (from {settings.database_url_source()})")
    click.echo(f"Open http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/")
    uvicorn.run("explaneat.api.app:app", host=host, port=port, reload=reload)


@cli.command()
@click.option("--database-url", default=None,
              help="SQLite URL to create (default: <data dir>/explaneat.db). Postgres URLs are never migrated.")
@click.option("--skip-ui", is_flag=True, help="Do not deploy web/react-explorer/dist.")
@click.option("--skip-db", is_flag=True, help="Do not create the database.")
def setup(database_url, skip_ui, skip_db):
    """Create the local SQLite database and deploy the built UI."""
    from .install import run_setup

    sys.exit(run_setup(database_url=database_url, skip_ui=skip_ui, skip_db=skip_db,
                       log=click.echo))


@cli.command()
def doctor():
    """Check Python, uv, Node, torch, the database and the UI build."""
    from .install import run_doctor

    sys.exit(run_doctor(log=click.echo))


@cli.command()
@click.option("--out", "out_dir", default="demo_output", show_default=True,
              type=click.Path(file_okay=False), help="Directory for report.md / report.html.")
@click.option("--tiny", is_flag=True, help="Seconds-long smoke configuration (CI).")
@click.option("--database-url", default=None,
              help="Database for the runs (default: the SQLite DB from setup, else <out>/demo.db).")
@click.option("--no-db", is_flag=True, help="Train in memory; no database.")
def demo(out_dir, tiny, database_url, no_db):
    """Hello world: synthetic data -> training -> explanation -> evidence -> report."""
    from .demo import run_demo

    result = run_demo(out_dir, tiny=tiny, database_url=database_url, use_db=not no_db,
                      log=click.echo)
    for s in result.stages:
        click.echo(f"  {s.status:4}  {s.title}")
    if result.report_md:
        click.echo(f"Report: {result.report_md}")
    click.echo("Demo PASSED" if result.ok else "Demo FAILED")
    sys.exit(0 if result.ok else 1)


if __name__ == '__main__':
    cli()
