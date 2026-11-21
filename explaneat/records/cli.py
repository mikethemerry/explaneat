"""``explaneat records``: the explanation records in the database.

    uv run python -m explaneat records import --dry-run          # every study folder
    uv run python -m explaneat records import --from experiments/churn_case/explanations
    uv run python -m explaneat records export                     # DB -> the committed JSON files
    uv run python -m explaneat records list --study cox_case

``DATABASE_URL`` selects the database (as everywhere). Import needs the ``explanation_records``
table (migration ``m6n7o8p9q0r1``: ``explaneat db upgrade``).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click


def _session_or_exit():
    from sqlalchemy import inspect

    from ..db import db
    from ..settings import redact, resolve_database_url
    if db.engine is None:
        db.init_db()
    if not inspect(db.engine).has_table("explanation_records"):
        click.echo(f"{redact(resolve_database_url())} has no explanation_records table: apply the "
                   "migration first (explaneat db upgrade).", err=True)
        sys.exit(2)
    return db


@click.group("records")
def records():
    """Explanation records in the database (import, export, list)."""


@records.command("import")
@click.option("--from", "dirs", multiple=True, type=click.Path(file_okay=False, exists=True),
              help="A records folder (repeatable; earlier folders keep their labels on a clash). "
                   "Default: $EXPLANEAT_RECORDS_DIR, then every experiments/*/explanations* folder.")
@click.option("--dry-run", is_flag=True, help="Show what would change; write nothing.")
@click.option("--force", is_flag=True, help="Take the file's content when it differs from the database's.")
def import_cmd(dirs, dry_run, force):
    """Load the JSON records into the database (idempotent)."""
    from ..analysis.explanation_records import resolve_dir
    from .sync import import_records
    db = _session_or_exit()
    with db.session_scope() as s:
        res = import_records(s, [resolve_dir(d) for d in dirs] if dirs else None, dry_run=dry_run, force=force)
        if dry_run:
            s.rollback()
    for r in res:
        if r["action"] != "unchanged":
            click.echo(f"{r['action']:9} {r['label']:32} {r['study']:18} {r['source']}"
                       + (f"  ({r['detail']})" if r.get("detail") else ""))
    c = Counter(r["action"] for r in res)
    click.echo(("[dry run] " if dry_run else "") + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
               + f" ({len(res)} record files)")
    if c.get("clash") or c.get("differs"):
        sys.exit(1)


@records.command("export")
@click.option("--out", "out_root", default=None, type=click.Path(file_okay=False),
              help="Write under this folder instead of the repository (same relative paths).")
@click.option("--label", "labels", multiple=True, help="Only these labels (repeatable).")
@click.option("--dry-run", is_flag=True)
def export_cmd(out_root, labels, dry_run):
    """Write the database's records (with Mike's current verification) to their JSON files."""
    from .sync import export_records
    db = _session_or_exit()
    with db.session_scope() as s:
        res = export_records(s, Path(out_root) if out_root else None, labels or None, dry_run=dry_run)
    for r in res:
        if r["action"] != "unchanged":
            click.echo(f"{r['action']:9} {r['label']:32} {r['path']}")
    c = Counter(r["action"] for r in res)
    click.echo(("[dry run] " if dry_run else "") + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


@records.command("list")
@click.option("--study", default=None)
def list_cmd(study):
    """Every record the explorer sees (database, then files), with where it is stored."""
    from ..analysis import explanation_records as er
    n = 0
    for e in er.catalog():
        if study and e.study != study:
            continue
        n += 1
        click.echo(f"{e.label:32} {e.study:18} {e.storage:5} {e.genome_id}  {e.source}")
    click.echo(f"{n} records")
