"""Database CLI commands for ExplaNEAT"""
import click
import os
import subprocess
import sys
from .base import db


@click.group()
def database():
    """Database management commands"""
    pass


@database.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
def init(url):
    """Initialize the database (create all tables)"""
    if url:
        db.init_db(url)
    else:
        db.init_db()
    
    click.echo("Creating database tables...")
    db.create_all()
    click.echo("Database initialized successfully!")


@database.command()
@click.option('--url', envvar='DATABASE_URL', help='Database URL')
@click.confirmation_option(prompt='Are you sure you want to drop all tables?')
def drop(url):
    """Drop all database tables"""
    if url:
        db.init_db(url)
    else:
        db.init_db()
    
    click.echo("Dropping all database tables...")
    db.drop_all()
    click.echo("All tables dropped!")


def _alembic(args, *, writes: bool):
    """Run Alembic for the configured database (``explaneat.settings``).

    ``sys.executable -m alembic`` works on every platform without relying on
    PATH. The resolved URL is passed as DATABASE_URL, which alembic/env.py
    reads first. Migrations are PostgreSQL-only: a SQLite file is created
    with ``explaneat setup`` / ``db init`` (create_all), so commands that
    would change a SQLite schema are refused.
    """
    from ..settings import is_sqlite, redact, resolve_database_url

    url = resolve_database_url()
    if writes and is_sqlite(url):
        click.echo(f"{redact(url)} is SQLite: migrations are PostgreSQL-only. "
                   "Use 'explaneat setup' or 'explaneat db init' for SQLite.", err=True)
        sys.exit(2)
    cmd = [sys.executable, '-m', 'alembic', *args]
    click.echo(f"Running: alembic {' '.join(args)}  (database: {redact(url)})")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, 'DATABASE_URL': url})
    if result.returncode:
        sys.exit(result.returncode)


@database.command()
@click.argument('message')
def revision(message):
    """Create a new database migration"""
    _alembic(['revision', '--autogenerate', '-m', message], writes=True)


@database.command()
@click.option('--revision', default='head', help='Revision to upgrade to')
def upgrade(revision):
    """Apply database migrations"""
    _alembic(['upgrade', revision], writes=True)


@database.command()
@click.option('--revision', default='-1', help='Revision to downgrade to')
def downgrade(revision):
    """Revert database migrations"""
    _alembic(['downgrade', revision], writes=True)


@database.command()
def current():
    """Show current database revision"""
    _alembic(['current'], writes=False)


@database.command()
def history():
    """Show migration history"""
    _alembic(['history'], writes=False)


@database.command()
@click.option('--database-name', default='explaneat_dev', help='Name for the new database')
@click.option('--user', default='postgres', help='PostgreSQL user')
@click.option('--host', default='localhost', help='PostgreSQL host')
@click.option('--port', default='5432', help='PostgreSQL port')
def create_db(database_name, user, host, port):
    """Create a new PostgreSQL database"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    click.echo(f"Creating database '{database_name}'...")
    
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            dbname='postgres',
            user=user,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE {database_name}")
        click.echo(f"Database '{database_name}' created successfully!")
        
        cursor.close()
        conn.close()
        
    except psycopg2.errors.DuplicateDatabase:
        click.echo(f"Database '{database_name}' already exists!")
    except Exception as e:
        click.echo(f"Error creating database: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    database()