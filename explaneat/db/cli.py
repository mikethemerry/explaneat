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


@database.command()
@click.argument('message')
def revision(message):
    """Create a new database migration"""
    cmd = ['alembic', 'revision', '--autogenerate', '-m', message]
    click.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


@database.command()
@click.option('--revision', default='head', help='Revision to upgrade to')
def upgrade(revision):
    """Apply database migrations"""
    cmd = ['alembic', 'upgrade', revision]
    click.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


@database.command()
@click.option('--revision', default='-1', help='Revision to downgrade to')
def downgrade(revision):
    """Revert database migrations"""
    cmd = ['alembic', 'downgrade', revision]
    click.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


@database.command()
def current():
    """Show current database revision"""
    cmd = ['alembic', 'current']
    subprocess.run(cmd)


@database.command()
def history():
    """Show migration history"""
    cmd = ['alembic', 'history']
    subprocess.run(cmd)


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