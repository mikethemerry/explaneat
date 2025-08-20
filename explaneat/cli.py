"""Main CLI entry point for ExplaNEAT"""
import click
from .db.cli import database


@click.group()
def cli():
    """ExplaNEAT - Explainable NEAT with database support"""
    pass


# Add database commands
cli.add_command(database, name='db')


if __name__ == '__main__':
    cli()