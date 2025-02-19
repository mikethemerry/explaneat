import os
import sys

# Add the backend directory to Python path
sys.path.append(os.path.abspath("backend"))

from backend import create_app
from flask.cli import FlaskGroup

app = create_app()
cli = FlaskGroup(app)


@cli.command("test")
def test():
    """Run unit tests."""
    import pytest

    pytest.main(["backend/tests"])


@cli.command("cov")
def cov():
    """Run unit tests with coverage."""
    import pytest

    pytest.main(["backend/tests", "--cov=backend", "--cov-report=term-missing"])


if __name__ == "__main__":
    cli()
