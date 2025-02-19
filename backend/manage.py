from flask_migrate import Migrate
from flask import cli
from backend import create_app, db

app = create_app()
migrate = Migrate(app, db)


@app.cli.command("test")
def test():
    """Run unit tests."""
    import pytest

    pytest.main(["tests"])


@app.cli.command("cov")
def cov():
    """Run unit tests with coverage."""
    import pytest

    pytest.main(["tests", "--cov=backend", "--cov-report=term-missing"])


if __name__ == "__main__":
    app.cli()
