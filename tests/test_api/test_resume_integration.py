"""Integration tests for POST /api/experiments/{id}/resume error cases.

The resume endpoint relies on JSONB columns (config_json) that don't render
on SQLite, so rather than spinning up a real database we override the
``get_db`` FastAPI dependency with a mock session. This keeps the tests
focused on the endpoint's error-handling branches (404 missing, 400 wrong
status) without pulling in NEAT or the full DB stack.
"""
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _patch_get_db(session):
    """Return a dependency override for get_db that yields the mock session."""
    def _override():
        yield session
    return _override


@pytest.fixture
def mock_experiment_not_found():
    """Mock session.get to return None for Experiment lookup."""
    session = MagicMock()
    session.get.return_value = None
    return session


@pytest.fixture
def mock_experiment_completed():
    """Mock session to return an Experiment with status='completed'."""
    exp = MagicMock()
    exp.status = "completed"
    session = MagicMock()
    session.get.return_value = exp
    return session


@pytest.fixture
def mock_experiment_running():
    """Mock session to return an Experiment with status='running'."""
    exp = MagicMock()
    exp.status = "running"
    session = MagicMock()
    session.get.return_value = exp
    return session


def test_resume_returns_404_for_missing_experiment(mock_experiment_not_found):
    """Resume returns 404 if the experiment doesn't exist."""
    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db

    app = create_app()
    app.dependency_overrides[get_db] = _patch_get_db(mock_experiment_not_found)

    client = TestClient(app)
    resp = client.post(f"/api/experiments/{uuid.uuid4()}/resume")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_resume_returns_400_when_completed(mock_experiment_completed):
    """Resume returns 400 if experiment status is 'completed' (not 'interrupted')."""
    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db

    app = create_app()
    app.dependency_overrides[get_db] = _patch_get_db(mock_experiment_completed)

    client = TestClient(app)
    resp = client.post(f"/api/experiments/{uuid.uuid4()}/resume")
    assert resp.status_code == 400
    detail = resp.json()["detail"].lower()
    # Message should reference the current status and/or the required 'interrupted' status.
    assert "interrupted" in detail


def test_resume_returns_400_when_running(mock_experiment_running):
    """Resume returns 400 if experiment is currently running."""
    from explaneat.api.app import create_app
    from explaneat.api.dependencies import get_db

    app = create_app()
    app.dependency_overrides[get_db] = _patch_get_db(mock_experiment_running)

    client = TestClient(app)
    resp = client.post(f"/api/experiments/{uuid.uuid4()}/resume")
    assert resp.status_code == 400
    assert "interrupted" in resp.json()["detail"].lower()
