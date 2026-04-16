"""Test that orphaned running experiments are marked interrupted on startup.

The full database stack uses PostgreSQL-specific JSONB columns that don't
render on SQLite, so these tests mock out ``db.session_scope`` rather than
spinning up a real database. That keeps the unit focused on the function's
logic (query running experiments, flip status, stamp end_time, return count).
"""
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import importlib

app_module = importlib.import_module("explaneat.api.app")


class _FakeQuery:
    """Minimal stand-in for ``session.query(Model).filter_by(...).all()``."""

    def __init__(self, results):
        self._results = results

    def filter_by(self, **kwargs):
        self._filter = kwargs
        return self

    def all(self):
        return self._results


class _FakeSession:
    def __init__(self, results):
        self._results = results

    def query(self, _model):
        return _FakeQuery(self._results)


def _patch_session(monkeypatch, session):
    @contextmanager
    def fake_scope():
        yield session

    fake_db = SimpleNamespace(session_scope=fake_scope)
    # Patch the `db` symbol used inside mark_orphaned_experiments_interrupted.
    # The function does a local import `from ..db.base import db`, so patch the
    # module where it actually lives.
    import explaneat.db.base as base_mod
    monkeypatch.setattr(base_mod, "db", fake_db)


def test_running_experiments_marked_interrupted(monkeypatch):
    """Experiments with status='running' are flipped to 'interrupted'."""
    orphan_a = MagicMock(spec=["status", "end_time"])
    orphan_a.status = "running"
    orphan_a.end_time = None
    orphan_b = MagicMock(spec=["status", "end_time"])
    orphan_b.status = "running"
    orphan_b.end_time = None

    session = _FakeSession([orphan_a, orphan_b])
    _patch_session(monkeypatch, session)

    count = app_module.mark_orphaned_experiments_interrupted()
    assert count == 2
    assert orphan_a.status == "interrupted"
    assert orphan_b.status == "interrupted"
    assert isinstance(orphan_a.end_time, datetime)
    assert isinstance(orphan_b.end_time, datetime)


def test_no_orphans_returns_zero(monkeypatch):
    """When no experiments are running, nothing is updated and count is 0."""
    session = _FakeSession([])
    _patch_session(monkeypatch, session)

    count = app_module.mark_orphaned_experiments_interrupted()
    assert count == 0


def test_filter_applied_to_running_status(monkeypatch):
    """The query must filter by ``status='running'`` so other statuses are untouched."""
    captured = {}

    class RecordingQuery(_FakeQuery):
        def filter_by(self, **kwargs):
            captured.update(kwargs)
            return self

    class RecordingSession(_FakeSession):
        def query(self, _model):
            return RecordingQuery([])

    _patch_session(monkeypatch, RecordingSession([]))
    app_module.mark_orphaned_experiments_interrupted()
    assert captured == {"status": "running"}
