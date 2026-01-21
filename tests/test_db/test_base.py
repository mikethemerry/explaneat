"""
Tests for Database base class and connection management.

Tests database initialization, session management, and cleanup operations.
"""

import pytest
from sqlalchemy.exc import OperationalError

from explaneat.db import db, Base, Experiment


@pytest.mark.db
@pytest.mark.unit
class TestDatabaseBase:
    """Test Database class base operations."""

    def test_init_db_with_url(self, test_db):
        """Test initializing database with a specific URL."""
        # Should already be initialized by fixture
        assert db.engine is not None
        assert db.Session is not None

    def test_init_db_creates_engine(self, test_db):
        """Test that init_db creates an engine."""
        assert db.engine is not None
        assert db.database_url == "sqlite:///:memory:"

    def test_create_all(self, test_db):
        """Test that create_all creates all tables."""
        # Tables should already be created by fixture
        # Verify by querying a table
        with db.session_scope() as session:
            # Should not raise an error
            count = session.query(Experiment).count()
            assert count >= 0  # Can be 0, but should not error

    def test_drop_all(self, test_db):
        """Test that drop_all removes all tables."""
        # Create a test record first
        with db.session_scope() as session:
            experiment = Experiment(
                experiment_sha="test_drop",
                name="Test Drop",
                config_json={},
                neat_config_text="# Test"
            )
            session.add(experiment)
        
        # Verify it exists
        with db.session_scope() as session:
            count = session.query(Experiment).count()
            assert count > 0
        
        # Drop all tables
        db.drop_all()
        
        # Verify tables are gone (should raise error on query)
        with pytest.raises(OperationalError):
            with db.session_scope() as session:
                session.query(Experiment).count()
        
        # Recreate for other tests
        db.create_all()

    def test_session_scope_commits(self, test_db):
        """Test that session_scope commits transactions."""
        experiment_id = None
        
        with db.session_scope() as session:
            experiment = Experiment(
                experiment_sha="test_commit",
                name="Test Commit",
                config_json={},
                neat_config_text="# Test"
            )
            session.add(experiment)
            session.flush()
            experiment_id = experiment.id
        
        # Verify it was committed
        with db.session_scope() as session:
            retrieved = session.get(Experiment, experiment_id)
            assert retrieved is not None
            assert retrieved.name == "Test Commit"

    def test_session_scope_rollback_on_error(self, test_db):
        """Test that session_scope rolls back on error."""
        experiment_id = None
        
        try:
            with db.session_scope() as session:
                experiment = Experiment(
                    experiment_sha="test_rollback",
                    name="Test Rollback",
                    config_json={},
                    neat_config_text="# Test"
                )
                session.add(experiment)
                session.flush()
                experiment_id = experiment.id
                # Raise an error to trigger rollback
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify it was NOT committed
        with db.session_scope() as session:
            retrieved = session.get(Experiment, experiment_id)
            assert retrieved is None

    def test_get_session(self, test_db):
        """Test that get_session returns a session."""
        session = db.get_session()
        assert session is not None
        
        # Verify it works
        count = session.query(Experiment).count()
        assert count >= 0
        
        session.close()

    def test_close(self, test_db):
        """Test that close cleans up resources."""
        # Get a session first
        session = db.get_session()
        assert session is not None
        
        # Close should clean up
        db.close()
        
        # After close, we should be able to reinitialize
        db.init_db("sqlite:///:memory:")
        db.create_all()
        
        # Verify it works
        with db.session_scope() as session:
            count = session.query(Experiment).count()
            assert count >= 0

    def test_multiple_sessions_independent(self, test_db):
        """Test that multiple sessions are independent."""
        experiment_id = None
        
        # Create in first session
        with db.session_scope() as session1:
            experiment = Experiment(
                experiment_sha="test_multi",
                name="Test Multi",
                config_json={},
                neat_config_text="# Test"
            )
            session1.add(experiment)
            session1.flush()
            experiment_id = experiment.id
        
        # Query in second session
        with db.session_scope() as session2:
            retrieved = session2.get(Experiment, experiment_id)
            assert retrieved is not None
            assert retrieved.name == "Test Multi"

    def test_session_isolation(self, test_db):
        """Test that sessions are isolated (uncommitted changes not visible)."""
        experiment_id = None
        
        # Start a session but don't commit
        session1 = db.get_session()
        experiment = Experiment(
            experiment_sha="test_isolation",
            name="Test Isolation",
            config_json={},
            neat_config_text="# Test"
        )
        session1.add(experiment)
        session1.flush()
        experiment_id = experiment.id
        
        # Try to query in another session (should not see it)
        with db.session_scope() as session2:
            retrieved = session2.get(Experiment, experiment_id)
            # Should be None because session1 hasn't committed
            assert retrieved is None
        
        # Commit session1
        session1.commit()
        session1.close()
        
        # Now should be visible
        with db.session_scope() as session3:
            retrieved = session3.get(Experiment, experiment_id)
            assert retrieved is not None
