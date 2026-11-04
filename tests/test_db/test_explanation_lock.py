"""Tests for lock_explanation: the serialization point for operations-stream writes."""

import pytest
from sqlalchemy import event, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from explaneat.db import Explanation
from explaneat.db.explanation_lock import lock_explanation


@pytest.fixture
def captured_selects():
    """Capture ORM SELECT statements executed by any session."""
    stmts = []

    def _on_execute(orm_execute_state):
        if orm_execute_state.is_select:
            stmts.append(orm_execute_state.statement)

    event.listen(Session, "do_orm_execute", _on_execute)
    yield stmts
    event.remove(Session, "do_orm_execute", _on_execute)


def _pg_sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


@pytest.mark.db
def test_lock_selects_genome_and_explanation_for_update(
    db_session, test_genome, test_explanation, captured_selects
):
    genome_id, expected_id = str(test_genome.id), test_explanation.id  # load before capture
    captured_selects.clear()
    explanation = lock_explanation(db_session, genome_id)

    assert explanation is not None and explanation.id == expected_id
    pg = [_pg_sql(s) for s in captured_selects]
    genome_sql = [s for s in pg if "FROM genomes" in s]
    expl_sql = [s for s in pg if "FROM explanations" in s]
    assert genome_sql and all("FOR UPDATE" in s for s in genome_sql)
    assert expl_sql and all("FOR UPDATE" in s for s in expl_sql)
    # Genome locked before explanation (makes get-or-create race-safe).
    assert pg.index(genome_sql[0]) < pg.index(expl_sql[0])


@pytest.mark.db
def test_lock_refreshes_stale_identity_map(db_session, test_genome, test_explanation):
    """A copy read before the lock must not be what the caller modifies."""
    stale = db_session.query(Explanation).filter_by(genome_id=test_genome.id).first()
    assert stale.operations in (None, [])

    # Simulate another writer committing new operations behind the ORM's back.
    db_session.execute(
        update(Explanation)
        .where(Explanation.id == test_explanation.id)
        .values(operations=[{"seq": 1, "type": "annotate", "params": {}}])
        .execution_options(synchronize_session=False)
    )

    locked = lock_explanation(db_session, test_genome.id)
    assert locked is stale  # same identity...
    assert locked.operations == [{"seq": 1, "type": "annotate", "params": {}}]  # ...fresh data


@pytest.mark.db
def test_lock_returns_none_without_create(db_session, test_genome):
    assert lock_explanation(db_session, test_genome.id) is None


@pytest.mark.db
def test_lock_creates_when_requested(db_session, test_genome):
    explanation = lock_explanation(db_session, test_genome.id, create=True)
    assert explanation is not None
    assert explanation.operations == []
    assert db_session.query(Explanation).filter_by(genome_id=test_genome.id).count() == 1
    # Idempotent: a second call returns the same row instead of creating another.
    again = lock_explanation(db_session, test_genome.id, create=True)
    assert again.id == explanation.id
    assert db_session.query(Explanation).filter_by(genome_id=test_genome.id).count() == 1


@pytest.mark.db
def test_lock_missing_genome_raises(db_session):
    import uuid

    with pytest.raises(ValueError, match="Genome not found"):
        lock_explanation(db_session, uuid.uuid4())
