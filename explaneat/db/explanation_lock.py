"""Serialized access to a genome's ``Explanation.operations`` event stream.

``Explanation.operations`` is a single JSONB array. Every writer (applying or
undoing an operation, adding evidence, updating a narrative, ...) does a
read-modify-write of the *whole* array, so two concurrent writers on the same
genome silently lose one of the updates unless they are serialized.

``lock_explanation`` is the one place that serialization happens. Call it at
the **start** of the transaction, before anything else reads the explanation,
then do the read-modify-write in the same transaction and let the commit
release the lock::

    with db.session_scope() as session:
        explanation = lock_explanation(session, genome_id)
        ops = list(explanation.operations or [])
        ...  # modify
        explanation.operations = ops
        flag_modified(explanation, "operations")

On PostgreSQL this takes row locks (``SELECT ... FOR UPDATE``) on the genome
row and the explanation row; a second writer blocks until the first commits
and then reads the committed value (READ COMMITTED). The genome row is locked
first so that *creating* a missing explanation is race-safe as well (there is
no unique constraint on ``explanations.genome_id``). On SQLite ``FOR UPDATE``
is not supported and SQLAlchemy omits it, so the lock is a no-op there.

``populate_existing()`` makes the locked SELECT overwrite any copy already in
the session's identity map, so the caller never modifies a stale snapshot that
was read before the lock was acquired.
"""

from __future__ import annotations

import uuid
from typing import Optional, Union

from sqlalchemy.orm import Session

from .models import Explanation, Genome


def _as_uuid(value: Union[str, uuid.UUID]) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def lock_explanation(
    session: Session,
    genome_id: Union[str, uuid.UUID],
    create: bool = False,
) -> Optional[Explanation]:
    """Lock and return the genome's Explanation row for a read-modify-write.

    Args:
        session: Active session; the lock is held until it commits/rolls back.
        genome_id: Genome UUID (string or ``uuid.UUID``).
        create: If True and no explanation exists, create (and flush) one
            while holding the genome lock.

    Returns:
        The locked, freshly loaded Explanation, or None if it does not exist
        and ``create`` is False.

    Raises:
        ValueError: If the genome does not exist.
    """
    gid = _as_uuid(genome_id)

    genome = (
        session.query(Genome)
        .filter(Genome.id == gid)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if genome is None:
        raise ValueError(f"Genome not found: {genome_id}")

    explanation = (
        session.query(Explanation)
        .filter(Explanation.genome_id == gid)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if explanation is None and create:
        explanation = Explanation(genome_id=gid, is_well_formed=False, operations=[])
        session.add(explanation)
        session.flush()
        # a new explanation is strict (no bare typed numbers); existing ones stay legacy
        from ..analysis.provenance import set_strict
        set_strict(session, gid, True)
    return explanation
