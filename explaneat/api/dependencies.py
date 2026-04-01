"""
FastAPI dependency injection for database sessions and common dependencies.
"""

from typing import Generator
from sqlalchemy.orm import Session

from ..db import db


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.

    Yields a session and ensures it's closed after the request.
    Uses session_factory directly (not scoped_session) so each
    concurrent request gets its own independent session.
    """
    if not db.session_factory:
        db.init_db()

    session = db.session_factory()
    try:
        yield session
    finally:
        session.close()
