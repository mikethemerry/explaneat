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
    """
    if not db.Session:
        db.init_db()

    session = db.Session()
    try:
        yield session
    finally:
        session.close()
