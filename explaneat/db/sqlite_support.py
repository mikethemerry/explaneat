"""Detect whether the schema can be created on SQLite.

The models use PostgreSQL JSONB / UUID column types. SQLite renderings for
them come from ``explaneat.db.sqlite_compat`` (added on the Cox survival
branch, ``exp/cox-survival``). Until that module is on this branch, a SQLite
file cannot hold the schema, and ``explaneat setup`` / ``demo`` say so
instead of failing halfway. The test suite registers the same renderings in
``tests/conftest.py``, so tests see SQLite as supported.
"""

from __future__ import annotations

from typing import Tuple

PENDING_NOTE = (
    "SQLite schema support (explaneat/db/sqlite_compat.py) is not on this "
    "branch yet; it arrives with the exp/cox-survival merge. Until then use "
    "Postgres via DATABASE_URL, or run the demo without a database."
)


def load_sqlite_compat() -> bool:
    """Import the SQLite type shim if this branch has it."""
    try:
        import explaneat.db.sqlite_compat  # noqa: F401
    except ImportError:
        return False
    return True


def sqlite_schema_supported() -> Tuple[bool, str]:
    """(supported, how): can JSONB and UUID columns compile for SQLite?"""
    has_module = load_sqlite_compat()
    from sqlalchemy import Column, MetaData, Table
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.dialects.postgresql import JSONB, UUID
    from sqlalchemy.schema import CreateTable

    table = Table("_probe", MetaData(), Column("j", JSONB), Column("u", UUID(as_uuid=True)))
    try:
        CreateTable(table).compile(dialect=sqlite.dialect())
    except Exception:  # sqlalchemy raises CompileError / UnsupportedCompilationError
        return False, PENDING_NOTE
    return True, ("explaneat.db.sqlite_compat" if has_module else "type renderings registered elsewhere")
