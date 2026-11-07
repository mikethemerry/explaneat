"""SQLite support for the PostgreSQL-typed schema.

The models use PostgreSQL JSONB / UUID column types; ``explaneat.db.sqlite_compat``
registers SQLite renderings for them (JSON, CHAR(32)). ``Database.init_db``
loads it for every SQLite URL, and ``explaneat setup`` / ``doctor`` / ``demo``
use the probe below to confirm the schema compiles before touching a file.
"""

from __future__ import annotations

from typing import Tuple


def load_sqlite_compat() -> bool:
    """Import the SQLite type renderings (idempotent)."""
    try:
        import explaneat.db.sqlite_compat  # noqa: F401
    except ImportError:  # pragma: no cover - the module ships with the package
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
    except Exception as e:  # CompileError / UnsupportedCompilationError
        return False, f"JSONB/UUID do not compile for SQLite ({type(e).__name__}); " \
                      "explaneat/db/sqlite_compat.py is missing or broken"
    return True, ("explaneat.db.sqlite_compat" if has_module else "renderings registered elsewhere")
