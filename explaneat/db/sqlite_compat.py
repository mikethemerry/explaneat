"""Opt-in SQLite support for the PostgreSQL-typed schema.

The models use PostgreSQL JSONB / UUID column types. Importing this module
registers SQLite renderings for them (JSON, CHAR(32)), the same shim the test
suite uses (tests/conftest.py), so a local file database works:

    import explaneat.db.sqlite_compat  # noqa: F401
    db.init_db("sqlite:////secure/local/explaneat.db"); db.create_all()

Use it for data that must stay inside a restricted environment (e.g. a
PREDICT / VAREANZ cohort under DAP/HDEC), where a shared Postgres server is not
allowed. Nothing changes for PostgreSQL. Schema changes still go through
Alembic on Postgres; a SQLite file is created directly with ``create_all``.
"""

from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "CHAR(32)"
