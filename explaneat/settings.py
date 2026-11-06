"""Local install settings: the data directory and the database URL.

Resolution order for the database URL (first match wins):

1. ``DATABASE_URL`` in the environment (e.g. a Postgres server);
2. ``database_url`` in ``<data dir>/config.json``, written by
   ``explaneat setup`` (a local SQLite file by default);
3. the legacy default ``postgresql://localhost/explaneat_dev``, so existing
   Postgres installs that never ran ``setup`` behave exactly as before.

The data directory is ``EXPLANEAT_HOME`` if set, otherwise ``~/.explaneat``
(``%USERPROFILE%\\.explaneat`` on Windows). It is deliberately outside the
repository, so restricted data (e.g. a cohort under a data access plan) never
lands in a git checkout. Nothing here makes a network connection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

LEGACY_DATABASE_URL = "postgresql://localhost/explaneat_dev"
CONFIG_FILENAME = "config.json"
SQLITE_FILENAME = "explaneat.db"


def data_dir() -> Path:
    """The data directory (not created here)."""
    env = os.environ.get("EXPLANEAT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".explaneat").resolve()


def config_path() -> Path:
    return data_dir() / CONFIG_FILENAME


def read_config() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_config(values: Dict[str, Any]) -> Path:
    """Merge ``values`` into the config file and return its path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = read_config()
    merged.update(values)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path


def sqlite_url(path: Path) -> str:
    """SQLAlchemy URL for a SQLite file; forward slashes work on Windows too."""
    return "sqlite:///" + Path(path).resolve().as_posix()


def default_sqlite_url() -> str:
    return sqlite_url(data_dir() / SQLITE_FILENAME)


def database_url_source() -> str:
    """Where the URL comes from: 'env', 'config' or 'legacy-default'."""
    if os.environ.get("DATABASE_URL"):
        return "env"
    if read_config().get("database_url"):
        return "config"
    return "legacy-default"


def resolve_database_url(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    configured = read_config().get("database_url")
    if configured:
        return configured
    return LEGACY_DATABASE_URL


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def redact(url: str) -> str:
    """Hide a password in a URL for display."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        if ":" in creds:
            creds = creds.split(":", 1)[0] + ":***"
        return f"{scheme}://{creds}@{host}"
    return url
