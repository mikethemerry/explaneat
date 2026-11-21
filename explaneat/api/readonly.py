"""Read-only mode for the API (``EXPLANEAT_READ_ONLY=1`` or ``explaneat api --read-only``).

In read-only mode:

* every request that is not GET/HEAD/OPTIONS gets 403, except a short list of
  POST routes that only compute (evidence plots, performance, SHAP, node
  classification, operation validation, linearisation, import preview);
* the database connection itself is read only (PostgreSQL
  ``default_transaction_read_only``, SQLite ``PRAGMA query_only``; see
  ``explaneat/db/base.py``), so a write that slips through fails instead of
  landing;
* ``GET /api/config`` reports ``read_only: true`` and the UI hides its editing
  controls.

GET routes never create rows in either mode (``GET /genomes/{id}/explanation``
and ``/model`` used to create an empty explanation row).
"""

from __future__ import annotations

import os
import re

from fastapi import Request
from fastapi.responses import JSONResponse

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

# POST routes that compute and return a result without writing anything.
COMPUTE_ROUTES = [
    re.compile(p)
    for p in (
        r"^/api/genomes/[^/]+/evidence/(viz-data|performance|shap|input-distribution|reference-case|case-rows)$",
        r"^/api/genomes/[^/]+/analyze/(split-detection|classify-nodes|auto-classify)$",
        r"^/api/genomes/[^/]+/operations/validate$",
        r"^/api/genomes/[^/]+/tools/linearise$",
        r"^/api/genomes/[^/]+/tools/operations/import-preview$",
    )
]

READ_ONLY_DETAIL = "This ExplaNEAT server is read only: changes are not saved."


def is_read_only() -> bool:
    return os.environ.get("EXPLANEAT_READ_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")


def is_allowed(method: str, path: str) -> bool:
    """Whether a request may run in read-only mode."""
    if method.upper() in SAFE_METHODS:
        return True
    if method.upper() == "POST":
        return any(p.match(path) for p in COMPUTE_ROUTES)
    return False


async def read_only_middleware(request: Request, call_next):
    if is_read_only() and request.url.path.startswith("/api") and not is_allowed(
        request.method, request.url.path
    ):
        return JSONResponse(status_code=403, content={"detail": READ_ONLY_DETAIL})
    return await call_next(request)


def client_config() -> dict:
    """What the UI needs to know about this server (GET /api/config)."""
    return {"read_only": is_read_only(), "static": False}
