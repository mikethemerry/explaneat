"""Canonical JSON serialisation and the content hash of an export document.

Conventions (documented in ``docs/export_pack.md`` §3):

* Keys are sorted at every level; indentation is one space; the file ends with a newline;
  non-ASCII characters are written as-is (UTF-8).
* Floats are written with Python's shortest round-trip ``repr`` (what ``json`` emits). Values
  read from the database (weights, biases, recorded metrics) are stored unchanged. Values the
  export *computes* (errors, performance) are first rounded to ``DERIVED_SIG_FIGS`` significant
  figures with :func:`sig`, so that last-bit floating-point noise cannot change the file.
* NaN and infinities are not valid JSON: they are written as ``null``.
* The content hash is SHA-256 over the canonical JSON of the document with the fields in
  ``HASH_EXCLUDED`` removed (the export timestamp, the hash itself and the tool version /
  commit), so the same explanation exported twice, or by a later commit that computes the same
  content, has the same hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict, Optional

DERIVED_SIG_FIGS = 6

# Dotted paths removed before hashing.
HASH_EXCLUDED = (
    "provenance.export_timestamp",
    "provenance.content_hash",
    "provenance.tool",
)


def sig(x: Optional[float], n: int = DERIVED_SIG_FIGS) -> Optional[float]:
    """Round a computed float to ``n`` significant figures (None / non-finite -> None)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{n}g}")


def _clean(obj: Any) -> Any:
    """Replace non-finite floats by None and tuples by lists, recursively."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return _clean(obj.item())
    return obj


def dumps(doc: Dict[str, Any]) -> str:
    """The canonical JSON text of ``doc``."""
    return json.dumps(_clean(doc), sort_keys=True, indent=1, ensure_ascii=False,
                      allow_nan=False) + "\n"


def _strip(doc: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(doc)
    for path in HASH_EXCLUDED:
        parts = path.split(".")
        cur = d
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    return d


def content_hash(doc: Dict[str, Any]) -> str:
    """SHA-256 (hex) of the canonical JSON of ``doc`` minus ``HASH_EXCLUDED``."""
    return hashlib.sha256(dumps(_strip(doc)).encode("utf-8")).hexdigest()


def seal(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``provenance.content_hash`` on ``doc`` (in place) and return it."""
    doc.setdefault("provenance", {})["content_hash"] = {
        "algorithm": "sha256",
        "excludes": list(HASH_EXCLUDED),
        "value": content_hash(doc),
    }
    return doc


def verify_hash(doc: Dict[str, Any]) -> bool:
    ch = (doc.get("provenance") or {}).get("content_hash") or {}
    return ch.get("value") == content_hash(doc)
