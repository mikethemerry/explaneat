"""Explanation records: the JSON reports of Claude-produced explanations.

Each record (``experiments/10seed/explanations/<label>.json``) holds an agent's
report for one genome plus the orchestrator's ``verified_by_orchestrator`` block.
Mike's verification is stored in the same file under ``mike_verification``
(``experiments/10seed/README.md`` §6), which ``build_explanations_csv.py`` reads.

The records directory is ``$EXPLANEAT_RECORDS_DIR`` or, by default, the
repository's ``experiments/10seed/explanations``. A relative path is read from the working
directory, else from the repository root, so ``EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4``
works from anywhere in the checkout. The plain-language layers follow it
(:func:`explaneat.analysis.plain_language_store.layers_dir`).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_dir(value: str) -> Path:
    """A directory given on the command line or in an environment variable: as given when it
    exists (relative to the working directory), else relative to the repository root."""
    p = Path(value).expanduser()
    if p.is_dir():
        return p.resolve()
    if p.is_absolute():
        return p
    q = REPO_ROOT / p
    return q if q.is_dir() else p


def records_dir() -> Path:
    env = os.environ.get("EXPLANEAT_RECORDS_DIR")
    return resolve_dir(env) if env else REPO_ROOT / "experiments" / "10seed" / "explanations"


def use_record_set(records: Optional[str] = None, layers: Optional[str] = None) -> None:
    """Point this process at another set of records (and layers): sets ``EXPLANEAT_RECORDS_DIR``
    (and ``EXPLANEAT_LAYERS_DIR``) so every reader and writer (review, export, recheck, sign-off)
    uses the same files. Used by the ``--records-dir`` / ``--layers-dir`` options."""
    for key, val in (("EXPLANEAT_RECORDS_DIR", records), ("EXPLANEAT_LAYERS_DIR", layers)):
        if val:
            d = resolve_dir(str(val))
            if not d.is_dir():
                raise FileNotFoundError(f"{val}: no such directory")
            os.environ[key] = str(d)


def record_set() -> Dict[str, str]:
    """The records and layers directories in use (repository-relative when inside it)."""
    from .plain_language_store import layers_dir

    def rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return {"records_dir": rel(records_dir()), "layers_dir": rel(layers_dir())}


def _read(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_label(label: str) -> str:
    if not label or any(c in label for c in "/\\") or label.startswith("."):
        raise ValueError(f"invalid record label {label!r}")
    return label


def record_path(label: str, directory: Optional[Path] = None) -> Path:
    return (directory or records_dir()) / f"{_safe_label(label)}.json"


def summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    v = rec.get("verified_by_orchestrator") or {}
    m = rec.get("mike_verification") or {}
    return {
        "label": rec.get("label"),
        "genome_id": rec.get("genome_id"),
        "full_coverage": rec.get("full_coverage"),
        "orchestrator_full_coverage": v.get("full_coverage"),
        "agrees_with_agent": v.get("agrees_with_agent"),
        "n_leaf_annotations": rec.get("n_leaf_annotations"),
        "n_composition_annotations": rec.get("n_composition_annotations"),
        "hidden_nodes": rec.get("hidden_nodes"),
        "closed_form_max_abs_err": rec.get("closed_form_max_abs_err"),
        "function_preservation_max_abs_err": rec.get("function_preservation_max_abs_err"),
        "summary": rec.get("summary"),
        "verified": bool(m.get("verified")),
        "mike_verification": m or None,
    }


def list_records(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    d = directory or records_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            rec = _read(p)
        except (OSError, ValueError):
            continue
        if "genome_id" not in rec:
            continue
        rec.setdefault("label", p.stem)
        out.append(summary(rec))
    return out


def load_record(label: str, directory: Optional[Path] = None) -> Dict[str, Any]:
    p = record_path(label, directory)
    if not p.is_file():
        raise FileNotFoundError(label)
    return _read(p)


def find_record_for_genome(genome_id: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    d = directory or records_dir()
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.json")):
        try:
            rec = _read(p)
        except (OSError, ValueError):
            continue
        if str(rec.get("genome_id")) == str(genome_id):
            rec.setdefault("label", p.stem)
            return rec
    return None


def write_verification(
    label: str,
    verified: bool,
    minutes: Optional[float] = None,
    changes: str = "",
    notes: str = "",
    directory: Optional[Path] = None,
    signoff: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``mike_verification`` into the record, atomically.

    ``signoff``: the ``signoff-v1`` block (``explaneat review signoff --write``). Under the PREREG
    amendment of 2026-11-11, ``verified_by_mike`` is "yes" only for a record verified with it.

    The rest of the record is left untouched and written back in the same
    format as ``verify_record.py`` (indent 2, key order kept).
    """
    p = record_path(label, directory)
    rec = _read(p)
    mv: Dict[str, Any] = {"verified": bool(verified)}
    if minutes is not None:
        mv["minutes"] = minutes
    mv["changes"] = changes or ""
    if notes:
        mv["notes"] = notes
    if signoff:
        mv["protocol"] = signoff.get("protocol")
        mv["signoff"] = signoff
    mv["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["mike_verification"] = mv
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, indent=2))
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return rec
