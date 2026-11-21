"""Explanation records: the JSON reports of Claude-produced explanations.

Each record holds an agent's report for one genome (plus, for the 10-seed study, the
orchestrator's ``verified_by_orchestrator`` block). Mike's verification is stored with it under
``mike_verification`` (``experiments/10seed/README.md`` §6), which ``build_explanations_csv.py``
reads.

Where records live
------------------
* **The database** (table ``explanation_records``, migration ``m6n7o8p9q0r1``; loaded with
  ``explaneat records import``). Used when the process has a database (``DATABASE_URL`` set, or
  the DB already initialised, or ``EXPLANEAT_RECORDS_DB=1``) and the table exists and holds the
  record. A verification written there is appended to ``explanation_record_events``.
  ``EXPLANEAT_RECORDS_DB=0`` forces the files.
* **Files**, the fallback (and the committed layer-2 snapshot: ``explaneat records export``
  writes the database's records back to them). The directories searched, in order:
  ``$EXPLANEAT_RECORDS_DIR`` (one directory, or several separated by ``os.pathsep``; default
  ``experiments/10seed/explanations``), then every study's ``experiments/<study>/explanations*``
  (and ``experiments/<a>/<b>/explanations*``) folder (``EXPLANEAT_RECORDS_DISCOVER=0`` turns that
  off). A record is any JSON file there (also one level down, e.g. ``shape-A/record.json``) with a
  ``genome_id``; in one folder, ``<label>.json`` beats ``record.json`` beats ``report.json``.

A relative directory is read from the working directory, else from the repository root, so
``EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4`` works from anywhere in the checkout.

Labels are unique: a record whose label is already taken (by the database, or by a folder
searched earlier) is listed as ``<study>.<label>`` (study = the folder above ``explanations``,
plus the folder's suffix: ``10seed-v4``, ``cox_case``). Writers write back where the record was
found (its database row, or its own file), never elsewhere. The plain-language layer of a record
is the file with its label in the sibling ``plain_language<suffix>`` folder, when there is one
(:func:`explaneat.analysis.plain_language_store.layer_path`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "experiments" / "10seed" / "explanations"
DISCOVER_ROOT = REPO_ROOT / "experiments"
ENTERED_BY = ("ui", "chat", "cli", "import")

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- directories
def resolve_dir(value: str) -> Path:
    """A directory given on the command line or in an environment variable: as given when it
    exists (relative to the working directory), else relative to the repository root."""
    p = Path(value).expanduser()
    if p.is_dir():
        return p.resolve()
    if p.is_absolute():
        return p
    q = REPO_ROOT / p
    return q.resolve() if q.is_dir() else p


def configured_dirs() -> List[Path]:
    """``$EXPLANEAT_RECORDS_DIR`` (``os.pathsep``-separated) or the default record set."""
    env = os.environ.get("EXPLANEAT_RECORDS_DIR")
    dirs = [resolve_dir(v) for v in env.split(os.pathsep) if v.strip()] if env else []
    return dirs or [DEFAULT_DIR]


def records_dir() -> Path:
    """The primary records directory (the first configured one)."""
    return configured_dirs()[0]


_DISCOVERED: Tuple[float, List[Path]] = (0.0, [])


def discovered_dirs(root: Optional[Path] = None, fresh: bool = False) -> List[Path]:
    """Every study's ``explanations*`` folder under ``experiments/`` (two levels deep), sorted;
    cached for 30 s unless ``fresh``. Empty when ``EXPLANEAT_RECORDS_DISCOVER=0``."""
    global _DISCOVERED
    if os.environ.get("EXPLANEAT_RECORDS_DISCOVER", "1").strip().lower() in ("0", "false", "no", "off"):
        return []
    base = root or DISCOVER_ROOT
    if root is None and not fresh and time.monotonic() - _DISCOVERED[0] < 30:
        return list(_DISCOVERED[1])
    out = []
    if base.is_dir():
        for pattern in ("*/explanations*", "*/*/explanations*"):
            out += [p.resolve() for p in base.glob(pattern) if p.is_dir()]
    out = sorted(set(out))
    if root is None:
        _DISCOVERED = (time.monotonic(), out)
    return out


def records_dirs(fresh: bool = False) -> List[Path]:
    """All directories searched for record files, in search order."""
    out: List[Path] = []
    for d in configured_dirs() + discovered_dirs(fresh=fresh):
        if d not in out:
            out.append(d)
    return out


def rel(p) -> str:
    """Repository-relative when inside it, else as given."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def study_of(directory: Path) -> str:
    """``experiments/10seed/explanations_v4`` -> ``10seed-v4``; ``.../cox_case/explanations`` ->
    ``cox_case``; another folder name -> that name."""
    name = Path(directory).name
    if name.startswith("explanations"):
        suffix = name[len("explanations"):].lstrip("_-")
        base = Path(directory).parent.name
        return f"{base}-{suffix}" if suffix else base
    return name


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


def record_set() -> Dict[str, Any]:
    """The primary records and layers directories (repository-relative when inside it), and
    every directory searched."""
    from .plain_language_store import layers_dir
    return {"records_dir": rel(records_dir()), "layers_dir": rel(layers_dir()),
            "records_dirs": [rel(d) for d in records_dirs()]}


# ----------------------------------------------------------------------------- files
def _read(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_FILE_CACHE: Dict[str, Tuple[int, int, Any]] = {}


def _read_cached(path: Path) -> Any:
    """Parsed JSON, cached by (mtime, size); None when unreadable."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    hit = _FILE_CACHE.get(key)
    if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    try:
        doc = _read(path)
    except (OSError, ValueError):
        doc = None
    _FILE_CACHE[key] = (st.st_mtime_ns, st.st_size, doc)
    return doc


def _safe_label(label: str) -> str:
    if not label or any(c in label for c in "/\\") or label.startswith("."):
        raise ValueError(f"invalid record label {label!r}")
    return label


@dataclass
class Entry:
    """Where one record lives. ``label`` is unique across studies; ``file_label`` is the label
    the record itself carries (the file names of its layer and review files)."""
    label: str
    file_label: str
    study: str
    genome_id: str
    records_dir: str            # repository-relative record set (the review store's guard)
    source: str                 # repository-relative file (for a DB record: its file of origin)
    storage: str                # "db" | "file"
    path: Optional[str] = None  # absolute file path (file storage)
    report_path: Optional[str] = None
    record_id: Optional[str] = None

    def public(self) -> Dict[str, Any]:
        return {"study": self.study, "storage": self.storage, "source": self.source,
                "records_dir": self.records_dir}


def dir_records(directory: Path) -> List[Dict[str, Any]]:
    """The records in one folder: ``[{label, genome_id, path, report_path}]``, one per label
    (``<label>.json`` at the top beats a top-level other name beats ``record.json`` beats
    ``report.json``), sorted by label."""
    d = Path(directory)
    if not d.is_dir():
        return []
    best: Dict[str, Tuple[int, str, Dict[str, Any]]] = {}
    for p in sorted(list(d.glob("*.json")) + list(d.glob("*/*.json"))):
        if p.name.startswith("."):
            continue
        doc = _read_cached(p)
        if not isinstance(doc, dict) or "genome_id" not in doc:
            continue
        stem = p.parent.name if p.stem in ("record", "report") and p.parent != d and not doc.get("label") else p.stem
        label = str(doc.get("label") or stem)
        top = p.parent == d
        rank = 0 if top and p.stem == label else 1 if top else 2 if p.name == "record.json" else \
            3 if p.name == "report.json" else 4
        cur = best.get(label)
        if cur is None or (rank, str(p)) < (cur[0], cur[1]):
            rp = p.parent / "report.json"
            best[label] = (rank, str(p), {
                "label": label, "genome_id": str(doc.get("genome_id")), "path": str(p),
                "report_path": str(rp) if p.name == "record.json" and rp.is_file() else None})
    return [best[k][2] for k in sorted(best)]


# ----------------------------------------------------------------------------- database
def _db():
    """The database to read records from, or None (files only)."""
    mode = os.environ.get("EXPLANEAT_RECORDS_DB", "auto").strip().lower()
    if mode in ("0", "off", "false", "no", "files"):
        return None
    from ..db import db
    if db.engine is None:
        if mode not in ("1", "on", "true", "yes") and not os.environ.get("DATABASE_URL"):
            return None
        try:
            db.init_db()
        except Exception as e:   # pragma: no cover - a broken URL: files only
            log.debug("records: no database (%s)", e)
            return None
    return db


def _with_db(fn: Callable, default=None):
    """``fn(session)`` on the records database; ``default`` when there is none, it is
    unreachable or the table does not exist yet (before the migration)."""
    d = _db()
    if d is None:
        return default
    from sqlalchemy.exc import SQLAlchemyError
    try:
        with d.session_scope() as s:
            return fn(s)
    except SQLAlchemyError as e:
        log.debug("records: database not used (%s)", e)
        return default


def _db_entries() -> Optional[List[Entry]]:
    from ..db.models import ExplanationRecord as R

    def q(s):
        rows = s.query(R.id, R.label, R.study, R.genome_id, R.records_dir, R.source_path,
                       R.record).order_by(R.label).all()
        out = []
        for rid, label, study, gid, rdir, src, rec in rows:
            out.append(Entry(label=label, file_label=str((rec or {}).get("label") or label), study=study,
                             genome_id=str(gid), records_dir=rdir or "", source=src or "", storage="db",
                             path=str(REPO_ROOT / src) if src else None, record_id=str(rid)))
        return out
    return _with_db(q, None)


def db_in_use() -> bool:
    """Whether the records table exists in this process's database (it may be empty)."""
    return _db_entries() is not None


# ----------------------------------------------------------------------------- catalogue
def catalog(directories: Optional[List[Path]] = None, use_db: bool = True) -> List[Entry]:
    """Every record: the database's (in the folders' search order), then the files not in it, in
    search order. A label taken earlier makes a later record ``<study>.<label>``; when that is
    taken too the later record is left out (the same file seen twice)."""
    entries: List[Entry] = []
    taken: Dict[str, Entry] = {}
    seen = set()
    order = {rel(d): i for i, d in enumerate(records_dirs())}
    db_rows = sorted(_db_entries() or [], key=lambda e: (order.get(e.records_dir, len(order)), e.records_dir,
                                                          e.label)) if use_db else []
    for e in db_rows:
        entries.append(e)
        taken[e.label] = e
        if e.source:
            seen.add(e.source)
    for d in directories if directories is not None else records_dirs():
        study, rdir = study_of(d), rel(d)
        for fr in dir_records(d):
            src = rel(fr["path"])
            if src in seen:
                continue
            seen.add(src)
            label = fr["label"]
            if label in taken:
                label = f"{study}.{fr['label']}"
                if label in taken:
                    continue
            e = Entry(label=label, file_label=fr["label"], study=study, genome_id=fr["genome_id"],
                      records_dir=rdir, source=src, storage="file", path=fr["path"],
                      report_path=fr["report_path"])
            entries.append(e)
            taken[label] = e
    return entries


def entry(label: str) -> Optional[Entry]:
    _safe_label(label)
    return next((e for e in catalog() if e.label == label), None)


def _load_entry(e: Entry) -> Dict[str, Any]:
    if e.storage == "db":
        from ..db.models import ExplanationRecord as R

        def q(s):
            row = s.query(R).filter(R.label == e.label).one_or_none()
            return None if row is None else (row.record, row.mike_verification)
        got = _with_db(q, None)
        if got is None:
            raise FileNotFoundError(e.label)
        rec, mv = got
        rec = dict(rec)
        if mv is not None:
            rec["mike_verification"] = mv
    else:
        rec = _read(Path(e.path))
    if e.label != e.file_label or "label" not in rec:
        rec["label"] = e.label
    return rec


def _report_of(e: Entry) -> Optional[Dict[str, Any]]:
    if e.storage == "db":
        from ..db.models import ExplanationRecord as R
        return _with_db(lambda s: s.query(R.report).filter(R.label == e.label).scalar(), None)
    return _read_cached(Path(e.report_path)) if e.report_path else None


# ----------------------------------------------------------------------------- reading
def record_path(label: str, directory: Optional[Path] = None) -> Path:
    """The record's file: in ``directory`` when given; else its own file (for a database record:
    the file it is exported to); else ``<records_dir>/<label>.json``."""
    if directory is not None:
        return Path(directory) / f"{_safe_label(label)}.json"
    e = entry(label)
    if e is not None and e.path:
        return Path(e.path)
    return records_dir() / f"{_safe_label(label)}.json"


def record_location(label: str) -> str:
    """Where the record is stored, for messages: ``database (from <file>)`` or the file."""
    e = entry(label)
    if e is None:
        return str(record_path(label))
    return f"database (exported to {e.source})" if e.storage == "db" else str(e.path)


def records_dir_of(label: Optional[str]) -> str:
    """The record set (repository-relative folder) the record belongs to, else the primary one."""
    e = entry(label) if label else None
    return e.records_dir if e is not None and e.records_dir else rel(records_dir())


def summary(rec: Dict[str, Any], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    v = rec.get("verified_by_orchestrator") or {}
    m = rec.get("mike_verification") or {}
    rep = report or {}
    g = lambda k: rec.get(k) if rec.get(k) is not None else rep.get(k)
    return {
        "label": rec.get("label"),
        "genome_id": rec.get("genome_id"),
        "full_coverage": g("full_coverage"),
        "orchestrator_full_coverage": v.get("full_coverage"),
        "agrees_with_agent": v.get("agrees_with_agent"),
        "n_leaf_annotations": g("n_leaf_annotations"),
        "n_composition_annotations": g("n_composition_annotations"),
        "hidden_nodes": g("hidden_nodes"),
        "closed_form_max_abs_err": g("closed_form_max_abs_err"),
        "function_preservation_max_abs_err": g("function_preservation_max_abs_err"),
        "summary": g("summary"),
        "verified": bool(m.get("verified")),
        "mike_verification": m or None,
    }


def list_records(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Summaries of the records: of one folder's ``*.json`` when ``directory`` is given (files
    only), else of every record (database and files), each with its ``study``, ``storage``,
    ``source`` and ``records_dir``."""
    if directory is not None:
        d = Path(directory)
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
    out = []
    for e in catalog():
        try:
            rec = _load_entry(e)
        except (OSError, ValueError):
            continue
        out.append({**summary(rec, _report_of(e)), **e.public()})
    return out


def load_record(label: str, directory: Optional[Path] = None) -> Dict[str, Any]:
    if directory is not None:
        p = record_path(label, directory)
        if not p.is_file():
            raise FileNotFoundError(label)
        return _read(p)
    e = entry(label)
    if e is None:
        raise FileNotFoundError(label)
    return _load_entry(e)


def load_record_file(path) -> Dict[str, Any]:
    """The record of a file: the database's copy when it holds that file's record, else the
    file. For scripts that name record files (``build_explanations_csv.py``)."""
    src = rel(path)
    from ..db.models import ExplanationRecord as R

    def q(s):
        row = s.query(R.record, R.mike_verification).filter(R.source_path == src).one_or_none()
        return None if row is None else (row[0], row[1])
    got = _with_db(q, None)
    if got is None:
        return _read(Path(path))
    rec = dict(got[0])
    if got[1] is not None:
        rec["mike_verification"] = got[1]
    return rec


def find_entry_for_genome(genome_id: str) -> Optional[Entry]:
    """The record of a genome. With several (a genome explained in two record sets), the one in
    a configured directory first, then one under its own label, then search order."""
    cands = [e for e in catalog() if e.genome_id == str(genome_id)]
    if not cands:
        return None
    conf = {rel(d) for d in configured_dirs()}
    return min(enumerate(cands), key=lambda ie: (ie[1].records_dir not in conf,
                                                 ie[1].label != ie[1].file_label, ie[0]))[1]


def find_record_for_genome(genome_id: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if directory is not None:
        d = Path(directory)
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
    e = find_entry_for_genome(genome_id)
    return _load_entry(e) if e is not None else None


def searched() -> Dict[str, Any]:
    """What a lookup searches, for messages: the database (if its table is in use) and the
    folders."""
    return {"database": db_in_use(), "records_dirs": [rel(d) for d in records_dirs()]}


# ----------------------------------------------------------------------------- writing
def dumps_like(rec: Dict[str, Any], existing: Optional[str] = None) -> str:
    """The record as JSON in the layout of the file it replaces (its indent, trailing newline
    and ASCII escaping), so a rewrite changes only what changed; indent 2 for a new file."""
    if not existing:
        return json.dumps(rec, indent=2)
    lines = existing.split("\n", 2)
    indent = len(lines[1]) - len(lines[1].lstrip(" ")) if len(lines) > 1 and lines[1].strip() else 2
    text = json.dumps(rec, indent=indent or None, ensure_ascii=existing.isascii())
    return text + "\n" if existing.endswith("\n") else text


def _write_json(p: Path, rec: Dict[str, Any]) -> None:
    try:
        existing = p.read_text(encoding="utf-8") if p.is_file() else None
    except (OSError, UnicodeDecodeError):
        existing = None
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(dumps_like(rec, existing))
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def content_hash(rec: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()


def write_verification(
    label: str,
    verified: bool,
    minutes: Optional[float] = None,
    changes: str = "",
    notes: str = "",
    directory: Optional[Path] = None,
    signoff: Optional[Dict[str, Any]] = None,
    entered_by: str = "cli",
) -> Dict[str, Any]:
    """Write ``mike_verification`` into the record, where the record is stored.

    ``signoff``: the ``signoff-v1`` block (``explaneat review signoff --write``). Under the PREREG
    amendment of 2026-11-11, ``verified_by_mike`` is "yes" only for a record verified with it.
    ``entered_by``: ui | chat | cli.

    A database record gets the new block and an ``explanation_record_events`` row holding the
    new and the previous block (nothing is lost). A file record is rewritten atomically, the
    rest left untouched, key order kept, in the file's own layout (``dumps_like``).
    """
    mv: Dict[str, Any] = {"verified": bool(verified)}
    if minutes is not None:
        mv["minutes"] = minutes
    mv["changes"] = changes or ""
    if notes:
        mv["notes"] = notes
    if signoff:
        mv["protocol"] = signoff.get("protocol")
        mv["signoff"] = signoff
    mv["entered_by"] = entered_by if entered_by in ENTERED_BY else "cli"
    mv["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    e = None if directory is not None else entry(label)
    if e is not None and e.storage == "db":
        return _write_db_verification(e, mv)
    p = record_path(label, directory) if e is None else Path(e.path)
    rec = _read(p)
    rec["mike_verification"] = mv
    _write_json(p, rec)
    if e is not None and e.label != e.file_label:
        rec = dict(rec, label=e.label)
    return rec


def _write_db_verification(e: Entry, mv: Dict[str, Any]) -> Dict[str, Any]:
    from ..db import db
    from ..db.models import ExplanationRecord as R, ExplanationRecordEvent as Ev
    with db.session_scope() as s:
        row = s.query(R).filter(R.label == e.label).one()
        prev = row.mike_verification
        rec = dict(row.record)
        rec["mike_verification"] = mv
        row.record = rec
        row.mike_verification = mv
        s.add(Ev(record_id=row.id, kind="verification", entered_by=mv["entered_by"], payload=mv, previous=prev))
    if e.label != e.file_label:
        rec = dict(rec, label=e.label)
    return rec


def verification_history(label: str) -> List[Dict[str, Any]]:
    """The record's events (database records only; empty for a file record)."""
    from ..db.models import ExplanationRecord as R, ExplanationRecordEvent as Ev

    def q(s):
        row = s.query(R.id).filter(R.label == label).one_or_none()
        if row is None:
            return []
        evs = s.query(Ev).filter(Ev.record_id == row[0]).order_by(Ev.id).all()
        return [{"kind": v.kind, "entered_by": v.entered_by, "payload": v.payload, "previous": v.previous,
                 "at": v.created_at.isoformat() if v.created_at else None} for v in evs]
    return _with_db(q, []) or []


def entry_dict(e: Optional[Entry]) -> Optional[Dict[str, Any]]:
    return asdict(e) if e is not None else None
