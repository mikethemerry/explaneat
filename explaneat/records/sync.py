"""Import explanation records from JSON files into ``explanation_records``, and export them back.

Import is idempotent: a file already imported (same ``source_path``) whose content matches the
database is left alone; one that differs is reported and skipped unless ``force`` (then the
database takes the file's content, and the event history keeps the previous verification). A new
record keeps its label unless the database already has that label for another file; then it is
``<study>.<label>`` (the same rule as the file catalogue). Each record is matched to the current
explanation of its genome (the latest ``explanations`` row), when there is one.

Export writes every database record (with its current ``mike_verification``) to its file of
origin, in that file's own layout (indent, trailing newline; key order kept), and leaves a file
alone when its content already matches. The committed record files are this export: the thesis generators read
them, so they stay reproducible from committed files.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..analysis import explanation_records as er


def _full(rec: Dict[str, Any], mv: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(rec)
    if mv is not None:
        out["mike_verification"] = mv
    return out


def _current_explanation(session, genome_id: str):
    from ..db.models import Explanation
    try:
        gid = uuid.UUID(str(genome_id))
    except (ValueError, TypeError):
        return None
    row = (session.query(Explanation.id).filter(Explanation.genome_id == gid)
           .order_by(Explanation.updated_at.desc(), Explanation.created_at.desc()).first())
    return row[0] if row else None


def import_records(session, directories: Optional[Sequence[Path]] = None, dry_run: bool = False,
                   force: bool = False) -> List[Dict[str, Any]]:
    """Import the records of ``directories`` (default: every folder the catalogue searches).
    Returns one row per record file: ``{label, study, source, action, detail}`` with action
    ``new`` / ``unchanged`` / ``differs`` (skipped) / ``updated`` (forced) / ``clash``."""
    from ..db.models import ExplanationRecord as R, ExplanationRecordEvent as Ev
    dirs = [Path(d) for d in directories] if directories is not None else er.records_dirs(fresh=True)
    rows = {r.source_path: r for r in session.query(R).all()}
    taken = {r.label: r.source_path for r in rows.values()}
    out: List[Dict[str, Any]] = []
    for fe in er.catalog(directories=dirs, use_db=False):
        rec = er._read(Path(fe.path))
        report = er._read(Path(fe.report_path)) if fe.report_path else None
        if report == rec:
            report = None
        mv = rec.get("mike_verification")
        item = {"label": fe.label, "study": fe.study, "source": fe.source, "genome_id": fe.genome_id}
        row = rows.get(fe.source)
        if row is not None:
            item["label"] = row.label
            same = _full(row.record, row.mike_verification) == rec and (row.report or None) == report
            if same:
                if row.explanation_id is None and not dry_run:
                    row.explanation_id = _current_explanation(session, fe.genome_id)
                out.append(dict(item, action="unchanged"))
                continue
            if not force:
                why = "the verification differs" if {k: v for k, v in _full(row.record, None).items()
                                                     if k != "mike_verification"} == \
                    {k: v for k, v in rec.items() if k != "mike_verification"} else "the record differs"
                out.append(dict(item, action="differs",
                                detail=f"{why}; kept the database's (export it, or import --force to take the file's)"))
                continue
            if not dry_run:
                prev = row.mike_verification
                row.record, row.report, row.mike_verification = rec, report, mv
                row.genome_id, row.content_hash = fe.genome_id, er.content_hash(rec)
                row.explanation_id = _current_explanation(session, fe.genome_id) or row.explanation_id
                session.add(Ev(record_id=row.id, kind="reimport", entered_by="import", payload=mv, previous=prev))
            out.append(dict(item, action="updated"))
            continue
        label = fe.file_label
        if label in taken:
            label = f"{fe.study}.{fe.file_label}"
        if label in taken:
            out.append(dict(item, action="clash", detail=f"labels {fe.file_label!r} and {label!r} are taken"))
            continue
        item["label"] = label
        taken[label] = fe.source
        if not dry_run:
            row = R(id=uuid.uuid4(), label=label, study=fe.study, genome_id=fe.genome_id,
                    explanation_id=_current_explanation(session, fe.genome_id), records_dir=fe.records_dir,
                    source_path=fe.source, record=rec, report=report, mike_verification=mv,
                    content_hash=er.content_hash(rec))
            session.add(row)
            session.flush()
            session.add(Ev(record_id=row.id, kind="import", entered_by="import", payload=mv, previous=None))
        out.append(dict(item, action="new"))
    return out


def export_records(session, out_root: Optional[Path] = None, labels: Optional[Sequence[str]] = None,
                   dry_run: bool = False) -> List[Dict[str, Any]]:
    """Write the database's records to their files (under ``out_root`` instead of the repository
    when given). ``{label, path, action}`` with action ``written`` / ``unchanged``."""
    from ..db.models import ExplanationRecord as R
    root = Path(out_root) if out_root is not None else er.REPO_ROOT
    q = session.query(R).order_by(R.label)
    if labels:
        q = q.filter(R.label.in_(list(labels)))
    out = []
    for row in q.all():
        src = row.source_path or f"{row.records_dir or row.study}/{row.record.get('label') or row.label}.json"
        p = root / src
        doc = _full(row.record, row.mike_verification)
        cur = None
        if p.is_file():
            try:
                cur = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                cur = None
        if cur == doc:
            out.append({"label": row.label, "path": str(p), "action": "unchanged"})
            continue
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            er._write_json(p, doc)
        out.append({"label": row.label, "path": str(p), "action": "written"})
    return out
