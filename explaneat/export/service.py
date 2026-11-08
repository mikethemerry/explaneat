"""Export jobs for the API (and anything else that wants a manifest of an export on disk).

Exports are written under ``$EXPLANEAT_HOME/exports`` (``settings.data_dir()``), never in the
repository:

* one genome: ``exports/<genome_id>/<content-hash>/`` (re-exporting unchanged content
  replaces the same directory);
* one experiment: ``exports/experiment-<experiment_id>/<UTC timestamp>/<label>/`` plus
  ``index.json`` / ``index.tex``.

Each directory gets a ``manifest.json``. The export id used by the download route is the
directory path relative to the exports root, with ``/`` written as ``~``.

Exports contain the model, the explanation and aggregate metrics only; evidence payloads are
scrubbed of long lists (``collect.scrub_payload``), and no data rows are written.
"""

from __future__ import annotations

import datetime as dt
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import settings


def exports_root() -> Path:
    return settings.data_dir() / "exports"


def export_id_for(path: Path) -> str:
    return str(path.resolve().relative_to(exports_root().resolve())).replace("/", "~").replace("\\", "~")


def path_for(export_id: str) -> Path:
    """The export directory for an id; refuses anything outside the exports root."""
    root = exports_root().resolve()
    p = (root / export_id.replace("~", "/")).resolve()
    if root not in p.parents:
        raise ValueError("invalid export id")
    return p


def file_for(export_id: str, name: str) -> Path:
    d = path_for(export_id)
    p = (d / name).resolve()
    if d not in p.parents or not p.is_file():
        raise FileNotFoundError(name)
    return p


def _files(d: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and "build" not in p.relative_to(d).parts and p.name != "manifest.json":
            out.append({"name": str(p.relative_to(d)).replace("\\", "/"), "size": p.stat().st_size})
    return out


def _manifest(d: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
    import json
    eid = export_id_for(d)
    m = {"export_id": eid, "path": str(d), **extra,
         "files": [{**f, "url": f"/api/exports/{eid}/{f['name']}"} for f in _files(d)]}
    (d / "manifest.json").write_text(json.dumps(m, indent=1, sort_keys=True, default=str) + "\n",
                                     encoding="utf-8")
    return m


def export_genome(genome_id: str, formats: List[str], review_marks: bool = True, db=None,
                  log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    from .collect import collect
    from .pack import export_doc, index_entry
    doc = collect(genome_id, db=db)
    tmp = exports_root() / f".tmp-{uuid.uuid4().hex}"
    try:
        res = export_doc(doc, tmp, formats, review_marks=review_marks, log=log)
        dest = exports_root() / doc["provenance"]["genome"]["id"] / doc["provenance"]["content_hash"]["value"]
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / doc["label"]), str(dest))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _manifest(dest, {"kind": "genome", "label": doc["label"],
                            "genome_id": doc["provenance"]["genome"]["id"],
                            "content_hash": doc["provenance"]["content_hash"]["value"],
                            "formats": res.get("formats", {}), "summary": index_entry(doc, ".")})


def export_experiment(experiment_id: str, formats: List[str], review_marks: bool = True, db=None,
                      log: Callable[[str], None] = lambda s: None,
                      progress: Callable[[int, int], None] = lambda i, n: None) -> Dict[str, Any]:
    from .collect import collect, explained_genomes_in_experiment
    from .pack import export_doc, index_entry, write_index
    if db is None:
        from mcp_server.server import get_db
        db = get_db()
    with db.session_scope() as s:
        gids = explained_genomes_in_experiment(s, experiment_id)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = exports_root() / f"experiment-{experiment_id}" / stamp
    out.mkdir(parents=True, exist_ok=True)
    entries, failures, formats_by = [], [], {}
    for i, g in enumerate(gids):
        progress(i, len(gids))
        try:
            doc = collect(g, db=db)
            res = export_doc(doc, out, formats, review_marks=review_marks, log=log)
            entries.append(index_entry(doc, doc["label"]))
            formats_by[doc["label"]] = res.get("formats", {})
        except Exception as e:  # keep going; report per genome
            failures.append({"genome_id": g, "error": f"{type(e).__name__}: {e}"})
    progress(len(gids), len(gids))
    write_index(entries, out, stamp)
    return _manifest(out, {"kind": "experiment", "experiment_id": experiment_id,
                           "n_genomes": len(gids), "exported": len(entries), "failures": failures,
                           "formats": formats_by})


# ----------------------------------------------------------------------------- jobs
_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def start_job(kind: str, fn: Callable[..., Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    jid = uuid.uuid4().hex[:12]
    job = {"job_id": jid, "kind": kind, "status": "running", "progress": None, "log": [],
           "manifest": None, "error": None,
           "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with _LOCK:
        _JOBS[jid] = job

    def log(s):
        job["log"] = (job["log"] + [s])[-200:]

    def progress(i, n):
        job["progress"] = {"done": i, "total": n}

    def run():
        try:
            extra = {"progress": progress} if kind == "experiment" else {}
            job["manifest"] = fn(log=log, **extra, **kwargs)
            job["status"] = "done"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"
            log(traceback.format_exc()[-2000:])
    threading.Thread(target=run, daemon=True, name=f"export-{jid}").start()
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)
