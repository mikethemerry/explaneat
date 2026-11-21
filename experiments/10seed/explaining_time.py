"""Explaining time and annotation counts of an explanation, from the database (read-only).

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
        uv run python experiments/10seed/explaining_time.py --dir explanations_v4
    # -> experiments/10seed/explaining_time_v4.csv (one row per record in explanations_v4/)

Replaces the explaining agent's own start/end ("claude_minutes") with a time read from what the
tools wrote while the agent worked. Every MCP write to an explanation is timestamped:

- ``explanations.created_at``: the explanation row, created by the agent's first operation;
- each operation's ``created_at`` (in the stored stream; naive timestamps are UTC);
- each evidence record's ``timestamp`` (``add_evidence``), ``amended_at`` and
  ``history[].replaced_at`` (``amend_evidence``) and ``deleted_at`` (``delete_evidence``);
- ``evidence_log.created_at`` for every provenance-stamped query and evidence tool call.

**Definition.** Explaining minutes = (last of those writes) - ``explanations.created_at``. It is a
lower bound on the agent's wall-clock time: reading the network before the first write and the
closing recheck and record writing after the last one are not in it.

Left out, because other writers made them after the explanation was finished:
``provenance_fixed_at`` (``experiments/10seed/fix_amended_provenance.py``), evidence saved
through the explorer API (categories ``analytical_methods``, ``visualizations``,
``counterfactuals``, ``other_evidence``; e.g. a review snapshot), and ``explanations.updated_at``
(bumped by any later edit). ``evidence_log`` entries before the explanation was created (a
runner's ``compute_performance`` call) are counted in ``n_log_before_start`` and left out.

Annotation counts are read from the same stream: an ``annotate`` operation with
``child_annotation_ids`` is a composition, any other is a leaf (as ``check_explanation.py``).

Each record (the agent's start/end and annotation counts) is read with
``explanation_records.load_record_file``: the database's ``explanation_records`` row when it holds
the record (``$EXPLANEAT_RECORDS_DATABASE_URL``, else ``DATABASE_URL``), else the record file.

Run it twice and compare: the output is byte-identical (no randomness; read-only).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# evidence categories written by the explorer API (explaneat/api/schemas.py), not by the agent's tools
API_CATEGORIES = ("analytical_methods", "visualizations", "counterfactuals", "other_evidence")
FIELDS = ["label", "genome_id", "explanation_id", "database", "record_start", "record_end", "record_minutes",
          "db_start", "db_end", "db_minutes", "difference_minutes", "n_operations", "n_evidence_records",
          "n_log_entries", "n_log_before_start", "n_excluded_writes", "n_writes_after_record_end",
          "n_leaf_db", "n_composition_db", "n_leaf_record", "n_composition_record"]


def ts(s) -> datetime:
    """An ISO timestamp (or datetime) as an aware UTC datetime; naive values are UTC."""
    d = s if isinstance(s, datetime) else datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _record_times(rec, out):
    for k in ("timestamp", "amended_at"):
        if rec.get(k):
            out.append(ts(rec[k]))
    for h in rec.get("history") or []:
        if h.get("replaced_at"):
            out.append(ts(h["replaced_at"]))


def stream_writes(operations):
    """(included, n_excluded, n_evidence_records): timestamps of the agent's writes in an operations stream."""
    inc, n_excl, n_rec = [], 0, 0
    for op in operations:
        if op.get("created_at"):
            inc.append(ts(op["created_at"]))
        ev = (op.get("params") or {}).get("evidence")
        if not isinstance(ev, dict):
            continue
        for r in ev.get("records") or []:
            n_rec += 1
            _record_times(r, inc)
            n_excl += bool(r.get("provenance_fixed_at"))
        for d in ev.get("deleted") or []:
            if d.get("deleted_at"):
                inc.append(ts(d["deleted_at"]))
            _record_times(d.get("record") or {}, inc)
        n_excl += sum(len(ev.get(c) or []) for c in API_CATEGORIES)
    return inc, n_excl, n_rec


def annotation_counts(operations):
    """(leaf, composition) annotate operations in the stream."""
    c = Counter("composition" if (o.get("params") or {}).get("child_annotation_ids") else "leaf"
                for o in operations if o.get("type") == "annotate")
    return c["leaf"], c["composition"]


def explaining_window(session, genome_id):
    """The explaining window of the genome's one explanation (read with the given session)."""
    from sqlalchemy import select
    from explaneat.db.models import EvidenceLogEntry, Explanation
    gid = uuid.UUID(str(genome_id))
    ex = session.execute(select(Explanation).where(Explanation.genome_id == gid)).scalars().all()
    assert len(ex) == 1, (genome_id, len(ex))
    ex = ex[0]
    ops = list(ex.operations or [])
    start = ts(ex.created_at)
    log = [ts(t) for t in session.execute(select(EvidenceLogEntry.created_at)
                                          .where(EvidenceLogEntry.genome_id == gid)).scalars()]
    inc, n_excl, n_rec = stream_writes(ops)
    inc += [t for t in log if t >= start]
    end = max(inc)
    leaf, comp = annotation_counts(ops)
    return {"explanation_id": str(ex.id), "start": start, "end": end,
            "minutes": round((end - start).total_seconds() / 60, 1),
            "n_operations": len(ops), "n_evidence_records": n_rec, "n_log_entries": len(log),
            "n_log_before_start": sum(t < start for t in log), "n_excluded_writes": n_excl,
            "writes": inc, "leaf": leaf, "composition": comp}


def iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def row(label, rec, w, database):
    rs, re_ = ts(rec["start"]), ts(rec["end"])
    rm = round((re_ - rs).total_seconds() / 60, 1)
    return {"label": label, "genome_id": rec["genome_id"], "explanation_id": w["explanation_id"],
            "database": database, "record_start": iso(rs), "record_end": iso(re_), "record_minutes": rm,
            "db_start": iso(w["start"]), "db_end": iso(w["end"]), "db_minutes": w["minutes"],
            "difference_minutes": round(rm - w["minutes"], 1), "n_operations": w["n_operations"],
            "n_evidence_records": w["n_evidence_records"], "n_log_entries": w["n_log_entries"],
            "n_log_before_start": w["n_log_before_start"], "n_excluded_writes": w["n_excluded_writes"],
            "n_writes_after_record_end": sum(t > re_ for t in w["writes"]),
            "n_leaf_db": w["leaf"], "n_composition_db": w["composition"],
            "n_leaf_record": rec["n_leaf_annotations"], "n_composition_record": rec["n_composition_annotations"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="explanations_v4", help="records folder under experiments/10seed/")
    ap.add_argument("--out", default=None, help="default: experiments/10seed/explaining_time<suffix>.csv")
    a = ap.parse_args()
    url = os.environ.get("DATABASE_URL")
    assert url, "set DATABASE_URL to the records' database (explaneat_v4 for explanations_v4)"
    database = url.rsplit("/", 1)[-1]
    # server-side read-only for every transaction of this process
    os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO))
    import compute_runs as cr
    cr.db.init_db(url)
    rec_dir = HERE / a.dir
    suffix = a.dir.replace("explanations", "")
    out = Path(a.out) if a.out else HERE / f"explaining_time{suffix}.csv"
    rows = []
    from explaneat.analysis.explanation_records import load_record_file
    for p in sorted(rec_dir.glob("*.json")):
        rec = load_record_file(p)   # the database's record when it holds it, else the file
        assert rec.get("database", database) == database, (p.name, rec.get("database"), database)
        with cr.ReadOnlySession() as s:
            w = explaining_window(s, rec["genome_id"])
        rows.append(row(p.stem, rec, w, database))
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    bad = [r["label"] for r in rows if (r["n_leaf_db"], r["n_composition_db"]) != (r["n_leaf_record"], r["n_composition_record"])]
    print(f"{out.relative_to(REPO)}: {len(rows)} rows; annotation counts differ from the record for: {bad or 'none'}")


if __name__ == "__main__":
    main()
