"""Correct the stale provenance ids left by amend_evidence(query_id=...) in the v4 run.

Before the fix, ``amend_evidence`` replaced an evidence record's payload but left its
``provenance`` list on the old query id, so ``recheck`` kept flagging the old (pre-structural-op)
query: the 12 false needs_review items in V4_RUN.md. The fixed tool recomputes the list with
``mcp_server.tools.snapshots.record_provenance``. This applies that same function to the
amended records (those with a ``history``) whose list holds an id the record no longer uses.
Nothing else is touched; the old list is kept on the record as ``provenance_before_fix``.

    DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python experiments/10seed/fix_amended_provenance.py [--write]
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(write: bool):
    url = os.environ.get("DATABASE_URL", "")
    if not url.endswith("/explaneat_v4"):
        raise SystemExit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (this fixes the v4 run only)")
    from sqlalchemy.orm.attributes import flag_modified
    from explaneat.db.explanation_lock import lock_explanation
    from mcp_server.server import get_db
    from mcp_server.tools.snapshots import record_provenance
    labels = {json.loads(p.read_text())["genome_id"]: p.stem for p in (HERE / "explanations_v4").glob("*.json")}
    db = get_db()
    total = 0
    for gid, lab in sorted(labels.items(), key=lambda kv: kv[1]):
        with db.session_scope() as session:
            import uuid
            ex = lock_explanation(session, uuid.UUID(gid))
            if ex is None or not ex.operations:
                continue
            ops = list(ex.operations)
            changed = []
            for op in ops:
                if op.get("type") != "annotate":
                    continue
                prm = op.get("params") or {}
                for j, r in enumerate((prm.get("evidence") or {}).get("records") or []):
                    if not r.get("history"):
                        continue
                    old = list(r.get("provenance") or [])
                    new = record_provenance(r)
                    if set(old) - set(new):           # an id the record no longer uses
                        changed.append((prm.get("name"), j, old, new))
                        if write:
                            r["provenance_before_fix"] = old
                            r["provenance"] = new
                            r["provenance_fixed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for name, j, old, new in changed:
                print(f"{lab} {name}.evidence[{j}]: {old} -> {new}")
            total += len(changed)
            if write and changed:
                ex.operations = ops
                flag_modified(ex, "operations")
                session.flush()
    print(f"{total} record(s) {'corrected' if write else 'to correct (dry run; --write to apply)'}")


if __name__ == "__main__":
    main("--write" in sys.argv[1:])
