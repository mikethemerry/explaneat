"""Collect one v4 explanation from the SQLite evaluation database (read only).

    DATABASE_URL=sqlite:///... uv run python experiments/authoring_v4/export_v4.py <label> <genome_id> <record.json>

Writes ``experiments/authoring_v4/v4/<label>/``: ``operations.json`` (the stored operations),
``record.json`` (the agent's final report) and ``evidence_log.json`` (every stored query).
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from explaneat.analysis.provenance import get_entry, list_entries  # noqa: E402
from explaneat.verify.model import load_db_operations  # noqa: E402
from mcp_server.server import get_db  # noqa: E402


def main(label, gid, record_path):
    out = HERE / "v4" / label
    out.mkdir(parents=True, exist_ok=True)
    ops = load_db_operations(gid)
    (out / "operations.json").write_text(json.dumps(ops, indent=1, default=str) + "\n")
    shutil.copy(record_path, out / "record.json")
    with get_db().session_scope() as s:
        log = [get_entry(s, gid, e["id"]) for e in list_entries(s, gid)]
    (out / "evidence_log.json").write_text(json.dumps(log, default=str) + "\n")
    print(label, len(ops), "operations,", len(log), "queries")


if __name__ == "__main__":
    main(*sys.argv[1:4])
