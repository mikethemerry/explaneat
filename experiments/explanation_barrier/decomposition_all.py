"""The coverage report's decomposition block for every ladder explanation (read only).

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/explanation_barrier/decomposition_all.py

Writes logs/decomposition.json: {label: decomposition block}.
"""
import json
from pathlib import Path

from explaneat.analysis.coverage_report import coverage_report
from mcp_server.helpers import build_engine
from mcp_server.server import get_db

HERE = Path(__file__).resolve().parent
out = {}
db = get_db()
for p in sorted((HERE / "explanations").glob("*.json")):
    rec = json.loads(p.read_text())
    with db.session_scope() as s:
        eng = build_engine(s, rec["genome_id"])
        d = coverage_report(eng.current_state, eng.annotations)["decomposition"]
    out[p.stem] = d
    print(p.stem, d["hidden_units"], d["granularity"], d["largest_part"]["hidden_units"], d["units_only_in_root"],
          "|", "; ".join(d["warnings"])[:160])
(HERE / "logs" / "decomposition.json").write_text(json.dumps(out, indent=1) + "\n")
