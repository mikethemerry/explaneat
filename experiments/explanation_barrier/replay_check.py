"""Replay stored explanations read only and hash the resulting model states (split-id change check).

    DATABASE_URL=<url> uv run python experiments/explanation_barrier/replay_check.py <out.json> [--limit N]

Run once with the current code and once with the previous code on the path (PYTHONPATH pointing
at an export of ``origin/main`` before the change); identical hashes mean identical model states.
Reads only: a read-only transaction; nothing is written to the database.
"""
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import text

from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.model_state import ModelStateEngine
from explaneat.db.models import Explanation
from mcp_server.helpers import load_genome_and_config
from mcp_server.server import get_db


def canon(state):
    nodes = sorted((n.id, n.type.value, n.bias, n.activation, n.response, n.aggregation, n.display_name)
                   for n in state.nodes)
    conns = sorted((c.from_node, c.to_node, c.weight, c.enabled) for c in state.connections)
    return json.dumps({"nodes": nodes, "conns": conns, "in": list(state.input_node_ids),
                       "out": list(state.output_node_ids)}, default=str, sort_keys=True)


def main(out, limit=None):
    db = get_db()
    res = {}
    with db.session_scope() as s:
        s.execute(text("SET TRANSACTION READ ONLY")) if db.engine.dialect.name == "postgresql" else None
        rows = s.query(Explanation).all()
        rows = [r for r in rows if r.operations]
        rows.sort(key=lambda r: str(r.genome_id))
        for r in rows[:limit]:
            gid = str(r.genome_id)
            try:
                genome, config, _ = load_genome_and_config(s, gid)
                eng = ModelStateEngine(ExplaNEAT(genome, config).get_phenotype_network())
                eng.load_operations({"operations": r.operations})
                st = eng.current_state
                ann = sorted((a.name, tuple(sorted(a.subgraph_nodes)), tuple(sorted(a.entry_nodes)),
                              tuple(sorted(a.exit_nodes))) for a in eng.annotations) \
                    if hasattr(eng.annotations[0] if eng.annotations else None, "subgraph_nodes") or not eng.annotations \
                    else len(eng.annotations)
                h = hashlib.sha256((canon(st) + json.dumps(ann, default=str)).encode()).hexdigest()
                n_split = sum(1 for o in r.operations if o.get("type") in ("split_node", "consolidate_node"))
                res[gid] = {"hash": h, "n_ops": len(r.operations), "n_split_ops": n_split, "n_nodes": len(st.nodes)}
            except Exception as e:  # noqa: BLE001
                res[gid] = {"error": f"{type(e).__name__}: {e}"[:300]}
        s.rollback()
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(len(res), "replayed;", sum(1 for v in res.values() if "error" in v), "errors")


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(sys.argv[1], lim)
