"""Model-state graphs for the experiment book's drawers (read only).

For every explanation in xdata/ (and each of its versions), replay the explanation's stored
operations on the genome's phenotype with ModelStateEngine and write the resulting structure
(nodes, edges, annotation hierarchy) to graphs/<id>__<version>.json.

Run with the repository's environment on a checkout of origin/main, one database at a time:
  DATABASE_URL=postgresql://localhost/explaneat_v4 PGOPTIONS='-c default_transaction_read_only=on' \
    /Users/mike/dev/explaneat/.venv/bin/python graphs.py <checkout> explaneat_v4
"""
import json
import re
import sys
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path(sys.argv[1]).resolve()
DBNAME = sys.argv[2]
sys.path.insert(0, str(MAIN))

from sqlalchemy import text  # noqa: E402

from mcp_server.server import get_db  # noqa: E402
from mcp_server.helpers import load_genome_and_config  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.verify.model import build_engine_from  # noqa: E402
from explaneat.db.models import Dataset, Experiment, Explanation, Genome, Population  # noqa: E402

OUT = HERE / "graphs"
OUT.mkdir(exist_ok=True)


def r6(x):
    return None if x is None else float(f"{float(x):.6g}")


def feature_names(s, exp, ds):
    names = list(ds.feature_names or [])
    return names


def build(s, genome_id):
    g = s.get(Genome, uuid.UUID(genome_id))
    pop = s.get(Population, g.population_id)
    exp = s.get(Experiment, pop.experiment_id)
    ds = s.get(Dataset, exp.dataset_id)
    neat_genome, config, _ = load_genome_and_config(s, genome_id)
    phen = ExplaNEAT(neat_genome, config).get_phenotype_network()
    ex = s.query(Explanation).filter(Explanation.genome_id == uuid.UUID(genome_id)).first()
    ops = json.loads(json.dumps(list(ex.operations or []), default=str)) if ex else []
    eng = build_engine_from(phen, ops)
    st = eng.current_state
    fnames = feature_names(s, exp, ds)
    in_ids = [str(i) for i in phen.input_node_ids]
    base_name = {}
    if len(fnames) == len(in_ids):
        for i, nid in enumerate(in_ids):
            base_name[nid] = fnames[i]
    renames = {}
    for o in ops:
        if o.get("type") == "rename_node":
            p = o.get("params") or {}
            if p.get("display_name"):
                renames[str(p.get("node_id"))] = p["display_name"]
    phen_ids = {str(n.id) for n in phen.nodes}
    nodes = []
    for n in st.nodes:
        nid = str(n.id)
        t = n.type.value if hasattr(n.type, "value") else str(n.type)
        kind = t
        base, sep, tok = nid.partition("_")
        if nid.startswith("identity_"):
            kind = "identity"
        elif sep and nid not in phen_ids and base in phen_ids | {str(i) for i in in_ids}:
            kind = "split_input" if t == "input" else "split"
        name = n.display_name or renames.get(nid)
        if not name and kind in ("input", "split_input"):
            b = renames.get(base) or base_name.get(base)
            name = (b + ("_" + tok if sep else "")) if b else None
        nodes.append([nid, kind, name, n.activation, r6(n.bias)])
    edges = []
    for c in st.connections:
        edges.append([str(c.from_node), str(c.to_node), r6(c.weight), 1 if c.enabled else 0])
    anns = []
    for a in eng.annotations:
        anns.append({
            "id": a.annotation_id, "name": a.name, "parent": a.parent_annotation_id,
            "nodes": [str(x) for x in a.subgraph_nodes], "entries": [str(x) for x in a.entry_nodes],
            "exits": [str(x) for x in a.exit_nodes],
        })
    # parent ids in the engine are annotation ids or names; normalise to names
    by_id = {a["id"]: a["name"] for a in anns}
    for a in anns:
        if a["parent"] in by_id:
            a["parent"] = by_id[a["parent"]]
    return {"nodes": nodes, "edges": edges, "anns": anns,
            "inputs": [str(i) for i in st.input_node_ids], "outputs": [str(i) for i in st.output_node_ids],
            "explanation_id": str(ex.id) if ex else None, "experiment": exp.name, "dataset": ds.name}


def main():
    db = get_db()
    todo = []
    for p in sorted((HERE / "xdata").glob("*.json")):
        d = json.loads(p.read_text())
        for vn, v in d["versions"].items():
            if v.get("db") == DBNAME and v.get("genome_id"):
                todo.append((d["id"], vn, v["genome_id"]))
    ok = 0
    for xid, vn, gid in todo:
        try:
            with db.session_scope() as s:
                s.execute(text("SET TRANSACTION READ ONLY"))
                g = build(s, gid)
            (OUT / f"{xid}__{vn}.json").write_text(json.dumps(g, separators=(",", ":")))
            ok += 1
            print(f"{xid}__{vn}: {len(g['nodes'])} nodes, {len(g['edges'])} edges, {len(g['anns'])} parts", flush=True)
        except Exception as e:
            print(f"FAILED {xid}__{vn}: {e!r}", flush=True)
            traceback.print_exc()
    print(f"{ok}/{len(todo)} on {DBNAME}")


if __name__ == "__main__":
    main()
