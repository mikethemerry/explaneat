"""Experiment 1 appendix networks: check each committed plain-language layer against the database (read-only).

    uv run python thesis/ch7/exp1/extract_networks.py        # -> data/network_figs.json

The appendix figures (networks.py) are drawn from committed plain-language layers (no database). This
extractor checks, for every Experiment 1 network, that its layer is the genome's current explanation:
  * the genome has exactly one explanation in explaneat_dev, and it is the layer's explanation_id;
  * the explained model state rebuilt from the stored operations (mcp_server.helpers.build_engine, the code
    the layer builder uses) has exactly the layer's connections (from, to) with the same weights, and the
    same hidden units;
and records the trained phenotype's size (experiments/10seed/compute_runs.structure, the definitions of the
Experiment 1 networks table: inputs used, hidden nodes, enabled connections), which networks.py asserts its
raw figure against. It also records a sha256 of the stored operations, so a later change to an explanation
shows up as a diff of data/network_figs.json.

Every transaction is READ ONLY (compute_runs.ReadOnlySession, plus PGOPTIONS default_transaction_read_only=on
for the library's own sessions). Run it twice: data/network_figs.json is byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E = REPO / "experiments"
for p in (str(REPO), str(E / "10seed"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/explaneat_dev")
os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()

from networks import CASES, layer_path  # noqa: E402  (the network list and each one's layer file)

OUT = HERE / "data" / "network_figs.json"
WEIGHT_TOL = 5e-6   # the layer stores weights to 6 significant figures


def check(cr, label, layer):
    from explaneat.db.models import Dataset, Explanation
    from mcp_server.helpers import build_engine
    from sqlalchemy import select
    gid = layer["genome_id"]
    with cr.ReadOnlySession() as s:
        g = s.get(cr.Genome, uuid.UUID(gid))
        exp = g.population.experiment
        cfg = cr.load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = g.to_neat_genome(cfg)
        names = list(s.get(Dataset, exp.dataset_id).feature_names)
        exs = s.execute(select(Explanation).where(Explanation.genome_id == g.id)).scalars().all()
        assert len(exs) == 1, (label, len(exs))
        ex_id = str(exs[0].id)
        ops = list(exs[0].operations or [])
        experiment = exp.name
        state = build_engine(s, gid).current_state
    assert ex_id == layer["explanation_id"], (label, ex_id, layer["explanation_id"])
    db_edges = {(c.from_node, c.to_node): float(c.weight) for c in state.connections if c.enabled}
    ly_edges = {(e["from"], e["to"]): float(e["weight"]) for e in layer["graph"]["edges"]}
    assert set(db_edges) == set(ly_edges), (label, sorted(set(db_edges) ^ set(ly_edges))[:10])
    werr = max(abs(db_edges[k] - ly_edges[k]) / max(1.0, abs(db_edges[k])) for k in db_edges)
    assert werr < WEIGHT_TOL, (label, werr)
    db_hidden = sorted(n.id for n in state.nodes if n.type.value == "hidden")
    ly_hidden = sorted(n["id"] for n in layer["graph"]["nodes"] if n["kind"] == "hidden")
    assert db_hidden == ly_hidden, (label, db_hidden, ly_hidden)
    st = cr.structure(genome, cfg, names)
    return {"label": label, "genome": gid, "experiment": experiment, "explanation_id": ex_id,
            "n_operations": len(ops),
            "operations_sha256": hashlib.sha256(json.dumps(ops, sort_keys=True).encode()).hexdigest(),
            "layer": str(layer_path(label).relative_to(REPO)),
            "model_state_matches_layer": True, "n_model_state_connections": len(db_edges),
            "n_inputs": st["n_inputs"], "inputs_connected": st["inputs_connected"],
            "hidden": st["hidden_nodes"], "conns": st["phenotype_connections_enabled"]}


def main():
    import torch
    torch.set_num_threads(1)
    import compute_runs as cr
    cr.db.init_db(os.environ["DATABASE_URL"])
    out = {"database": os.environ["DATABASE_URL"].rsplit("/", 1)[-1] + " (read-only)", "networks": {}}
    for case in CASES:
        label = case["label"]
        if label == "xor":
            continue
        layer = json.loads(layer_path(label).read_text())
        out["networks"][label] = check(cr, label, layer)
        print("checked", label, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
