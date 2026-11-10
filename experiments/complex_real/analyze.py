"""Post-import parity check and structure statistics for an imported complex-real run
(adapted from experiments/mnist/analyze.py; binary tasks only).

1. Parity: the imported best genome loaded from the DB as the MCP tools load it
   (load_genome_and_config -> ExplaNEAT phenotype -> StructureNetwork), on the DB dataset and
   split (scaler applied by load_split_data), must reproduce the in-memory train/test metrics
   (NeuralNeat), and the DB genome's NeuralNeat output must be bit-equal to the in-memory one.
2. Structure: hidden nodes, depth, connections, inputs used, and the share of hidden ReLU units
   that switch / are always on / dead on the training rows (strict and 1% tolerance).

    uv run python experiments/complex_real/analyze.py krk-default-s0
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402
from mcp_server.helpers import load_genome_and_config, load_split_data  # noqa: E402
from mcp_server.server import get_db  # noqa: E402
from run_seed import RUNS_DIR, load_config  # noqa: E402

TOL = 0.01


def metrics(p, y):
    p = p.ravel()
    eps = 1e-12
    return {"auc": float(roc_auc_score(y, p)), "acc": float(((p >= 0.5) == (y == 1)).mean()),
            "bce": float(-np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))}


def structure_stats(pheno, sn, X_train):
    ids_in, ids_out = set(pheno.input_node_ids), set(pheno.output_node_ids)
    valid = set(sn.node_info)
    conns = [c for c in pheno.connections if c.enabled and c.from_node in valid and c.to_node in valid]
    hidden = [n for n in valid if n not in ids_in and n not in ids_out]
    sn.forward(torch.as_tensor(X_train, dtype=torch.float64))
    frac = {h: float((sn.get_node_activation(h) > 0).mean()) for h in hidden}
    strict = Counter("dead" if f == 0 else "always_on" if f == 1 else "switching" for f in frac.values())
    loose = Counter("dead" if f <= TOL else "always_on" if f >= 1 - TOL else "switching" for f in frac.values())
    n_h = max(len(hidden), 1)
    return dict(
        hidden_nodes=len(hidden), depth_layers=int(max(i["depth"] for i in sn.node_info.values())),
        connections=len(conns),
        direct_input_output=sum(1 for c in conns if c.from_node in ids_in and c.to_node in ids_out),
        inputs_used=len({c.from_node for c in conns if c.from_node in ids_in}), inputs_total=len(ids_in),
        hidden_activity_strict={k: strict.get(k, 0) for k in ("switching", "always_on", "dead")},
        switching_share_strict=strict.get("switching", 0) / n_h if hidden else None,
        hidden_activity_1pct={k: loose.get(k, 0) for k in ("switching", "always_on", "dead")},
        switching_share_1pct=loose.get("switching", 0) / n_h if hidden else None,
        hidden_active_fraction=dict(sorted(frac.items(), key=lambda kv: kv[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    args = ap.parse_args()
    imp = json.loads((HERE / "logs" / f"{args.name}.import.json").read_text())
    summ = json.loads((HERE / "logs" / f"{args.name}.json").read_text())
    gid = imp["best_genome_id"]
    with gzip.open(RUNS_DIR / args.name / "best_genome.pkl.gz", "rb") as f:
        mem_genome = pickle.load(f)["genome"]
    db = get_db()
    with db.session_scope() as s:
        neat_genome, config, _ = load_genome_and_config(s, gid)
        Xtr, ytr, *_ = load_split_data(s, imp["split_id"], "train", 1.0, 10**9)
        Xte, yte, *_ = load_split_data(s, imp["split_id"], "test", 1.0, 10**9)
    pheno = ExplaNEAT(neat_genome, config).get_phenotype_network()
    sn = StructureNetwork(pheno)
    parity = {}
    for nm, X, y in (("train", Xtr, ytr), ("test", Xte, yte)):
        out = sn.forward(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
        m = metrics(out, np.asarray(y, dtype=float))
        parity[nm] = dict(structure_network=m, in_memory=summ[nm],
                          max_abs_diff=max(abs(m[k] - summ[nm][k]) for k in m))
    xt = torch.as_tensor(Xte, dtype=torch.float64)
    with torch.no_grad():
        a = NeuralNeat(neat_genome, config).forward(xt).numpy()
        b = NeuralNeat(mem_genome, load_config(summ["task"])).forward(xt).numpy()
        s_ = sn.forward(xt).detach().numpy()
    parity["neuralneat_db_vs_memory_bit_equal"] = bool(np.array_equal(a, b))
    parity["structure_vs_neuralneat_max_abs_output_diff"] = float(np.max(np.abs(s_.ravel() - b.ravel())))
    parity["ok"] = (all(parity[k]["max_abs_diff"] < 1e-9 for k in ("train", "test"))
                    and parity["neuralneat_db_vs_memory_bit_equal"]
                    and parity["structure_vs_neuralneat_max_abs_output_diff"] < 1e-9)
    report = dict(name=args.name, genome_id=gid, experiment_id=imp["experiment_id"], parity=parity,
                  structure=structure_stats(pheno, sn, Xtr))
    (HERE / "logs" / f"{args.name}.analysis.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    brief = {k: v for k, v in report["structure"].items() if k != "hidden_active_fraction"}
    print(json.dumps(dict(parity=parity, structure=brief), indent=2, default=str))


if __name__ == "__main__":
    main()
