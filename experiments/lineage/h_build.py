"""Study H, stage 1: per-lineage data (PLAN_H.md). Read-only on the DB.

For one run, this loads the explained (final best) genome's fitter_parent lineage (and the
parent1 lineage for the sensitivity analysis) and computes, for every lineage genome:
- train AUC (checked against the stored fitness) and test AUC;
- gene sets and the gene diff from the previous lineage genome;
- for every leaf annotation of the final explanation, the decodability d_L(g)
  = 1 - novelty of the leaf's signal on the ancestor's inputs + hidden
  activations (genome_surgery.signal_basis / linear_novelty).

Output: experiments/lineage/replay_db/h_cache/<label>-s<seed>.pkl (not committed).

    uv run python experiments/lineage/h_build.py churn100 0
"""
import argparse
import json
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "churn"), str(REPO / "experiments" / "10seed")):
    if p not in sys.path:
        sys.path.insert(0, p)

threadpool_limits(1)

import torch  # noqa: E402

torch.set_num_threads(1)

from explaneat.analysis import lineage as L  # noqa: E402
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment, Explanation, Genome  # noqa: E402
from explaneat.db.serialization import deserialize_genome  # noqa: E402
import genome_surgery as gs  # noqa: E402
from compute_runs import load_split  # noqa: E402

CACHE = HERE / "replay_db" / "h_cache"
SPLITS = {"churn": "45962ea6-3c78-49f9-a01e-3665f6809b1e",
          "heart": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
          "monk2": "63257a45-49c9-42a6-8b28-8e346eab2bee"}
LABELS = {  # label -> (experiment name pattern, split, final-genome source)
    "churn100": ("churn100-s{s}", "churn", "explanation"),
    "churn200": ("churn200-s{s}", "churn", "explanation"),
    "heart100": ("10seed-heart100-s{s}", "heart", "explanation"),
    "monk2": ("10seed-monk2-s{s}", "monk2", "runs"),
}


def final_genome_id(label, seed):
    if LABELS[label][2] == "explanation":
        return json.load(open(REPO / f"experiments/10seed/explanations/{label}-s{seed}.json"))[
            "genome_id"]
    import csv
    for r in csv.DictReader(open(REPO / "experiments/10seed/runs.csv")):
        if r["dataset"] == label and int(r["seed"]) == seed:
            return r["genome_id"]
    raise KeyError((label, seed))


def gene_sets(data):
    nodes = {int(k) for k in data["nodes"]}
    conns = {(int(c["in_node"]), int(c["out_node"])): bool(c["enabled"])
             for c in data["connections"].values()}
    return nodes, conns


def leaf_signals(ops, final, config, X):
    """[{name, s, kind, G_nodes, G_conns, reason}] for every leaf annotation."""
    anns = L.annotations_from_operations(ops)
    res = L._Resolver(ops)
    out = []
    basis_final = None
    for a in anns:
        if a["children"]:
            continue
        row = dict(name=a["name"], s=None, kind=None, reason="")
        try:
            sub = gs.annotation_to_genome_subgraph(ops, a["name"], final, config)
            row["G_nodes"] = sorted(sub.hidden_nodes)
            row["G_conns"] = sorted(set(map(tuple, sub.internal_connections)) |
                                    set(map(tuple, sub.input_output_connections)))
            if sub.readout_connections or sub.input_output_connections:
                s, _, _ = gs.readout_contribution(final, config, X, sub.readout_connections,
                                                  sub.input_output_connections)
                row.update(s=s, kind="readout_contribution")
            else:
                row["reason"] = "no readout edge to the output"
        except gs.NonInputEntryError:
            row["reason"] = "entries not all inputs"
        except Exception as e:  # noqa: BLE001 - recorded, counted
            row["reason"] = f"{type(e).__name__}: {e}"
        if "G_nodes" not in row:  # genes from the lineage resolver
            nodes = {res.node(n) for n in a["subgraph_nodes"]} - {None}
            row["G_nodes"] = sorted(n for n in nodes if n in final.nodes and n not in
                                    config.genome_config.output_keys)
            conns = set()
            for u, v in a["subgraph_connections"]:
                conns.update(res.connection(u, v))
            row["G_conns"] = sorted(conns)
        if row["s"] is None:
            # Fallback: exit activations (hidden exits only).
            if basis_final is None:
                basis_final = gs.signal_basis(final, config, X)
            B, labels = basis_final
            col = {lab: i for i, lab in enumerate(labels)}
            ex = [res.node(x) for x in a["exit_nodes"]]
            idx = [col.get(f"h:{n}") for n in ex]
            if ex and all(i is not None for i in idx):
                row.update(s=B[:, idx], kind="exit_activations")
                row["reason"] += "; fallback: exit activations"
            else:
                row["reason"] += "; no usable signal (identity/output exit)"
        out.append(row)
    return out


def decodability(s, B):
    """R^2 (1 - novelty); multi-column s: sum of explained SS / sum of total SS."""
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        r = gs.linear_novelty(s, B)
        return r["r2"]
    num = den = 0.0
    for j in range(s.shape[1]):
        c = s[:, j] - s[:, j].mean()
        sst = float(c @ c)
        if sst <= 1e-12:
            continue
        r = gs.linear_novelty(s[:, j], B)
        num += r["r2"] * sst
        den += sst
    return num / den if den > 0 else float("nan")


def build(label, seed):
    pattern, split_key, _ = LABELS[label]
    name = pattern.format(s=seed)
    final_id = final_genome_id(label, seed)
    db.init_db()
    L.clear_cache()
    lin = L.get_lineage(final_id, mode="fitter_parent")
    lin_p1 = L.get_lineage(final_id, mode="parent1")
    assert lin[-1]["experiment_name"] == name, (lin[-1]["experiment_name"], name)
    with db.session_scope() as s:
        exp = s.query(Experiment).filter(Experiment.name == name).one()
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        data = load_split(s, SPLITS[split_key])
        ids = [uuid.UUID(r["genome_id"]) for r in lin]
        gd = {str(i): d for i, d in s.query(Genome.id, Genome.genome_data).filter(Genome.id.in_(ids))}
        ex = s.query(Explanation).filter(Explanation.genome_id == uuid.UUID(final_id)).all()
        ops = list(ex[0].operations) if len(ex) == 1 else None
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    genomes = [deserialize_genome(gd[r["genome_id"]], config) for r in lin]
    final = genomes[-1]

    leaves = leaf_signals(ops, final, config, Xtr) if ops else []
    rows = []
    prev = None
    for r, g in zip(lin, genomes):
        p_tr = gs.predict_proba(g, config, Xtr)
        p_te = gs.predict_proba(g, config, Xte)
        auc_tr = float(roc_auc_score(ytr, p_tr))
        nodes, conns = gene_sets(gd[r["genome_id"]])
        row = dict(r, train_auc_recomputed=auc_tr, test_auc=float(roc_auc_score(yte, p_te)),
                   nodes=nodes, conns=conns)
        if prev is not None:
            row["diff"] = L.diff_genome_data(gd[prev["genome_id"]], gd[r["genome_id"]])
        try:
            B, _ = gs.signal_basis(g, config, Xtr)
            row["d"] = {lf["name"]: (decodability(lf["s"], B) if lf["s"] is not None else np.nan)
                        for lf in leaves}
        except Exception as e:  # noqa: BLE001
            row["d"] = {lf["name"]: np.nan for lf in leaves}
            row["basis_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        prev = r
    # Leaf membership of every gene ever added along the lineage (final explanation).
    added = set()
    for row in rows[1:]:
        added.update(row["diff"]["added_nodes"])
        added.update(tuple(int(x) for x in k.rsplit("_", 1)) if k.count("_") == 1 else
                     (int(gd[row["genome_id"]]["connections"][k]["in_node"]),
                      int(gd[row["genome_id"]]["connections"][k]["out_node"]))
                     for k in row["diff"]["added_connections"])
    gene_map = None
    if ops:
        gene_map = L.map_genes_to_annotations(sorted(added, key=str), final_id,
                                              explanation=ops)["genes"]
    out = dict(label=label, seed=seed, experiment=name, final_genome_id=final_id,
               rows=rows, lineage_parent1=lin_p1,
               leaves=[{k: v for k, v in lf.items() if k != "s"} | {
                   "s_sd": float(np.std(lf["s"])) if lf["s"] is not None else None}
                   for lf in leaves],
               gene_map=gene_map, has_explanation=ops is not None,
               fitness_check_max_abs=float(max(abs(r["train_auc_recomputed"] - r["fitness"])
                                               for r in rows)))
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{label}-s{seed}.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=sorted(LABELS))
    ap.add_argument("seeds", nargs="+", type=int)
    args = ap.parse_args()
    for s in args.seeds:
        o = build(args.label, s)
        print(f"{o['experiment']}: {len(o['rows'])} lineage genomes, {len(o['leaves'])} leaves, "
              f"fitness check {o['fitness_check_max_abs']:.2e}", flush=True)


if __name__ == "__main__":
    main()
