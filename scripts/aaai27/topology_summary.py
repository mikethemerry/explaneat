"""Emit a topology summary for an evolved genome (AAAI-27 Exp3 candidate choice).

Summarises the best genome of an experiment: node/connection counts, connected
inputs, hidden units and their fan-in, max active path length, train/test AUC,
and — for monk2 — which of the rule-relevant '=1' inputs the network wired in.

Usage:
  uv run python scripts/aaai27/topology_summary.py <experiment_id> [out.md]
  uv run python scripts/aaai27/topology_summary.py monk2        # by known name
"""
import sys
import uuid
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from explaneat.db import db
from explaneat.db.models import Experiment, Population, Genome, Dataset, DatasetSplit
from explaneat.core.config_utils import load_neat_config
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat

KNOWN = {"monk2": "monk2 - OHE - 250-100-5"}


def best_genome_row(session, experiment_id):
    from sqlalchemy import func
    return (session.query(Genome)
            .join(Population, Genome.population_id == Population.id)
            .filter(Population.experiment_id == experiment_id,
                    Genome.fitness.isnot(None))
            .order_by(Genome.fitness.desc())
            .first())


def summarize(experiment_id, out_path=None):
    lines = []
    with db.session_scope() as s:
        exp = s.query(Experiment).filter_by(id=experiment_id).first()
        if not exp:
            sys.exit(f"experiment {experiment_id} not found")
        g_row = best_genome_row(s, exp.id)
        if not g_row:
            sys.exit("no scored genome in experiment yet (training incomplete?)")
        split = s.query(DatasetSplit).filter_by(id=exp.split_id).first()
        dataset = s.query(Dataset).filter_by(id=split.dataset_id).first()
        config = load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = g_row.to_neat_genome(config)
        gid = str(g_row.id)
        stored_fitness = g_row.fitness
        X, y = dataset.get_data()
        mean = np.array(split.scaler_params["mean"])
        scale = np.where(np.array(split.scaler_params["scale"]) == 0, 1.0,
                         np.array(split.scaler_params["scale"]))
        Xs = (X - mean) / scale
        feat_names = dataset.feature_names
        meta = dataset.additional_metadata or {}
        exp_name = exp.name
        tr, te = split.train_indices, split.test_indices

    # metrics
    net = NeuralNeat(genome, config)
    import torch
    def auc(idx):
        p = net.forward(torch.tensor(Xs[idx], dtype=torch.float64)).detach().numpy().ravel()
        return roc_auc_score(y[idx], p)
    auc_tr, auc_te = auc(tr), auc(te)

    # topology
    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    input_ids = set(pheno.input_node_ids)
    output_ids = set(pheno.output_node_ids)
    connected_in = {c.from_node for c in pheno.connections if c.enabled} & input_ids
    hidden = [n.id for n in pheno.nodes if n.id not in input_ids and n.id not in output_ids]
    enabled_edges = [c for c in pheno.connections if c.enabled]
    # fan-in per hidden unit
    fanin = Counter()
    for c in enabled_edges:
        if c.to_node in hidden:
            fanin[c.to_node] += 1
    fanin_vals = sorted(fanin.values())
    try:
        depth = ExplaNEAT(genome, config).depth
    except Exception:
        depth = "n/a"

    lines.append(f"# Topology summary: {exp_name}")
    lines.append("")
    lines.append(f"- experiment_id: `{experiment_id}`")
    lines.append(f"- best genome: `{gid}`  (stored fitness = train AUC = {stored_fitness:.4f})")
    lines.append(f"- dataset: {dataset.name}  (split {split.id})")
    lines.append("")
    lines.append(f"- nodes (genome): {len(genome.nodes)}")
    lines.append(f"- connection genes: {len(genome.connections)}  "
                 f"(enabled in phenotype: {len(enabled_edges)})")
    lines.append(f"- connected inputs: {len(connected_in)} of {len(feat_names)}")
    lines.append(f"- hidden units: {len(hidden)}")
    lines.append(f"- hidden fan-in (enabled): {fanin_vals}  "
                 f"(min={min(fanin_vals) if fanin_vals else 0}, "
                 f"max={max(fanin_vals) if fanin_vals else 0})")
    lines.append(f"- max active path length (depth): {depth}")
    lines.append(f"- train AUC: {auc_tr:.4f} (n={len(tr)})")
    lines.append(f"- test AUC:  {auc_te:.4f} (n={len(te)})")

    # monk2 rule-relevant coverage
    rule_cols = meta.get("rule_relevant_columns")
    if rule_cols:
        name_to_key = {nm: -(i + 1) for i, nm in enumerate(feat_names)}
        covered, missing = [], []
        for col in rule_cols:
            k = name_to_key.get(col)
            (covered if k in connected_in else missing).append(col)
        lines.append("")
        lines.append(f"- rule-relevant inputs connected: {len(covered)}/{len(rule_cols)} "
                     f"({', '.join(covered) or 'none'})")
        if missing:
            lines.append(f"  - NOT connected: {', '.join(missing)}")

    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
        print(f"\nwrote {out_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    with db.session_scope() as s:
        if arg in KNOWN:
            exp = s.query(Experiment).filter_by(name=KNOWN[arg]).order_by(
                Experiment.created_at.desc()).first()
            if not exp:
                sys.exit(f"no experiment named {KNOWN[arg]!r} yet")
            exp_id = exp.id
        else:
            exp_id = uuid.UUID(arg)
    summarize(exp_id, out)


if __name__ == "__main__":
    main()
