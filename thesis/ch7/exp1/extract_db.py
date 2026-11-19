"""Experiment 1: per-network measures of each dataset's explained network (read-only).

    uv run python thesis/ch7/exp1/extract_db.py        # -> data/networks.json

The same measures as Experiment 2's ten-run tables (thesis/ch7/exp2/per_run.py), for the one
explained network of each Experiment 1 dataset. Sources:

- structure: the genome from explaneat_dev, in a READ ONLY transaction
  (experiments/10seed/compute_runs.ReadOnlySession), rebuilt with its experiment's NEAT config;
  compute_runs.structure (inputs used, hidden nodes, enabled phenotype connections, depth, skip
  connections; the definitions of runs.csv) and the library's ExplaNEAT.density / skippines /
  depth (layers), as extract_db.structure_metrics for Experiment 2.
- performance: experiments/benchmarks/results/propneat.json (classification; recomputed there and
  checked against each study's run record) and results/survival.json (WHAS500, Harrell's C).
- explanation: the explanation's operations in explaneat_dev (latest explanation of the genome;
  counts of split_node and add_identity_node, as experiments/10seed/build_explanations_csv.py) and
  the study's explanation record (annotations, coverage, function-preservation error, start/end
  times). The AAAI paper explanations have no record: their annotation counts and coverage come
  from experiments/aaai27/recheck/results/paper_coverage_options.json ("current", as stored), the
  function-preservation error from experiments/explanation_levels/levels/paper-*.json.
- XOR: the 30 seeds' reduced networks in experiments/xor/xor_results.json (no database), rebuilt
  as neat genomes with run_xor.py's config so the same ExplaNEAT measures apply; the rebuilt
  logits are checked against the stored ones.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
E = REPO / "experiments"
DATA = HERE / "data"
for p in (str(REPO), str(E / "10seed"), str(E / "xor")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONHASHSEED", "0")

DB = "postgresql://localhost/explaneat_dev"

# label -> metadata. protocol: which explanation method produced the explanation.
V4 = "authoring v4 (provenance-stamped, strict mode)"
PAPER = "AAAI paper explanation (July 2026, before the v4 method)"
NETS = {
    "paper-heart": dict(dataset="Heart", genome="9022a9eb-ecf9-41ba-b138-28ed4d131c9d", record=None,
                        paper="heart", levels="explanation_levels/levels/paper-heart.json", protocol=PAPER,
                        selection="the AAAI paper's Heart genome (40 generations)"),
    "paper-monk2": dict(dataset="MONK-2", genome="23dde0fa-a635-4974-91aa-c05266ea0747", record=None,
                        paper="monk2", levels="explanation_levels/levels/paper-monk2.json", protocol=PAPER,
                        selection="the AAAI paper's MONK-2 genome"),
    "cox-whas500-s0": dict(dataset="WHAS500", genome="f96ebc8d-5ab6-4b78-866f-5c82f1640771",
                           record="survival/cox_case/explanations/cox-whas500-s0.json", protocol=V4,
                           selection="lowest validation Cox loss of five seeds"),
    "c4cap-main-h10-e50-s1": dict(dataset="Connect-4 (cells)", genome="f0eb7635-7f85-437b-815b-993c5ad6cc14",
                                  record="c4_capacity/explanations/record.json", protocol=V4,
                                  selection="largest training AUC of the three main runs"),
    "c4lf-lines-heavy-s0": dict(dataset="Connect-4 (lines)", genome="6ebf7e0c-55ec-4215-9852-0ac982e43c0e",
                                record="c4_line_features/explanations/lines/report.json", protocol=V4,
                                selection="largest training AUC of the six line-feature runs"),
    "shape-A": dict(dataset="Synthetic U", genome="2aac39c4-20d1-48f1-8f43-e8dfcaf50a52",
                    record="ushape_saddle/explanations/shape-A.json", protocol=V4,
                    selection="largest training AUC of five seeds (s0)"),
    "shape-B": dict(dataset="Synthetic saddle", genome="25f5830e-33cc-4dda-8ea8-16e092c09a3e",
                    record="ushape_saddle/explanations/shape-B.json", protocol=V4,
                    selection="largest training AUC of five seeds (s0)"),
    "shape-C": dict(dataset="Synthetic U + saddle", genome="a1cbdd4a-42e8-4c1a-b060-00aa3e22c76b",
                    record="ushape_saddle/explanations/shape-C.json", protocol=V4,
                    selection="largest training AUC of five seeds (s4)"),
}


def jload(p):
    return json.loads(Path(p).read_text())


def minutes(rec):
    a, b = (datetime.fromisoformat(rec[k].replace("Z", "+00:00")) for k in ("start", "end"))
    return (b - a).total_seconds() / 60


def network(cr, label, m):
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.db.models import Dataset, Explanation
    from sqlalchemy import select
    with cr.ReadOnlySession() as s:
        g = s.get(cr.Genome, uuid.UUID(m["genome"]))
        exp = g.population.experiment
        cfg = cr.load_neat_config(exp.neat_config_text or "", exp.config_json)
        genome = g.to_neat_genome(cfg)
        names = list(s.get(Dataset, exp.dataset_id).feature_names)
        ex = s.execute(select(Explanation).where(Explanation.genome_id == g.id)
                       .order_by(Explanation.updated_at.desc())).scalars().first()
        ops = list(ex.operations or [])
        ex_id = str(ex.id)
    st = cr.structure(genome, cfg, names)
    e = ExplaNEAT(genome, cfg)
    oc = Counter(o["type"] for o in ops)
    out = {"label": label, **{k: m[k] for k in ("dataset", "genome", "protocol", "selection")},
           "explanation_id": ex_id, "n_inputs": st["n_inputs"], "inputs": st["inputs_connected"],
           "hidden": st["hidden_nodes"], "conns": st["phenotype_connections_enabled"], "depth": st["depth"],
           "skips": st["skip_connections"], "direct_io": st["direct_io_edges"],
           "density": float(e.density()), "skippiness": float(e.skippines()), "layers": int(e.depth()),
           "splits": oc.get("split_node", 0), "identity": oc.get("add_identity_node", 0),
           "annotate_ops": oc.get("annotate", 0), "mike": None}
    assert out["layers"] == out["depth"] + 1, (label, out["layers"], out["depth"])
    if m["record"]:
        rec = jload(E / m["record"])
        assert rec["genome_id"] == m["genome"], label
        out.update(leaf=rec["n_leaf_annotations"], comp=rec["n_composition_annotations"],
                   cov_s=rec["structural_coverage"], cov_c=rec["compositional_coverage"],
                   fp=rec["function_preservation_max_abs_err"], agent=round(minutes(rec), 1),
                   record=f"experiments/{m['record']}")
        assert rec["n_splits"] == out["splits"] and rec["n_identity_nodes"] == out["identity"], \
            (label, rec["n_splits"], out["splits"], rec["n_identity_nodes"], out["identity"])
        assert rec["n_leaf_annotations"] + rec["n_composition_annotations"] == out["annotate_ops"], label
        out["mike"] = (rec.get("mike_verification") or {}).get("minutes")
    else:
        po = jload(E / "aaai27/recheck/results/paper_coverage_options.json")[m["paper"]]
        assert po["genome_id"] == m["genome"] and po["explanation_id"] == ex_id, label
        cur = po["current"]
        lv = jload(E / m["levels"])["levels"]
        out.update(leaf=cur["leaf"], comp=cur["composition"], cov_s=cur["structural_coverage"],
                   cov_c=cur["compositional_coverage"], fp=lv["faithful"]["function_preservation_max_abs_err"],
                   agent=None, record=None)
        assert cur["annotations"] == out["annotate_ops"], (label, cur["annotations"], out["annotate_ops"])
    return out


def xor():
    import neat
    import torch
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.core.neuralneat import NeuralNeat
    from run_xor import load_config
    cfg = load_config()
    X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float64)
    nets = []
    for run in jload(E / "xor/xor_results.json")["runs"]:
        g = neat.DefaultGenome(run["seed"])
        out = cfg.genome_config.node_gene_type(0)
        out.bias, out.response, out.activation, out.aggregation = run["output_bias"], 1.0, "relu", "sum"
        g.nodes[0] = out
        conns = []
        for h in run["hidden"]:
            n = cfg.genome_config.node_gene_type(h["node"])
            n.bias, n.response, n.activation, n.aggregation = h["bias"], 1.0, "relu", "sum"
            g.nodes[h["node"]] = n
            conns += [((int(a), h["node"]), w) for a, w in h["w_in"].items()]
        conns += [((int(a), 0), w) for a, w in run["skip_weights"].items()]
        for h in run["hidden"]:
            conns += [((h["node"], int(b)), w) for b, w in h["w_out"].items() if int(b) == 0]
        for key, w in dict(conns).items():
            c = cfg.genome_config.connection_gene_type(key)
            c.weight, c.enabled = w, True
            g.connections[key] = c
        logit = torch.logit(NeuralNeat(g, cfg).forward(X).detach().ravel()).numpy()
        assert np.allclose(np.clip(logit, -30, 30), np.clip(run["logits"], -30, 30), atol=0.05), (run["seed"], logit, run["logits"])
        st = __import__("compute_runs").structure(g, cfg, ["x1", "x2"])
        e = ExplaNEAT(g, cfg)
        nets.append({"seed": run["seed"], "inputs": st["inputs_connected"], "n_inputs": 2, "hidden": st["hidden_nodes"],
                     "conns": st["phenotype_connections_enabled"], "depth": st["depth"], "skips": st["skip_connections"],
                     "density": float(e.density()), "skippiness": float(e.skippines()),
                     "raw_hidden": run["raw"]["n_hidden"], "raw_conns": run["raw"]["n_connections"],
                     "train_accuracy": float(np.mean((np.array(run["preds"]) > 0.5) == np.array([0, 1, 1, 0])))})
    return nets


def main():
    import compute_runs as cr
    cr.db.init_db(DB)
    DATA.mkdir(exist_ok=True)
    out = {"database": "explaneat_dev (read-only)", "networks": {}}
    for label, m in NETS.items():
        out["networks"][label] = network(cr, label, m)
        print(label, {k: v for k, v in out["networks"][label].items() if k not in ("protocol", "selection")}, flush=True)
    out["xor"] = xor()
    (DATA / "networks.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
