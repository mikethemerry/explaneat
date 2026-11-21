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
- explanation (computed, read-only, on the genome's one explanation in explaneat_dev):
  - counts of split_node and add_identity_node operations (as experiments/10seed/build_explanations_csv.py)
    and of leaf / composition annotate operations (experiments/10seed/explaining_time.annotation_counts);
  - structural and compositional coverage and hierarchy completeness from the library's own check,
    MCP ``get_coverage`` (``coverage_report``; the current definition, under which annotations of
    disconnected inputs are exempt from the single-root rule, decision D4), and the
    function-preservation error and check_explanation's stricter hierarchy check (no exemption;
    ``hierarchy_complete_check``) from experiments/10seed/check_explanation.check on the genome's
    training split: the two checks experiments/10seed/verify_record.py runs for Experiment 2;
  - explaining minutes from the database's timestamped writes (explaining_time.explaining_window:
    explanation created -> last operation, evidence record or evidence_log entry).
  Each is compared with what the explaining agent reported in the study's record (kept as rec_* for
  audit): annotation counts, splits and identity nodes must agree (asserted); coverage, the error
  and minutes may differ and are printed. The AAAI paper explanations have no record: their
  rec_* values are experiments/aaai27/recheck/results/paper_coverage_options.json ("current", the
  stored stream under the definition before D4: MONK-2's and Adult's compositional coverage are
  0.75 and 0.9 there because their pruned-input annotations sit outside the root) and
  experiments/explanation_levels/levels/paper-*.json (the error); their annotation counts are
  asserted equal. They predate the evidence log, so they have no explaining minutes.
- performance of the original Adult network (``perf``; 21 Nov 2026, while
  experiments/benchmarks/results/propneat.json has no Experiment 1 Adult entry): train / test AUC,
  the 10,000-resample 95% interval and accuracy at 0.5, computed here read-only with
  experiments/10seed/compute_runs (load_split on the run's split, bootstrap_indices, performance), the
  code that wrote experiments/10seed/runs_paper_reference.csv; asserted equal to that file's adult row.
  make_tables.py prefers propneat.json's entry when it has one and asserts the two agree.
- parts: each stored annotation (name, entry and exit counts, the model inputs among its entries, its
  children), read from the explanation's annotate operations; for the book's case cards.
The whole process runs with PGOPTIONS default_transaction_read_only=on (no write can reach the
database). Run it twice and compare: data/networks.json is byte-identical.
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
# the library checks (mcp_server.get_db) open their own connection from DATABASE_URL; every
# transaction of this process is read-only on the server side
os.environ["DATABASE_URL"] = DB
os.environ["PGOPTIONS"] = (os.environ.get("PGOPTIONS", "") + " -c default_transaction_read_only=on").strip()

# label -> metadata. protocol: which explanation method produced the explanation.
V4 = "authoring v4 (provenance-stamped, strict mode)"
PAPER = "AAAI paper explanation (July 2026, before the v4 method)"
NETS = {
    "paper-heart": dict(dataset="Heart", genome="9022a9eb-ecf9-41ba-b138-28ed4d131c9d", record=None,
                        paper="heart", levels="explanation_levels/levels/paper-heart.json", protocol=PAPER,
                        selection="the AAAI paper's Heart genome (40 generations)"),
    "paper-adult": dict(dataset="Adult", genome="4bc8fa07-bbf5-48e4-96a8-77be8b577ba3", record=None,
                        paper="adult", levels="explanation_levels/levels/paper-adult.json", protocol=PAPER,
                        selection="the AAAI paper's Adult genome (\"No pressure\", 100 generations)", perf="adult"),
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


def checks(genome_id):
    """The library's coverage and explanation checks (as experiments/10seed/verify_record.py)."""
    from check_explanation import check, split_for_genome
    from mcp_server.tools.coverage import get_coverage
    cov = json.loads(get_coverage(genome_id))
    c = check(genome_id, split_for_genome(genome_id))
    return cov, c


def performance(cr, s, label, m, g, exp, genome, cfg):
    """Train / test AUC, 95% interval and accuracy of a network, by the code of runs_paper_reference.csv."""
    sid = cr.SPLITS[m["perf"]]
    assert str(exp.split_id) == sid, (label, str(exp.split_id), sid)
    data = cr.load_split(s, sid)
    p = cr.performance(genome, cfg, data, cr.bootstrap_indices(len(data["y_test"])), g.fitness)
    path = "10seed/runs_paper_reference.csv"
    with open(E / path) as f:
        rec = [r for r in csv.DictReader(f) if r["dataset"] == m["perf"] and r["seed"] == "paper"]
    assert len(rec) == 1 and rec[0]["genome_id"] == m["genome"], (label, len(rec))
    rec = rec[0]
    out = {"split_id": sid, "train_auc": float(p["train_auc"]), "test_auc": float(p["test_auc"]),
           "test_auc_ci95": [float(p["test_auc_ci_lo"]), float(p["test_auc_ci_hi"])],
           "n_boot_used": int(p["n_boot_used"]), "test_accuracy": float(p["test_acc_0.5"]),
           "train_accuracy": float(p["train_acc_0.5"]), "n_train": int(p["n_train"]), "n_test": int(p["n_test"]),
           "record": {"file": f"experiments/{path}", **{k: float(rec[k]) for k in (
               "train_auc", "test_auc", "test_auc_ci_lo", "test_auc_ci_hi")}}}
    for k, v in (("train_auc", out["train_auc"]), ("test_auc", out["test_auc"]),
                 ("test_auc_ci_lo", out["test_auc_ci95"][0]), ("test_auc_ci_hi", out["test_auc_ci95"][1])):
        assert abs(v - out["record"][k]) < 1e-9, (label, k, v, out["record"][k])
    assert int(rec["n_boot_used"]) == out["n_boot_used"] and int(rec["n_test"]) == out["n_test"], label
    return out


def network(cr, label, m):
    from explaneat.core.explaneat import ExplaNEAT
    from explaneat.db.models import Dataset, Explanation
    from explaining_time import explaining_window
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
        w = explaining_window(s, m["genome"])   # asserts the genome has exactly one explanation
        perf = performance(cr, s, label, m, g, exp, genome, cfg) if m.get("perf") else None
    assert w["explanation_id"] == ex_id, label
    cov, chk = checks(m["genome"])
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
    if perf is not None:
        out["perf"] = perf
    # the parts of the explanation, as stored (annotate operations): name, entry / exit counts, the
    # model inputs among the entries (input key i <-> data column i, as compute_runs.structure) and children
    col = {str(k): names[i] for i, k in enumerate(cfg.genome_config.input_keys)}
    out["parts"] = [{"name": o["params"]["name"], "n_entry": len(o["params"]["entry_nodes"]),
                     "n_exit": len(o["params"]["exit_nodes"]),
                     "entry_inputs": [col[n] for n in o["params"]["entry_nodes"] if n in col],
                     "children": list(o["params"].get("child_annotation_ids") or [])}
                    for o in ops if o["type"] == "annotate"]
    assert len(out["parts"]) == out["annotate_ops"], label
    assert out["layers"] == out["depth"] + 1, (label, out["layers"], out["depth"])
    # computed: annotation counts (operations), coverage and hierarchy (get_coverage), hierarchy
    # completeness and function preservation (check_explanation), explaining minutes (database writes)
    out.update(leaf=w["leaf"], comp=w["composition"], cov_s=cov["structural_coverage"],
               cov_c=cov["compositional_coverage"], hierarchy_complete=cov["hierarchy"]["complete"],
               hierarchy_complete_check=chk["hierarchy"]["complete"],
               exempt_disconnected=cov["hierarchy"].get("exempt_disconnected", []),
               fp=chk["function_preservation_max_abs_err"],
               agent=w["minutes"] if w["n_log_entries"] else None,
               db_start=w["start"].isoformat(timespec="seconds"), db_end=w["end"].isoformat(timespec="seconds"),
               n_log_entries=w["n_log_entries"])
    assert (cov["leaf_count"], cov["composition_count"]) == (out["leaf"], out["comp"]), (label, cov["leaf_count"], cov["composition_count"])
    assert out["leaf"] + out["comp"] == out["annotate_ops"], label
    if m["record"]:
        rec = jload(E / m["record"])
        assert rec["genome_id"] == m["genome"], label
        out.update(rec_leaf=rec["n_leaf_annotations"], rec_comp=rec["n_composition_annotations"],
                   rec_cov_s=rec["structural_coverage"], rec_cov_c=rec["compositional_coverage"],
                   rec_hierarchy_complete=rec.get("hierarchy_complete"),
                   rec_fp=rec["function_preservation_max_abs_err"], rec_agent=round(minutes(rec), 1),
                   record=f"experiments/{m['record']}")
        assert rec["n_splits"] == out["splits"] and rec["n_identity_nodes"] == out["identity"], \
            (label, rec["n_splits"], out["splits"], rec["n_identity_nodes"], out["identity"])
        assert (rec["n_leaf_annotations"], rec["n_composition_annotations"]) == (out["leaf"], out["comp"]), \
            (label, "record", rec["n_leaf_annotations"], rec["n_composition_annotations"], "database", out["leaf"], out["comp"])
        assert out["agent"] is not None, (label, "a v4 explanation without an evidence log")
        out["mike"] = (rec.get("mike_verification") or {}).get("minutes")
    else:
        po = jload(E / "aaai27/recheck/results/paper_coverage_options.json")[m["paper"]]
        assert po["genome_id"] == m["genome"] and po["explanation_id"] == ex_id, label
        cur = po["current"]
        lv = jload(E / m["levels"])["levels"]
        out.update(rec_leaf=cur["leaf"], rec_comp=cur["composition"], rec_cov_s=cur["structural_coverage"],
                   rec_cov_c=cur["compositional_coverage"], rec_hierarchy_complete=cur["hierarchy_complete"],
                   rec_fp=lv["faithful"]["function_preservation_max_abs_err"], rec_agent=None, record=None)
        assert (cur["leaf"], cur["composition"]) == (out["leaf"], out["comp"]), (label, cur, out)
        assert out["agent"] is None, (label, "the paper explanations predate the evidence log")
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
    import torch
    torch.set_num_threads(1)
    import compute_runs as cr
    cr.db.init_db(DB)
    DATA.mkdir(exist_ok=True)
    out = {"database": "explaneat_dev (read-only)", "networks": {}}
    for label, m in NETS.items():
        out["networks"][label] = network(cr, label, m)
        r = out["networks"][label]
        print(label, {k: v for k, v in r.items() if k not in ("protocol", "selection")}, flush=True)
        print(f"  {label}: minutes record {r['rec_agent']} computed {r['agent']}; coverage record "
              f"{r['rec_cov_s']}/{r['rec_cov_c']} computed {r['cov_s']}/{r['cov_c']}; hierarchy complete record "
              f"{r['rec_hierarchy_complete']} computed {r['hierarchy_complete']} (strict check "
              f"{r['hierarchy_complete_check']}); error record {r['rec_fp']:.3g} "
              f"computed {r['fp']:.3g}", flush=True)
    out["xor"] = xor()
    (DATA / "networks.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
