"""Per-run metrics for the 10-seed consistency experiment (PREREG.md, "Metrics").

Read-only: every DB session runs inside a READ ONLY Postgres transaction and is
rolled back. Nothing is written to the DB; only CSVs in this directory.

Usage (re-runnable; Adult rows appear once its runs are 'completed'):

    uv run python experiments/10seed/compute_runs.py            # all outputs
    uv run python experiments/10seed/compute_runs.py --skip-paper

Outputs (experiments/10seed/):
    runs.csv                   heart, monk2, adult (pre-registered 30 runs)
    runs_heart100.csv          Heart, 100 generations (additional analysis)
    runs_paper_reference.csv   the three paper genomes through the same code

Definitions (pre-registered; judgment calls marked *):
- Best genome: highest stored training fitness over all generations; ties ->
  earliest generation, then lowest NEAT genome key (PREREG amendment 2026-11-03).
  NaN fitnesses are ignored.
- Forward pass: NeuralNeat on X scaled with the split's stored StandardScaler,
  float64, as reproducibility/load.py. Train AUC is recomputed and compared to
  the stored fitness (fitness_abs_diff).
- Test AUC CI: percentile 95% CI over 10,000 bootstrap resamples of the test set.
  Indices come from np.random.RandomState(42).randint(0, n, n) drawn 10,000
  times in sequence (the paper's reproducibility/scripts/common.py procedure).
  One index matrix per dataset, shared by every model on that dataset (paired;
  heart100 and the paper genome reuse their dataset's matrix). Resamples with a
  single class are skipped (same for every model of a dataset).
- Accuracy at threshold 0.5 (p > 0.5, as reproducibility/load.py).
- Phenotype: ExplaNEAT(genome, config).get_phenotype_network() (enabled edges on
  some input->output path).
- connected inputs: input nodes with >=1 enabled phenotype edge; unused = rest.
- genome genes: node genes (hidden+output; neat-python stores no input genes)
  + connection genes incl. disabled.
- depth(node): longest path (edges) from any input, inputs depth 0; network
  depth = max depth over output nodes.
- skip connections*: enabled phenotype edges with depth(to) > depth(from) + 1,
  EXCLUDING direct input->output edges, which are reported separately
  (direct_io_edges). skip_connections_incl_direct is given for transparency.
- fan-in: number of enabled phenotype edges into each hidden node (mean, max;
  NaN if no hidden nodes).
- Structural monosemanticity proxy (NOT an annotation result): for each hidden
  node, the distinct original features among the input nodes with a phenotype
  path to it; one-hot columns map to their feature via the name before ':'.
  Share with <=2 features and median (NaN if no hidden nodes).
- wall_clock_hours: experiments.end_time - start_time (both stored the same way,
  so the difference is valid even though the stored tz offset is not).
"""

import argparse
import csv
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit, Experiment, Genome  # noqa: E402

SPLITS = {
    "heart": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
    "heart100": "165cebbd-6609-46ca-867e-c09e17b2d6ec",
    "monk2": "63257a45-49c9-42a6-8b28-8e346eab2bee",
    "adult": "1df50169-8095-40d1-b61a-30bb541cd313",
}
MAIN_DATASETS = ("heart", "monk2", "adult")
SEEDS = range(10)
PAPER = {
    "heart": ("9022a9eb-ecf9-41ba-b138-28ed4d131c9d", 0.8966),
    "monk2": ("23dde0fa-a635-4974-91aa-c05266ea0747", 1.0),
    "adult": ("4bc8fa07-bbf5-48e4-96a8-77be8b577ba3", 0.9055),
}
RERUN_NOTES = {
    ("adult", s): ("re-run with same seed after startup crash during data load "
                   "(psycopg2 OperationalError, 30 concurrent starts); crashed "
                   "attempt wrote no Experiment row; see RUN_INCIDENTS.md")
    for s in (1, 3, 8)
}
BOOT_SEED, N_BOOT = 42, 10_000
FITNESS_TOL = 1e-6

COLUMNS = [
    "dataset", "seed", "experiment_id", "experiment_name", "genome_id", "genome_key",
    "generation_of_best", "total_generations", "n_rows_at_best_fitness",
    "n_keys_tied_in_best_generation",
    "code_commit", "hardware", "wall_clock_hours", "rerun_note",
    # performance
    "stored_fitness", "train_auc", "fitness_abs_diff", "fitness_match_1e-6",
    "test_auc", "test_auc_ci_lo", "test_auc_ci_hi", "n_boot_used",
    "train_acc_0.5", "test_acc_0.5", "n_train", "n_test",
    # size
    "n_inputs", "inputs_connected", "inputs_unused", "hidden_nodes",
    "phenotype_connections_enabled", "genome_node_genes", "genome_conn_genes_total",
    "genome_conn_genes_enabled", "genome_genes_total",
    # shape
    "depth", "skip_connections", "direct_io_edges", "skip_connections_incl_direct",
    "hidden_fanin_mean", "hidden_fanin_max",
    # structural monosemanticity proxy
    "structproxy_share_hidden_le2_features", "structproxy_median_features_per_hidden",
]


# ---------------------------------------------------------------- DB access

class ReadOnlySession:
    """Session in a READ ONLY transaction, always rolled back."""

    def __enter__(self):
        self.s = db.Session()
        self.s.execute(text("SET TRANSACTION READ ONLY"))
        return self.s

    def __exit__(self, *exc):
        self.s.rollback()
        self.s.close()
        db.Session.remove()
        return False


def load_split(session, split_id):
    split = session.get(DatasetSplit, uuid.UUID(split_id))
    dataset = session.get(Dataset, split.dataset_id)
    X, y = dataset.get_data()
    assert split.scaler_type == "StandardScaler", split.scaler_type
    mean = np.asarray(split.scaler_params["mean"], dtype=np.float64)
    scale = np.asarray(split.scaler_params["scale"], dtype=np.float64)
    assert mean.shape[0] == X.shape[1], "scaler/dataset width mismatch"
    Xs = (X.astype(np.float64) - mean) / scale
    tr, te = list(split.train_indices), list(split.test_indices)
    return dict(X_train=Xs[tr], y_train=np.asarray(y)[tr].ravel(),
                X_test=Xs[te], y_test=np.asarray(y)[te].ravel(),
                feature_names=list(dataset.feature_names))


def best_genome_row(session, experiment_id):
    """(genome uuid, key, fitness, generation, ties) per the PREREG tie-break.

    ties = (genome rows at the best fitness in any generation -- elites carried
    over count again --, distinct keys at the best fitness in the best genome's
    generation, i.e. the candidates the key tie-break actually decided between).
    """
    rows = session.execute(text("""
        SELECT g.id, g.genome_id, g.fitness, p.generation
        FROM genomes g JOIN populations p ON g.population_id = p.id
        WHERE p.experiment_id = :eid AND g.fitness IS NOT NULL
          AND g.fitness <> 'NaN'::float8
          AND g.fitness = (SELECT max(g2.fitness) FROM genomes g2
                           JOIN populations p2 ON g2.population_id = p2.id
                           WHERE p2.experiment_id = :eid
                             AND g2.fitness <> 'NaN'::float8)
        ORDER BY g.fitness DESC, p.generation ASC, g.genome_id ASC"""),
        {"eid": experiment_id}).fetchall()
    top = rows[0]
    n_keys_same_gen = len({r[1] for r in rows if r[3] == top[3]})
    return str(top[0]), top[1], top[2], top[3], (len(rows), n_keys_same_gen)


def experiment_meta(session, exp):
    hw = exp.hardware_info or {}
    rp = hw.get("run_params") or {}
    commit = rp.get("code_commit") or exp.git_commit_sha or ""
    hardware = "; ".join(str(x) for x in (
        hw.get("processor", ""), hw.get("device", ""),
        f"torch_threads={hw.get('torch_threads', '')}") if x)
    hours = ((exp.end_time - exp.start_time).total_seconds() / 3600
             if exp.end_time and exp.start_time else float("nan"))
    total_gens = session.execute(
        text("SELECT count(*) FROM populations WHERE experiment_id = :eid"),
        {"eid": exp.id}).scalar()
    return dict(code_commit=commit, hardware=hardware, wall_clock_hours=hours,
                total_generations=total_gens)


# ---------------------------------------------------------------- metrics

def bootstrap_indices(n):
    rng = np.random.RandomState(BOOT_SEED)
    return np.stack([rng.randint(0, n, size=n) for _ in range(N_BOOT)])


def predict(genome, config, X):
    net = NeuralNeat(genome, config)
    return net.forward(torch.tensor(X, dtype=torch.float64)).detach().numpy().ravel()


def performance(genome, config, data, boot_idx, stored_fitness):
    p_tr = predict(genome, config, data["X_train"])
    p_te = predict(genome, config, data["X_test"])
    ytr, yte = data["y_train"], data["y_test"]
    train_auc = roc_auc_score(ytr, p_tr)
    aucs = []
    for idx in boot_idx:
        yb = yte[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, p_te[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    diff = abs(train_auc - stored_fitness)
    return {
        "stored_fitness": stored_fitness, "train_auc": train_auc,
        "fitness_abs_diff": diff, "fitness_match_1e-6": diff < FITNESS_TOL,
        "test_auc": roc_auc_score(yte, p_te), "test_auc_ci_lo": lo,
        "test_auc_ci_hi": hi, "n_boot_used": len(aucs),
        "train_acc_0.5": accuracy_score(ytr, p_tr > 0.5),
        "test_acc_0.5": accuracy_score(yte, p_te > 0.5),
        "n_train": len(ytr), "n_test": len(yte),
    }


def feature_of(name):
    return name.split(":", 1)[0]


def structure(genome, config, feature_names):
    ph = ExplaNEAT(genome, config).get_phenotype_network()
    input_keys = [str(k) for k in config.genome_config.input_keys]
    assert len(input_keys) == len(feature_names), "inputs != features"
    # input key i <-> data column i (NeuralNeat input order)
    feat = {k: feature_of(feature_names[i]) for i, k in enumerate(input_keys)}
    inputs = set(input_keys)
    outputs = {str(k) for k in config.genome_config.output_keys}
    edges = [(c.from_node, c.to_node) for c in ph.connections if c.enabled]
    hidden = [n.id for n in ph.nodes if n.type == NodeType.HIDDEN]
    nodes = {n.id for n in ph.nodes} | inputs

    succ, pred = defaultdict(list), defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)

    # topological order (Kahn) over nodes touched by edges
    indeg = {n: len(pred[n]) for n in nodes}
    order, queue = [], [n for n in nodes if indeg[n] == 0]
    while queue:
        n = queue.pop()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    assert len(order) == len(nodes), "phenotype is not a DAG"

    depth = {n: 0 for n in nodes}
    for n in order:
        for m in succ[n]:
            depth[m] = max(depth[m], depth[n] + 1)
    # nodes with no inputs in the phenotype other than inputs should not exist
    for n in hidden:
        assert pred[n], f"hidden node {n} without incoming edge"

    net_depth = max(depth[o] for o in outputs if o in depth)
    direct_io = [(a, b) for a, b in edges if a in inputs and b in outputs]
    skips_all = [(a, b) for a, b in edges if depth[b] > depth[a] + 1]
    skips = [e for e in skips_all if not (e[0] in inputs and e[1] in outputs)]

    fanin = [len(pred[h]) for h in hidden]

    # upstream original features per hidden node
    upstream = {}
    for n in order:
        s = {feat[n]} if n in inputs else set()
        for p in pred[n]:
            s |= upstream[p]
        upstream[n] = s
    nfeat = [len(upstream[h]) for h in hidden]

    connected = {a for a, _ in edges if a in inputs}
    nan = float("nan")
    conn_genes = list(genome.connections.values())
    return {
        "n_inputs": len(inputs),
        "inputs_connected": len(connected),
        "inputs_unused": len(inputs) - len(connected),
        "hidden_nodes": len(hidden),
        "phenotype_connections_enabled": len(edges),
        "genome_node_genes": len(genome.nodes),
        "genome_conn_genes_total": len(conn_genes),
        "genome_conn_genes_enabled": sum(1 for c in conn_genes if c.enabled),
        "genome_genes_total": len(genome.nodes) + len(conn_genes),
        "depth": net_depth,
        "skip_connections": len(skips),
        "direct_io_edges": len(direct_io),
        "skip_connections_incl_direct": len(skips_all),
        "hidden_fanin_mean": float(np.mean(fanin)) if fanin else nan,
        "hidden_fanin_max": int(max(fanin)) if fanin else nan,
        "structproxy_share_hidden_le2_features":
            float(np.mean([k <= 2 for k in nfeat])) if nfeat else nan,
        "structproxy_median_features_per_hidden":
            float(np.median(nfeat)) if nfeat else nan,
    }


def genome_metrics(session, genome_uuid, data, boot_idx):
    g_row = session.get(Genome, uuid.UUID(genome_uuid))
    exp = g_row.population.experiment
    config = load_neat_config(exp.neat_config_text or "", exp.config_json)
    genome = g_row.to_neat_genome(config)
    row = {"genome_id": genome_uuid, "genome_key": g_row.genome_id,
           "generation_of_best": g_row.population.generation}
    row.update(performance(genome, config, data, boot_idx, g_row.fitness))
    row.update(structure(genome, config, data["feature_names"]))
    return row, exp


# ---------------------------------------------------------------- driver

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print(f"wrote {path.relative_to(REPO)} ({len(rows)} rows)")


def summarise(label, rows):
    if not rows:
        return
    keys = ["train_auc", "test_auc", "test_acc_0.5", "fitness_abs_diff",
            "generation_of_best", "wall_clock_hours", "inputs_connected",
            "hidden_nodes", "phenotype_connections_enabled", "genome_genes_total",
            "depth", "skip_connections", "direct_io_edges", "hidden_fanin_mean",
            "structproxy_share_hidden_le2_features",
            "structproxy_median_features_per_hidden"]
    print(f"\n== {label}: n={len(rows)}  (median / min / max)")
    for k in keys:
        v = np.array([float(r[k]) for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            print(f"  {k:42s} all NaN")
            continue
        print(f"  {k:42s} {np.median(v):10.4g} {v.min():10.4g} {v.max():10.4g}"
              + (f"   (NaN in {len(rows) - len(v)})" if len(v) < len(rows) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-paper", action="store_true")
    args = ap.parse_args()

    db.init_db()
    data_cache, boot_cache = {}, {}

    def data_for(ds):
        sid = SPLITS[ds]
        if sid not in data_cache:
            with ReadOnlySession() as s:
                data_cache[sid] = load_split(s, sid)
            boot_cache[sid] = bootstrap_indices(len(data_cache[sid]["y_test"]))
        return data_cache[sid], boot_cache[sid]

    results = {ds: [] for ds in (*MAIN_DATASETS, "heart100")}
    pending = []
    for ds in results:
        data, boot = data_for(ds)
        for seed in SEEDS:
            name = f"10seed-{ds}-s{seed}"
            with ReadOnlySession() as s:
                exps = s.query(Experiment).filter_by(name=name).all()
                if len(exps) != 1 or exps[0].status != "completed":
                    st = exps[0].status if len(exps) == 1 else f"{len(exps)} rows"
                    pending.append((name, st))
                    continue
                exp = exps[0]
                assert str(exp.split_id) == SPLITS[ds], f"{name}: split mismatch"
                gid, key, fit, gen, (n_rows, n_keys) = best_genome_row(s, exp.id)
                row, _ = genome_metrics(s, gid, data, boot)
                assert row["generation_of_best"] == gen and row["genome_key"] == key
                row.update(experiment_meta(s, exp))
                row.update(dataset=ds, seed=seed, experiment_id=str(exp.id),
                           experiment_name=name, n_rows_at_best_fitness=n_rows,
                           n_keys_tied_in_best_generation=n_keys,
                           rerun_note=RERUN_NOTES.get((ds, seed), ""))
            results[ds].append(row)
            print(f"{name}: genome {gid[:8]} key {key} gen {gen} "
                  f"train {row['train_auc']:.4f} (|d|={row['fitness_abs_diff']:.1e}) "
                  f"test {row['test_auc']:.4f} [{row['test_auc_ci_lo']:.4f},"
                  f"{row['test_auc_ci_hi']:.4f}]", flush=True)

    write_csv(HERE / "runs.csv", [r for ds in MAIN_DATASETS for r in results[ds]])
    write_csv(HERE / "runs_heart100.csv", results["heart100"])

    paper_rows = []
    if not args.skip_paper:
        for ds, (gid, expected) in PAPER.items():
            data, boot = data_for(ds)
            with ReadOnlySession() as s:
                row, exp = genome_metrics(s, gid, data, boot)
                if str(exp.split_id) != SPLITS[ds]:
                    print(f"NOTE paper {ds}: experiment split {exp.split_id} "
                          f"!= {SPLITS[ds]} (evaluated on the latter)")
                row.update(experiment_meta(s, exp))
                row.update(dataset=ds, seed="paper", experiment_id=str(exp.id),
                           experiment_name=exp.name,
                           rerun_note=f"paper reference genome; expected test AUC {expected}")
            ok = abs(row["test_auc"] - expected) < 5e-5
            print(f"paper {ds}: test AUC {row['test_auc']:.4f} vs {expected} "
                  f"{'MATCH' if ok else 'MISMATCH'}; train {row['train_auc']:.4f} "
                  f"(|d| vs stored fitness {row['fitness_abs_diff']:.1e})")
            paper_rows.append(row)
        write_csv(HERE / "runs_paper_reference.csv", paper_rows)

    for ds in results:
        summarise(ds, results[ds])
    all_rows = [r for ds in results for r in results[ds]] + paper_rows
    if all_rows:
        print(f"\nmax |train AUC - stored fitness| over {len(all_rows)} genomes: "
              f"{max(r['fitness_abs_diff'] for r in all_rows):.3e}")
    if pending:
        print("\nPENDING (not completed):", ", ".join(f"{n} [{st}]" for n, st in pending))


if __name__ == "__main__":
    main()
