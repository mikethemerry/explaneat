"""Checkpoint and curve measurements for the very long runs (RESULTS.md section 2.4). Read-only on the database.

For each run (experiment ``longrun-<shape-C|churn>-s<seed>``):

* ``curves/<name>.csv.gz``: one row per generation, from the generation's stored best genome
  (light storage keeps it): training AUC on the fit rows (= the stored fitness; checked), monitoring
  AUC, test AUC, hidden units and enabled phenotype connections are not recomputed here (the
  genotype sizes are in logs/<name>.gen.csv); test AUC is computed for reporting only.
* ``checkpoints.csv`` / ``checkpoints.json``: at the checkpoints (100, 200, 500, 1000, 2000, 5000,
  10000 reached, and the stop) and on the log grid, the best-by-train genome up to that generation
  (highest fitness, ties earliest generation then lowest key): train / monitoring / test AUC, size
  and depth (compute_runs.structure), for shape-C U r, saddle r (score_g300's scoring), the
  spurious-structure share (1 - R^2 of the model logit vs the oracle logit on 20,000 fresh rows
  from the generator) and the noise-input variance share; for both datasets the marginal-capacity
  share (hidden nodes whose mean-ablation lowers the fit-row training AUC by < 0.01).

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/long_run/score_checkpoints.py [names...]
"""
import csv
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "ushape_saddle")):
    if p not in sys.path:
        sys.path.insert(0, p)

from threadpoolctl import threadpool_limits  # noqa: E402

threadpool_limits(1)
import torch  # noqa: E402

torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score  # noqa: E402
from sqlalchemy import text  # noqa: E402

import data as shape_data  # noqa: E402
from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat as nneat  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.serialization import deserialize_genome  # noqa: E402

_spec = importlib.util.spec_from_file_location("compute_runs_10seed", REPO / "experiments" / "10seed" / "compute_runs.py")
_cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cr)
ReadOnlySession, load_split, structure = _cr.ReadOnlySession, _cr.load_split, _cr.structure

CHECKPOINTS = (100, 200, 500, 1000, 2000, 5000, 10000)
GRID = (10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000)
MARGINAL_DROP = 0.01
FRESH_N, FRESH_SEED = 20_000, 20261116
RUNS = ["longrun-shape-C-s0", "longrun-shape-C-s1", "longrun-shape-C-s2", "longrun-churn-s0", "longrun-churn-s1"]


def auc(y, s):
    return float(roc_auc_score(y, s))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def task_of(name):
    return "both" if "shape-C" in name else "churn"


def run_info(s, name):
    e = s.execute(text("select id, neat_config_text, config_json, split_id, hardware_info from experiments where name = :n"),
                  {"n": name}).one()
    return dict(id=str(e[0]), config=load_neat_config(e[1] or "", e[2]), split_id=str(e[3]), hw=e[4])


def stored_rows(s, eid):
    """(generation, key, fitness, genome uuid) for every stored genome."""
    return [(int(g), int(k), float(f), str(i)) for g, k, f, i in s.execute(text("""
        select p.generation, g.genome_id, g.fitness, g.id from genomes g join populations p on g.population_id = p.id
        where p.experiment_id = :e and g.fitness is not null and g.fitness <> 'NaN'::float8"""), {"e": eid})]


def best_upto(rows, last_index):
    c = [r for r in rows if r[0] <= last_index]
    return max(c, key=lambda r: (r[2], -r[0], -r[1]))


def genome_of(s, gid, config):
    d = s.execute(text("select genome_data from genomes where id = :i"), {"i": gid}).scalar()
    return deserialize_genome(d, config)


def predict(genome, config, X):
    with torch.no_grad():
        return nneat(genome, config).forward(torch.tensor(X, dtype=torch.float64)).detach().cpu().numpy().ravel()


def fresh_rows():
    return shape_data.inputs(n=FRESH_N, seed=FRESH_SEED)


def shape_scores(vm, fit_pos, Xfresh):
    from explaneat.verify.probe import hold_row, to_z
    names = list(vm.base.feature_names)
    assert names == shape_data.FEATURES, names
    col = {n: i for i, n in enumerate(names)}
    base = hold_row(vm, "typical")
    xs = np.arange(128.0, 151.0)
    R = np.repeat(base[None, :], len(xs), axis=0)
    R[:, col["sodium"]] = xs
    su = vm.m.logit(to_z(vm, R))
    u_r = corr(su, next(iter(shape_data.u_term(R).values())))
    lo_h, hi_h = np.percentile(vm.m.raw_train[:, col["heart_rate"]], [1, 99])
    lo_p, hi_p = np.percentile(vm.m.raw_train[:, col["sbp"]], [1, 99])
    H, P = np.meshgrid(np.linspace(lo_h, hi_h, 41), np.linspace(lo_p, hi_p, 41))
    R = np.repeat(base[None, :], H.size, axis=0)
    R[:, col["heart_rate"]] = H.ravel()
    R[:, col["sbp"]] = P.ravel()
    S = vm.m.logit(to_z(vm, R))
    T = next(iter(shape_data.saddle_term(R).values()))
    s_r = corr(S - S.mean(), T - T.mean())
    # spurious share and noise share on fresh rows from the generator
    Z = to_z(vm, Xfresh)
    eta = vm.m.logit(Z)
    oracle, _ = shape_data.ground_truth("both", Xfresh)
    r = corr(eta, oracle)
    Xn = Xfresh.copy()
    Xn[:, col["noise"]] = float(np.mean(vm.m.raw_train[fit_pos, col["noise"]]))
    dn = eta - vm.m.logit(to_z(vm, Xn))
    return dict(u_r=u_r, u_min_at=float(xs[int(np.argmin(su))]), saddle_r=s_r, oracle_r=r,
                spurious_share=1.0 - r * r, noise_var_share=float(dn.var() / eta.var()) if eta.var() > 0 else 0.0)


def marginal(vm, fit_pos):
    from explaneat.verify.probe import Ablation
    X, y = vm.m.X[fit_pos], vm.m.y[fit_pos]
    a0 = auc(y, vm.m.logit(X))
    hidden = [n.id for n in vm.structure.nodes if n.type == NodeType.HIDDEN]
    drops = {}
    for h in hidden:
        ab = Ablation(vm, nodes=[h], to="mean")
        drops[h] = a0 - auc(y, ab.logit(X))
    n_marg = sum(1 for d in drops.values() if d < MARGINAL_DROP)
    return dict(marginal_share=(n_marg / len(hidden)) if hidden else None, n_marginal=n_marg,
                n_hidden_ablated=len(hidden), drop_max=max(drops.values()) if drops else None,
                drop_median=float(np.median(list(drops.values()))) if drops else None, fit_auc_vm=a0)


def score_point(s, info, dat, fit_pos, mon_pos, row, task, Xfresh, kind):
    from explaneat.verify.model import VerifyModel, load_base
    gen, key, fit, gid = row
    genome = genome_of(s, gid, info["config"])
    p_tr = predict(genome, info["config"], dat["X_train"])
    out = dict(kind=kind, generation_of_best=gen + 1, genome_key=key, genome_id=gid, fitness=fit,
               train_auc=auc(dat["y_train"][fit_pos], p_tr[fit_pos]),
               monitor_auc=auc(dat["y_train"][mon_pos], p_tr[mon_pos]),
               full_train_auc=auc(dat["y_train"], p_tr),
               test_auc=auc(dat["y_test"], predict(genome, info["config"], dat["X_test"])))
    out["fitness_check_abs"] = abs(out["train_auc"] - fit)
    st = structure(genome, info["config"], dat["feature_names"])
    out.update({k: st[k] for k in ("hidden_nodes", "phenotype_connections_enabled", "depth", "inputs_connected",
                                   "genome_genes_total", "skip_connections", "direct_io_edges")})
    vm = VerifyModel(load_base(gid), operations=[])
    if task == "both":
        out.update(shape_scores(vm, fit_pos, Xfresh))
    out.update(marginal(vm, fit_pos))
    return out


def main(names):
    db.init_db()
    Xfresh = fresh_rows()
    allrows = []
    (HERE / "curves").mkdir(exist_ok=True)
    for name in names:
        t0 = time.time()
        task = task_of(name)
        split = json.loads((HERE / f"monitor_split_{task}.json").read_text())
        fit_pos, mon_pos = np.array(split["fit_positions"]), np.array(split["monitor_positions"])
        stop_p = HERE / "logs" / f"{name}.stop.json"
        stop = json.loads(stop_p.read_text()) if stop_p.exists() else None
        with ReadOnlySession() as s:
            info = run_info(s, name)
            dat = load_split(s, info["split_id"])
            rows = stored_rows(s, info["id"])
            last = max(r[0] for r in rows)
            # curves: the generation's best stored genome, every generation
            per_gen = {}
            for r in rows:
                if r[0] not in per_gen or (r[2], -r[1]) > (per_gen[r[0]][2], -per_gen[r[0]][1]):
                    per_gen[r[0]] = r
            cpath = HERE / "curves" / f"{name}.csv.gz"
            with gzip.open(cpath, "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["generation", "genome_key", "fitness", "train_auc", "monitor_auc", "test_auc"])
                for g in sorted(per_gen):
                    _, k, f, gid = per_gen[g]
                    gn = genome_of(s, gid, info["config"])
                    p = predict(gn, info["config"], dat["X_train"])
                    w.writerow([g + 1, k, f, auc(dat["y_train"][fit_pos], p[fit_pos]), auc(dat["y_train"][mon_pos], p[mon_pos]),
                                auc(dat["y_test"], predict(gn, info["config"], dat["X_test"]))])
            print(f"{name}: curves {len(per_gen)} generations ({time.time() - t0:.0f} s)", flush=True)
            finished = stop is not None
            points = {}
            for c in CHECKPOINTS:
                if c - 1 <= last:
                    points.setdefault(c, set()).add("checkpoint")
            for c in GRID:
                if c - 1 <= last:
                    points.setdefault(c, set()).add("grid")
            points.setdefault(last + 1, set()).add("stop" if finished else "latest")
            for c in sorted(points):
                row = best_upto(rows, c - 1)
                r = score_point(s, info, dat, fit_pos, mon_pos, row, task, Xfresh, "+".join(sorted(points[c])))
                r = dict(run=name, task=task, generation=c, **r)
                allrows.append(r)
                print(f"  g{c}: best gen {r['generation_of_best']} key {r['genome_key']} train {r['train_auc']:.4f} "
                      f"mon {r['monitor_auc']:.4f} test {r['test_auc']:.4f} hidden {r['hidden_nodes']} "
                      f"conns {r['phenotype_connections_enabled']} depth {r['depth']} "
                      + (f"U {r['u_r']:.3f} saddle {r['saddle_r']:.3f} spur {r['spurious_share']:.3f} "
                         f"noise {r['noise_var_share']:.4f} " if task == "both" else "")
                      + f"marg {r['marginal_share']} ({time.time() - t0:.0f} s)", flush=True)
    out = HERE / "checkpoints.json"
    prev = json.loads(out.read_text()) if out.exists() else {"rows": []}
    keep = [r for r in prev["rows"] if r["run"] not in names]
    rows_all = keep + allrows
    rows_all.sort(key=lambda r: (r["run"], r["generation"]))
    out.write_text(json.dumps(dict(marginal_drop=MARGINAL_DROP, fresh_rows=FRESH_N, fresh_seed=FRESH_SEED,
                                   rows=rows_all), indent=1, default=float) + "\n")
    cols = sorted({k for r in rows_all for k in r}, key=lambda k: list(rows_all[0]).index(k) if k in rows_all[0] else 999)
    with open(HERE / "checkpoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_all)


if __name__ == "__main__":
    main(sys.argv[1:] or RUNS)
