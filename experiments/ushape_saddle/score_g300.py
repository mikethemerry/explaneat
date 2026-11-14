"""Shape scores for the longer runs on the both dataset (read-only on the database; no explanations).

For each shape-C-s{n}-g300 run and each checkpoint (generations 100, 200, 300 = generation indices
<= 99, 199, 299), the best-by-train genome up to that point (PREREG tie-break: highest fitness,
earliest generation, lowest key) is scored:
  * train / test AUC, hidden units, enabled phenotype connections (experiments/10seed/compute_runs.py),
    species count in that generation (from the run log: the populations table's num_species is not filled);
  * U: the model logit over sodium 128..150 mmol/L (1 mmol/L steps; other inputs at the typical case,
    as in shape_plots.py), Pearson r with the true U term, and whether both ends rise (logit at 128 and at
    150 each more than 1 above the curve's minimum);
  * saddle: the logit on a 41 x 41 heart rate x SBP grid (training 1st-99th percentiles, other inputs
    typical), mean removed, Pearson r with the true -zh*zp surface, and the interaction share of variance
    (variance left after the best additive f(hr) + g(sbp) fit, as in shape_plots.py).
A shape counts as found when r >= 0.9. Also checks that the generation-100 best fitness of each rerun
equals the original 100-generation run's best fitness exactly.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/score_g300.py
Writes g300_scores.json and g300_scores.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("compute_runs_10seed", REPO / "experiments" / "10seed" / "compute_runs.py")
_cr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cr)
ReadOnlySession, bootstrap_indices, genome_metrics, load_split = _cr.ReadOnlySession, _cr.bootstrap_indices, _cr.genome_metrics, _cr.load_split

import data  # noqa: E402
from explaneat.db import db  # noqa: E402

FOUND_R = 0.9
CHECKPOINTS = (100, 200, 300)


def best_upto(s, eid, last_gen):
    r = s.execute(text("""
        SELECT g.id, g.genome_id, g.fitness, p.generation, p.num_species FROM genomes g
        JOIN populations p ON g.population_id = p.id
        WHERE p.experiment_id = :eid AND p.generation <= :lg AND g.fitness IS NOT NULL AND g.fitness <> 'NaN'::float8
        ORDER BY g.fitness DESC, p.generation ASC, g.genome_id ASC LIMIT 1"""), {"eid": eid, "lg": last_gen}).fetchone()
    nsp = s.execute(text("SELECT num_species FROM populations WHERE experiment_id = :eid AND generation = :lg"),
                    {"eid": eid, "lg": last_gen}).scalar()
    return str(r[0]), r[1], r[2], r[3], nsp


def log_species(seed, gen_index):
    import re
    lines = [l for l in (HERE / "logs" / f"both-s{seed}-g300.log").read_text().splitlines() if "Species:" in l]
    return int(re.search(r"Species: (\d+)", lines[gen_index]).group(1))


def exp_id(s, name):
    return str(s.execute(text("SELECT id FROM experiments WHERE name = :n"), {"n": name}).scalar())


def shapes(genome_id):
    from explaneat.verify.model import load
    from explaneat.verify.probe import hold_row, to_z
    vm = load(genome_id, operations=[])
    names = list(vm.base.feature_names)
    assert names == data.FEATURES, names
    col = {n: i for i, n in enumerate(names)}
    base = hold_row(vm, "typical")
    xs = np.arange(128.0, 151.0)
    R = np.repeat(base[None, :], len(xs), axis=0)
    R[:, col["sodium"]] = xs
    score = vm.m.logit(to_z(vm, R))
    tu = next(iter(data.u_term(R).values()))
    u_r = float(np.corrcoef(score, tu)[0, 1])
    mn = score.min()
    ends = bool(score[0] - mn > 1 and score[-1] - mn > 1)
    lo_h, hi_h = np.percentile(vm.m.raw_train[:, col["heart_rate"]], [1, 99])
    lo_p, hi_p = np.percentile(vm.m.raw_train[:, col["sbp"]], [1, 99])
    H, P = np.meshgrid(np.linspace(lo_h, hi_h, 41), np.linspace(lo_p, hi_p, 41))
    R = np.repeat(base[None, :], H.size, axis=0)
    R[:, col["heart_rate"]] = H.ravel(); R[:, col["sbp"]] = P.ravel()
    S = vm.m.logit(to_z(vm, R)).reshape(H.shape)
    T = next(iter(data.saddle_term(R).values())).reshape(H.shape)
    Sc, Tc = S - S.mean(), T - T.mean()
    s_r = float(np.corrcoef(Sc.ravel(), Tc.ravel())[0, 1]) if Sc.std() > 0 else 0.0
    resid = Sc - Sc.mean(axis=0, keepdims=True) - Sc.mean(axis=1, keepdims=True)
    inter = float(resid.var() / Sc.var()) if Sc.var() > 0 else 0.0
    return dict(u_r=u_r, u_min_at=float(xs[int(np.argmin(score))]), u_rise_128=float(score[0] - mn),
                u_rise_150=float(score[-1] - mn), u_both_ends=ends, saddle_r=s_r, saddle_interaction_share=inter)


def main():
    reg = json.loads((HERE / "datasets.json").read_text())["both"]
    db.init_db()
    rows, checks = [], {}
    with ReadOnlySession() as s:
        dat = load_split(s, reg["split_id"])
        boot = bootstrap_indices(len(dat["y_test"]))
        for seed in range(5):
            eid = exp_id(s, f"shape-C-s{seed}-g300")
            orig = exp_id(s, f"shape-C-s{seed}")
            _, ok, of, og, _ = best_upto(s, orig, 99)
            for g in CHECKPOINTS:
                gid, key, fit, gen, nsp = best_upto(s, eid, g - 1)
                if g == 100:
                    checks[seed] = dict(original_best_fitness=of, rerun_best_fitness=fit, exact=(of == fit),
                                        original_key=ok, rerun_key=key, original_gen=og, rerun_gen=gen)
                m, _ = genome_metrics(s, gid, dat, boot)
                row = dict(seed=seed, generation=g, experiment_id=eid, genome_id=gid, genome_key=key, generation_of_best=gen,
                           fitness=fit, train_auc=m["train_auc"], test_auc=m["test_auc"], hidden=m["hidden_nodes"],
                           connections=m["phenotype_connections_enabled"], species=log_species(seed, g - 1))
                row.update(shapes(gid))
                row["found_u"] = row["u_r"] >= FOUND_R
                row["found_saddle"] = row["saddle_r"] >= FOUND_R
                row["found_both"] = row["found_u"] and row["found_saddle"]
                print(f"s{seed} g{g}: key {key} (gen {gen}) train {row['train_auc']:.3f} test {row['test_auc']:.3f} "
                      f"U r {row['u_r']:.2f} ends {row['u_both_ends']} saddle r {row['saddle_r']:.2f} "
                      f"inter {row['saddle_interaction_share']:.2f} hidden {row['hidden']} conns {row['connections']} "
                      f"species {row['species']}", flush=True)
                rows.append(row)
    (HERE / "g300_scores.json").write_text(json.dumps(dict(found_threshold_r=FOUND_R, gen100_replay_check=checks,
                                                           rows=rows), indent=1, default=str) + "\n")
    with open(HERE / "g300_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(checks, indent=1, default=str))
    for g in CHECKPOINTS:
        r = [x for x in rows if x["generation"] == g]
        print(f"gen {g}: U {sum(x['found_u'] for x in r)}/5, saddle {sum(x['found_saddle'] for x in r)}/5, "
              f"both {sum(x['found_both'] for x in r)}/5")


if __name__ == "__main__":
    main()
