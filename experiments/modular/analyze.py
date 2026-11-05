"""Analysis for the modular-crossover study (PLAN.md "Outcomes" / "Analysis").

Reads results/<run set>/*.json (+ .detail.json) and writes, into
experiments/modular/results/:
- runs.csv           one row per (task, arm, seed), incl. library, operator and
                     module-carrier statistics
- arm_summary.csv    per (task, arm) medians / counts
- comparisons.csv    paired-by-seed comparisons 2v1, 2v3, 2v4 (+ secondary pairs)
- library.csv        every library variant added (all runs), with truth matches (mux11)

    uv run python experiments/modular/analyze.py [--set main]
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ARMS = ["plain", "modules", "sham", "topology"]
PRIMARY = [("modules", "plain"), ("modules", "sham"), ("modules", "topology")]
SECONDARY = [("sham", "plain"), ("topology", "plain"), ("sham", "topology")]
TRUTH_R = 0.9
N_BOOT = 10_000
BOOT_SEED = 42


def load(run_set):
    d = HERE / "results" / run_set
    runs = {}
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".detail.json"):
            continue
        s = json.loads(f.read_text())
        det = json.loads((d / f.name.replace(".json", ".detail.json")).read_text())
        runs[(s["task"], s["arm"], s["seed"])] = (s, det)
    return runs


def gens_to_solve(s):
    """First generation (0-based) the run's best train AUC on the final stage
    was 1.0; unsolved runs are censored at n_gen."""
    return s["solved_gen"] if s["solved_gen"] is not None else s["n_gen"]


def run_row(s, det):
    ops = Counter((r.get("op") or "none", bool(r.get("applied"))) for r in det["operators"])
    last = det["generations"][-1]
    n_any = sum(i["n_carrying_any_module"] for i in last["islands"])
    n_foreign = sum(i["n_carrying_foreign_module"] for i in last["islands"])
    lib = det["library_history"]
    truth_hits = sorted({h["truth_best"] for h in lib
                         if h.get("truth_r") is not None and h["truth_r"] >= TRUTH_R})
    indep = [h for h in lib if not h["new_module"] and h.get("n_nodes_already_tagged_same", 0) == 0]
    row = {
        "task": s["task"], "arm": s["arm"], "seed": s["seed"],
        "n_gen": s["n_gen"], "pop_size": s["pop_size"],
        "final_test_auc": s["final_test_auc"], "final_test_acc": s["final_test_acc"],
        "final_train_auc": s["final_train_auc"], "final_train_acc": s["final_train_acc"],
        "final_test_bce": s["final_test_bce"],
        "solved": s["solved_gen"] is not None, "solved_gen": s["solved_gen"],
        "gens_to_solve_censored": gens_to_solve(s),
        "final_hidden_nodes": s["final_hidden_nodes"],
        "final_connections": s["final_connections"],
        "final_best_island": s["best_island"],
        "final_best_modules": " ".join(str(m) for m in s["final_modules"]),
        "final_best_n_modules": len(s["final_modules"]),
        "wall_s": s["wall_s"],
        "n_library_modules": s["n_library_modules"],
        "n_library_variants_added": s["n_library_variants_added"],
        "n_alias_variants_independent": len(indep),
        "truth_features_matched": " ".join(truth_hits),
        "n_truth_features_matched": len(truth_hits),
        "ops_insertion": ops[("insertion", True)], "ops_crossover": ops[("crossover", True)],
        "ops_replacement": ops[("replacement", True)], "ops_migration": ops[("migration", True)],
        "ops_not_applied": sum(v for (o, a), v in ops.items() if not a),
        "final_pop_carrying_any_module": n_any,
        "final_pop_carrying_foreign_module": n_foreign,
        "final_pop_size": s["pop_size"] * s["n_islands"],
        "final_island_bests_with_modules": sum(1 for i in last["islands"] if i["best_modules"]),
    }
    return row


def boot_ci(d, stat, rng):
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    vals = stat(d[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compare(rows, task, a, b, outcome, higher_better=True):
    A = {r["seed"]: r for r in rows if r["task"] == task and r["arm"] == a}
    B = {r["seed"]: r for r in rows if r["task"] == task and r["arm"] == b}
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return None
    d = np.array([float(A[s][outcome]) - float(B[s][outcome]) for s in seeds])
    if not higher_better:
        d = -d  # positive = arm a better
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    zero = len(d) - pos - neg
    p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    mlo, mhi = boot_ci(d, np.median, rng)
    rng = np.random.default_rng(BOOT_SEED)
    alo, ahi = boot_ci(d, np.mean, rng)
    return {"task": task, "arm_a": a, "arm_b": b, "outcome": outcome,
            "direction": "a - b" if higher_better else "b - a (fewer generations = better)",
            "n_seeds": len(seeds), "n_a_better": pos, "n_b_better": neg, "n_tied": zero,
            "sign_test_p": p, "median_delta": float(np.median(d)),
            "median_delta_ci_lo": mlo, "median_delta_ci_hi": mhi,
            "mean_delta": float(np.mean(d)), "mean_delta_ci_lo": alo, "mean_delta_ci_hi": ahi,
            "median_a": float(np.median([float(A[s][outcome]) for s in seeds])),
            "median_b": float(np.median([float(B[s][outcome]) for s in seeds]))}


def holm(ps):
    order = np.argsort([np.inf if not np.isfinite(p) else p for p in ps])
    adj = [float("nan")] * len(ps)
    running = 0.0
    m = sum(1 for p in ps if np.isfinite(p))
    for rank, i in enumerate(order):
        if not np.isfinite(ps[i]):
            continue
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def write(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main")
    args = ap.parse_args()
    runs = load(args.set)
    out = HERE / "results" if args.set == "main" else HERE / "results" / args.set
    rows = [run_row(s, det) for (_, _, _), (s, det) in sorted(runs.items())]
    write(out / "runs.csv", rows)

    tasks = sorted({r["task"] for r in rows})
    summ = []
    for t in tasks:
        for a in ARMS:
            R = [r for r in rows if r["task"] == t and r["arm"] == a]
            if not R:
                continue
            solved = [r["solved_gen"] for r in R if r["solved"]]
            summ.append({"task": t, "arm": a, "n_runs": len(R),
                         "median_test_auc": float(np.median([r["final_test_auc"] for r in R])),
                         "min_test_auc": float(np.min([r["final_test_auc"] for r in R])),
                         "max_test_auc": float(np.max([r["final_test_auc"] for r in R])),
                         "median_test_acc": float(np.median([r["final_test_acc"] for r in R])),
                         "median_train_auc": float(np.median([r["final_train_auc"] for r in R])),
                         "n_solved": len(solved),
                         "median_gens_to_solve_censored": float(np.median(
                             [r["gens_to_solve_censored"] for r in R])),
                         "median_solved_gen_among_solved": float(np.median(solved))
                         if solved else float("nan"),
                         "median_hidden_nodes": float(np.median([r["final_hidden_nodes"] for r in R])),
                         "median_connections": float(np.median([r["final_connections"] for r in R])),
                         "median_wall_min": float(np.median([r["wall_s"] for r in R])) / 60,
                         "median_library_modules": float(np.median([r["n_library_modules"] for r in R])),
                         "median_final_pop_carrying_foreign": float(np.median(
                             [r["final_pop_carrying_foreign_module"] for r in R])),
                         "median_truth_features_matched": float(np.median(
                             [r["n_truth_features_matched"] for r in R]))})
    write(out / "arm_summary.csv", summ)

    comps = []
    for t in tasks:
        for role, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
            for a, b in pairs:
                for outcome, hb in (("final_test_auc", True), ("gens_to_solve_censored", False),
                                    ("final_test_acc", True)):
                    c = compare(rows, t, a, b, outcome, hb)
                    if c is None:
                        continue
                    c["role"] = role if outcome != "final_test_acc" else role + " (accuracy)"
                    comps.append(c)
    prim = [i for i, c in enumerate(comps) if c["role"] == "primary"]
    adj = holm([comps[i]["sign_test_p"] for i in prim])
    for i, a in zip(prim, adj):
        comps[i]["holm_p_primary_family"] = a
    write(out / "comparisons.csv", comps)

    libs = []
    for (t, a, s), (summary, det) in sorted(runs.items()):
        for h in det["library_history"]:
            libs.append({"task": t, "arm": a, "seed": s, **{k: (" ".join(map(str, v))
                                                                if isinstance(v, list) else v)
                                                            for k, v in h.items()}})
    write(out / "library.csv", libs)


if __name__ == "__main__":
    main()
