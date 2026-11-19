"""Experiment 2 (three-study set): recompute the headline numbers from the repo files.
Run: python3 e2_stats.py <repo root> > e2_stats.json (read by e2figs.py; data/exp2.py copies its numbers)."""
import csv, json, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
def rows(p, **flt):
    with open(R / p) as f:
        return [r for r in csv.DictReader(f) if all(r[k] == v for k, v in flt.items())]
def q(xs):
    xs = sorted(xs); n = len(xs)
    def pct(p):  # numpy default (linear)
        k = (n - 1) * p; lo = int(k); hi = min(lo + 1, n - 1); return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return {"median": st.median(xs), "q1": pct(.25), "q3": pct(.75), "min": xs[0], "max": xs[-1]}
S = {
  "adult":    ("experiments/10seed/runs.csv", {"dataset": "adult"}, "experiments/10seed/explanations_v4.csv", {"dataset": "adult"}),
  "churn100": ("experiments/churn/runs_churn100.csv", {}, "experiments/churn/explanations_churn100_v4.csv", {}),
  "heart100": ("experiments/10seed/runs_heart100.csv", {}, "experiments/10seed/explanations_heart100_v4.csv", {}),
}
out = {}
for k, (rp, rf, ep, ef) in S.items():
    runs = rows(rp, **rf); ex = rows(ep, **ef)
    assert len(runs) == 10 and len(ex) == 10, (k, len(runs), len(ex))
    recs = [json.loads((R / "experiments/10seed/explanations_v4" / f"{e['label']}.json").read_text()) for e in ex]
    o = {
      "seeds": sorted(int(r["seed"]) for r in runs),
      "test_auc": q([float(r["test_auc"]) for r in runs]),
      "test_auc_by_seed": {int(r["seed"]): float(r["test_auc"]) for r in runs},
      "test_ci_by_seed": {int(r["seed"]): [float(r["test_auc_ci_lo"]), float(r["test_auc_ci_hi"])] for r in runs},
      "train_auc": q([float(r["train_auc"]) for r in runs]),
      "hidden": q([int(r["hidden_nodes"]) for r in runs]),
      "conns": q([int(r["phenotype_connections_enabled"]) for r in runs]),
      "n_train": runs[0]["n_train"], "n_test": runs[0]["n_test"], "n_inputs": runs[0]["n_inputs"],
      "gens": sorted({r["total_generations"] for r in runs}),
      "train_h": q([float(r["wall_clock_hours"]) for r in runs]),
      "full_cov": sum(e["full_coverage"] == "True" for e in ex),
      "within_budget": sum(e["within_budget"] == "True" for e in ex),
      "protocol": sorted({e["protocol"] for e in ex}),
      "leaves": q([int(e["n_leaf"]) for e in ex]),
      "comps": q([int(e["n_composition"]) for e in ex]),
      "minutes": q([float(e["agent_minutes_db"]) for e in ex]),
      "minutes_by_seed": {int(e["seed"]): float(e["agent_minutes_db"]) for e in ex},
      "hidden_by_seed": {int(r["seed"]): int(r["hidden_nodes"]) for r in runs},
      "fp_max": max(float(e["function_preservation_max_abs_err"]) for e in ex),
      "verified_by_mike": sorted({e["verified_by_mike"] for e in ex}),
      "complete_ok": sum(bool(r["recheck_summary"]["complete_ok"]) for r in recs),
      "mismatch": sum(r["recheck_summary"]["mismatch"] for r in recs),
      "needs_review": sum(r["recheck_summary"]["needs_review"] for r in recs),
      "mike_verified_true": sum(bool((r.get("mike_verification") or {}).get("verified")) for r in recs),
      "mike_minutes": q([float((r.get("mike_verification") or {})["minutes"]) for r in recs
                         if (r.get("mike_verification") or {}).get("minutes") is not None]),
      "genome_match": all(e["genome_id"] == next(r for r in runs if r["seed"] == e["seed"])["genome_id"] for e in ex),
    }
    out[k] = o
print(json.dumps(out, indent=1))
