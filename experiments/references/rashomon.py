"""Task 6: Rashomon-set membership in AUC terms.

A model is inside the epsilon-set for reference r when test AUC >= r - eps.
References per dataset:
  (i)   best observed: the highest best-genome test AUC among that run set's seeds;
  (ii)  logistic regression on the same split (paper protocol; churn from
        experiments/churn/baselines.json as the reported value);
  (iii) gradient boosting on the same split (HistGradientBoosting, CV-tuned on
        train only; results/thesis/baselines.json).
eps in {0.005, 0.01, 0.02}.

Inputs: runs CSVs (per-seed best genome test AUC, unchanged), baselines.json,
rashomon_population.csv (final-generation genomes).
Outputs: rashomon_seeds.csv, rashomon_counts.csv, rashomon_population_summary.csv,
rashomon_summary.json (all under results/thesis/).

Run: uv run python scripts/thesis/rashomon.py
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "thesis"
EPS = [0.005, 0.01, 0.02]

RUN_SETS = [  # (run_set, csv, dataset filter, baseline key, label)
    ("heart", "experiments/10seed/runs.csv", "heart", "heart", "pre-registered"),
    ("monk2", "experiments/10seed/runs.csv", "monk2", "monk2", "pre-registered"),
    ("adult", "experiments/10seed/runs.csv", "adult", "adult", "pre-registered"),
    ("heart100", "experiments/10seed/runs_heart100.csv", None, "heart", "supplementary"),
    ("churn100", "experiments/churn/runs_churn100.csv", None, "churn", "supplementary"),
    ("churn200", "experiments/churn/runs_churn200.csv", None, "churn", "supplementary"),
]
PAPER_LR = {"adult": 0.905, "heart": 0.919, "monk2": 0.423}


def main():
    bl = json.loads((RES / "baselines.json").read_text())
    churn_file = json.loads((ROOT / "experiments/churn/baselines.json").read_text())
    pop = list(csv.DictReader(open(RES / "rashomon_population.csv")))

    seeds_out, counts_out, pop_out, summary = [], [], [], {}
    for rs, path, ds, bkey, kind in RUN_SETS:
        runs = [r for r in csv.DictReader(open(ROOT / path)) if ds is None or r["dataset"] == ds]
        auc = {int(r["seed"]): float(r["test_auc"]) for r in runs}
        refs = {"best_observed": max(auc.values())}
        if bkey == "churn":
            refs["logistic"] = churn_file["l2_logistic"]["test_auc"]
        else:
            refs["logistic"] = bl[bkey]["logistic"]["test_auc"]
        refs["gradient_boosting"] = bl[bkey]["gradient_boosting"]["test_auc"]
        summary[rs] = {"kind": kind, "n_seeds": len(auc), "refs": refs,
                       "logistic_refit_same_protocol": bl[bkey]["logistic"]["test_auc"],
                       "paper_logistic": PAPER_LR.get(bkey),
                       "seed_auc_min": min(auc.values()), "seed_auc_median": float(np.median(list(auc.values()))),
                       "seed_auc_max": max(auc.values()), "counts": {}, "population": {}}
        for s, a in sorted(auc.items()):
            row = {"run_set": rs, "kind": kind, "seed": s, "test_auc": round(a, 4)}
            for rn, rv in refs.items():
                row[f"gap_to_{rn}"] = round(a - rv, 4)
                for e in EPS:
                    row[f"in_{rn}_eps{e}"] = int(a >= rv - e)
            seeds_out.append(row)
        P = np.array([float(r["test_auc"]) for r in pop if r["run_set"] == rs])
        by_run = {}
        for r in pop:
            if r["run_set"] == rs:
                by_run.setdefault(r["seed"], []).append(float(r["test_auc"]))
        for rn, rv in refs.items():
            for e in EPS:
                n_in = int(sum(a >= rv - e for a in auc.values()))
                counts_out.append({"run_set": rs, "kind": kind, "reference": rn, "reference_auc": round(rv, 4),
                                   "eps": e, "threshold_auc": round(rv - e, 4),
                                   "seeds_in_set": n_in, "n_seeds": len(auc)})
                summary[rs]["counts"][f"{rn}@{e}"] = n_in
                if len(P):
                    share = float(np.mean(P >= rv - e))
                    per_run = [float(np.mean(np.array(v) >= rv - e)) for v in by_run.values()]
                    pop_out.append({"run_set": rs, "kind": kind, "reference": rn, "reference_auc": round(rv, 4),
                                    "eps": e, "n_genomes": len(P), "share_in_set": round(share, 4),
                                    "per_run_share_min": round(min(per_run), 4),
                                    "per_run_share_median": round(float(np.median(per_run)), 4),
                                    "per_run_share_max": round(max(per_run), 4),
                                    "runs_with_any_genome_in_set": int(sum(x > 0 for x in per_run))})
                    summary[rs]["population"][f"{rn}@{e}"] = round(share, 4)
        if len(P):
            summary[rs]["population_test_auc"] = {
                "n": int(len(P)), "nan": int(np.isnan(P).sum()),
                "min": float(np.nanmin(P)), "p10": float(np.nanpercentile(P, 10)),
                "median": float(np.nanmedian(P)), "p90": float(np.nanpercentile(P, 90)),
                "max": float(np.nanmax(P)),
                "per_run_median_auc": {k: float(np.median(v)) for k, v in sorted(by_run.items(), key=lambda t: int(t[0]))}}
    for name, rows in (("rashomon_seeds.csv", seeds_out), ("rashomon_counts.csv", counts_out),
                       ("rashomon_population_summary.csv", pop_out)):
        with open(RES / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (RES / "rashomon_summary.json").write_text(json.dumps(summary, indent=2))
    for rs, s in summary.items():
        print(rs, {k: round(v, 4) for k, v in s["refs"].items()}, s["counts"])
        print("   pop", s["population"])


if __name__ == "__main__":
    main()
