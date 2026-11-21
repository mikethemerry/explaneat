"""Final runs against the fixed runs and the comparators; harness check.

Reads selection.json, runs/<dataset>/<selected>_full_s<seed>.json (seeds 0-4), the harness-check runs
(runs/<dataset>/<fixed>_full_s0.json), the fixed runs' records (seeds 0-4 of
experiments/churn/runs_churn100.csv and experiments/ushape_saddle/runs.csv) and
experiments/benchmarks/benchmarks.csv. Writes final.csv (one row per dataset x arm x seed),
summary.json and summary.md (generated tables).

    uv run python experiments/propneat_sensitivity/summarise.py
"""
import csv
import json
import statistics

from common import DATASETS, FINAL_SEEDS, HERE, fixed_config_id, run_path

COMPARATORS = ["lr_l2", "lr_l1", "tree", "rf", "gb", "ebm", "mlp"]
BENCH = HERE.parents[0] / "benchmarks" / "benchmarks.csv"


def fixed_records(d):
    spec = DATASETS[d]
    with open(spec["fixed_runs"]) as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == spec["fixed_dataset"]]
    by_seed = {int(r["seed"]): r for r in rows}
    return [by_seed[s] for s in FINAL_SEEDS]


def med_range(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def main():
    sel = json.loads((HERE / "selection.json").read_text())
    with open(BENCH) as f:
        bench = list(csv.DictReader(f))
    out_rows, summary = [], {}
    for d, spec in DATASETS.items():
        cid = sel[d]["selected"]
        tuned = [json.loads(run_path(d, cid, "full", s).read_text()) for s in FINAL_SEEDS]
        fixed = fixed_records(d)
        for r in tuned:
            assert r["eval_split"] == "test" and r["generations"] == 100
            b = r["best"]
            out_rows.append({"dataset": d, "arm": "tuned", "config": cid, "seed": r["seed"],
                             "train_auc": b["train_auc"], "test_auc": b["test_auc"],
                             "generation_of_best": b["generation"], "genome_key": b["key"],
                             "hidden_nodes": b["hidden_nodes"],
                             "phenotype_connections_enabled": b["phenotype_connections_enabled"],
                             "inputs_connected": b["inputs_connected"], "source": str(run_path(d, cid, "full", r["seed"]).relative_to(HERE))})
        for r in fixed:
            out_rows.append({"dataset": d, "arm": "fixed", "config": fixed_config_id(d), "seed": int(r["seed"]),
                             "train_auc": float(r["train_auc"]), "test_auc": float(r["test_auc"]),
                             "generation_of_best": int(r["generation_of_best"]), "genome_key": int(r["genome_key"]),
                             "hidden_nodes": int(r["hidden_nodes"]),
                             "phenotype_connections_enabled": int(r["phenotype_connections_enabled"]),
                             "inputs_connected": int(r["inputs_connected"]),
                             "source": str(spec["fixed_runs"].relative_to(HERE.parents[1]))})

        # Harness check: the in-memory loop with the fixed config, seed 0, equals the stored run.
        chk = json.loads(run_path(d, fixed_config_id(d), "full", 0).read_text())["best"]
        f0 = fixed[0]
        harness = {"generation": chk["generation"] == int(f0["generation_of_best"]),
                   "key": chk["key"] == int(f0["genome_key"]),
                   "train_auc": abs(chk["train_auc"] - float(f0["train_auc"])) < 1e-9,
                   "test_auc": abs(chk["test_auc"] - float(f0["test_auc"])) < 1e-9,
                   "hidden_nodes": chk["hidden_nodes"] == int(f0["hidden_nodes"]),
                   "connections": chk["phenotype_connections_enabled"] == int(f0["phenotype_connections_enabled"])}

        comps = {r["model_key"]: r for r in bench if r["dataset"] == spec["bench_key"] and r["model_key"] in COMPARATORS}
        best_k = max(comps, key=lambda k: float(comps[k]["test_value"]))
        t_auc = [r["best"]["test_auc"] for r in tuned]
        f_auc = [float(r["test_auc"]) for r in fixed]
        summary[d] = {
            "label": spec["label"], "fixed_config": fixed_config_id(d), "selected_config": cid,
            "selected_is_fixed": sel[d]["selected_is_fixed"],
            "cv_mean_auc_selected": sel[d]["selected_mean_auc"], "cv_mean_auc_fixed": sel[d]["fixed_mean_auc"],
            "tuned_test_auc": med_range(t_auc), "fixed_test_auc": med_range(f_auc),
            "tuned_minus_fixed_median": statistics.median(t_auc) - statistics.median(f_auc),
            "tuned_hidden": med_range([r["best"]["hidden_nodes"] for r in tuned]),
            "fixed_hidden": med_range([int(r["hidden_nodes"]) for r in fixed]),
            "tuned_connections": med_range([r["best"]["phenotype_connections_enabled"] for r in tuned]),
            "fixed_connections": med_range([int(r["phenotype_connections_enabled"]) for r in fixed]),
            "best_comparator": {"model_key": best_k, "model": comps[best_k]["model"],
                                "test_auc": float(comps[best_k]["test_value"])},
            "gb_test_auc": float(comps["gb"]["test_value"]),
            "tuned_seeds_above_gb": sum(a > float(comps["gb"]["test_value"]) for a in t_auc),
            "fixed_seeds_above_gb": sum(a > float(comps["gb"]["test_value"]) for a in f_auc),
            "harness_check_seed0": harness, "harness_ok": all(harness.values()),
            "wall_clock_h_tuned": [json.loads(run_path(d, cid, "full", s).with_suffix(".meta.json").read_text())["wall_clock_s"] / 3600
                                   for s in FINAL_SEEDS],
        }
        if sel[d]["selected_is_fixed"]:
            summary[d]["final_equals_fixed_per_seed"] = [abs(a - b) < 1e-9 for a, b in zip(t_auc, f_auc)]

    with open(HERE / "final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_md(summary)
    print(json.dumps({d: {k: v for k, v in s.items() if k in ("selected_config", "tuned_test_auc", "fixed_test_auc", "harness_ok")}
                      for d, s in summary.items()}, indent=1))


def f3(x):
    return f"{x:.3f}"


def mr(m):
    return f"{f3(m['median'])} ({f3(m['min'])}-{f3(m['max'])})"


def write_md(summary):
    cv = list(csv.DictReader(open(HERE / "cv.csv")))
    lines = ["<!-- Generated by experiments/propneat_sensitivity/summarise.py from cv.csv, selection.json, "
             "final.csv and experiments/benchmarks/benchmarks.csv; do not edit by hand. -->", "",
             "## CV surface: mean (SD) held-out-fold AUC, seed 0, 5 folds of the training rows", "",
             "| configuration | " + " | ".join(DATASETS[d]["label"] for d in DATASETS) + " |",
             "|---|" + "---|" * len(DATASETS)]
    configs = [r["config"] for r in cv if r["dataset"] == next(iter(DATASETS))]
    for c in configs:
        cells = []
        for d in DATASETS:
            r = next(r for r in cv if r["dataset"] == d and r["config"] == c)
            s = f"{float(r['mean_auc']):.3f} ({float(r['sd_auc']):.3f})"
            if r["selected"] == "True":
                s = f"**{s}**"
            if r["is_fixed"] == "True":
                s += " (fixed)"
            cells.append(s)
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Bold: selected. (fixed): the fixed runs' configuration.", "",
              "## Test AUC, median (range) over seeds 0-4", "",
              "| dataset | fixed config | fixed test AUC | selected | tuned test AUC | hidden fixed / tuned | "
              "connections fixed / tuned | best comparator | GB | seeds > GB fixed / tuned | harness check |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, s in summary.items():
        lines.append(f"| {s['label']} | {s['fixed_config']} | {mr(s['fixed_test_auc'])} | {s['selected_config']} | "
                     f"{mr(s['tuned_test_auc'])} | {s['fixed_hidden']['median']:g} / {s['tuned_hidden']['median']:g} | "
                     f"{s['fixed_connections']['median']:g} / {s['tuned_connections']['median']:g} | "
                     f"{s['best_comparator']['model_key']} {f3(s['best_comparator']['test_auc'])} | {f3(s['gb_test_auc'])} | "
                     f"{s['fixed_seeds_above_gb']} / {s['tuned_seeds_above_gb']} | {'ok' if s['harness_ok'] else 'FAIL'} |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
