"""Assemble benchmarks.csv and check the earlier reference results against this run.

    uv run python experiments/benchmarks/make_csv.py

Reads results/classification/*.json, results/survival.json and results/propneat.json. Writes
benchmarks.csv (one row per dataset and model, PropNEAT rows included) and results/reuse.json:
every earlier comparator number on the same split (the files named in EARLIER), set beside this
run's, with whether the earlier protocol is this one and whether the numbers agree. Then the grid
check (check_grids.py): writes results/grid_check.md and fails if any CV choice sits at a grid edge.
"""
from __future__ import annotations

import csv
import json

import check_grids
from common import HERE, REPO, RESULTS, jdump, jload

E = REPO / "experiments"
MODEL_NAMES = {
    "lr_l2": "Logistic regression (L2)", "lr_l1": "Logistic regression (L1, sparse)", "tree": "Decision tree",
    "rf": "Random forest", "gb": "Gradient boosting (HistGradientBoosting)", "ebm": "Explainable boosting machine",
    "mlp": "MLP (one or two hidden layers)",
    "cox_ph": "Cox PH (unpenalised)", "cox_l2": "Cox PH, L2 (ridge)", "cox_enet": "Cox PH, elastic net",
    "rsf": "Random survival forest", "gbs": "Gradient-boosted survival (Cox loss)",
}
COMPARATORS = ["lr_l2", "lr_l1", "tree", "rf", "gb", "ebm", "mlp"]
SURV_MODELS = ["cox_ph", "cox_l2", "cox_enet", "rsf", "gbs"]
EXP = {"heart": "1,2", "monk2": "1", "c4_cells": "1", "c4_lines": "1", "shape_u": "1", "shape_saddle": "1",
       "shape_both": "1", "adult": "1,2", "churn": "1,2", "whas500": "1", "gbsg2": "1 (miscellany)", "xor": "1"}
EXP1_NET = {"heart": "paper-heart", "monk2": "paper-monk2", "adult": "paper-adult", "churn": "churn-case-s10", "c4_cells": "c4cap-main-h10-e50-s1",
            "c4_lines": "c4lf-lines-heavy-s0", "shape_u": "shape-A", "shape_saddle": "shape-B", "shape_both": "shape-C"}
EXP2_STUDY = {"adult": "adult", "churn": "churn100", "heart": "heart100"}
NARROW = "no: the narrower grids (experiments/references, 19 Nov 2026; results/narrow_grids/)"

COLS = ["experiment", "dataset", "model_key", "model", "params", "cv_metric", "cv_score", "test_metric", "test_value",
        "ci_lo", "ci_hi", "test_accuracy", "train_value", "uno_c", "td_auc_mean", "ibs", "n_train", "n_test", "note"]


def p(params):
    return json.dumps(params, sort_keys=True) if params else ""


def classification_rows():
    out = []
    pn = jload(RESULTS / "propneat.json")
    for f in sorted((RESULTS / "classification").glob("*.json")):
        ds = f.stem
        if ds == "xor":
            continue
        res = jload(f)[ds]
        n_tr, n_te = res["lr_l2"]["n_train"], res["lr_l2"]["n_test"]
        if ds in EXP1_NET:
            r = pn["exp1"][EXP1_NET[ds]]
            out.append(dict(experiment="1", dataset=ds, model_key="propneat_explained",
                            model=f"PropNEAT, explained network ({EXP1_NET[ds]})", cv_metric="", cv_score="",
                            test_metric="auc", test_value=r["test_auc"], ci_lo=r["test_auc_ci95"][0], ci_hi=r["test_auc_ci95"][1],
                            test_accuracy=r["test_accuracy"], train_value=r["train_auc"], n_train=n_tr, n_test=n_te,
                            note="recomputed from the genome (explaneat_dev, read-only); equals " + r["record"]["file"]))
        if ds in EXP2_STUDY:
            s = pn["exp2"][EXP2_STUDY[ds]]
            b = s["best_by_train"]
            out.append(dict(experiment="2", dataset=ds, model_key="propneat_best_by_train",
                            model=f"PropNEAT, best seed by training AUC ({EXP2_STUDY[ds]}-s{b['seed']})",
                            test_metric="auc", test_value=b["test_auc"], ci_lo=b["test_auc_ci95"][0], ci_hi=b["test_auc_ci95"][1],
                            test_accuracy=b["test_accuracy"], train_value=b["train_auc"], n_train=n_tr, n_test=n_te,
                            note="from " + s["source"]))
            out.append(dict(experiment="2", dataset=ds, model_key="propneat_median10",
                            model=f"PropNEAT, median of ten seeds ({EXP2_STUDY[ds]})", test_metric="auc",
                            test_value=s["test_auc_median"], ci_lo="", ci_hi="", test_accuracy=s["test_acc_median"],
                            n_train=n_tr, n_test=n_te,
                            note=f"range {s['test_auc_min']:.4f}-{s['test_auc_max']:.4f} over the ten seeds; from {s['source']}"))
        for m in COMPARATORS:
            r = res[m]
            out.append(dict(experiment=EXP[ds], dataset=ds, model_key=m, model=MODEL_NAMES[m], params=p(r["params"]),
                            cv_metric="auc (5-fold stratified, train rows)", cv_score=r["cv_auc"], test_metric="auc",
                            test_value=r["test_auc"], ci_lo=r["test_auc_ci95"][0], ci_hi=r["test_auc_ci95"][1],
                            test_accuracy=r["test_accuracy"], train_value=r["train_auc"], n_train=n_tr, n_test=n_te))
    x = jload(RESULTS / "classification" / "xor.json")["xor"]
    acc = pn["exp1"]["xor"]["train_accuracy_all"]
    out.append(dict(experiment="1", dataset="xor", model_key="propneat_explained", model="PropNEAT, 30 seeds (reduced networks)",
                    test_metric="accuracy on the 4 rows", test_value=min(acc), n_train=4, n_test=0,
                    note=f"{sum(a == 1.0 for a in acc)} of {len(acc)} networks classify all four rows (min shown)"))
    for m in ("lr_l2", "tree"):
        out.append(dict(experiment="1", dataset="xor", model_key=m, model=MODEL_NAMES[m], params=x[m]["params"],
                        test_metric="accuracy on the 4 rows", test_value=x[m]["train_accuracy"], n_train=4, n_test=0,
                        note=f"predictions {x[m]['predictions']}"))
    return out


def survival_rows():
    out = []
    S = jload(RESULTS / "survival.json")
    for ds, res in S.items():
        base = dict(experiment=EXP[ds], dataset=ds, test_metric="harrell_c", n_train=res["n_train"], n_test=res["n_test"])
        r = res["propneat"]
        out.append(dict(base, model_key="propneat_explained" if ds == "whas500" else "propneat_selected",
                        model=f"PropNEAT Cox, selected network ({r['tag']})", test_value=r["harrell_c"],
                        ci_lo=r["harrell_c_ci95"][0], ci_hi=r["harrell_c_ci95"][1], train_value=r["train_harrell_c"],
                        uno_c=r["uno_c"], td_auc_mean=r["td_auc"]["mean_auc"], ibs=r["ibs"],
                        note="scored from the genome pickle; equals experiments/survival/cox_case/runs/" + r["tag"] + ".json"))
        for m in SURV_MODELS:
            r = res[m]
            out.append(dict(base, model_key=m, model=MODEL_NAMES[m], params=p(r["params"]),
                            cv_metric="harrell_c (5-fold, stratified on event, train rows)" if m != "cox_ph" else "",
                            cv_score=r["cv_c"] if r["cv_c"] is not None else "", test_value=r["harrell_c"],
                            ci_lo=r["harrell_c_ci95"][0], ci_hi=r["harrell_c_ci95"][1], train_value=r["train_harrell_c"],
                            uno_c=r["uno_c"], td_auc_mean=r["td_auc"]["mean_auc"], ibs=r["ibs"]))
    return out


# ------------------------------------------------------------------ earlier results on the same splits

def reuse(rows):
    now = {(r["dataset"], r["model_key"]): r for r in rows}
    checks = []

    def add(src, ds, key, earlier, same_protocol, what="test", tol=1e-9):
        r = now[(ds, key)]
        val = r["test_value"] if what == "test" else r["cv_score"]
        checks.append({"source": src, "dataset": ds, "model": key, "quantity": what, "earlier": earlier, "now": val,
                       "same_protocol": same_protocol, "agree": abs(float(earlier) - float(val)) < tol})

    bench = jload(E / "references/heart_benchmark/bench.json")
    keys = {"Sparse logistic regression (L1)": "lr_l1", "Decision tree": "tree", "Random forest": "rf", "EBM": "ebm",
            "MLP (1 hidden layer)": "mlp"}
    for k, v in bench.items():
        add("experiments/references/heart_benchmark/bench.json", "heart", keys[k], v["test_auc"], NARROW)
        add("experiments/references/heart_benchmark/bench.json", "heart", keys[k], v["cv_auc"], NARROW, "cv")
    base = jload(E / "references/results/baselines.json")
    for ds in ("heart", "monk2", "adult", "churn"):
        b = base[ds]
        add("experiments/references/results/baselines.json", ds, "lr_l2", b["logistic"]["test_auc"],
            "no: C chosen by unstratified KFold on the narrower grid")
        add("experiments/references/results/baselines.json", ds, "gb", b["gradient_boosting"]["test_auc"], NARROW)
        add("experiments/references/results/baselines.json", ds, "gb", b["gradient_boosting"]["cv_auc"], NARROW, "cv")
    ch = jload(E / "churn/baselines.json")
    add("experiments/churn/baselines.json", "churn", "lr_l2", ch["l2_logistic"]["test_auc"], "no: untuned (C = 1)")
    add("experiments/churn/baselines.json", "churn", "gb", ch["gradient_boosting"]["test_auc"], "no: default settings")
    us = jload(E / "ushape_saddle/references.json")
    for t, ds in (("u", "shape_u"), ("saddle", "shape_saddle"), ("both", "shape_both")):
        add("experiments/ushape_saddle/references.json", ds, "lr_l2", us[t]["l2_logistic"]["test_auc"], "no: untuned (C = 1)")
        add("experiments/ushape_saddle/references.json", ds, "gb", us[t]["gradient_boosting"]["test_auc"], "no: default settings")
    c4 = jload(E / "c4_capacity/baselines.json")
    add("experiments/c4_capacity/baselines.json", "c4_cells", "lr_l2", c4["lr_cells"]["test"], "no: untuned (C = 1)")
    add("experiments/c4_capacity/baselines.json (from complex_real)", "c4_cells", "gb", c4["gradient_boosting"]["test"],
        "no: default settings")
    lf = jload(E / "c4_line_features/baselines.json")["lines"]
    add("experiments/c4_line_features/baselines.json", "c4_lines", "lr_l2", lf["lr"]["test"], "no: untuned (C = 1)")
    add("experiments/c4_line_features/baselines.json", "c4_lines", "gb", lf["gradient_boosting"]["test"], "no: default settings")
    with open(E / "survival/cox_case/references.csv") as f:
        ref = {(r["dataset"], r["model"]): r for r in csv.DictReader(f)}
    for ds in ("whas500", "gbsg2"):
        add("experiments/survival/cox_case/references.csv", ds, "cox_ph", float(ref[(ds, "cox_ph")]["harrell_c"]), True)
        add("experiments/survival/cox_case/references.csv", ds, "cox_l2", float(ref[(ds, "ridge_cox")]["harrell_c"]),
            "no: alpha tuned on the validation rows")
        add("experiments/survival/cox_case/references.csv", ds, "rsf", float(ref[(ds, "rsf")]["harrell_c"]),
            "no: fixed settings (min_samples_leaf 10)")
        key = "propneat_explained" if ds == "whas500" else "propneat_selected"
        add("experiments/survival/cox_case/references.csv", ds, key, float(ref[(ds, "propneat_cox")]["harrell_c"]), True)
    # this suite's own first run (19 Nov 2026, the narrower grids): kept in results/narrow_grids/
    with open(RESULTS / "narrow_grids" / "benchmarks.csv") as f:
        for r in csv.DictReader(f):
            if (r["dataset"], r["model_key"]) in now and r["test_value"] != "" and r["params"] != "" \
                    and not r["model_key"].startswith("propneat"):
                add("experiments/benchmarks/results/narrow_grids/benchmarks.csv", r["dataset"], r["model_key"],
                    r["test_value"], NARROW)
    return checks


def main():
    rows = classification_rows() + survival_rows()
    with open(HERE / "benchmarks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    checks = reuse(rows)
    jdump(checks, RESULTS / "reuse.json")
    for c in checks:
        print(f"{c['dataset']:12s} {c['model']:20s} {c['quantity']:4s} earlier {float(c['earlier']):.4f} now {float(c['now']):.4f} "
              f"agree={c['agree']} same_protocol={c['same_protocol']}")
    print(len(rows), "rows ->", HERE / "benchmarks.csv")
    check_grids.main()                               # exits non-zero if a choice is at a grid edge


if __name__ == "__main__":
    main()
