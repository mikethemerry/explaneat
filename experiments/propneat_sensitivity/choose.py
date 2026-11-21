"""CV table and configuration choice (training rows only; no test numbers are read).

Reads runs/<dataset>/<config>_fold<k>_s0.json; writes cv.csv (one row per dataset x configuration:
fold AUCs, mean, SD, network size of the fold runs) and selection.json (the chosen configuration per
dataset by the PLAN.md rule: highest mean fold AUC; means equal at TIE_DECIMALS are tied and the
simplest wins: fewer epochs, then lower weight decay, then the Heart family). Also checks the repeat
CV (repeat/ against runs/, byte for byte).

    uv run python experiments/propneat_sensitivity/choose.py
"""
import csv
import json
import statistics

from common import (CV_SEED, DATASETS, HERE, N_FOLDS, TIE_DECIMALS, all_configs, fixed_config_id,
                    run_path, simplicity_key)
from launch import REPEAT


def main():
    rows, selection = [], {}
    for d in DATASETS:
        scored = []
        for c in all_configs():
            runs = [json.loads(run_path(d, c["id"], k, CV_SEED).read_text()) for k in range(N_FOLDS)]
            aucs = [r["best"]["heldout_auc"] for r in runs]
            assert all(r["eval_split"] == "heldout" for r in runs)
            mean, sd = statistics.fmean(aucs), statistics.stdev(aucs)
            rows.append({"dataset": d, "config": c["id"], "epochs": c["epochs"], "family": c["family"],
                         "weight_decay": c["weight_decay"], "is_fixed": c["id"] == fixed_config_id(d),
                         **{f"fold{k}_auc": a for k, a in enumerate(aucs)},
                         "mean_auc": mean, "sd_auc": sd,
                         "mean_train_auc": statistics.fmean(r["best"]["train_auc"] for r in runs),
                         "median_hidden": statistics.median(r["best"]["hidden_nodes"] for r in runs),
                         "median_connections": statistics.median(
                             r["best"]["phenotype_connections_enabled"] for r in runs)})
            scored.append((c, mean))
        top = max(round(m, TIE_DECIMALS) for _, m in scored)
        tied = sorted((c for c, m in scored if round(m, TIE_DECIMALS) == top), key=simplicity_key)
        best_c = tied[0]
        ranked = sorted(scored, key=lambda t: -t[1])
        selection[d] = {"selected": best_c["id"], "selected_mean_auc": dict(scored_map(scored))[best_c["id"]],
                        "tied_at_top": [c["id"] for c in tied], "fixed": fixed_config_id(d),
                        "fixed_mean_auc": dict(scored_map(scored))[fixed_config_id(d)],
                        "fixed_rank": [c["id"] for c, _ in ranked].index(fixed_config_id(d)) + 1,
                        "selected_is_fixed": best_c["id"] == fixed_config_id(d),
                        "range_of_means": [min(m for _, m in scored), max(m for _, m in scored)]}
        for r in rows:
            if r["dataset"] == d:
                r["selected"] = r["config"] == best_c["id"]
                r["rank"] = [c["id"] for c, _ in ranked].index(r["config"]) + 1

    # Repeat check: the second CV of one configuration must equal the first byte for byte.
    rd, rc = REPEAT
    same = [run_path(rd, rc, k, CV_SEED, "repeat").read_bytes() == run_path(rd, rc, k, CV_SEED).read_bytes()
            for k in range(N_FOLDS)]
    selection["_repeat_check"] = {"dataset": rd, "config": rc, "folds_identical": same,
                                  "all_identical": all(same)}

    with open(HERE / "cv.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    for d in DATASETS:
        print(d, selection[d]["selected"], "fixed rank", selection[d]["fixed_rank"])
    print("repeat identical:", all(same))


def scored_map(scored):
    return [(c["id"], m) for c, m in scored]


if __name__ == "__main__":
    main()
