"""The PropNEAT rows of the comparator tables (read-only).

    uv run python experiments/benchmarks/propneat.py        # after export_data.py

Experiment 1 (one explained network per dataset): each classification network is loaded from
explaneat_dev in a READ ONLY transaction (experiments/10seed/compute_runs.ReadOnlySession),
rebuilt with its experiment's stored NEAT config and run forward with NeuralNeat on the same
cached arrays as the comparators (cache/<name>.npz). Train and test AUC, test accuracy at 0.5
and the 10,000-resample 95% interval of the test AUC (RandomState(42), as the comparators) are
computed here and checked against the study's own run record (the AUCs must agree to 1e-9).
The survival networks are scored in survival.py; XOR's 30 networks are read from
experiments/xor/xor_results.json (accuracy on the four rows).

Experiment 2 (ten seeds per study): read from the runs CSVs, not recomputed (they already use
this test set, interval and resampling): median and range of the test AUC over the ten seeds,
and the seed with the best training AUC (the study's selection quantity; ties to the lower seed)
with its interval. Output: results/propneat.json.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
import uuid

import numpy as np

from common import CACHE, CLASSIFICATION, REPO, RESULTS, auc_ci, boot_indices, jdump, jload

sys.path.insert(0, str(REPO / "experiments" / "10seed"))

E = REPO / "experiments"

# label -> (cache, genome id, (runs file, filter, train column, test column))
EXP1 = {
    "paper-heart": ("heart", "9022a9eb-ecf9-41ba-b138-28ed4d131c9d",
                    ("10seed/runs_paper_reference.csv", {"dataset": "heart"}, "train_auc", "test_auc")),
    "paper-monk2": ("monk2", "23dde0fa-a635-4974-91aa-c05266ea0747",
                    ("10seed/runs_paper_reference.csv", {"dataset": "monk2"}, "train_auc", "test_auc")),
    "paper-adult": ("adult", "4bc8fa07-bbf5-48e4-96a8-77be8b577ba3",
                    ("10seed/runs_paper_reference.csv", {"dataset": "adult"}, "train_auc", "test_auc")),
    "churn-case-s10": ("churn", "9cacd431-a3b0-4649-89fb-961d5a6fe5b3",
                       ("churn_case/runs.csv", {"experiment_name": "churn-case-s10"}, "train_auc", "test_auc")),
    "c4cap-main-h10-e50-s1": ("c4_cells", "f0eb7635-7f85-437b-815b-993c5ad6cc14",
                              ("c4_capacity/runs.csv", {"name": "main-h10-e50-s1"}, "train_auc", "test_auc")),
    "c4lf-lines-heavy-s0": ("c4_lines", "6ebf7e0c-55ec-4215-9852-0ac982e43c0e",
                            ("c4_line_features/runs.csv", {"name": "lines-heavy-s0"}, "train_auc", "test_auc")),
    "shape-A": ("shape_u", "2aac39c4-20d1-48f1-8f43-e8dfcaf50a52",
                ("ushape_saddle/runs.csv", {"dataset": "u", "seed": "0"}, "train_auc", "test_auc")),
    "shape-B": ("shape_saddle", "25f5830e-33cc-4dda-8ea8-16e092c09a3e",
                ("ushape_saddle/runs.csv", {"dataset": "saddle", "seed": "0"}, "train_auc", "test_auc")),
    "shape-C": ("shape_both", "a1cbdd4a-42e8-4c1a-b060-00aa3e22c76b",
                ("ushape_saddle/runs.csv", {"dataset": "both", "seed": "4"}, "train_auc", "test_auc")),
}
EXP2 = {"adult": ("10seed/runs.csv", {"dataset": "adult"}, "adult"),
        "churn100": ("churn/runs_churn100.csv", {}, "churn"),
        "heart100": ("10seed/runs_heart100.csv", {}, "heart")}


def rows(path, flt):
    with open(E / path) as f:
        return [r for r in csv.DictReader(f) if all(r[k] == v for k, v in flt.items())]


def exp1():
    import compute_runs as cr
    from sklearn.metrics import accuracy_score, roc_auc_score
    cr.db.init_db("postgresql://localhost/explaneat_dev")
    out = {}
    for label, (cache, gid, (path, flt, ctr, cte)) in EXP1.items():
        d = np.load(CACHE / f"{cache}.npz")
        with cr.ReadOnlySession() as s:
            g = s.get(cr.Genome, uuid.UUID(gid))
            exp = g.population.experiment
            cfg = cr.load_neat_config(exp.neat_config_text or "", exp.config_json)
            genome = g.to_neat_genome(cfg)
            split_id = str(exp.split_id)
            assert split_id == CLASSIFICATION[cache][1], (label, split_id)  # trained on the cached split
        p_tr, p_te = cr.predict(genome, cfg, d["Xtr"]), cr.predict(genome, cfg, d["Xte"])
        ytr, yte = d["ytr"], d["yte"]
        ci, nb = auc_ci(yte, p_te, boot_indices(len(yte)))
        rec = rows(path, flt)
        assert len(rec) == 1, (label, len(rec))
        r = {"dataset": cache, "genome_id": gid, "split_id": split_id,
             "train_auc": float(roc_auc_score(ytr, p_tr)), "test_auc": float(roc_auc_score(yte, p_te)),
             "test_auc_ci95": ci, "n_boot_used": nb, "test_accuracy": float(accuracy_score(yte, p_te > 0.5)),
             "train_accuracy": float(accuracy_score(ytr, p_tr > 0.5)),
             "record": {"file": f"experiments/{path}", "train_auc": float(rec[0][ctr]), "test_auc": float(rec[0][cte])}}
        assert abs(r["train_auc"] - r["record"]["train_auc"]) < 1e-9, (label, r["train_auc"], r["record"])
        assert abs(r["test_auc"] - r["record"]["test_auc"]) < 1e-9, (label, r["test_auc"], r["record"])
        for k in ("test_auc_ci_lo", "test_auc_ci_hi"):
            if k in rec[0]:
                r["record"][k] = float(rec[0][k])
        out[label] = r
        print(label, {k: r[k] for k in ("train_auc", "test_auc", "test_auc_ci95", "test_accuracy")}, flush=True)
    x = jload(E / "xor/xor_results.json")
    acc = [float(np.mean((np.array(run["preds"]) > 0.5) == np.array([0, 1, 1, 0]))) for run in x["runs"]]
    out["xor"] = {"dataset": "xor", "n_networks": len(acc), "train_accuracy_all": acc,
                  "n_solved": x["summary"]["n_solved"], "source": "experiments/xor/xor_results.json"}
    return out


def exp2():
    out = {}
    for study, (path, flt, cache) in EXP2.items():
        R = sorted(rows(path, flt), key=lambda r: int(r["seed"]))
        assert [int(r["seed"]) for r in R] == list(range(10)), study
        te = [float(r["test_auc"]) for r in R]
        best = max(R, key=lambda r: (float(r["train_auc"]), -int(r["seed"])))
        out[study] = {"dataset": cache, "source": f"experiments/{path}", "n_seeds": len(R),
                      "test_auc_median": st.median(te), "test_auc_min": min(te), "test_auc_max": max(te),
                      "test_acc_median": st.median(float(r["test_acc_0.5"]) for r in R),
                      "best_by_train": {"seed": int(best["seed"]), "train_auc": float(best["train_auc"]),
                                        "test_auc": float(best["test_auc"]),
                                        "test_auc_ci95": [float(best["test_auc_ci_lo"]), float(best["test_auc_ci_hi"])],
                                        "test_accuracy": float(best["test_acc_0.5"])},
                      "n_boot_used": int(R[0]["n_boot_used"])}
    return out


def main():
    res = {"exp1": exp1(), "exp2": exp2()}
    jdump(res, RESULTS / "propneat.json")


if __name__ == "__main__":
    main()
