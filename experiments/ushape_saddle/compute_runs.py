"""Per-seed metrics for the study's runs (the definitions of experiments/10seed/compute_runs.py:
best genome by training fitness, ties to the earliest generation then the lowest NEAT key; test
AUC with a 10,000-resample paired bootstrap CI, seed 42; structure counts) and the share of the
LR-to-oracle gap closed. Read-only on the database. Writes runs.csv and summary.json.

    DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/ushape_saddle/compute_runs.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "10seed"))
sys.path.insert(0, str(REPO))

import compute_runs as cr  # noqa: E402
from compute_runs import (ReadOnlySession, best_genome_row, bootstrap_indices,  # noqa: E402
                          experiment_meta, genome_metrics, load_split, write_csv)

if "gap_closed" not in cr.COLUMNS:
    cr.COLUMNS.append("gap_closed")
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment  # noqa: E402


def main():
    reg = json.loads((HERE / "datasets.json").read_text())
    refs = json.loads((HERE / "references.json").read_text())
    db.init_db()
    rows, summary = [], {}
    with ReadOnlySession() as s:
        for task, spec in reg.items():
            data = load_split(s, spec["split_id"])
            boot = bootstrap_indices(len(data["y_test"]))
            exps = (s.query(Experiment).filter(Experiment.name.like(f"{spec['label']}-s%"))
                    .filter(Experiment.status == "completed").order_by(Experiment.random_seed).all())
            exps = [e for e in exps if e.name == f"{spec['label']}-s{e.random_seed}"]
            lr, orc = refs[task]["l2_logistic"]["test_auc"], refs[task]["oracle"]["test_auc"]
            trows = []
            for exp in exps:
                assert str(exp.split_id) == spec["split_id"]
                gid, key, fit, gen, (n_rows, n_keys) = best_genome_row(s, exp.id)
                row, _ = genome_metrics(s, gid, data, boot)
                row.update(experiment_meta(s, exp))
                row.update(dataset=task, seed=exp.random_seed, experiment_id=str(exp.id), experiment_name=exp.name,
                           n_rows_at_best_fitness=n_rows, n_keys_tied_in_best_generation=n_keys, rerun_note="")
                row["gap_closed"] = (row["test_auc"] - lr) / (orc - lr)
                print(f"{exp.name}: genome {gid[:8]} key {key} gen {gen} train {row['train_auc']:.4f} "
                      f"test {row['test_auc']:.4f} [{row['test_auc_ci_lo']:.4f},{row['test_auc_ci_hi']:.4f}] "
                      f"hidden {row['hidden_nodes']} conns {row['phenotype_connections_enabled']} "
                      f"inputs {row['inputs_connected']} gap closed {row['gap_closed']:.2f} {row['wall_clock_hours']:.2f} h")
                trows.append(row)
            rows += trows
            if trows:
                best = max(trows, key=lambda r: (r["train_auc"], -r["generation_of_best"], -r["genome_key"]))
                tests = [r["test_auc"] for r in trows]
                summary[task] = dict(label=spec["label"], n_seeds=len(trows), lr=lr, oracle=orc,
                                     gb=refs[task]["gradient_boosting"]["test_auc"],
                                     test_auc_median=float(np.median(tests)), test_auc_min=min(tests), test_auc_max=max(tests),
                                     gap_closed_median=float(np.median([r["gap_closed"] for r in trows])),
                                     best_by_train=dict(seed=best["seed"], genome_id=best["genome_id"], train_auc=best["train_auc"],
                                                        test_auc=best["test_auc"], gap_closed=best["gap_closed"],
                                                        hidden=best["hidden_nodes"], conns=best["phenotype_connections_enabled"]),
                                     hidden_median=float(np.median([r["hidden_nodes"] for r in trows])),
                                     conns_median=float(np.median([r["phenotype_connections_enabled"] for r in trows])))
    write_csv(HERE / "runs.csv", rows)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for t, v in summary.items():
        print(f"{t}: median test {v['test_auc_median']:.3f} (LR {v['lr']:.3f}, GB {v['gb']:.3f}, oracle {v['oracle']:.3f}); "
              f"gap closed median {v['gap_closed_median']:.2f}, best-by-train seed {v['best_by_train']['seed']} "
              f"test {v['best_by_train']['test_auc']:.3f} ({v['best_by_train']['gap_closed']:.2f})")


if __name__ == "__main__":
    main()
