"""Metrics for the churn runs (same definitions as experiments/10seed/compute_runs.py).

    uv run python experiments/churn/compute_churn_runs.py [--prefix churn-pilot]
Writes experiments/churn/runs_<prefix>.csv.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "10seed"))
sys.path.insert(0, str(HERE.parents[1]))

from compute_runs import (ReadOnlySession, best_genome_row, bootstrap_indices,  # noqa: E402
                          experiment_meta, genome_metrics, load_split, summarise, write_csv)
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment  # noqa: E402

SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="churn-pilot")
    args = ap.parse_args()
    db.init_db()
    with ReadOnlySession() as s:
        data = load_split(s, SPLIT)
        exps = (s.query(Experiment).filter(Experiment.name.like(f"{args.prefix}-s%"))
                .order_by(Experiment.name).all())
        rows = []
        boot = bootstrap_indices(len(data["y_test"]))
        for exp in exps:
            if exp.status != "completed":
                print(f"{exp.name}: {exp.status} (skipped)")
                continue
            assert str(exp.split_id) == SPLIT
            gid, key, fit, gen, (n_rows, n_keys) = best_genome_row(s, exp.id)
            row, _ = genome_metrics(s, gid, data, boot)
            row.update(experiment_meta(s, exp))
            row.update(dataset="churn", seed=exp.random_seed, experiment_id=str(exp.id),
                       experiment_name=exp.name, n_rows_at_best_fitness=n_rows,
                       n_keys_tied_in_best_generation=n_keys, rerun_note="")
            rows.append(row)
            print(f"{exp.name}: genome {gid[:8]} gen {gen} train {row['train_auc']:.4f} "
                  f"test {row['test_auc']:.4f} [{row['test_auc_ci_lo']:.4f},{row['test_auc_ci_hi']:.4f}] "
                  f"acc {row['test_acc_0.5']:.3f} hidden {row['hidden_nodes']} "
                  f"conns {row['phenotype_connections_enabled']} inputs {row['inputs_connected']}")
    write_csv(HERE / f"runs_{args.prefix}.csv", rows)
    summarise(args.prefix, rows)


if __name__ == "__main__":
    main()
