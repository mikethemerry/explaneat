"""Metrics for the churn case run (the definitions of experiments/10seed/compute_runs.py: best
genome by training fitness, ties to the earliest generation then the lowest NEAT key; test AUC
with a 10,000-resample paired bootstrap CI, seed 42; structure counts).

    uv run python experiments/churn_case/compute_run.py [--name churn-case-s10]
Read-only on the database. Writes experiments/churn_case/runs.csv.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "10seed"))
sys.path.insert(0, str(HERE.parents[1]))

from compute_runs import (ReadOnlySession, best_genome_row, bootstrap_indices,  # noqa: E402
                          experiment_meta, genome_metrics, load_split, write_csv)
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Experiment  # noqa: E402

SPLIT = "45962ea6-3c78-49f9-a01e-3665f6809b1e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="churn-case-s10")
    args = ap.parse_args()
    db.init_db()
    with ReadOnlySession() as s:
        data = load_split(s, SPLIT)
        exp = s.query(Experiment).filter(Experiment.name == args.name).one()
        assert exp.status == "completed", exp.status
        assert str(exp.split_id) == SPLIT
        gid, key, fit, gen, (n_rows, n_keys) = best_genome_row(s, exp.id)
        row, _ = genome_metrics(s, gid, data, bootstrap_indices(len(data["y_test"])))
        row.update(experiment_meta(s, exp))
        row.update(dataset="churn", seed=exp.random_seed, experiment_id=str(exp.id),
                   experiment_name=exp.name, n_rows_at_best_fitness=n_rows,
                   n_keys_tied_in_best_generation=n_keys, rerun_note="")
        print(f"{exp.name}: genome {gid} key {key} gen {gen} train {row['train_auc']:.4f} "
              f"test {row['test_auc']:.4f} [{row['test_auc_ci_lo']:.4f},{row['test_auc_ci_hi']:.4f}] "
              f"acc {row['test_acc_0.5']:.3f} hidden {row['hidden_nodes']} "
              f"conns {row['phenotype_connections_enabled']} inputs {row['inputs_connected']}")
    write_csv(HERE / "runs.csv", [row])


if __name__ == "__main__":
    main()
