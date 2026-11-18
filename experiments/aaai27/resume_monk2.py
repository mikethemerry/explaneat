"""Resume the interrupted monk2 experiment to its target generation count.

Replicates the API resume path (routes/experiments.py::resume_experiment)
synchronously via ExperimentRunner._resume_evolution_loop, on the same
correct dataset/split the run was already using.

KNOWN BLOCKER (2026-07-10): the framework resume path is broken. Both this
script and the API resume route crash with:

    AttributeError: 'NoneType' object has no attribute 'info'
    at neat/species.py speciate() <- population.py:185 resume_from_db()

`DatabaseBackpropPopulation.resume_from_db` speciates before any reporter is
attached, so `species.reporters` is None. This is a pre-existing core bug (it
also affects resuming the paper's interrupted experiments), out of scope for
this analysis task. Until it is fixed, use a fresh full run instead
(scripts/aaai27/ingest_monk2.py) rather than resume.

Run: uv run python scripts/aaai27/resume_monk2.py
"""
import sys
import uuid

import numpy as np

from explaneat.db import db
from explaneat.db.models import Experiment, Dataset, DatasetSplit
from explaneat.db.population import (compute_remaining_generations,
                                     DatabaseBackpropPopulation)

EXP_NAME = "monk2 - OHE - 250-100-5"


def main():
    with db.session_scope() as s:
        exp = (s.query(Experiment).filter_by(name=EXP_NAME)
               .order_by(Experiment.created_at.desc()).first())
        if not exp:
            sys.exit(f"no experiment named {EXP_NAME!r}")
        exp_id = str(exp.id)
        if exp.status != "interrupted":
            sys.exit(f"experiment status is {exp.status!r}, expected 'interrupted' "
                     f"(restart instead if you want a fresh run)")
        resolved = (exp.config_json or {}).get("resolved_config") or {}
        tr = resolved.get("training", {})
        target = tr.get("n_generations", 100)
        n_epochs = tr.get("n_epochs_backprop", 5)
        fitness = tr.get("fitness_function", "auc")
        config_text = exp.neat_config_text
        dataset = s.query(Dataset).filter_by(id=exp.dataset_id).first()
        split = s.query(DatasetSplit).filter_by(id=exp.split_id).first()
        X_full, y_full = dataset.get_data()
        train_idx = split.train_indices or []
        mean = np.array(split.scaler_params["mean"])
        scale = np.where(np.array(split.scaler_params["scale"]) == 0, 1.0,
                         np.array(split.scaler_params["scale"]))
        split_id = str(split.id)
        ds_name = dataset.name

    last_gen = DatabaseBackpropPopulation._get_latest_generation(exp_id)
    remaining = compute_remaining_generations(last_gen, target)
    print(f"resuming {EXP_NAME} on dataset {ds_name}: last_gen={last_gen}, "
          f"target={target}, remaining={remaining}")
    if remaining == 0:
        with db.session_scope() as s:
            s.query(Experiment).filter_by(id=uuid.UUID(exp_id)).first().status = "completed"
        print("already at target; marked completed")
        return

    X_train = (X_full[train_idx] - mean) / scale
    y_train = y_full[train_idx].astype(np.float64)

    from explaneat.api.experiment_runner import ExperimentRunner, ExperimentProgress
    progress = ExperimentProgress(job_id="monk2-resume", experiment_id=exp_id,
                                  total_generations=remaining)
    ExperimentRunner._resume_evolution_loop(
        progress=progress, experiment_id=exp_id, config_text=config_text,
        X_train=X_train, y_train=y_train, fitness_function=fitness,
        remaining_generations=remaining, n_epochs_backprop=n_epochs,
        split_id=split_id,
    )
    # resume loop does not always flip status; ensure completed on clean finish
    with db.session_scope() as s:
        e = s.query(Experiment).filter_by(id=uuid.UUID(exp_id)).first()
        if e.status != "completed":
            e.status = "completed"
    print(f"resume done: best_fitness={progress.best_fitness}")


if __name__ == "__main__":
    main()
