"""Task 6, population level, final generation: test AUC of every genome in
each run's final evaluated generation (read-only).

For each run in experiments/10seed/runs.csv, runs_heart100.csv and
experiments/churn/runs_churn{100,200}.csv: take the last generation that has
evaluated genomes, forward every genome through NeuralNeat (the training
engine, as compute_runs.py does) on the split's test rows (stored scaler),
and record its test AUC next to its stored training fitness.

Run: PYTHONHASHSEED=0 uv run python scripts/thesis/rashomon_population.py [run_set ...]
Writes results/thesis/rashomon_population.csv (appends per run set; reruns of
a run set replace its rows).
"""
import csv
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sqlalchemy import text

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from explaneat.core.config_utils import load_neat_config  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.db import db  # noqa: E402
from explaneat.db.models import Dataset, DatasetSplit, Experiment, Genome  # noqa: E402

RUN_SETS = {
    "heart": (ROOT / "experiments/10seed/runs.csv", "heart"),
    "monk2": (ROOT / "experiments/10seed/runs.csv", "monk2"),
    "adult": (ROOT / "experiments/10seed/runs.csv", "adult"),
    "heart100": (ROOT / "experiments/10seed/runs_heart100.csv", None),
    "churn100": (ROOT / "experiments/churn/runs_churn100.csv", None),
    "churn200": (ROOT / "experiments/churn/runs_churn200.csv", None),
}
OUT = ROOT / "results" / "thesis" / "rashomon_population.csv"
FIELDS = ["run_set", "seed", "experiment_id", "generation", "genome_id", "genome_key",
          "train_fitness", "test_auc", "is_best_genome_of_run"]


def runs(run_set):
    path, ds = RUN_SETS[run_set]
    with open(path) as f:
        return [r for r in csv.DictReader(f) if ds is None or r["dataset"] == ds]


def ro_session():
    if db.Session is None:
        db.init_db()
    s = db.Session()
    s.execute(text("SET TRANSACTION READ ONLY"))
    return s


def main():
    sets = sys.argv[1:] or list(RUN_SETS)
    existing = []
    if OUT.exists():
        with open(OUT) as f:
            existing = [r for r in csv.DictReader(f) if r["run_set"] not in sets]
    rows = list(existing)
    data_cache = {}
    for rs in sets:
        for run in runs(rs):
            t0 = time.time()
            s = ro_session()
            try:
                exp = s.get(Experiment, uuid.UUID(run["experiment_id"]))
                config = load_neat_config(exp.neat_config_text or "", exp.config_json)
                sid = str(exp.split_id)
                if sid not in data_cache:
                    split = s.get(DatasetSplit, exp.split_id)
                    X, y = s.get(Dataset, split.dataset_id).get_data()
                    mean = np.asarray(split.scaler_params["mean"], dtype=np.float64)
                    scale = np.asarray(split.scaler_params["scale"], dtype=np.float64)
                    te = list(split.test_indices)
                    data_cache[sid] = (torch.tensor((X[te].astype(np.float64) - mean) / scale),
                                       np.asarray(y)[te].ravel())
                Xte, yte = data_cache[sid]
                gen = s.execute(text("""
                    SELECT max(p.generation) FROM populations p JOIN genomes g ON g.population_id = p.id
                    WHERE p.experiment_id = :e AND g.fitness IS NOT NULL"""), {"e": exp.id}).scalar()
                gids = [r[0] for r in s.execute(text("""
                    SELECT g.id FROM genomes g JOIN populations p ON g.population_id = p.id
                    WHERE p.experiment_id = :e AND p.generation = :g ORDER BY g.genome_id"""),
                    {"e": exp.id, "g": gen})]
                for gid in gids:
                    g = s.get(Genome, gid)
                    ng = g.to_neat_genome(config)
                    p = NeuralNeat(ng, config).forward(Xte).detach().numpy().ravel()
                    auc = float(roc_auc_score(yte, p)) if np.all(np.isfinite(p)) else float("nan")
                    rows.append({"run_set": rs, "seed": run["seed"], "experiment_id": run["experiment_id"],
                                 "generation": gen, "genome_id": str(gid), "genome_key": g.genome_id,
                                 "train_fitness": g.fitness, "test_auc": auc,
                                 "is_best_genome_of_run": str(gid) == run["genome_id"]})
            finally:
                s.rollback()
                s.close()
                db.Session.remove()
            print(f"{rs} s{run['seed']} gen {gen}: {len(gids)} genomes, {time.time()-t0:.0f}s", flush=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
