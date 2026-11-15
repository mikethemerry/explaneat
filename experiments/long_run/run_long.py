"""One very long PropNEAT run with a monitoring split and a pre-fixed stopping rule (RESULTS.md section 2).

The same harness, config and fitness as the earlier runs of each dataset (the 10-seed harness:
seeding, 1 torch thread, DatabaseBackpropPopulation.run with auc_fitness, 5 backprop epochs,
unmasked), with three changes fixed in RESULTS.md before any run:

* 20% of the training rows (stratified, seed MONITOR_SEED, the same rows for every seed of a
  dataset) are held out as a monitoring split; evolution and backprop see the other 80%;
* the run stops when the generation's best-by-training-fitness genome's monitoring AUC has not
  improved by >= MIN_DELTA for PATIENCE generations, or at MAX_GENERATIONS;
* light storage: each generation stores only its best genome; CHECKPOINT generations (and the
  stop generation) are stored in full; parents go to a sidecar CSV
  (experiments/lineage/parents/<name>.csv.gz, the replay-sidecar format of lineage.py).

The test split is never read. Per-generation summary: logs/<name>.gen.csv.

    PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
        uv run python experiments/long_run/run_long.py both 0
"""
import argparse
import csv
import gzip
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import neat
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "10seed"))

from run_seed import DATASETS, N_EPOCHS_BACKPROP, TORCH_THREADS, load_training_data, seed_everything  # noqa: E402
from explaneat.core.lineage_capture import parent_rows  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat as nneat  # noqa: E402
from explaneat.db.base import db  # noqa: E402
from explaneat.db.models import Experiment  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

MAX_GENERATIONS = 10_000
PATIENCE = 1_000
MIN_DELTA = 0.001
MONITOR_SHARE = 0.20
MONITOR_SEED = 20261115
# checkpoint generations (1-based counts; generation index = count - 1); 300 is the first
# explanation point
CHECKPOINTS = (100, 200, 300, 500, 1000, 2000, 5000, 10000)
MIN_FREE_GB = 30
SIDECAR_DIR = REPO / "experiments" / "lineage" / "parents"
DESCRIPTION = "Very long runs (experiments/long_run), exploratory"


def task_spec(task):
    if task == "both":
        reg = json.loads((REPO / "experiments" / "ushape_saddle" / "datasets.json").read_text())["both"]
        return dict(dataset_id=reg["dataset_id"], split_id=reg["split_id"],
                    config=REPO / "experiments" / "ushape_saddle" / "configs" / "syn10.ini", prefix="longrun-shape-C")
    if task == "churn":
        d = DATASETS["churn"]
        return dict(dataset_id=d["dataset_id"], split_id=d["split_id"], config=d["neat_config"], prefix="longrun-churn")
    raise SystemExit(f"unknown task {task}")


def monitor_split(y):
    """Positions (into the split's training rows) of the fit and monitoring rows."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    fit, mon = train_test_split(idx, test_size=MONITOR_SHARE, stratify=y, random_state=MONITOR_SEED)
    return np.sort(fit), np.sort(mon)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def free_gb():
    return shutil.disk_usage("/").free / 1e9


class Monitor:
    """The stopping rule and the per-generation log; reads no global RNG state it does not restore."""

    def __init__(self, pop, config, X_mon, y_mon, name, max_gen):
        self.pop, self.config = pop, config
        self.X_mon = torch.tensor(X_mon, dtype=torch.float64)
        self.y_mon = np.asarray(y_mon).ravel()
        self.best_mon, self.best_gen = -1.0, -1
        self.max_gen = max_gen
        self.t0 = time.time()
        self.log_path = HERE / "logs" / f"{name}.gen.csv"
        self.fh = open(self.log_path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["generation", "wall_s", "best_key", "best_fitness", "mean_fitness", "n_species",
                         "best_hidden_genotype", "best_conns_enabled", "mean_hidden_genotype", "mean_conns_enabled",
                         "monitor_auc", "monitor_ref", "gens_since_improvement", "free_gb"])
        SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
        self.sidecar = gzip.open(SIDECAR_DIR / f"{name}.csv.gz", "wt", newline="")
        self.sw = csv.writer(self.sidecar)
        self.sw.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])

    def _size(self, g):
        outs = set(self.config.genome_config.output_keys)
        hidden = sum(1 for k in g.nodes if k not in outs)
        conns = sum(1 for c in g.connections.values() if c.enabled)
        return hidden, conns

    def __call__(self, gen, population, best):
        # parents of this generation -> sidecar; prune what is no longer needed (bounded memory)
        for gg in sorted(k for k in self.pop.parent_log if k <= gen):
            for r in parent_rows(gg, self.pop.parent_log.pop(gg)):
                self.sw.writerow(["" if v is None else (int(v) if isinstance(v, bool) else v) for v in r])
        anc = self.pop.reproduction.ancestors
        for k in [k for k in anc if k not in population]:
            del anc[k]
        del self.pop.backprop_times[:]
        # monitoring AUC of the generation's best-by-training-fitness genome (ties: lowest key)
        k, g = max(((k, g) for k, g in population.items() if g.fitness is not None),
                   key=lambda kg: (kg[1].fitness, -kg[0]))
        states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        try:
            try:
                with torch.no_grad():
                    p = nneat(g, self.config).forward(self.X_mon).detach().cpu().numpy().ravel()
                mon = auc(self.y_mon, p)
            except Exception:  # noqa: BLE001  (an invalid network scores chance, as auc_fitness does)
                mon = 0.5
        finally:
            random.setstate(states[0]); np.random.set_state(states[1]); torch.random.set_rng_state(states[2])
        if mon >= self.best_mon + MIN_DELTA:
            self.best_mon, self.best_gen = mon, gen
        since = gen - self.best_gen
        fits = [x.fitness for x in population.values() if x.fitness is not None]
        sizes = [self._size(x) for x in population.values()]
        bh, bc = self._size(g)
        fg = free_gb()
        self.w.writerow([gen, round(time.time() - self.t0, 1), k, g.fitness, float(np.mean(fits)),
                         len(self.pop.species.species), bh, bc, float(np.mean([s[0] for s in sizes])),
                         float(np.mean([s[1] for s in sizes])), mon, self.best_mon, since, round(fg, 1)])
        if gen % 10 == 0:
            self.fh.flush(); self.sidecar.flush()
        if fg < MIN_FREE_GB:
            return f"free disk below {MIN_FREE_GB} GB ({fg:.1f} GB)"
        if since >= PATIENCE:
            return (f"monitoring AUC plateau: no improvement >= {MIN_DELTA} for {PATIENCE} generations "
                    f"(reference {self.best_mon:.4f} at generation index {self.best_gen})")
        return None

    def close(self):
        self.fh.close(); self.sidecar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["both", "churn"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--max-generations", type=int, default=MAX_GENERATIONS)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")
    if free_gb() < MIN_FREE_GB:
        sys.exit(f"free disk below {MIN_FREE_GB} GB")
    spec = task_spec(args.task)
    name = args.name or f"{spec['prefix']}-s{args.seed}"
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("explaneat.db.population").setLevel(logging.INFO)
    db.init_db()
    X_train, y_train, dataset_name = load_training_data(spec["dataset_id"], spec["split_id"])
    fit, mon = monitor_split(np.asarray(y_train).ravel())
    split_file = HERE / f"monitor_split_{args.task}.json"
    rec = dict(task=args.task, split_id=spec["split_id"], monitor_seed=MONITOR_SEED, share=MONITOR_SHARE,
               positions_are="indices into the split's sorted training rows (split.train_indices order)",
               fit_positions=fit.tolist(), monitor_positions=mon.tolist())
    if split_file.exists():
        assert json.loads(split_file.read_text())["monitor_positions"] == rec["monitor_positions"]
    else:
        split_file.write_text(json.dumps(rec) + "\n")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(spec["config"]))
    seed_everything(args.seed)
    t0 = time.time()
    full = [c - 1 for c in CHECKPOINTS if c <= args.max_generations] + [args.max_generations - 1]
    population = DatabaseBackpropPopulation(config, X_train[fit], y_train[fit], experiment_name=name,
                                            dataset_name=dataset_name, description=DESCRIPTION,
                                            dataset_id=spec["dataset_id"], mask_phantom=False,
                                            genome_storage="best", full_save_generations=full)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    with db.session_scope() as session:
        exp = session.get(Experiment, population.experiment_id)
        exp.split_id = UUID(spec["split_id"])
        exp.random_seed = args.seed
        exp.hardware_info = {
            "platform": os.name, "cpu_count": os.cpu_count(), "machine": platform.machine(),
            "processor": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "device": "cpu", "torch_threads": TORCH_THREADS, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "neat": "0.92",
            "run_params": {"dataset": args.task, "seed": args.seed, "max_generations": args.max_generations,
                           "n_epochs_backprop": N_EPOCHS_BACKPROP, "fitness_function": "auc",
                           "neat_config": str(Path(spec["config"]).relative_to(REPO)), "code_commit": commit,
                           "mask_phantom": False, "genome_storage": "best",
                           "full_save_generation_indices": full,
                           "monitor_split": {"share": MONITOR_SHARE, "seed": MONITOR_SEED,
                                             "file": str(split_file.relative_to(REPO)),
                                             "n_fit": int(len(fit)), "n_monitor": int(len(mon))},
                           "stopping": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                                        "metric": "monitoring AUC of each generation's best-by-training-fitness genome"},
                           "parents_sidecar": f"experiments/lineage/parents/{name}.csv.gz"},
        }
    print(f"EXPERIMENT_ID {population.experiment_id}", flush=True)
    m = Monitor(population, config, X_train[mon], y_train[mon], name, args.max_generations)
    try:
        population.run(auc_fitness, n=args.max_generations, nEpochs=N_EPOCHS_BACKPROP, on_generation=m)
    finally:
        m.close()
    reason = population.stop_reason or f"generation cap ({args.max_generations})"
    out = {"experiment_id": str(population.experiment_id), "name": name, "task": args.task, "seed": args.seed,
           "stop_reason": reason, "last_generation_index": int(population.current_generation),
           "generations_run": int(population.current_generation) + 1,
           "best_monitor_auc": m.best_mon, "best_monitor_generation_index": m.best_gen,
           "wall_clock_s": round(time.time() - t0, 1), "best_fitness": population.best_genome.fitness}
    with db.session_scope() as session:
        exp = session.get(Experiment, population.experiment_id)
        hi = dict(exp.hardware_info)
        hi["run_params"] = dict(hi["run_params"], stop=out)
        exp.hardware_info = hi
    (HERE / "logs" / f"{name}.stop.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
