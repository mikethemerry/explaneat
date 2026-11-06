"""Replay a recorded 10-seed / churn run in memory and capture its parents.

None of the recorded evolutionary runs stored parents (the runners never
passed an AncestryReporter). The runs are bit-identical from their seeds
(experiments/10seed/README.md), so this script replays a run exactly as
``experiments/10seed/run_seed.py`` launched it, with the same dataset, split,
NEAT config, seed, torch settings and ``mask_phantom=False``, and records
``reproduction.ancestors`` every generation (via ``parent_log``, see
explaneat/core/lineage_capture.py).

The actual ``DatabaseBackpropPopulation.run`` loop is executed. Only the
database is replaced: the module-level ``db`` handle is stubbed, and the save
methods are overridden to capture each generation's genomes instead of writing
them. Nothing is written to any database. The training data are read from the
DB (read-only), as run_seed.py does.

Output, under experiments/lineage/replay_db/: ``<label>-s<seed>.replay.csv.gz``
with one row per saved genome:
``generation, genome_key, fitness, genome_md5, parent1_key, parent2_key, is_elite_copy``.
``genome_md5`` is md5 of ``json.dumps(serialize_genome(g), sort_keys=True)``,
the same canonical form verify_replay.py computes from the stored genome_data.

    PYTHONHASHSEED=0 uv run python experiments/lineage/replay_lineage.py churn 3 --generations 200
    PYTHONHASHSEED=0 uv run python experiments/lineage/replay_lineage.py heart100 3
"""
import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import neat  # noqa: E402

from explaneat.db import population as popmod  # noqa: E402
from explaneat.db.base import db as real_db  # noqa: E402
from explaneat.db.population import DatabaseBackpropPopulation  # noqa: E402
from explaneat.db.serialization import serialize_genome  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_seed", REPO / "experiments/10seed/run_seed.py")
run_seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_seed)

OUT_DIR = HERE / "replay_db"


def genome_md5(data: dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class _NullSession:
    def get(self, *args, **kwargs):
        return SimpleNamespace()

    def __getattr__(self, name):  # any other DB use is a bug in the replay
        raise RuntimeError(f"replay touched the database: session.{name}")


class _NullDB:
    @contextlib.contextmanager
    def session_scope(self):
        yield _NullSession()


class ReplayPopulation(DatabaseBackpropPopulation):
    """DatabaseBackpropPopulation with every database write replaced by capture."""

    def __init__(self, config, x_train, y_train, mask_phantom=False):
        # Same RNG path as DatabaseBackpropPopulation.__init__: everything that
        # consumes randomness happens in BackpropPopulation.__init__.
        BackpropPopulation.__init__(self, config, x_train, y_train,
                                    mask_phantom=mask_phantom)
        self.experiment_id = None
        self.current_population_id = None
        self.current_generation = 0
        self.ancestry_reporter = None
        self.captured = []  # (generation, key, fitness, md5)

    def _save_population_state(self, generation):
        return generation

    def _save_genomes(self, population_id, ancestry_reporter=None):
        gen = population_id
        for key, g in self.population.items():
            data = serialize_genome(g)
            self.captured.append((gen, key, data["fitness"], genome_md5(data)))

    def _save_training_metrics_batch(self, population_id, nEpochs):
        pass

    def _save_result(self, *args, **kwargs):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(run_seed.DATASETS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--label", default=None,
                    help="output label (default: churn<N>/heart100/monk2/...)")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = run_seed.DATASETS[args.dataset]
    n_gen = args.generations or spec["n_generations"]
    label = args.label or (f"churn{n_gen}" if args.dataset == "churn" else args.dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{label}-s{args.seed}.replay.csv.gz"

    real_db.init_db()
    X_train, y_train, _ = run_seed.load_training_data(spec["dataset_id"], spec["split_id"])
    real_db.close() if hasattr(real_db, "close") else None

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(spec["neat_config"]))
    run_seed.seed_everything(args.seed)
    t0 = time.time()
    pop = ReplayPopulation(config, X_train, y_train, mask_phantom=False)
    popmod.db = _NullDB()  # the run loop's status updates become no-ops
    pop.run(auc_fitness, n=n_gen, nEpochs=run_seed.N_EPOCHS_BACKPROP)

    with gzip.open(out, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "fitness", "genome_md5",
                    "parent1_key", "parent2_key", "is_elite_copy"])
        for gen, key, fit, h in pop.captured:
            p1, p2, elite = pop.parent_log[gen][key]
            w.writerow([gen, key, repr(fit), h,
                        "" if p1 is None else p1, "" if p2 is None else p2, int(elite)])
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "generations": n_gen,
                      "rows": len(pop.captured), "wall_s": round(time.time() - t0, 1),
                      "out": str(out.relative_to(REPO))}), flush=True)


if __name__ == "__main__":
    main()
