"""One seeded portal run, in its own process.

    python -m explaneat.batch.worker <run_dir>/spec.json

The spec (written by ``explaneat.batch.jobs``) names the task, the dataset
split, the template, the seed and the run settings.  The worker:

1. seeds python ``random``, numpy and torch, one torch thread, deterministic
   algorithms (``PYTHONHASHSEED=0`` is set by the manager);
2. builds the NEAT config from the template (written to ``config.ini`` in the
   run directory, so the experiment records its text);
3. runs ``DatabaseBackpropPopulation`` (ancestry recorded), with the Cox
   criterion and identity output for survival;
4. after every generation scores the training-fitness champion on the
   validation fold (Cox loss for survival, AUC for classification) and writes
   ``progress.json``; a ``cancel`` file in the run directory stops the run at
   the next generation boundary (status ``interrupted``, resumable);
5. at the end records the selection (the champion with the best validation
   score) in ``Experiment.config_json["batch"]``.

Resume (``spec["resume_experiment_id"]``) continues an interrupted run from its
last saved generation with ``DatabaseBackpropPopulation.resume_from_db``; the
RNG state is not restored, so a resumed run is not bit-identical to an
uninterrupted one.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


class Cancelled(Exception):
    pass


def _write_json(path: Path, obj) -> None:
    from .common import write_json_atomic

    write_json_atomic(path, obj)


class RunReporter:
    """NEAT reporter: progress file, cancellation, validation champion tracking."""

    def __init__(self, run_dir: Path, task, config, X_val, y_val, total, state):
        import torch

        self.run_dir, self.task, self.config = run_dir, task, config
        self.total = total
        self.state = state  # progress dict, shared with main()
        self.X_val = torch.as_tensor(X_val, dtype=torch.float64) if X_val is not None else None
        self.y_val = y_val
        self.generation = state.get("generation", 0)
        self.trace = state.setdefault("trace", [])
        self.risk_sets = None
        if task == "survival" and y_val is not None:
            from ..survival.cox import CoxRiskSets

            self.risk_sets = CoxRiskSets(y_val[:, 0], y_val[:, 1])

    def _flush(self):
        self.state["updated_at"] = time.time()
        _write_json(self.run_dir / "progress.json", self.state)

    def start_generation(self, generation):
        if (self.run_dir / "cancel").exists():
            raise Cancelled("cancelled")
        self.generation = generation
        self.state.update(status="running", generation=generation)
        self._flush()

    def _val_score(self, genome):
        """Validation score of a genome; higher is better (−Cox loss or AUC)."""
        import torch

        from ..core.neuralneat import NeuralNeat

        if self.X_val is None:
            return None
        try:
            with torch.no_grad():
                out = NeuralNeat(genome, self.config).forward(self.X_val).numpy().ravel()
        except Exception:
            return None
        if self.task == "survival":
            from ..survival.cox import cox_ph_loss

            loss = float(cox_ph_loss(torch.as_tensor(out), self.risk_sets, ties="efron"))
            return -loss if np.isfinite(loss) else None
        from sklearn.metrics import roc_auc_score

        try:
            return float(roc_auc_score(self.y_val.ravel(), out))
        except ValueError:
            return None

    def post_evaluate(self, config, population, species, best_genome):
        g = max(population.values(), key=lambda x: x.fitness if x.fitness is not None else -1e18)
        val = self._val_score(g)
        self.trace.append({"generation": self.generation, "genome_key": int(g.key),
                           "train_fitness": g.fitness, "val_score": val})
        fits = [x.fitness for x in population.values() if x.fitness is not None]
        self.state.update(best_fitness=g.fitness,
                          mean_fitness=float(np.mean(fits)) if fits else None,
                          generations_done=self.generation + 1)
        best = [t for t in self.trace if t["val_score"] is not None]
        if best:
            self.state["best_val_score"] = max(t["val_score"] for t in best)
        self._flush()

    def __getattr__(self, name):  # every other reporter hook is a no-op
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _select(trace, experiment_id):
    """Champion with the best validation score (else the best training fitness),
    resolved to its DB genome row (genomes are saved after evaluation)."""
    from ..db import db
    from ..db.models import Genome, Population

    cand = [t for t in trace if t.get("val_score") is not None]
    rule = "best validation score among generation champions"
    if not cand:
        cand = [t for t in trace if t.get("train_fitness") is not None]
        rule = "best training fitness (no validation fold)"
        key = "train_fitness"
    else:
        key = "val_score"
    if not cand:
        return None
    best = max(cand, key=lambda t: (t[key], -t["generation"]))
    with db.session_scope() as s:
        row = (s.query(Genome.id)
               .join(Population, Population.id == Genome.population_id)
               .filter(Population.experiment_id == uuid.UUID(str(experiment_id)),
                       Population.generation == best["generation"],
                       Genome.genome_id == best["genome_key"]).first())
    return {**best, "genome_id": str(row[0]) if row else None, "rule": rule}


def main(spec_path: str) -> int:
    spec_path = Path(spec_path)
    run_dir = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    state = {"status": "starting", "pid": os.getpid(), "generation": 0,
             "total_generations": spec["generations"], "seed": spec["seed"],
             "started_at": time.time()}
    prev = run_dir / "progress.json"
    if spec.get("resume_experiment_id") and prev.exists():
        old = json.loads(prev.read_text(encoding="utf-8"))
        state["trace"] = old.get("trace", [])
        state["experiment_id"] = spec["resume_experiment_id"]
    _write_json(prev, state)
    (run_dir / "cancel").unlink(missing_ok=True)

    import neat

    from ..core.neuralneat import set_output_activation
    from ..db import db
    from ..db.models import Experiment
    from ..db.population import DatabaseBackpropPopulation
    from .common import load_split_arrays, template_ini_text

    if str(spec["database_url"]).startswith("sqlite"):
        import explaneat.db.sqlite_compat  # noqa: F401  (JSONB/UUID on SQLite)
    db.init_db(spec["database_url"])
    task = spec["task"]
    try:
        with db.session_scope() as s:
            data = load_split_arrays(s, spec["dataset_split_id"])
        Xtr, ytr = data["X_train"], data["y_train"]
        Xva, yva = data["X_val"], data["y_val"]
        ini = template_ini_text(spec["template"], Xtr.shape[1], spec.get("pop_size"),
                                spec.get("config_overrides"))
        cfg_path = run_dir / "config.ini"
        cfg_path.write_text(ini, encoding="utf-8")
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, str(cfg_path))
        kwargs = {"mask_phantom": True}
        if task == "survival":
            from ..survival import CoxCriterion, make_cox_fitness

            set_output_activation(config, "identity")
            kwargs["criterion"] = CoxCriterion("efron")
            fitness = make_cox_fitness("efron", float(spec.get("complexity_penalty", 1.0)))
        else:
            from ..evaluators.evaluators import auc_fitness, binary_cross_entropy

            fitness = auc_fitness if spec.get("fitness", "auc") == "auc" else binary_cross_entropy

        _seed(int(spec["seed"]))
        if spec.get("resume_experiment_id"):
            pop = DatabaseBackpropPopulation.resume_from_db(
                spec["resume_experiment_id"], config, Xtr, ytr, **kwargs)
            done = DatabaseBackpropPopulation._get_latest_generation(spec["resume_experiment_id"])
            remaining = max(0, int(spec["generations"]) - (int(done or 0) + 1))
            start_gen = (done or 0) + 1
        else:
            pop = DatabaseBackpropPopulation(
                config, Xtr, ytr, experiment_name=spec["experiment_name"],
                dataset_name=data["dataset_name"], dataset_id=data["dataset_id"],
                description=spec.get("description", "Portal batch run"), **kwargs)
            remaining = int(spec["generations"])
            start_gen = 0
        if task == "survival":
            pop.skip_invalid_genomes = True
        exp_id = str(pop.experiment_id)
        state["experiment_id"] = exp_id
        with db.session_scope() as s:
            exp = s.get(Experiment, uuid.UUID(exp_id))
            exp.split_id = uuid.UUID(data["split_id"])
            exp.dataset_id = uuid.UUID(data["dataset_id"])
            exp.random_seed = int(spec["seed"])
            cj = copy.deepcopy(exp.config_json or {})
            cj["batch"] = {**cj.get("batch", {}), "batch_id": spec["batch_id"], "run_id": spec["run_id"],
                           "task": task, "template": spec["template"], "seed": spec["seed"],
                           "generations": spec["generations"],
                           "complexity_penalty": spec.get("complexity_penalty")}
            exp.config_json = cj

        reporter = RunReporter(run_dir, task, config, Xva, yva, spec["generations"], state)
        reporter.generation = start_gen  # run() continues at the saved generation
        pop.reporters.add(reporter)
        pop.run(fitness, n=remaining, nEpochs=int(spec.get("epochs", 5)))

        sel = _select(state.get("trace", []), exp_id)
        with db.session_scope() as s:
            exp = s.get(Experiment, uuid.UUID(exp_id))
            cj = copy.deepcopy(exp.config_json or {})
            cj.setdefault("batch", {})["selected"] = sel
            cj["batch"]["trace"] = state.get("trace", [])
            exp.config_json = cj
        state.update(status="completed", selected=sel, finished_at=time.time())
        _write_json(prev, state)
        return 0
    except Cancelled:
        exp_id = state.get("experiment_id")
        if exp_id:
            with db.session_scope() as s:
                exp = s.get(Experiment, uuid.UUID(exp_id))
                if exp is not None:
                    exp.status = "interrupted"
        state.update(status="interrupted", finished_at=time.time())
        _write_json(prev, state)
        return 3
    except Exception as e:  # report, then fail
        state.update(status="failed", error=f"{type(e).__name__}: {e}",
                     traceback=traceback.format_exc()[-4000:], finished_at=time.time())
        _write_json(prev, state)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
