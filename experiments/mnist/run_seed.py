"""Run one seeded in-memory PropNEAT evolution on MNIST (exploratory, not pre-registered).

Tasks (see ``data.py``):

    mnist38   3 vs 8, 14x14 (196 inputs), sigmoid output, BCE backprop, AUC fitness
              (the 10-seed study's training path; config configs/mnist38.ini)
    mnist10   10 classes, 784 inputs, identity output (logits), softmax
              cross-entropy backprop, fitness = -train cross-entropy;
              OutputSinkGenome (outputs never feed other nodes); config
              configs/mnist10.ini (partial_direct 0.1)

Seeding follows experiments/10seed/run_seed.py: python ``random`` (neat-python),
numpy and torch seeded before the population is built, 1 torch thread,
deterministic algorithms, ``PYTHONHASHSEED=0`` required.  In memory
(``BackpropPopulation``): nothing is written to the database during the run.
mask_phantom = True (the library default): the stored genome is the trained
network.  The best genome is the best *training* fitness over all generations
(no validation selection, as in the 10-seed study); the test split is only
logged.

Outputs:
    experiments/mnist/logs/<name>.csv       one row per generation
    experiments/mnist/logs/<name>.json      summary (timings, final metrics)
    ~/.explaneat/mnist_runs/<name>/ckpt.pkl.gz         resume checkpoint (every --ckpt-every gens)
    ~/.explaneat/mnist_runs/<name>/final.pkl.gz        last evaluated population, species,
                                                        parents, best genome, config text
    ~/.explaneat/mnist_runs/<name>/best_genome.pkl.gz
    ~/.explaneat/mnist_runs/<name>/parents.csv.gz      full parent log (lineage sidecar format)

Resume: rerun the same command; if ckpt.pkl.gz exists the run continues from it
(RNG states restored, so the continuation equals an uninterrupted run).
Use --fresh to ignore a checkpoint.

    PYTHONHASHSEED=0 uv run python experiments/mnist/run_seed.py mnist38 0
    PYTHONHASHSEED=0 uv run python experiments/mnist/run_seed.py mnist10 0 --generations 100
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import count  # noqa: E402
from pathlib import Path  # noqa: E402

import neat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from data import prepare  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.multiclass import (  # noqa: E402
    SoftmaxCrossEntropyCriterion,
    make_multiclass_fitness,
    use_output_sink_genome,
)
from explaneat.core.neuralneat import NeuralNeat, set_output_activation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

TORCH_THREADS = 1
N_EPOCHS_BACKPROP = 5
RUNS_DIR = Path(os.environ.get("EXPLANEAT_MNIST_RUNS", Path.home() / ".explaneat" / "mnist_runs"))
LOG_DIR = HERE / "logs"

TASKS = {
    "mnist38": dict(config=HERE / "configs/mnist38.ini", n_train=None, generations=100,
                    multiclass=False),
    "mnist10": dict(config=HERE / "configs/mnist10.ini", n_train=10000, generations=100,
                    multiclass=True),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


def load_config(task: str) -> neat.Config:
    spec = TASKS[task]
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(spec["config"]))
    if spec["multiclass"]:
        use_output_sink_genome(config)
        set_output_activation(config, "identity")
    return config


def evaluate(genome, config, X, y, multiclass: bool) -> dict:
    """Metrics of one genome via NeuralNeat (the training engine)."""
    try:
        net = NeuralNeat(genome, config)
    except GenomeNotValidError:
        return {}
    with torch.no_grad():
        out = net.forward(torch.as_tensor(X, dtype=torch.float64))
    if multiclass:
        yt = torch.as_tensor(y, dtype=torch.long)
        return {"ce": float(F.cross_entropy(out, yt)),
                "acc": float((out.argmax(1) == yt).double().mean())}
    p = out.numpy().ravel()
    yy = np.asarray(y).ravel()
    eps = 1e-12
    bce = float(-np.mean(yy * np.log(np.clip(p, eps, 1)) + (1 - yy) * np.log(np.clip(1 - p, eps, 1))))
    return {"auc": float(roc_auc_score(yy, p)), "acc": float(((p >= 0.5) == (yy == 1)).mean()),
            "bce": bce}


def genome_size(genome, config) -> dict:
    """Hidden nodes / enabled connections / inputs used in the *phenotype* (NeuralNeat-valid)."""
    try:
        net = NeuralNeat(genome, config)
    except GenomeNotValidError:
        return {"hidden": None, "conns": None, "inputs_used": None, "depth": None}
    valid = set(net.node_mapping.valid_nodes)
    ins = set(config.genome_config.input_keys)
    outs = set(config.genome_config.output_keys)
    conns = [k for k, c in genome.connections.items()
             if c.enabled and k[0] in valid and k[1] in valid]
    return {"hidden": len([n for n in valid if n not in ins and n not in outs]),
            "conns": len(conns), "inputs_used": len({a for a, _ in conns if a in ins}),
            "depth": net.node_mapping.depth}


def species_summary(species_set, generation) -> list:
    out = []
    for sid, s in species_set.species.items():
        fits = [m.fitness for m in s.members.values() if m.fitness is not None]
        out.append(dict(species_id=sid, size=len(s.members), members=sorted(s.members),
                        representative=s.representative.key if s.representative else None,
                        created=s.created, last_improved=s.last_improved,
                        fitness=s.fitness, adjusted_fitness=s.adjusted_fitness,
                        fitness_mean=float(np.mean(fits)) if fits else None,
                        fitness_max=float(max(fits)) if fits else None,
                        fitness_min=float(min(fits)) if fits else None,
                        generation=generation))
    return out


def dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class RunReporter:
    """Per-generation CSV log, resume checkpoints, and capture of the last evaluated generation."""

    FIELDS = ["generation", "wall_s", "gen_s", "best_fitness", "mean_fitness", "n_species",
              "champ_key", "champ_hidden", "champ_conns", "champ_inputs_used", "champ_depth",
              "champ_test_metric", "best_so_far_fitness"]

    def __init__(self, run, data, last_generation):
        self.run, self.data, self.last_generation = run, data, last_generation
        self.pop = None
        self.t_gen = None
        self.row = None
        self.final = None
        self._evaluated = None
        self.best_generation = None

    def start_generation(self, generation):
        self.t_gen = time.time()

    def post_evaluate(self, config, population, species, best):
        pop = self.pop
        gen = pop.generation
        self._evaluated = population
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        m = evaluate(best, config, self.data["X_test"], self.data["y_test"], self.run["multiclass"])
        size = genome_size(best, config)
        self.row = dict(generation=gen, best_fitness=best.fitness, mean_fitness=float(np.mean(fits)),
                        n_species=len(species.species), champ_key=best.key,
                        champ_hidden=size["hidden"], champ_conns=size["conns"],
                        champ_inputs_used=size["inputs_used"], champ_depth=size["depth"],
                        champ_test_metric=m.get("acc" if self.run["multiclass"] else "auc"))
        if gen == self.last_generation:
            # the last *evaluated* population (run() then reproduces once more)
            self.final = dict(generation=gen, population=copy.deepcopy(population),
                              species=species_summary(species, gen),
                              parents=dict(pop.parent_log.get(gen, {})))

    def end_generation(self, config, population, species_set):
        pop = self.pop
        best_so_far = pop.best_genome.fitness if pop.best_genome is not None else None
        if any(g is pop.best_genome for g in self._evaluated.values()):
            self.best_generation = self.row["generation"]
        self.row.update(gen_s=round(time.time() - self.t_gen, 2),
                        wall_s=round(time.time() - self.run["t0"] + self.run["wall_before"], 1),
                        best_so_far_fitness=best_so_far)
        new = not self.run["csv"].exists()
        with open(self.run["csv"], "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(self.row)
        logging.info("gen %d  %.1fs  best %.5f  test %.4f  hidden %s conns %s  species %d",
                     self.row["generation"], self.row["gen_s"], self.row["best_fitness"],
                     self.row["champ_test_metric"] or float("nan"), self.row["champ_hidden"],
                     self.row["champ_conns"], self.row["n_species"])
        next_gen = pop.generation + 1
        if next_gen <= self.last_generation and next_gen % self.run["ckpt_every"] == 0:
            self.checkpoint(next_gen)

    def checkpoint(self, next_gen):
        pop = self.pop
        rep = pop.reproduction
        nxt = next(rep.genome_indexer)
        rep.genome_indexer = count(nxt)
        sp = pop.species
        reporters, sp.reporters = sp.reporters, None
        sp_indexer_next = next(sp.indexer)
        sp.indexer = count(sp_indexer_next)
        try:
            state = dict(generation=next_gen, population=pop.population, species=sp,
                         best_genome=pop.best_genome, genome_indexer_next=nxt,
                         species_indexer_next=sp_indexer_next, ancestors=rep.ancestors,
                         parent_log=pop.parent_log, best_generation=self.best_generation,
                         rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state()),
                         wall_s=time.time() - self.run["t0"] + self.run["wall_before"])
            dump(state, self.run["dir"] / "ckpt.pkl.gz")
        finally:
            sp.reporters = reporters
        logging.info("checkpoint at generation %d", next_gen)

    def __getattr__(self, name):  # other reporter hooks are no-ops
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                     cwd=REPO, text=True).splitlines()
    dirty = any(not line.rstrip().endswith(".pyc") for line in status if line.strip())
    return commit, dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = TASKS[args.task]
    n_gen = args.generations or spec["generations"]
    n_train = args.n_train if args.n_train is not None else spec["n_train"]
    name = args.name or f"{args.task}-s{args.seed}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data = prepare(args.task, n_train)
    config = load_config(args.task)
    multiclass = spec["multiclass"]
    y_train = data["y_train"].astype(np.float64)
    criterion = SoftmaxCrossEntropyCriterion() if multiclass else None
    fitness = make_multiclass_fitness("neg_ce") if multiclass else auc_fitness
    csv_path = LOG_DIR / f"{name}.csv"
    ckpt = run_dir / "ckpt.pkl.gz"

    kwargs = {} if criterion is None else {"criterion": criterion}
    seed_everything(args.seed)
    wall_before = 0.0
    if ckpt.exists() and not args.fresh:
        with gzip.open(ckpt, "rb") as f:
            st = pickle.load(f)
        pop = BackpropPopulation(config, data["X_train"], y_train,
                                 initial_state=(st["population"], st["species"], st["generation"]),
                                 mask_phantom=True, **kwargs)
        pop.reproduction.genome_indexer = count(st["genome_indexer_next"])
        pop.reproduction.ancestors = st["ancestors"]
        pop.species.reporters = pop.reporters
        pop.species.indexer = count(st["species_indexer_next"])
        pop.best_genome = st["best_genome"]
        pop.parent_log = st["parent_log"]
        random.setstate(st["rng"]["random"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        wall_before = st["wall_s"]
        start_gen = st["generation"]
        # drop log rows the checkpoint does not cover
        if csv_path.exists():
            rows = list(csv.DictReader(open(csv_path)))
            keep = [r for r in rows if int(r["generation"]) < start_gen]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RunReporter.FIELDS)
                w.writeheader()
                w.writerows(keep)
        logging.info("resumed %s at generation %d", name, start_gen)
    else:
        if csv_path.exists():
            csv_path.unlink()
        pop = BackpropPopulation(config, data["X_train"], y_train, mask_phantom=True, **kwargs)
        start_gen = 0

    run = dict(multiclass=multiclass, csv=csv_path, dir=run_dir, ckpt_every=args.ckpt_every,
               t0=time.time(), wall_before=wall_before)
    reporter = RunReporter(run, data, last_generation=n_gen - 1)
    reporter.pop = pop
    if start_gen:
        reporter.best_generation = st.get("best_generation")
    pop.add_reporter(reporter)
    commit, dirty = git_state()
    logging.info("%s: train %s test %s, %d generations from %d, commit %s%s", name,
                 data["X_train"].shape, data["X_test"].shape, n_gen, start_gen, commit[:9],
                 " (dirty)" if dirty else "")

    pop.run(fitness, n=n_gen - start_gen, nEpochs=N_EPOCHS_BACKPROP)
    wall = time.time() - run["t0"] + wall_before

    best = copy.deepcopy(pop.best_genome)
    final = reporter.final
    train_m = evaluate(best, config, data["X_train"], data["y_train"], multiclass)
    test_m = evaluate(best, config, data["X_test"], data["y_test"], multiclass)
    config_text = Path(spec["config"]).read_text()
    meta = dict(task=args.task, seed=args.seed, name=name, n_generations=n_gen,
                best_generation=reporter.best_generation, n_train=int(len(y_train)),
                n_epochs_backprop=N_EPOCHS_BACKPROP, fitness_function="neg_ce" if multiclass else "auc",
                criterion="softmax_cross_entropy" if multiclass else "bce",
                output_activation="identity" if multiclass else "sigmoid",
                genome_type="OutputSinkGenome" if multiclass else "DefaultGenome",
                mask_phantom=True, torch_threads=TORCH_THREADS,
                neat_config=str(Path(spec["config"]).relative_to(REPO)),
                code_commit=commit, code_dirty=dirty, data=data["meta"],
                hardware=dict(machine=platform.machine(), cpu_count=os.cpu_count(),
                              python=platform.python_version(), torch=torch.__version__,
                              numpy=np.__version__, neat="0.92"),
                wall_clock_s=round(wall, 1))
    dump(dict(meta=meta, config_text=config_text, best_genome=best, final=final,
              train_rows=data["train_rows"], test_rows=data["test_rows"]), run_dir / "final.pkl.gz")
    dump(dict(meta=meta, config_text=config_text, genome=best), run_dir / "best_genome.pkl.gz")
    with gzip.open(run_dir / "parents.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for r in pop.ancestry_rows():
            w.writerow([r[0], r[1], "" if r[2] is None else r[2], "" if r[3] is None else r[3],
                        int(bool(r[4]))])
    summary = dict(meta, best_key=best.key, best_fitness=best.fitness, best_size=genome_size(best, config),
                   train=train_m, test=test_m, final_generation=final["generation"] if final else None)
    (LOG_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
