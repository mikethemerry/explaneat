"""Seeded in-memory PropNEAT run on a complex real dataset (exploratory, not pre-registered).

Tasks (``data.py``): ``krk`` (chess KRK, draw vs win, 6 ordinal inputs) and ``c4``
(Connect-4, win vs not win, 42 ordinal cells). Sigmoid output, BCE backprop (5 epochs per
generation), AUC fitness: the 10-seed study's training path (``configs/<task>.ini`` is
``experiments/10seed/configs/heart.ini`` with ``num_inputs`` changed).

Variants:
    default  the config as is (static compatibility threshold 3.0)
    tuned    a dynamic compatibility threshold targeting 5-10 species: after each
             generation's speciation, if there are fewer than ``--species-min`` species the
             threshold drops by ``--threshold-step``, if more than ``--species-max`` it rises
             by the step (floor ``--threshold-min``). Harness only: the threshold lives on
             ``config.species_set_config`` and neat-python reads it at the next speciation.

Seeding as experiments/10seed/run_seed.py and experiments/mnist/run_seed.py (python random,
numpy and torch seeded before the population is built; 1 torch thread; deterministic
algorithms; PYTHONHASHSEED=0 required). In memory (BackpropPopulation, mask_phantom=True,
the library default: the stored genome is the trained network). The best genome is the best
*training* fitness over all generations; the test split is only logged.

Outputs:
    experiments/complex_real/logs/<name>.csv      one row per generation (incl. n_species, threshold)
    experiments/complex_real/logs/<name>.json     summary
    ~/.explaneat/complex_real_runs/<name>/{final,best_genome,ckpt}.pkl.gz, parents.csv.gz

    PYTHONHASHSEED=0 uv run python experiments/complex_real/run_seed.py krk 0
    PYTHONHASHSEED=0 uv run python experiments/complex_real/run_seed.py c4 0 --variant tuned
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
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from data import prepare  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

TORCH_THREADS = 1
N_EPOCHS_BACKPROP = 5
RUNS_DIR = Path(os.environ.get("EXPLANEAT_COMPLEX_RUNS", Path.home() / ".explaneat" / "complex_real_runs"))
LOG_DIR = HERE / "logs"

TASKS = {
    "krk": dict(config=HERE / "configs/krk.ini", n_train=None, generations=100),
    "c4": dict(config=HERE / "configs/c4.ini", n_train=None, generations=100),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)


def load_config(task: str) -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                       neat.DefaultStagnation, str(TASKS[task]["config"]))


def evaluate(genome, config, X, y) -> dict:
    try:
        net = NeuralNeat(genome, config)
    except GenomeNotValidError:
        return {}
    with torch.no_grad():
        p = net.forward(torch.as_tensor(X, dtype=torch.float64)).numpy().ravel()
    yy = np.asarray(y).ravel()
    eps = 1e-12
    bce = float(-np.mean(yy * np.log(np.clip(p, eps, 1)) + (1 - yy) * np.log(np.clip(1 - p, eps, 1))))
    return {"auc": float(roc_auc_score(yy, p)), "acc": float(((p >= 0.5) == (yy == 1)).mean()), "bce": bce}


def genome_size(genome, config) -> dict:
    try:
        net = NeuralNeat(genome, config)
    except GenomeNotValidError:
        return {"hidden": None, "conns": None, "inputs_used": None, "depth": None}
    valid = set(net.node_mapping.valid_nodes)
    ins = set(config.genome_config.input_keys)
    outs = set(config.genome_config.output_keys)
    conns = [k for k, c in genome.connections.items() if c.enabled and k[0] in valid and k[1] in valid]
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
                        fitness_min=float(min(fits)) if fits else None, generation=generation))
    return out


def dump(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class ThresholdController:
    """Dynamic compatibility threshold: one step toward the target species band per generation."""

    def __init__(self, config, lo, hi, step, floor):
        self.config, self.lo, self.hi, self.step, self.floor = config, lo, hi, step, floor

    @property
    def threshold(self):
        return self.config.species_set_config.compatibility_threshold

    def update(self, n_species):
        t = self.threshold
        if n_species < self.lo:
            t = max(self.floor, t - self.step)
        elif n_species > self.hi:
            t = t + self.step
        self.config.species_set_config.compatibility_threshold = t
        return t


class RunReporter:
    FIELDS = ["generation", "wall_s", "gen_s", "best_fitness", "mean_fitness", "n_species",
              "threshold_used", "threshold_next", "champ_key", "champ_hidden", "champ_conns",
              "champ_inputs_used", "champ_depth", "champ_test_auc", "best_so_far_fitness"]

    def __init__(self, run, data, last_generation, controller=None):
        self.run, self.data, self.last_generation, self.controller = run, data, last_generation, controller
        self.pop = None
        self.row = None
        self.final = None
        self._evaluated = None
        self.best_generation = None
        self.t_gen = None

    def start_generation(self, generation):
        self.t_gen = time.time()

    def post_evaluate(self, config, population, species, best):
        gen = self.pop.generation
        self._evaluated = population
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        m = evaluate(best, config, self.data["X_test"], self.data["y_test"])
        size = genome_size(best, config)
        self.row = dict(generation=gen, best_fitness=best.fitness, mean_fitness=float(np.mean(fits)),
                        n_species=len(species.species),
                        threshold_used=config.species_set_config.compatibility_threshold,
                        champ_key=best.key, champ_hidden=size["hidden"], champ_conns=size["conns"],
                        champ_inputs_used=size["inputs_used"], champ_depth=size["depth"],
                        champ_test_auc=m.get("auc"))
        if gen == self.last_generation:
            self.final = dict(generation=gen, population=copy.deepcopy(population),
                              species=species_summary(species, gen),
                              parents=dict(self.pop.parent_log.get(gen, {})),
                              threshold=config.species_set_config.compatibility_threshold)

    def end_generation(self, config, population, species_set):
        pop = self.pop
        if any(g is pop.best_genome for g in self._evaluated.values()):
            self.best_generation = self.row["generation"]
        # species of the *next* generation are formed now (speciate ran before this hook);
        # the threshold change applies from the following speciation
        nxt = self.controller.update(len(species_set.species)) if self.controller else \
            config.species_set_config.compatibility_threshold
        self.row.update(gen_s=round(time.time() - self.t_gen, 2),
                        wall_s=round(time.time() - self.run["t0"] + self.run["wall_before"], 1),
                        best_so_far_fitness=pop.best_genome.fitness if pop.best_genome else None,
                        threshold_next=nxt)
        new = not self.run["csv"].exists()
        with open(self.run["csv"], "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(self.row)
        logging.info("gen %d  %.1fs  best %.5f  test %.4f  hidden %s conns %s  species %d  thr %.2f",
                     self.row["generation"], self.row["gen_s"], self.row["best_fitness"],
                     self.row["champ_test_auc"] or float("nan"), self.row["champ_hidden"],
                     self.row["champ_conns"], self.row["n_species"], self.row["threshold_used"])
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
        sp_next = next(sp.indexer)
        sp.indexer = count(sp_next)
        try:
            state = dict(generation=next_gen, population=pop.population, species=sp,
                         best_genome=pop.best_genome, genome_indexer_next=nxt,
                         species_indexer_next=sp_next, ancestors=rep.ancestors,
                         parent_log=pop.parent_log, best_generation=self.best_generation,
                         threshold=pop.config.species_set_config.compatibility_threshold,
                         rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state()),
                         wall_s=time.time() - self.run["t0"] + self.run["wall_before"])
            dump(state, self.run["dir"] / "ckpt.pkl.gz")
        finally:
            sp.reporters = reporters

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                     cwd=REPO, text=True).splitlines()
    return commit, any(line.strip() and not line.rstrip().endswith(".pyc") for line in status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--variant", choices=["default", "tuned"], default="default")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--species-min", type=int, default=5)
    ap.add_argument("--species-max", type=int, default=10)
    ap.add_argument("--threshold-step", type=float, default=0.3)
    ap.add_argument("--threshold-min", type=float, default=0.3)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = TASKS[args.task]
    n_gen = args.generations or spec["generations"]
    n_train = args.n_train if args.n_train is not None else spec["n_train"]
    name = args.name or f"{args.task}-{args.variant}-s{args.seed}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data = prepare(args.task, n_train)
    config = load_config(args.task)
    y_train = data["y_train"].astype(np.float64)
    csv_path = LOG_DIR / f"{name}.csv"
    ckpt = run_dir / "ckpt.pkl.gz"
    controller = (ThresholdController(config, args.species_min, args.species_max,
                                      args.threshold_step, args.threshold_min)
                  if args.variant == "tuned" else None)

    seed_everything(args.seed)
    wall_before, st = 0.0, None
    if ckpt.exists() and not args.fresh:
        with gzip.open(ckpt, "rb") as f:
            st = pickle.load(f)
        config.species_set_config.compatibility_threshold = st["threshold"]
        pop = BackpropPopulation(config, data["X_train"], y_train,
                                 initial_state=(st["population"], st["species"], st["generation"]),
                                 mask_phantom=True)
        pop.reproduction.genome_indexer = count(st["genome_indexer_next"])
        pop.reproduction.ancestors = st["ancestors"]
        pop.species.reporters = pop.reporters
        pop.species.indexer = count(st["species_indexer_next"])
        pop.best_genome = st["best_genome"]
        pop.parent_log = st["parent_log"]
        random.setstate(st["rng"]["random"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        wall_before, start_gen = st["wall_s"], st["generation"]
        if csv_path.exists():
            rows = [r for r in csv.DictReader(open(csv_path)) if int(r["generation"]) < start_gen]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=RunReporter.FIELDS)
                w.writeheader()
                w.writerows(rows)
        logging.info("resumed %s at generation %d", name, start_gen)
    else:
        if csv_path.exists():
            csv_path.unlink()
        pop = BackpropPopulation(config, data["X_train"], y_train, mask_phantom=True)
        start_gen = 0

    run = dict(csv=csv_path, dir=run_dir, ckpt_every=args.ckpt_every, t0=time.time(),
               wall_before=wall_before)
    reporter = RunReporter(run, data, last_generation=n_gen - 1, controller=controller)
    reporter.pop = pop
    if st:
        reporter.best_generation = st.get("best_generation")
    pop.add_reporter(reporter)
    commit, dirty = git_state()
    logging.info("%s: train %s test %s, %d generations from %d, commit %s%s", name,
                 data["X_train"].shape, data["X_test"].shape, n_gen, start_gen, commit[:9],
                 " (dirty)" if dirty else "")

    pop.run(auc_fitness, n=n_gen - start_gen, nEpochs=N_EPOCHS_BACKPROP)
    wall = time.time() - run["t0"] + wall_before

    best = copy.deepcopy(pop.best_genome)
    final = reporter.final
    train_m = evaluate(best, config, data["X_train"], data["y_train"])
    test_m = evaluate(best, config, data["X_test"], data["y_test"])
    config_text = Path(spec["config"]).read_text()
    rows = list(csv.DictReader(open(csv_path)))
    nsp = [int(r["n_species"]) for r in rows]
    meta = dict(task=args.task, variant=args.variant, seed=args.seed, name=name, n_generations=n_gen,
                best_generation=reporter.best_generation, n_train=int(len(y_train)),
                n_epochs_backprop=N_EPOCHS_BACKPROP, fitness_function="auc", criterion="bce",
                output_activation="sigmoid", mask_phantom=True, torch_threads=TORCH_THREADS,
                neat_config=str(Path(spec["config"]).relative_to(REPO)),
                dynamic_threshold=(dict(species_min=args.species_min, species_max=args.species_max,
                                        step=args.threshold_step, floor=args.threshold_min)
                                   if controller else None),
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
    summary = dict(meta, best_key=best.key, best_fitness=best.fitness,
                   best_size=genome_size(best, config), train=train_m, test=test_m,
                   species=dict(mean=float(np.mean(nsp)), min=min(nsp), max=max(nsp),
                                last=nsp[-1], mean_last50=float(np.mean(nsp[-50:]))),
                   final_threshold=final["threshold"] if final else None,
                   final_generation=final["generation"] if final else None)
    (LOG_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
