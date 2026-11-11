"""MNIST-10 linear-start series (exploratory, not pre-registered).

Question: if PropNEAT starts from the linear model (every pixel wired to every
output), does evolution then progress past linear?

Reuses the mnist10 path of ``run_seed.py`` (``configs/mnist10.ini``,
``data.prepare("mnist10", 10000)``: the same 10k train / 10k test rows,
OutputSinkGenome, identity outputs, softmax-CE backprop, fitness = -train CE,
mask_phantom=True, seeding, checkpoints).  Differences, all opt-in flags:

    --init full    initial_connection = full_direct, weights/biases N(0, 1)
                   (the config's own init)
    --init lr      full_direct, then every genome's weights and output biases are
                   set to the fitted multinomial LR coefficients (sklearn
                   LogisticRegression(C=1, max_iter=5000) on the 10k train rows,
                   as in baselines.py; coefficients used as is)
    --set S.k=v    override a NEAT config value (written to the run dir)
    --controller lo,hi
                   species-target controller: after each speciation the
                   compatibility threshold is multiplied by 0.8 if there are fewer
                   than ``lo`` species and by 1.25 if more than ``hi``.
                   Multiplicative (not the +-0.3 of experiments/complex_real/)
                   because full_direct genomes are ~0.01-0.1 apart.
    --hidden-penalty L
                   fitness = -train CE - L * (enabled connections that are not
                   input -> output); the linear backbone is never charged
    --addnode-weighted
                   add-node splits an enabled connection chosen with probability
                   proportional to |weight| (default: uniform over all connections)

Validation: 5,000 standard-train rows *outside* the 10k training subset (seeded,
``VAL_SEED``) are scored for the champion every generation.  They are never used
for training or for selection inside a run; they exist to compare runs without
touching test.  Test is logged as in run_seed.py.  The best genome is the best
training fitness over all generations, as in run_seed.py.

    PYTHONHASHSEED=0 uv run python experiments/mnist/linear_start.py ls-full-s0 0 --init full
    PYTHONHASHSEED=0 uv run python experiments/mnist/linear_start.py ls-lr-s0 0 --init lr

Outputs: experiments/mnist/logs/linear_start/<name>.{csv,json}; run files in
~/.explaneat/mnist_runs/<name>/ (config.ini, ckpt, final, best_genome, parents).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import configparser  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import count  # noqa: E402
from pathlib import Path  # noqa: E402

import neat  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from data import N_STD_TRAIN, SUBSET_SEED, load_raw, prepare  # noqa: E402
from run_seed import (RUNS_DIR, TASKS, dump, evaluate, genome_size, git_state,  # noqa: E402
                      seed_everything, species_summary)
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.multiclass import (INVALID_CE, OutputSinkGenome,  # noqa: E402
                                       SoftmaxCrossEntropyCriterion, make_multiclass_fitness)
from explaneat.core.neuralneat import NeuralNeat, set_output_activation  # noqa: E402

LOG_DIR = HERE / "logs" / "linear_start"
VAL_SEED = SUBSET_SEED + 1
N_VAL = 5000


# ---------------------------------------------------------------- data / LR
def validation_rows(train_rows, n_val=N_VAL):
    X, y = load_raw()
    pool = np.setdiff1d(np.arange(N_STD_TRAIN), train_rows)
    rows = np.sort(np.random.default_rng(VAL_SEED).choice(pool, size=n_val, replace=False))
    return rows, X[rows].astype(np.float64) / 255.0, y[rows].astype(np.int64)


def load_data():
    d = prepare("mnist10", TASKS["mnist10"]["n_train"])
    d["val_rows"], d["X_val"], d["y_val"] = validation_rows(d["train_rows"])
    assert not set(d["val_rows"]) & set(d["train_rows"]) and d["val_rows"].max() < N_STD_TRAIN
    return d


def fit_lr(d):
    """Multinomial LR as in baselines.py; cached in RUNS_DIR (deterministic lbfgs)."""
    path = RUNS_DIR / "lr_mnist10_coef.npz"
    if path.exists():
        z = np.load(path)
        return z["coef"], z["intercept"]
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1.0, max_iter=5000).fit(d["X_train"], d["y_train"].astype(int))
    assert list(m.classes_) == list(range(10))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, coef=m.coef_, intercept=m.intercept_)
    return m.coef_, m.intercept_


# ---------------------------------------------------------------- genome / config
class WeightedAddNodeGenome(OutputSinkGenome):
    """OutputSinkGenome whose add-node splits an enabled connection ~ |weight|."""

    def mutate_add_node(self, config):
        conns = [c for c in self.connections.values() if c.enabled]
        if not conns:
            return super().mutate_add_node(config)
        w = [abs(c.weight) + 1e-12 for c in conns]
        conn_to_split = random.choices(conns, weights=w, k=1)[0]
        new_node_id = config.get_new_node_key(self.nodes)
        self.nodes[new_node_id] = self.create_node(config, new_node_id)
        conn_to_split.enabled = False
        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)


def write_config(run_dir: Path, overrides: list[str]) -> Path:
    cp = configparser.ConfigParser()
    cp.read(TASKS["mnist10"]["config"])
    cp["DefaultGenome"]["initial_connection"] = "full_direct"
    for ov in overrides:
        key, val = ov.split("=", 1)
        sec, k = key.split(".", 1)
        assert k in cp[sec], f"unknown config key {key}"
        cp[sec][k] = val
    path = run_dir / "config.ini"
    with open(path, "w") as f:
        cp.write(f)
    return path


def load_config(path: Path, weighted_addnode: bool) -> neat.Config:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(path))
    config.genome_type = WeightedAddNodeGenome if weighted_addnode else OutputSinkGenome
    set_output_activation(config, "identity")
    return config


def warm_start(population, config, coef, intercept):
    gc = config.genome_config
    for g in population.values():
        for oi, o in enumerate(gc.output_keys):
            g.nodes[o].bias = float(intercept[oi])
            for ii, i in enumerate(gc.input_keys):
                g.connections[(i, o)].weight = float(coef[oi, ii])


def backbone_count(genome, config):
    ins, outs = set(config.genome_config.input_keys), set(config.genome_config.output_keys)
    n_bb = sum(1 for (a, b), c in genome.connections.items() if c.enabled and a in ins and b in outs)
    n_other = sum(1 for c in genome.connections.values() if c.enabled) - n_bb
    return n_bb, n_other


class CachedCEPopulation(BackpropPopulation):
    """BackpropPopulation whose backprop step also records each genome's post-training
    train CE (``genome._post_ce``), so the fitness need not rebuild the network.

    The training step is ``BackpropPopulation.backpropagate`` without its timing and
    logging statistics: NeuralNeat(mask_phantom=True), a fresh Adadelta(lr=1.5),
    ``nEpochs`` full-batch steps, weights written back.  Phantom entries are 0 and
    masked, so the recorded CE equals ``make_multiclass_fitness("neg_ce")`` on the
    written-back genome (checked by ``--check-fitness``).  Rebuilding NeuralNeat for a
    7,840-connection genome is about a third of the generation time.
    """

    def backpropagate(self, xs, ys, nEpochs=5):
        for k, genome in self.population.items():
            net = NeuralNeat(genome, self.config, criterion=self.criterion,
                             mask_phantom=self.mask_phantom)
            optimizer = torch.optim.Adadelta(net.parameters(), lr=1.5)
            optimizer.zero_grad()
            for _ in range(nEpochs):
                loss = self.criterion(net.forward(xs), ys)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                post = float(self.criterion(net.forward(xs), ys))
            net.update_genome_weights()
            self.population[k] = net.genome
            net.genome._post_ce = post


def make_fitness(penalty: float, check: bool = False):
    slow = make_multiclass_fitness("neg_ce")

    def base(genomes, config, xs, ys, device):
        if check:
            cached = {k: g._post_ce for k, g in genomes.items()}
            slow(genomes, config, xs, ys, device)
            diff = max(abs(-g.fitness - cached[k]) for k, g in genomes.items())
            logging.info("fitness check: max |cached CE - rebuilt CE| = %.3g", diff)
            assert diff < 1e-9, diff
        for g in genomes.values():
            ce = g.__dict__.pop("_post_ce")
            g.fitness = -ce if np.isfinite(ce) else -INVALID_CE

    if not penalty:
        return base

    def fitness(genomes, config, xs, ys, device):
        base(genomes, config, xs, ys, device)
        for g in genomes.values():
            g.train_neg_ce = g.fitness
            g.fitness = g.fitness - penalty * backbone_count(g, config)[1]

    return fitness


class Controller:
    def __init__(self, config, lo, hi, down=0.8, up=1.25, floor=1e-4):
        self.config, self.lo, self.hi, self.down, self.up, self.floor = config, lo, hi, down, up, floor

    def update(self, n_species):
        sc = self.config.species_set_config
        if n_species < self.lo:
            sc.compatibility_threshold = max(self.floor, sc.compatibility_threshold * self.down)
        elif n_species > self.hi:
            sc.compatibility_threshold *= self.up
        return sc.compatibility_threshold


# ---------------------------------------------------------------- reporter
class Reporter:
    FIELDS = ["generation", "wall_s", "gen_s", "best_fitness", "best_train_ce", "mean_fitness",
              "n_species", "species_sizes", "threshold_used", "threshold_next", "champ_key",
              "champ_hidden", "champ_conns", "champ_backbone", "champ_inputs_used", "champ_depth",
              "champ_val_acc", "champ_val_ce", "champ_test_acc", "champ_test_ce",
              "pop_hidden_mean", "pop_hidden_max", "pop_with_hidden", "best_so_far_fitness"]

    def __init__(self, run, data, last_generation, controller):
        self.run, self.data, self.last_generation, self.controller = run, data, last_generation, controller
        self.pop = self.row = self.final = self._evaluated = self.best_generation = self.t_gen = None

    def start_generation(self, generation):
        self.t_gen = time.time()

    def post_evaluate(self, config, population, species, best):
        gen = self.pop.generation
        self._evaluated = population
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        d = self.data
        va = evaluate(best, config, d["X_val"], d["y_val"], True)
        te = evaluate(best, config, d["X_test"], d["y_test"], True)
        size = genome_size(best, config)
        ins, outs = set(config.genome_config.input_keys), set(config.genome_config.output_keys)
        nh = [len([n for n in g.nodes if n not in outs]) for g in population.values()]
        self.row = dict(generation=gen, best_fitness=best.fitness,
                        best_train_ce=-getattr(best, "train_neg_ce", best.fitness),
                        mean_fitness=float(np.mean(fits)), n_species=len(species.species),
                        species_sizes=" ".join(str(len(s.members)) for s in species.species.values()),
                        threshold_used=config.species_set_config.compatibility_threshold,
                        champ_key=best.key, champ_hidden=size["hidden"], champ_conns=size["conns"],
                        champ_backbone=backbone_count(best, config)[0],
                        champ_inputs_used=size["inputs_used"], champ_depth=size["depth"],
                        champ_val_acc=va.get("acc"), champ_val_ce=va.get("ce"),
                        champ_test_acc=te.get("acc"), champ_test_ce=te.get("ce"),
                        pop_hidden_mean=float(np.mean(nh)), pop_hidden_max=int(max(nh)),
                        pop_with_hidden=int(sum(1 for x in nh if x)))
        if gen == self.last_generation:
            self.final = dict(generation=gen, population=copy.deepcopy(population),
                              species=species_summary(species, gen),
                              parents=dict(self.pop.parent_log.get(gen, {})),
                              threshold=config.species_set_config.compatibility_threshold)

    def end_generation(self, config, population, species_set):
        pop = self.pop
        if any(g is pop.best_genome for g in self._evaluated.values()):
            self.best_generation = self.row["generation"]
        nxt = (self.controller.update(len(species_set.species)) if self.controller
               else config.species_set_config.compatibility_threshold)
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
        r = self.row
        logging.info("gen %d %.1fs fit %.5f val %.4f test %.4f hidden %s conns %s species %d thr %.4g",
                     r["generation"], r["gen_s"], r["best_fitness"], r["champ_val_acc"] or float("nan"),
                     r["champ_test_acc"] or float("nan"), r["champ_hidden"], r["champ_conns"],
                     r["n_species"], r["threshold_used"])
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
        gc = pop.config.genome_config
        node_next = None
        if gc.node_indexer is not None:
            node_next = next(gc.node_indexer)
            gc.node_indexer = count(node_next)
        try:
            state = dict(generation=next_gen, population=pop.population, species=sp,
                         best_genome=pop.best_genome, genome_indexer_next=nxt,
                         species_indexer_next=sp_next, ancestors=rep.ancestors,
                         parent_log=pop.parent_log, best_generation=self.best_generation,
                         threshold=pop.config.species_set_config.compatibility_threshold,
                         node_indexer_next=node_next,
                         rng=dict(random=random.getstate(), numpy=np.random.get_state(),
                                  torch=torch.get_rng_state()),
                         wall_s=time.time() - self.run["t0"] + self.run["wall_before"])
            dump(state, self.run["dir"] / "ckpt.pkl.gz")
        finally:
            sp.reporters = reporters
        logging.info("checkpoint at generation %d", next_gen)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("seed", type=int)
    ap.add_argument("--init", choices=["full", "lr"], required=True)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--controller", default=None, help="lo,hi species band")
    ap.add_argument("--hidden-penalty", type=float, default=0.0)
    ap.add_argument("--addnode-weighted", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--check-fitness", action="store_true",
                    help="also rebuild every genome and assert the cached CE equals it")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    d = load_data()
    cfg_path = write_config(run_dir, args.overrides)
    config = load_config(cfg_path, args.addnode_weighted)
    y_train = d["y_train"].astype(np.float64)
    fitness = make_fitness(args.hidden_penalty, args.check_fitness)
    controller = None
    if args.controller:
        lo, hi = (int(x) for x in args.controller.split(","))
        controller = Controller(config, lo, hi)
    csv_path = LOG_DIR / f"{args.name}.csv"
    ckpt = run_dir / "ckpt.pkl.gz"
    coef = intercept = None
    if args.init == "lr":
        coef, intercept = fit_lr(d)

    seed_everything(args.seed)
    wall_before, st = 0.0, None
    crit = SoftmaxCrossEntropyCriterion()
    if ckpt.exists() and not args.fresh:
        with gzip.open(ckpt, "rb") as f:
            st = pickle.load(f)
        config.species_set_config.compatibility_threshold = st["threshold"]
        if st["node_indexer_next"] is not None:
            config.genome_config.node_indexer = count(st["node_indexer_next"])
        pop = CachedCEPopulation(config, d["X_train"], y_train, criterion=crit, mask_phantom=True,
                                 initial_state=(st["population"], st["species"], st["generation"]))
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
                w = csv.DictWriter(f, fieldnames=Reporter.FIELDS)
                w.writeheader()
                w.writerows(rows)
        logging.info("resumed %s at generation %d", args.name, start_gen)
    else:
        if csv_path.exists():
            csv_path.unlink()
        pop = CachedCEPopulation(config, d["X_train"], y_train, criterion=crit, mask_phantom=True)
        if args.init == "lr":
            warm_start(pop.population, config, coef, intercept)
            # re-speciate on the warm-started weights
            pop.species = config.species_set_type(config.species_set_config, pop.reporters)
            pop.species.speciate(config, pop.population, 0)
        start_gen = 0

    run = dict(csv=csv_path, dir=run_dir, ckpt_every=args.ckpt_every, t0=time.time(),
               wall_before=wall_before)
    rep = Reporter(run, d, args.generations - 1, controller)
    rep.pop = pop
    if st:
        rep.best_generation = st.get("best_generation")
    pop.add_reporter(rep)
    commit, dirty = git_state()
    logging.info("%s: init %s, %d gens x %d epochs, overrides %s, controller %s, penalty %g, "
                 "weighted add-node %s, commit %s%s", args.name, args.init, args.generations,
                 args.epochs, args.overrides, args.controller, args.hidden_penalty,
                 args.addnode_weighted, commit[:9], " (dirty)" if dirty else "")

    pop.run(fitness, n=args.generations - start_gen, nEpochs=args.epochs)
    wall = time.time() - run["t0"] + wall_before

    best = copy.deepcopy(pop.best_genome)
    m = {s: evaluate(best, config, d[f"X_{s}"], d[f"y_{s}"], True) for s in ("train", "val", "test")}
    config_text = cfg_path.read_text()
    meta = dict(series="MNIST-10 linear-start series", name=args.name, seed=args.seed,
                init=args.init, n_generations=args.generations, n_epochs_backprop=args.epochs,
                overrides=args.overrides, controller=args.controller,
                hidden_penalty=args.hidden_penalty, addnode_weighted=args.addnode_weighted,
                best_generation=rep.best_generation, n_train=int(len(y_train)),
                n_val=int(len(d["y_val"])), val_seed=VAL_SEED, fitness_function="neg_ce",
                criterion="softmax_cross_entropy", output_activation="identity",
                genome_type=config.genome_type.__name__, mask_phantom=True,
                code_commit=commit, code_dirty=dirty, data=d["meta"],
                hardware=dict(machine=platform.machine(), cpu_count=os.cpu_count(),
                              python=platform.python_version(), torch=torch.__version__),
                wall_clock_s=round(wall, 1))
    dump(dict(meta=meta, config_text=config_text, best_genome=best, final=rep.final,
              train_rows=d["train_rows"], val_rows=d["val_rows"], test_rows=d["test_rows"]),
         run_dir / "final.pkl.gz")
    dump(dict(meta=meta, config_text=config_text, genome=best), run_dir / "best_genome.pkl.gz")
    with gzip.open(run_dir / "parents.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for r in pop.ancestry_rows():
            w.writerow([r[0], r[1], "" if r[2] is None else r[2], "" if r[3] is None else r[3],
                        int(bool(r[4]))])
    summary = dict(meta, best_key=best.key, best_fitness=best.fitness,
                   best_size=genome_size(best, config),
                   best_backbone=backbone_count(best, config)[0], **m)
    (LOG_DIR / f"{args.name}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
