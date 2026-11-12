"""Connect-4 capacity study: seeded in-memory PropNEAT run with config overrides
(exploratory, not pre-registered).

Same training path as experiments/complex_real/run_seed.py (BackpropPopulation,
mask_phantom=True, skip_invalid_genomes=True, Adadelta lr 1.5 full batch, BCE, AUC fitness, no
complexity penalty, sigmoid output, best genome = best *training* fitness), reusing its
reporter, threshold controller and checkpointing. What changes per run:

    --set Section.key=value   NEAT config overrides on configs/c4_base.ini (= complex_real c4.ini)
                              e.g. --set DefaultGenome.num_hidden=10 --set NEAT.pop_size=400
    --epochs N                backprop epochs per generation (complex_real: 5)
    --tuned                   species-band threshold controller (--species-min/max, --threshold-step)
    --val-frac F              pilot mode: train on (1-F) of the training rows, and the per-generation
                              monitor AUC is on the held-out F of the *training* split. The test
                              split is not read at all in pilot mode.

In main runs (no --val-frac) the network trains on the full training split and the monitor AUC is
on the training rows; test metrics are computed once, at the end, for the best genome.

The derived config is written to configs/<name>.ini. Outputs: logs/<name>.csv (per generation;
``champ_test_auc`` there is the monitor AUC described above), logs/<name>.json, and
~/.explaneat/c4_capacity_runs/<name>/{final,best_genome,ckpt}.pkl.gz, parents.csv.gz.

    PYTHONHASHSEED=0 uv run python experiments/c4_capacity/run_seed.py 0 --name p1-h10 \
        --generations 40 --val-frac 0.1 --set NEAT.pop_size=150 --set DefaultGenome.num_hidden=10
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
CR = REPO / "experiments" / "complex_real"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CR))

import run_seed as cr  # noqa: E402  (experiments/complex_real/run_seed.py)
from data import prepare  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

RUNS_DIR = Path(os.environ.get("EXPLANEAT_C4CAP_RUNS", Path.home() / ".explaneat" / "c4_capacity_runs"))
LOG_DIR = HERE / "logs"
CONF_DIR = HERE / "configs"
BASE_INI = CONF_DIR / "c4_base.ini"


def build_config(name, overrides):
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(BASE_INI)
    for o in overrides:
        key, val = o.split("=", 1)
        sec, k = key.split(".", 1)
        if not cp.has_option(sec, k):
            sys.exit(f"unknown option {key}")
        cp.set(sec, k, val)
    path = CONF_DIR / f"{name}.ini"
    with open(path, "w") as f:
        cp.write(f)
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, str(path))
    return cfg, path


def align_initial_hidden(pop, config):
    """neat-python gives every new genome's initial hidden nodes fresh global ids, so with
    ``num_hidden > 0`` no two initial genomes share a hidden gene: every connection through a
    hidden node is disjoint, crossover is non-homologous and each genome starts as its own
    species. Renumber each initial genome's hidden nodes (in creation order) to the shared ids
    1..H, so the initial population is H homologous hidden units with different random weights,
    then re-speciate. RNG-neutral (no random draws)."""
    H = config.genome_config.num_hidden
    outs = set(config.genome_config.output_keys)
    for g in pop.population.values():
        hid = sorted(k for k in g.nodes if k not in outs)
        m = {old: i + 1 for i, old in enumerate(hid)}
        rn = lambda k: m.get(k, k)  # noqa: E731
        nodes = {}
        for k, n in g.nodes.items():
            n.key = rn(k)
            nodes[n.key] = n
        g.nodes = nodes
        conns = {}
        for (a, b), c in g.connections.items():
            c.key = (rn(a), rn(b))
            conns[c.key] = c
        g.connections = conns
    config.genome_config.node_indexer = count(max(outs | {H}) + 1)
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)


def lr_warm_start(pop, config, X_fit, y_fit):
    """Warm start from the linear model: set every initial genome's direct input->output weights
    and output bias to a logistic regression fitted on the same (z-scored) fit rows. Requires
    initial_connection = full_direct. Hidden units (if any) keep their random init. The LR fit
    is deterministic, so this is RNG-neutral."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(X_fit, y_fit)
    out = config.genome_config.output_keys[0]
    for g in pop.population.values():
        for j, k in enumerate(config.genome_config.input_keys):
            g.connections[(k, out)].weight = float(lr.coef_[0, j])
        g.nodes[out].bias = float(lr.intercept_[0])
    g0 = next(iter(pop.population.values()))
    logging.info("LR warm start: genome %s fit-row AUC before backprop %.5f (LR %.5f)", g0.key,
                 cr.evaluate(g0, config, X_fit, y_fit)["auc"],
                 cr.roc_auc_score(y_fit, lr.decision_function(X_fit)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--name", required=True)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--species-min", type=int, default=5)
    ap.add_argument("--species-max", type=int, default=10)
    ap.add_argument("--threshold-step", type=float, default=0.1)
    ap.add_argument("--threshold-min", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--lr-warm", action="store_true",
                    help="warm-start the direct input->output weights from a fitted logistic regression")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    name = args.name
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    d = prepare("c4", None)
    if args.val_frac:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(d["y_train"]))
        fit_i, val_i = train_test_split(idx, test_size=args.val_frac, stratify=d["y_train"], random_state=777)
        X_fit, y_fit = d["X_train"][fit_i], d["y_train"][fit_i]
        mon = dict(X_test=d["X_train"][val_i], y_test=d["y_train"][val_i])
    else:
        X_fit, y_fit = d["X_train"], d["y_train"]
        mon = dict(X_test=d["X_train"], y_test=d["y_train"])
    config, cfg_path = build_config(name, args.set)
    y_fit = y_fit.astype(np.float64)
    csv_path = LOG_DIR / f"{name}.csv"
    ckpt = run_dir / "ckpt.pkl.gz"
    controller = (cr.ThresholdController(config, args.species_min, args.species_max,
                                         args.threshold_step, args.threshold_min) if args.tuned else None)

    cr.seed_everything(args.seed)
    wall_before, st = 0.0, None
    if ckpt.exists() and not args.fresh:
        with gzip.open(ckpt, "rb") as f:
            st = pickle.load(f)
        config.species_set_config.compatibility_threshold = st["threshold"]
        pop = BackpropPopulation(config, X_fit, y_fit, initial_state=(st["population"], st["species"], st["generation"]),
                                 mask_phantom=True, skip_invalid_genomes=True)
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
                w = csv.DictWriter(f, fieldnames=cr.RunReporter.FIELDS)
                w.writeheader()
                w.writerows(rows)
        logging.info("resumed %s at generation %d", name, start_gen)
    else:
        if csv_path.exists():
            csv_path.unlink()
        pop = BackpropPopulation(config, X_fit, y_fit, mask_phantom=True, skip_invalid_genomes=True)
        start_gen = 0
        if config.genome_config.num_hidden > 0:
            align_initial_hidden(pop, config)
        if args.lr_warm:
            lr_warm_start(pop, config, X_fit, y_fit)

    run = dict(csv=csv_path, dir=run_dir, ckpt_every=args.ckpt_every, t0=time.time(), wall_before=wall_before)
    reporter = cr.RunReporter(run, mon, last_generation=args.generations - 1, controller=controller)
    reporter.pop = pop
    if st:
        reporter.best_generation = st.get("best_generation")
    pop.add_reporter(reporter)
    commit, dirty = cr.git_state()
    logging.info("%s: fit %s monitor %s, %d generations from %d, epochs %d, overrides %s, commit %s%s", name,
                 X_fit.shape, mon["X_test"].shape, args.generations, start_gen, args.epochs, args.set,
                 commit[:9], " (dirty)" if dirty else "")

    pop.run(auc_fitness, n=args.generations - start_gen, nEpochs=args.epochs)
    wall = time.time() - run["t0"] + wall_before

    best = copy.deepcopy(pop.best_genome)
    final = reporter.final
    res = dict(train=cr.evaluate(best, config, d["X_train"], d["y_train"]))
    if args.val_frac:
        res["fit_rows"] = cr.evaluate(best, config, X_fit, y_fit)
        res["val"] = cr.evaluate(best, config, mon["X_test"], mon["y_test"])
    else:
        res["test"] = cr.evaluate(best, config, d["X_test"], d["y_test"])
    config_text = cfg_path.read_text()
    rows = list(csv.DictReader(open(csv_path)))
    nsp = [int(r["n_species"]) for r in rows]
    meta = dict(task="c4", study="c4_capacity", seed=args.seed, name=name, variant="tuned" if args.tuned else "static",
                n_generations=args.generations, best_generation=reporter.best_generation,
                n_fit=int(len(y_fit)), val_frac=args.val_frac, lr_warm=args.lr_warm, n_epochs_backprop=args.epochs, overrides=args.set,
                fitness_function="auc", criterion="bce", output_activation="sigmoid", mask_phantom=True,
                torch_threads=cr.TORCH_THREADS, neat_config=str(cfg_path.relative_to(REPO)),
                dynamic_threshold=(dict(species_min=args.species_min, species_max=args.species_max,
                                        step=args.threshold_step, floor=args.threshold_min) if controller else None),
                code_commit=commit, code_dirty=dirty, data=d["meta"],
                hardware=dict(machine=platform.machine(), cpu_count=os.cpu_count(),
                              python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                wall_clock_s=round(wall, 1))
    if not args.val_frac:
        cr.dump(dict(meta=meta, config_text=config_text, best_genome=best, final=final,
                     train_rows=d["train_rows"], test_rows=d["test_rows"]), run_dir / "final.pkl.gz")
        with gzip.open(run_dir / "parents.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
            for r in pop.ancestry_rows():
                w.writerow([r[0], r[1], "" if r[2] is None else r[2], "" if r[3] is None else r[3], int(bool(r[4]))])
    cr.dump(dict(meta=meta, config_text=config_text, genome=best), run_dir / "best_genome.pkl.gz")
    if ckpt.exists():
        ckpt.unlink()  # keep disk small: the run is finished
    summary = dict(meta, best_key=best.key, best_fitness=best.fitness, best_size=cr.genome_size(best, config),
                   **res, species=dict(mean=float(np.mean(nsp)), min=min(nsp), max=max(nsp), last=nsp[-1],
                                       mean_last20=float(np.mean(nsp[-20:]))),
                   final_threshold=final["threshold"] if final else None)
    (LOG_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
