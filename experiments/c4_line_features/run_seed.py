"""Connect-4 line-features base test: seeded in-memory PropNEAT run on a derived-input base
(exploratory, not pre-registered).

Same training path as experiments/complex_real/run_seed.py and experiments/c4_capacity/run_seed.py
(BackpropPopulation, mask_phantom=True, skip_invalid_genomes=True, Adadelta lr 1.5 full batch,
BCE, AUC fitness, no complexity penalty, sigmoid output, best genome = best *training* fitness;
the initial hidden ids are aligned across genomes as in the capacity study). What changes:

    --inputs lines|lines_cells|perline   the base (lfdata.py); num_inputs is set from it
    --config heavy|default               heavy  = the capacity study's main config (pop 150, 10 started
                                         hidden units, weight_init_stdev 0.15, 50 backprop epochs per
                                         generation, 80 generations);
                                         default = complex_real's (pop 250, no started hidden units,
                                         weight SD 1, 5 epochs, 100 generations)

The network trains on the full training split; the per-generation monitor AUC (``champ_test_auc``
in the CSV, an inherited column name) is on the training rows; test metrics are computed once, at
the end, for the best genome. Outputs: configs/<name>.ini, logs/<name>.csv, logs/<name>.json and
~/.explaneat/c4_line_features_runs/<name>/{final,best_genome}.pkl.gz, parents.csv.gz.

    PYTHONHASHSEED=0 uv run python experiments/c4_line_features/run_seed.py 0 --inputs lines --config heavy
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
sys.path.insert(0, str(HERE))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("complex_real_run_seed", CR / "run_seed.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)  # experiments/complex_real/run_seed.py (same file name as this one)
from lfdata import prepare  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

RUNS_DIR = Path(os.environ.get("EXPLANEAT_C4LF_RUNS", Path.home() / ".explaneat" / "c4_line_features_runs"))
LOG_DIR = HERE / "logs"
CONF_DIR = HERE / "configs"
BASE_INI = CR / "configs" / "c4.ini"

CONFIGS = {
    "heavy": dict(generations=80, epochs=50,
                  overrides={"NEAT.pop_size": "150", "DefaultGenome.num_hidden": "10",
                             "DefaultGenome.weight_init_stdev": "0.15"}),
    "default": dict(generations=100, epochs=5, overrides={}),
}


def build_config(name, n_inputs, overrides):
    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(BASE_INI)
    cp.set("DefaultGenome", "num_inputs", str(n_inputs))
    for key, val in overrides.items():
        sec, k = key.split(".", 1)
        assert cp.has_option(sec, k), key
        cp.set(sec, k, val)
    CONF_DIR.mkdir(exist_ok=True)
    path = CONF_DIR / f"{name}.ini"
    with open(path, "w") as f:
        cp.write(f)
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                      neat.DefaultStagnation, str(path))
    return cfg, path


def align_initial_hidden(pop, config):
    """As in the capacity study: renumber each initial genome's hidden nodes to the shared ids
    1..H so the initial population is H homologous hidden units with different random weights,
    then re-speciate. RNG-neutral."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--inputs", required=True, choices=("lines", "lines_cells", "perline"))
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=10)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    spec = CONFIGS[args.config]
    generations = args.generations or spec["generations"]
    name = f"{args.inputs}-{args.config}-s{args.seed}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("experimenter.backproppop", "experimenter.evaluators", "experimenter.methodTimer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    d = prepare(args.inputs)
    X_fit, y_fit = d["X_train"], d["y_train"].astype(np.float64)
    mon = dict(X_test=d["X_train"], y_test=d["y_train"])
    config, cfg_path = build_config(name, len(d["names"]), spec["overrides"])
    csv_path = LOG_DIR / f"{name}.csv"
    ckpt = run_dir / "ckpt.pkl.gz"

    cr.seed_everything(args.seed)
    wall_before, st = 0.0, None
    if ckpt.exists():
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

    run = dict(csv=csv_path, dir=run_dir, ckpt_every=args.ckpt_every, t0=time.time(), wall_before=wall_before)
    reporter = cr.RunReporter(run, mon, last_generation=generations - 1, controller=None)
    reporter.pop = pop
    if st:
        reporter.best_generation = st.get("best_generation")
    pop.add_reporter(reporter)
    commit, dirty = cr.git_state()
    logging.info("%s: fit %s, %d generations from %d, epochs %d, overrides %s, commit %s%s", name,
                 X_fit.shape, generations, start_gen, spec["epochs"], spec["overrides"],
                 commit[:9], " (dirty)" if dirty else "")

    pop.run(auc_fitness, n=generations - start_gen, nEpochs=spec["epochs"])
    wall = time.time() - run["t0"] + wall_before

    best = copy.deepcopy(pop.best_genome)
    final = reporter.final
    res = dict(train=cr.evaluate(best, config, d["X_train"], d["y_train"]),
               test=cr.evaluate(best, config, d["X_test"], d["y_test"]))
    config_text = cfg_path.read_text()
    rows = list(csv.DictReader(open(csv_path)))
    nsp = [int(r["n_species"]) for r in rows]
    meta = dict(task="c4", study="c4_line_features", inputs=args.inputs, config=args.config, seed=args.seed,
                name=name, variant="static", n_generations=generations, best_generation=reporter.best_generation,
                n_fit=int(len(y_fit)), val_frac=None, n_epochs_backprop=spec["epochs"], overrides=spec["overrides"],
                fitness_function="auc", criterion="bce", output_activation="sigmoid", mask_phantom=True,
                torch_threads=cr.TORCH_THREADS, neat_config=str(cfg_path.relative_to(REPO)),
                dynamic_threshold=None, code_commit=commit, code_dirty=dirty, data=d["meta"],
                hardware=dict(machine=platform.machine(), cpu_count=os.cpu_count(),
                              python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                wall_clock_s=round(wall, 1))
    cr.dump(dict(meta=meta, config_text=config_text, best_genome=best, final=final,
                 train_rows=d["train_rows"], test_rows=d["test_rows"]), run_dir / "final.pkl.gz")
    with gzip.open(run_dir / "parents.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "genome_key", "parent1_key", "parent2_key", "is_elite_copy"])
        for r in pop.ancestry_rows():
            w.writerow([r[0], r[1], "" if r[2] is None else r[2], "" if r[3] is None else r[3], int(bool(r[4]))])
    cr.dump(dict(meta=meta, config_text=config_text, genome=best), run_dir / "best_genome.pkl.gz")
    if ckpt.exists():
        ckpt.unlink()
    summary = dict(meta, best_key=best.key, best_fitness=best.fitness, best_size=cr.genome_size(best, config),
                   **res, species=dict(mean=float(np.mean(nsp)), min=min(nsp), max=max(nsp), last=nsp[-1],
                                       mean_last20=float(np.mean(nsp[-20:]))),
                   max_logged_fitness=max(float(r["best_fitness"]) for r in rows) if rows else None)
    (LOG_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
