"""One seeded PropNEAT run on Heart with a held-out validation set (in memory).

Heart-100 settings (configs/heart.ini, 100 generations, 5 backprop epochs,
AUC fitness) except: trained on the 193 fit rows only, mask_phantom=True,
Adadelta weight_decay=<wd>, best genome chosen by validation AUC across all
generations (BackpropPopulation's validation path). No DB writes.

Per generation it also logs the best-by-train and best-by-validation genomes'
train/val/test AUC (logging only; nothing is selected on test).

    PYTHONHASHSEED=0 uv run python experiments/heart_regularise/run_arm.py 1e-3 0
"""

import argparse
import json
import os
import random
import sys
import time

import neat
import numpy as np
import torch
from neat.reporting import BaseReporter
from sklearn.metrics import roc_auc_score

from common import HERE, NEAT_CONFIG, load_splits

from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

N_GENERATIONS = 100
N_EPOCHS_BACKPROP = 5


def auc(genome, config, X, y):
    try:
        net = NeuralNeat(genome, config)
    except GenomeNotValidError:
        return 0.5
    with torch.no_grad():
        p = net.forward(torch.as_tensor(X, dtype=torch.float64)).cpu().numpy().ravel()
    return float(roc_auc_score(y, p))


def size(genome, config):
    ph = ExplaNEAT(genome, config).get_phenotype_network()
    return {
        "hidden_nodes": sum(1 for n in ph.nodes if n.type == NodeType.HIDDEN),
        "phenotype_connections_enabled": sum(1 for c in ph.connections if c.enabled),
    }


class GenLog(BaseReporter):
    """Log per generation: best-by-train and best-by-val genomes on all three splits."""

    def __init__(self, pop, data):
        self.pop, self.data, self.rows = pop, data, []

    def post_evaluate(self, config, population, species, best_genome):
        (_, _), (Xv, yv), (Xt, yt) = self.data
        bt = max(population.values(), key=lambda g: g.fitness)
        bv = max(population.values(),
                 key=lambda g: getattr(g, "validation_fitness", -float("inf")))
        self.rows.append({
            "generation": self.pop.generation,
            "bytrain_key": bt.key, "bytrain_train": bt.fitness,
            "bytrain_val": getattr(bt, "validation_fitness", None),
            "bytrain_test": auc(bt, config, Xt, yt),
            "byval_key": bv.key, "byval_train": bv.fitness,
            "byval_val": getattr(bv, "validation_fitness", None),
            "byval_test": auc(bv, config, Xt, yt),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weight_decay", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=N_GENERATIONS)
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.exit("Set PYTHONHASHSEED=0 for reproducibility")

    data = load_splits()
    (Xf, yf), (Xv, yv), (Xt, yt) = data
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, str(NEAT_CONFIG))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    t0 = time.time()
    pop = BackpropPopulation(config, Xf, yf, xs_val=Xv, ys_val=yv,
                             mask_phantom=True, weight_decay=args.weight_decay)
    log = GenLog(pop, data)
    pop.reporters.add(log)
    best = pop.run(auc_fitness, n=args.generations, nEpochs=N_EPOCHS_BACKPROP)

    sel = {
        "generation": pop.best_generation, "key": best.key,
        "train_auc": auc(best, config, Xf, yf),
        "val_auc": auc(best, config, Xv, yv),
        "test_auc": auc(best, config, Xt, yt),
        "stored_train_fitness": best.fitness,
        "stored_val_fitness": best.validation_fitness,
        **size(best, config),
    }
    # Best-by-train over the whole run (the Heart-100 rule), for comparison only.
    bt = max(log.rows, key=lambda r: r["bytrain_train"])  # first max = earliest
    out = {
        "weight_decay": args.weight_decay, "seed": args.seed,
        "generations": args.generations, "n_fit": len(yf), "n_val": len(yv),
        "n_test": len(yt), "wall_clock_s": round(time.time() - t0, 1),
        "selected_by_val": sel,
        "bytrain_rule": {"generation": bt["generation"], "train_auc": bt["bytrain_train"],
                         "val_auc": bt["bytrain_val"], "test_auc": bt["bytrain_test"]},
        "per_generation": log.rows,
    }
    rdir = HERE / "runs"
    rdir.mkdir(exist_ok=True)
    path = rdir / f"wd{args.weight_decay:g}_s{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_generation"}), flush=True)


if __name__ == "__main__":
    main()
