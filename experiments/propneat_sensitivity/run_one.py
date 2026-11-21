"""One seeded, in-memory PropNEAT run of the sensitivity check (no database writes).

    PYTHONHASHSEED=0 uv run python experiments/propneat_sensitivity/run_one.py <dataset> <config_id> <fold|full> <seed>
        [--generations N] [--subdir runs]

<fold> 0-4: trained on the other four folds of the training rows (scaler fit on those rows), the
best genome by training fitness (max over all generations, ties to the earliest generation then
the lowest NEAT key, as the fixed runs' records) scored by AUC on the held-out fold. Test rows are
not loaded into the run. <full>: trained on all training rows (scaler = the split's stored scaler,
asserted), best genome scored on the test rows.

Everything else as the fixed runs: BackpropPopulation (the training loop of
DatabaseBackpropPopulation without storage), AUC fitness, mask_phantom=False, Adadelta lr 1.5,
population and topology from the config. Only epochs, the structural family and weight_decay vary.

skip_invalid_genomes=True (added 21 Nov 2026 after one CV run, both/e5_heart_wd0.01/fold 1, raised
GenomeNotValidError: a genome with no path to the output): such a genome is left untrained and
auc_fitness scores it 0.5. A run with no invalid genome is unchanged (the flag only replaces the
raise), shown by the repeat CV rerun with it (PLAN.md, amendment).
"""
import argparse
import copy
import json
import platform
import random
import subprocess
import time

import neat
import numpy as np
import torch
from neat.reporting import BaseReporter
from sklearn.metrics import roc_auc_score

from common import (DATASETS, N_GENERATIONS, REPO, config_path, load_folds, load_raw,
                    parse_config_id, require_env, run_path, standardise)

from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.errors import GenomeNotValidError  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.genome_network import NodeType  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402
from explaneat.evaluators.evaluators import auc_fitness  # noqa: E402

TORCH_THREADS = 1


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
    enabled = [c for c in ph.connections if c.enabled]
    inputs_used = {c.from_node for c in enabled} & {n.id for n in ph.nodes if n.type == NodeType.INPUT}
    return {"hidden_nodes": sum(1 for n in ph.nodes if n.type == NodeType.HIDDEN),
            "phenotype_connections_enabled": len(enabled), "inputs_connected": len(inputs_used)}


class BestByTrain(BaseReporter):
    """Best genome over the run by training fitness; ties: earliest generation, then lowest key.
    Consumes no randomness."""

    def __init__(self, pop):
        self.pop, self.best, self.best_gen, self.curve = pop, None, None, []

    def post_evaluate(self, config, population, species, best_genome):
        gen = self.pop.generation
        top = max(g.fitness for g in population.values())
        g = min((g for g in population.values() if g.fitness == top), key=lambda g: g.key)
        self.curve.append(top)
        if self.best is None or top > self.best.fitness:
            self.best, self.best_gen = copy.deepcopy(g), gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("config_id")
    ap.add_argument("fold")
    ap.add_argument("seed", type=int)
    ap.add_argument("--generations", type=int, default=N_GENERATIONS)
    ap.add_argument("--subdir", default="runs")
    args = ap.parse_args()
    require_env()
    cfg = parse_config_id(args.config_id)

    X, y, train_idx, test_idx, stored = load_raw(args.dataset)
    if args.fold == "full":
        fit_rows, eval_rows, eval_name = train_idx, test_idx, "test"
        Xs, mean, scale = standardise(X, fit_rows)
        np.testing.assert_allclose(mean, stored["mean"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(scale, stored["scale"], rtol=0, atol=1e-9)
    else:
        k = int(args.fold)
        held = load_folds()[args.dataset]["held_out"]
        eval_rows = held[k]
        fit_rows = sorted(set(train_idx) - set(eval_rows))
        fit_rows = [i for i in train_idx if i in set(fit_rows)]  # stored training order
        assert not set(eval_rows) & set(test_idx)
        eval_name = "heldout"
        Xs, mean, scale = standardise(X, fit_rows)
        del test_idx  # never used in a CV run
    Xf, yf = Xs[fit_rows], y[fit_rows]
    Xe, ye = Xs[eval_rows], y[eval_rows]

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, str(config_path(args.dataset, cfg["family"])))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)

    t0 = time.time()
    pop = BackpropPopulation(config, Xf, yf, mask_phantom=False, weight_decay=cfg["weight_decay"],
                             skip_invalid_genomes=True)
    tracker = BestByTrain(pop)
    pop.reporters.add(tracker)
    pop.run(auc_fitness, n=args.generations, nEpochs=cfg["epochs"])
    wall = time.time() - t0
    best = tracker.best
    train_auc = auc(best, config, Xf, yf)
    assert abs(train_auc - best.fitness) < 1e-9, (train_auc, best.fitness)

    out = {
        "dataset": args.dataset, "config": cfg, "fold": args.fold, "seed": args.seed,
        "generations": args.generations, "n_fit": len(fit_rows), "n_eval": len(eval_rows),
        "eval_split": eval_name,
        "best": {"generation": tracker.best_gen, "key": best.key, "train_auc": train_auc,
                 f"{eval_name}_auc": auc(best, config, Xe, ye), **size(best, config)},
        "best_train_fitness_per_generation": tracker.curve,
    }
    meta = {"wall_clock_s": round(wall, 1),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "explaneat"],
                                                       cwd=REPO, text=True).strip()),
            "machine": platform.machine(), "torch": torch.__version__, "torch_threads": TORCH_THREADS}
    path = run_path(args.dataset, args.config_id, args.fold, args.seed, args.subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Results and run metadata in separate files, so two runs can be compared byte for byte.
    path.write_text(json.dumps(out, indent=1) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({**{k: v for k, v in out.items() if k != "best_train_fitness_per_generation"}, **meta}),
          flush=True)


if __name__ == "__main__":
    main()
