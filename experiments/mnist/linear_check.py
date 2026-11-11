"""Sanity check: a fully connected 784 -> 10 genome (no hidden nodes) trained with
PropNEAT's own backprop path should match multinomial logistic regression.

Arms (same train/test rows as mnist10, same initial weights from the genome):

  propneat_gen    BackpropPopulation.backpropagate on a one-genome population,
                  5 epochs per call (one call = one PropNEAT generation): a fresh
                  Adadelta(lr=1.5) every 5 epochs, softmax CE, identity outputs,
                  mask_phantom=True.  The exact training step of the MNIST runs.
  propneat_cont   the same NeuralNeat net and loss, one Adadelta(lr=1.5) kept
                  for all epochs (optimiser state not reset).
  torch_gen       nn.Linear(784, 10) + F.cross_entropy, initialised to the
                  genome's weights, fresh Adadelta(lr=1.5) every 5 epochs.
  torch_cont      the same, one Adadelta for all epochs.
  torch_adam      the same, Adam(lr=0.01) for all epochs (a convergence reference).

    PYTHONHASHSEED=0 uv run python experiments/mnist/linear_check.py [--epochs 2000]

Writes experiments/mnist/logs/linear_check.json (per-epoch train CE + periodic
accuracies).  No DB access.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from data import prepare  # noqa: E402
from run_seed import load_config  # noqa: E402
from explaneat.core.backproppop import BackpropPopulation  # noqa: E402
from explaneat.core.multiclass import SoftmaxCrossEntropyCriterion  # noqa: E402
from explaneat.core.neuralneat import NeuralNeat  # noqa: E402

EPOCHS_PER_GEN = 5


def full_direct_genome(config, key=1):
    """A 784 -> 10 genome with every input wired to every output, no hidden nodes."""
    gc = config.genome_config
    gc.initial_connection = "full_direct"
    gc.connection_fraction = None
    g = config.genome_type(key)
    g.configure_new(gc)
    n_in, n_out = len(gc.input_keys), len(gc.output_keys)
    assert len(g.connections) == n_in * n_out and set(g.nodes) == set(gc.output_keys)
    return g


def genome_matrix(genome, config):
    """(W [784, 10], b [10]) in input_keys x output_keys order."""
    gc = config.genome_config
    W = np.array([[genome.connections[(i, o)].weight for o in gc.output_keys] for i in gc.input_keys])
    b = np.array([genome.nodes[o].bias for o in gc.output_keys])
    return W, b


def metrics(logits, y):
    return dict(ce=float(F.cross_entropy(logits, y)), acc=float((logits.argmax(1) == y).double().mean()))


def run_torch(W0, b0, Xtr, ytr, Xte, yte, epochs, mode, every):
    W = torch.nn.Parameter(torch.tensor(W0.copy()))
    b = torch.nn.Parameter(torch.tensor(b0.copy()))
    make = (lambda: torch.optim.Adam([W, b], lr=0.01)) if mode == "adam" else \
        (lambda: torch.optim.Adadelta([W, b], lr=1.5))
    opt = make()
    hist, snaps = [], []
    for ep in range(epochs):
        if mode == "gen" and ep % EPOCHS_PER_GEN == 0:
            opt = make()
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        hist.append(float(loss))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (ep + 1) % every == 0 or ep + 1 == epochs:
            with torch.no_grad():
                snaps.append(dict(epoch=ep + 1, train=metrics(Xtr @ W + b, ytr),
                                  test=metrics(Xte @ W + b, yte)))
    return hist, snaps


def run_propneat_gen(genome, config, Xtr, ytr_f, ytr, Xte, yte, epochs, every):
    """The MNIST runs' training step: BackpropPopulation.backpropagate, 5 epochs per call."""
    config.pop_size = 2  # the population built at init is replaced below
    pop = BackpropPopulation(config, Xtr, ytr_f, criterion=SoftmaxCrossEntropyCriterion(),
                             mask_phantom=True)
    pop.population = {genome.key: genome}
    hist, snaps = [], []
    crit = SoftmaxCrossEntropyCriterion()
    for gen in range(epochs // EPOCHS_PER_GEN):
        pop.backpropagate(pop.xs, pop.ys, nEpochs=EPOCHS_PER_GEN)
        g = pop.population[genome.key]
        net = NeuralNeat(g, config, mask_phantom=True)
        with torch.no_grad():
            hist.append(float(crit(net.forward(pop.xs), pop.ys)))
            ep = (gen + 1) * EPOCHS_PER_GEN
            if ep % every == 0 or ep == epochs:
                snaps.append(dict(epoch=ep, train=metrics(net.forward(pop.xs), ytr),
                                  test=metrics(net.forward(Xte), yte)))
    return hist, snaps  # hist: train CE after each generation (every 5 epochs)


def run_propneat_cont(genome, config, Xtr, ytr_f, ytr, Xte, yte, epochs, every):
    crit = SoftmaxCrossEntropyCriterion()
    net = NeuralNeat(genome, config, criterion=crit, mask_phantom=True)
    opt = torch.optim.Adadelta(net.parameters(), lr=1.5)
    hist, snaps = [], []
    for ep in range(epochs):
        loss = crit(net.forward(Xtr), ytr_f)
        hist.append(float(loss))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (ep + 1) % every == 0 or ep + 1 == epochs:
            with torch.no_grad():
                snaps.append(dict(epoch=ep + 1, train=metrics(net.forward(Xtr), ytr),
                                  test=metrics(net.forward(Xte), yte)))
    return hist, snaps


def run_evolved(names, config, data, epochs, every):
    """Keep training each saved best mnist10 genome's own weights (its structure fixed).

    Two optimisers: the PropNEAT step (fresh Adadelta lr 1.5 every 5 epochs) and
    Adam lr 0.01 throughout.  Shows how far the evolved *structure* can go."""
    import copy
    import gzip
    import pickle

    runs = Path(os.environ.get("EXPLANEAT_MNIST_RUNS", Path.home() / ".explaneat" / "mnist_runs"))
    crit = SoftmaxCrossEntropyCriterion()
    Xtr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    Xte = torch.as_tensor(data["X_test"], dtype=torch.float64)
    ytr = torch.as_tensor(data["y_train"], dtype=torch.long)
    yte = torch.as_tensor(data["y_test"], dtype=torch.long)
    yf = ytr.double().unsqueeze(1)
    out = {}
    for name in names:
        with gzip.open(runs / name / "best_genome.pkl.gz", "rb") as f:
            g0 = pickle.load(f)["genome"]
        res = {}
        for mode in ("gen", "adam"):
            net = NeuralNeat(copy.deepcopy(g0), config, criterion=crit, mask_phantom=True)
            with torch.no_grad():
                start = dict(train=metrics(net.forward(Xtr), ytr), test=metrics(net.forward(Xte), yte))
            make = (lambda: torch.optim.Adam(net.parameters(), lr=0.01)) if mode == "adam" else \
                (lambda: torch.optim.Adadelta(net.parameters(), lr=1.5))
            opt = make()
            snaps = []
            for ep in range(epochs):
                if mode == "gen" and ep % EPOCHS_PER_GEN == 0:
                    opt = make()
                loss = crit(net.forward(Xtr), yf)
                loss.backward()
                opt.step()
                opt.zero_grad()
                if (ep + 1) % every == 0 or ep + 1 == epochs:
                    with torch.no_grad():
                        snaps.append(dict(epoch=ep + 1, train=metrics(net.forward(Xtr), ytr),
                                          test=metrics(net.forward(Xte), yte)))
            res[mode] = dict(start=start, snaps=snaps)
            print(f"{name:16s} {mode:5s} start test acc {start['test']['acc']:.4f} -> after {epochs} "
                  f"more epochs train acc {snaps[-1]['train']['acc']:.4f} ce {snaps[-1]['train']['ce']:.4f} "
                  f"test acc {snaps[-1]['test']['acc']:.4f}", flush=True)
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="propneat_gen,propneat_cont,torch_gen,torch_cont,torch_adam")
    ap.add_argument("--evolved", default="", help="comma list of saved mnist10 runs to continue training")
    ap.add_argument("--out", default=str(HERE / "logs" / "linear_check.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data = prepare("mnist10")
    config = load_config("mnist10")
    if args.evolved:
        res = run_evolved(args.evolved.split(","), config, data, args.epochs, args.every)
        Path(args.out).write_text(json.dumps(dict(epochs=args.epochs, evolved=res)) + "\n")
        return
    genome = full_direct_genome(config)
    W0, b0 = genome_matrix(genome, config)
    Xtr = torch.as_tensor(data["X_train"], dtype=torch.float64)
    Xte = torch.as_tensor(data["X_test"], dtype=torch.float64)
    ytr = torch.as_tensor(data["y_train"], dtype=torch.long)
    yte = torch.as_tensor(data["y_test"], dtype=torch.long)
    ytr_f = data["y_train"].astype(np.float64)

    # Same function at init: NeuralNeat == X @ W + b with outputs in output_keys order.
    with torch.no_grad():
        nn_logits = NeuralNeat(genome, config, mask_phantom=True).forward(Xtr)
        lin_logits = Xtr @ torch.tensor(W0) + torch.tensor(b0)
    init_maxdiff = float((nn_logits - lin_logits).abs().max())
    print(f"init: NeuralNeat vs X@W+b max |diff| = {init_maxdiff:.2e}; "
          f"train CE {metrics(lin_logits, ytr)}", flush=True)

    out = dict(epochs=args.epochs, epochs_per_generation=EPOCHS_PER_GEN, seed=args.seed,
               init_maxdiff=init_maxdiff, init_train=metrics(lin_logits, ytr),
               baseline_lr=json.loads((HERE / "baselines.json").read_text())["mnist10"]["logistic_regression"],
               arms={})
    import copy
    for arm in args.arms.split(","):
        t0 = time.time()
        g = copy.deepcopy(genome)
        if arm == "propneat_gen":
            hist, snaps = run_propneat_gen(g, config, data["X_train"], ytr_f, ytr, Xte, yte, args.epochs, args.every)
        elif arm == "propneat_cont":
            hist, snaps = run_propneat_cont(g, config, Xtr, torch.as_tensor(ytr_f).unsqueeze(1), ytr, Xte, yte,
                                            args.epochs, args.every)
        else:
            hist, snaps = run_torch(W0, b0, Xtr, ytr, Xte, yte, args.epochs, arm.split("_")[1], args.every)
        out["arms"][arm] = dict(loss=hist, snaps=snaps, seconds=round(time.time() - t0, 1))
        last = snaps[-1]
        print(f"{arm:14s} {time.time() - t0:6.1f}s  epoch {last['epoch']}  "
              f"train acc {last['train']['acc']:.4f} ce {last['train']['ce']:.4f}  "
              f"test acc {last['test']['acc']:.4f} ce {last['test']['ce']:.4f}", flush=True)
        for s in snaps:
            if s["epoch"] in (5, 25, 100, 500, 1000, 2500, 5000):
                print(f"    epoch {s['epoch']:5d}  train acc {s['train']['acc']:.4f}  test acc {s['test']['acc']:.4f}"
                      f"  train ce {s['train']['ce']:.4f}", flush=True)
        Path(args.out).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
