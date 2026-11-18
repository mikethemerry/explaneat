"""Pick the headline network from the scarcity (prune) phase and evaluate it once on test.

  uv run python experiments/conv_neat/select_pruned.py [--run results/mnist10_pruned]

The prune run's own finalize tests its monitored best-by-validation genome, which can come from
the phase's first generation (offspring of the unpruned run, before any deletion). The headline is
instead the best-by-validation genome of the phase's **last evaluated population** (every member has
been through the deletion-only phase; ties: fewer genes). Test is read once for it, then it is
fine-tuned (20 epochs on the run's training rows, best-validation epoch). Writes
``headline_genome.json`` and ``headline_results.json`` in the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from explaneat.conv.data import mnist_splits  # noqa: E402
from explaneat.conv.network import accuracy, build_engine, pick_device, predict  # noqa: E402
from explaneat.conv.population import genome_to_dict  # noqa: E402

sys.path.insert(0, HERE)
from run_long import finetune  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(HERE, "results", "mnist10_pruned"))
    p.add_argument("--finetune", type=int, default=20)
    a = p.parse_args()
    conf = json.load(open(os.path.join(a.run, "config.json")))
    st = pickle.load(open(os.path.join(a.run, "checkpoint.pkl"), "rb"))
    pop = [g for g in st["population"] if g.metrics.get("val_acc") is not None]
    assert pop, "the checkpoint's population has not been evaluated (use the finished checkpoint)"
    g = sorted(pop, key=lambda g: (-g.metrics["val_acc"], g.complexity(), g.key))[0]
    cfg = st["cfg"]
    args = conf["args"]
    data = mnist_splits(args["seed"], classes=tuple(range(10)), n_train=args["n_train"], n_val=args["n_val"])
    dev = pick_device("auto")
    test = accuracy(predict(build_engine(g, cfg), torch.tensor(data["X_test"])), data["y_test"])
    sizes = [x.complexity() for x in st["population"]]
    res = {"rule": "best validation accuracy in the prune phase's last evaluated population (ties: fewer genes)",
           "key": g.key, "generation": g.metrics["generation"], "val_acc": g.metrics["val_acc"],
           "train_acc": g.metrics.get("train_acc"), "test_acc": test, "complexity": g.complexity(),
           "n_params": g.n_params(), "n_hidden": len(g.hidden_keys()), "n_enabled_edges": len(g.enabled_conns()),
           "population_mean_complexity": float(np.mean(sizes)), "population_min_complexity": int(min(sizes))}
    print(json.dumps(res), flush=True)
    res["finetuned"] = finetune(g, cfg, data, a.finetune, args["seed"], dev, cfg.batch_size)
    with open(os.path.join(a.run, "headline_genome.json"), "w") as f:
        json.dump(genome_to_dict(g), f, indent=1, default=float)
    with open(os.path.join(a.run, "headline_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
