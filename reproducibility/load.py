"""Offline loader: rebuild a paper genome and reproduce its AUC — no database.

Usage: uv run python reproducibility/load.py <adult|heart|monk2>

Requires the `explaneat` package (the repo) importable, but NO database.
"""
import json
import os
import sys

import neat
import numpy as np
import torch
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from sklearn.metrics import roc_auc_score, accuracy_score

from explaneat.core.neuralneat import NeuralNeat

HERE = os.path.dirname(os.path.abspath(__file__))


def build_genome(gj, config):
    """Rebuild a neat genome from genome.json (mirrors deserialize_genome)."""
    g = neat.DefaultGenome(gj["key"])
    for n in gj["nodes"]:
        node = DefaultNodeGene(n["id"])
        node.bias = n["bias"]; node.response = n["response"]
        node.activation = n["activation"]; node.aggregation = n["aggregation"]
        g.nodes[n["id"]] = node
    for c in gj["connections"]:
        key = (c["from"], c["to"])
        conn = DefaultConnectionGene(key)
        conn.weight = c["weight"]; conn.enabled = c["enabled"]
        g.connections[key] = conn
    return g


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("adult", "heart", "monk2"):
        sys.exit("usage: load.py <adult|heart|monk2>")
    d = os.path.join(HERE, sys.argv[1])
    gj = json.load(open(os.path.join(d, "genome.json")))
    perf = json.load(open(os.path.join(d, "performance.json")))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(d, "neat_config.ini"))
    genome = build_genome(gj, config)
    net = NeuralNeat(genome, config)

    data = np.load(os.path.join(d, "data.npz"))
    print(f"{sys.argv[1]}: rebuilt genome ({len(genome.nodes)} nodes, "
          f"{len(genome.connections)} connections)")
    for split in ("train", "test"):
        X = torch.tensor(data[f"X_{split}"].astype(np.float64), dtype=torch.float64)
        y = data[f"y_{split}"]
        p = net.forward(X).detach().numpy().ravel()
        auc = roc_auc_score(y, p); acc = accuracy_score(y, (p > 0.5))
        exp = perf[split]
        ok = abs(auc - exp["auc"]) < 1e-3 and abs(acc - exp["accuracy"]) < 1e-3
        print(f"  {split}: AUC={auc:.4f} (reported {exp['auc']})  "
              f"acc={acc:.4f} (reported {exp['accuracy']})  "
              f"{'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
