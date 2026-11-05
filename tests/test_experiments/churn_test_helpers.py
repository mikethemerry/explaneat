"""Synthetic NEAT genome builder for the churn tooling tests."""

import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene


def build_genome(key, nodes, conns):
    """nodes: {id: bias}; conns: {(a, b): weight} (all enabled)."""
    g = neat.DefaultGenome(key)
    for nid, bias in nodes.items():
        n = DefaultNodeGene(nid)
        n.bias, n.response, n.activation, n.aggregation = float(bias), 1.0, "relu", "sum"
        g.nodes[nid] = n
    for k, w in conns.items():
        c = DefaultConnectionGene(k)
        c.weight, c.enabled = float(w), True
        g.connections[k] = c
    return g
