"""Module operators (PLAN.md "Operators").

Weight modes (one per arm):
- ``explained`` (arm 2, modules): grafted genes carry the library variant's
  weights and biases; then readout refit.
- ``sham`` (arm 3): same topology, every grafted weight and grafted-node bias
  redrawn from N(0, 1); then readout refit.
- ``topology`` (arm 4): same topology, internal weights and biases N(0, 1),
  readout weights 0 (the child computes exactly its pre-graft function);
  no refit; only the normal per-generation backprop trains the new genes.

Readout refit = genome_surgery.refit_readout: logistic regression of the
grafted readout weights plus the output bias on the current train rows,
every other weight fixed (L2 1e-6 on readout weights only).

Grafted nodes get fresh ids from neat's shared node indexer
(config.genome_config.get_new_node_key), so ids stay unique across all
islands of a run and later add-node mutations cannot collide with them.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from neat.genes import DefaultConnectionGene, DefaultNodeGene

from chunks import Chunk, gs

WEIGHT_MODES = ("explained", "sham", "topology")


class OperatorNotApplicable(Exception):
    pass


def valid_feedforward(genome, config) -> bool:
    """Enabled graph acyclic and NeuralNeat builds a net that reaches the output."""
    if not gs.is_acyclic(genome):
        return False
    try:
        net = gs.build_net(genome, config)
    except Exception:  # GenomeNotValidError, ValueError
        return False
    outs = set(config.genome_config.output_keys)
    return any(L["layer_type"] == "OUTPUT" and outs & set(L["nodes"]) for L in net.layers.values())


def _new_node(nid, attrs):
    ng = DefaultNodeGene(nid)
    ng.bias, ng.response, ng.activation, ng.aggregation = attrs
    return ng


def _new_conn(key, w):
    cg = DefaultConnectionGene(key)
    cg.weight = float(w)
    cg.enabled = True
    return cg


def graft_chunk(recipient, config, chunk: Chunk, module_id: int, weight_mode: str,
                rng: Optional[np.random.Generator] = None):
    """Copy ``chunk`` into a deep copy of ``recipient`` with fresh hidden ids.

    Returns (new genome, new readout keys, id_map)."""
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(weight_mode)
    if weight_mode != "explained" and rng is None:
        raise ValueError("sham/topology grafts need an rng")
    new = copy.deepcopy(recipient)
    gc = config.genome_config
    inputs = set(gc.input_keys)
    outputs = set(gc.output_keys)
    id_map = {}
    for d in sorted(chunk.hidden_nodes):
        nid = gc.get_new_node_key(new.nodes)
        id_map[d] = nid
        attrs = list(chunk.node_attrs[d])
        if weight_mode != "explained":
            attrs[0] = float(rng.normal(0.0, 1.0))
        new.nodes[nid] = _new_node(nid, tuple(attrs))

    def m(n):
        return n if (n in inputs or n in outputs) else id_map[n]

    for a, b, w in sorted(chunk.internal):
        key = (m(a), m(b))
        if key in new.connections:  # cannot happen: b is fresh
            raise ValueError(f"graft collision {key}")
        if weight_mode != "explained":
            w = float(rng.normal(0.0, 1.0))
        new.connections[key] = _new_conn(key, w)
    readout = []
    for a, b, w in sorted(chunk.readout):
        key = (m(a), b)
        if weight_mode == "sham":
            w = float(rng.normal(0.0, 1.0))
        elif weight_mode == "topology":
            w = 0.0
        new.connections[key] = _new_conn(key, w)
        readout.append(key)
    tags = dict(getattr(new, "module_tags", {}))
    for nid in id_map.values():
        tags[nid] = module_id
    new.module_tags = tags
    return new, readout, id_map


def refit(genome, config, X, y, readout_keys) -> Tuple[object, dict]:
    # NeuralNeat's layer parsing does not terminate on a cyclic genome (a
    # module crossover can create a cycle through edges the child already
    # had): reject before building it; the operator result is then treated
    # as invalid (PLAN.md deviation 1, RESULTS.md).
    if not gs.is_acyclic(genome):
        raise ValueError("operator result has a cycle")
    keys =[k for k in readout_keys if k in genome.connections and genome.connections[k].enabled]
    # only readouts NeuralNeat computes (a grafted exit can be dead when an
    # inherited open-chunk node lost its non-module inputs)
    t = gs.output_preactivation_terms(genome, config, X)
    imap = t["net"].layers[t["layer"]]["input_map"]
    keys = [k for k in keys if k in imap]
    if not keys:
        return genome, {"skipped": "no live readout"}
    g, info = gs.refit_readout(genome, config, X, y, keys)
    return g, {"train_logloss_before": info["train_logloss_before"],
               "train_logloss_after": info["train_logloss_after"],
               "converged": info["converged"],
               "readout_weights_after": info["readout_weights_after"],
               "output_bias_after": info["output_bias_after"]}


def knockout_module(genome, module_id: int):
    """Deep copy with every node tagged ``module_id`` and every connection
    touching one deleted. Returns (genome, deleted node ids)."""
    g = copy.deepcopy(genome)
    nodes = set(g.module_nodes(module_id))
    for k in [k for k in g.connections if k[0] in nodes or k[1] in nodes]:
        del g.connections[k]
    for n in nodes:
        del g.nodes[n]
    g.module_tags = {n: m for n, m in g.module_tags.items() if n not in nodes}
    return g, sorted(nodes)


def module_content(genome) -> Dict[int, frozenset]:
    """module id -> frozenset of its tagged node ids (the variant signature)."""
    out: Dict[int, set] = {}
    for n, m in genome.tags().items():
        out.setdefault(m, set()).add(n)
    return {m: frozenset(v) for m, v in out.items()}


def differing_modules(pa, pb) -> List[int]:
    """Modules carried by ``pb`` with node ids that ``pa`` does not have exactly."""
    ca, cb = module_content(pa), module_content(pb)
    return sorted(m for m, nodes in cb.items() if ca.get(m) != nodes)


def module_as_chunk(genome, config, module_id: int) -> Chunk:
    """The genes of ``genome`` tagged ``module_id`` as a Chunk (open chunk:
    whatever the tagged nodes are now): nodes, connections into them from
    inputs or other tagged nodes, and their enabled edges to the output."""
    gc = config.genome_config
    inputs, outputs = set(gc.input_keys), set(gc.output_keys)
    nodes = set(genome.module_nodes(module_id))
    internal, readout = [], []
    for (a, b), c in sorted(genome.connections.items()):
        if not c.enabled:
            continue
        if b in nodes and (a in nodes or a in inputs):
            internal.append((a, b, float(c.weight)))
        elif a in nodes and b in outputs:
            readout.append((a, b, float(c.weight)))
    attrs = {n: (float(genome.nodes[n].bias), float(genome.nodes[n].response),
                 genome.nodes[n].activation, genome.nodes[n].aggregation) for n in sorted(nodes)}
    return Chunk(sorted(nodes), sorted({a for a, _, _ in internal if a in inputs}),
                 sorted({a for a, _, _ in readout}), internal, readout, attrs)


def add_module_keep_ids(child, config, donor, module_id: int, weight_mode: str, rng=None):
    """Module crossover step: replace ``child``'s version of ``module_id`` (if
    any) by ``donor``'s, keeping the donor's node ids (globally unique within
    a run, so the inherited module stays aligned with its lineage). Returns
    (new genome, readout keys of the inherited version)."""
    g, _ = knockout_module(child, module_id)
    ch = module_as_chunk(donor, config, module_id)
    gc = config.genome_config
    for n in ch.hidden_nodes:
        attrs = list(ch.node_attrs[n])
        if weight_mode != "explained":
            attrs[0] = float(rng.normal(0.0, 1.0))
        g.nodes[n] = _new_node(n, tuple(attrs))
    for a, b, w in ch.internal:
        if weight_mode != "explained":
            w = float(rng.normal(0.0, 1.0))
        g.connections[(a, b)] = _new_conn((a, b), w)
    readout = []
    for a, b, w in ch.readout:
        if weight_mode == "sham":
            w = float(rng.normal(0.0, 1.0))
        elif weight_mode == "topology":
            w = 0.0
        g.connections[(a, b)] = _new_conn((a, b), w)
        readout.append((a, b))
    tags = dict(g.module_tags)
    for n in ch.hidden_nodes:
        tags[n] = module_id
    g.module_tags = tags
    # the shared node indexer must stay ahead of every id in use
    _ = gc  # ids came from the same indexer originally; nothing to advance
    return g, readout
