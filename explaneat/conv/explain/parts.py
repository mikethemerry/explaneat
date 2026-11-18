"""Automatic parts (a two-level annotation hierarchy) for a ConvNEAT network.

Rule (documented, deterministic):

1. Units are the functional hidden nodes (on an input -> output path).
2. Each unit gets a *label*: on the synthetic task, its set of strong planted
   concepts under spatial-max pooling (tau = 0.5), e.g. {hbar} or
   {vbar, diag}; on digits, its strong class (|2 AUC - 1| >= 0.5 under max
   pooling). No strong concept: no label.
3. Edges u -> v between units are visited in order of decreasing edge
   importance (rise in log-loss under mean-ablation; ties by innovation
   number). The groups of u and v are merged if
   (a) every node of both groups carries the same non-empty label
       (a connected sub-DAG sharing a dominant concept), or
   (b) u is unlabelled or shares v's label, u's only functional consumer is v,
       and v's only hidden producer is u (a chain feeding one node);
   and the merge keeps the graph of groups acyclic (so the parts compose).
4. Every group is a leaf part. Its entries are the outside sources of its
   edges, its exits the members with an edge leaving it.
5. The root is the head: ``logit = head_b + head_w * GAP(h_out)`` with
   ``h_out = b_out + sum_i K_{i->out} * R(h_i)``, where the h_i are exits
   of the leaves (or the input). The output map node belongs to the root.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from .engine import ExplainNet, functional_nodes
from .formula import FormulaContext, System, evaluate, node_formula, part_system


def _quotient_acyclic(g: ConvGenome, group_of: Dict[int, int]) -> bool:
    adj: Dict[int, Set[int]] = {}
    for c in g.enabled_conns():
        a, b = group_of.get(c.src, ("n", c.src)), group_of.get(c.dst, ("n", c.dst))
        if a != b:
            adj.setdefault(a, set()).add(b)
    state: Dict = {}

    def dfs(u) -> bool:
        state[u] = 1
        for v in adj.get(u, ()):
            s = state.get(v, 0)
            if s == 1 or (s == 0 and not dfs(v)):
                return False
        state[u] = 2
        return True

    return all(state.get(u, 0) == 2 or dfs(u) for u in list(adj))


def propose_parts(g: ConvGenome, labels: Dict[int, Optional[frozenset]],
                  edge_score: Dict[int, float]) -> List[List[int]]:
    fn = functional_nodes(g)
    units = [k for k in topo_order(g) if k in fn and g.nodes[k].kind == "hidden"]
    group_of = {u: u for u in units}
    members = {u: [u] for u in units}
    edges = [c for c in g.enabled_conns() if c.src in group_of and c.dst in group_of]
    edges.sort(key=lambda c: (-edge_score.get(c.innov, 0.0), c.innov))
    consumers = {u: {c.dst for c in g.enabled_conns() if c.src == u and c.dst in fn} for u in units}
    hidden_prod = {u: {c.src for c in g.enabled_conns() if c.dst == u and c.src in group_of} for u in units}
    for c in edges:
        a, b = group_of[c.src], group_of[c.dst]
        if a == b:
            continue
        la = {labels.get(n) for n in members[a]}
        lb = {labels.get(n) for n in members[b]}
        same = len(la) == 1 and la == lb and None not in la
        chain = (consumers[c.src] == {c.dst} and hidden_prod[c.dst] == {c.src}
                 and (labels.get(c.src) is None or labels.get(c.src) == labels.get(c.dst)))
        if not (same or chain):
            continue
        trial = dict(group_of)
        for n in members[b]:
            trial[n] = a
        if not _quotient_acyclic(g, trial):
            continue
        group_of = trial
        members[a] = members[a] + members.pop(b)
    order = {k: i for i, k in enumerate(topo_order(g))}
    parts = [sorted(m, key=order.get) for m in members.values()]
    # topological order of the groups (they form a DAG by construction)
    gid = {n: i for i, p in enumerate(parts) for n in p}
    deps = {i: {gid[c.src] for c in g.enabled_conns() if c.dst in p and c.src in gid and gid[c.src] != i}
            for i, p in enumerate(parts)}
    out, done = [], set()
    while len(out) < len(parts):
        ready = [i for i in range(len(parts)) if i not in done and deps[i] <= done]
        i = min(ready, key=lambda i: order[parts[i][0]])
        done.add(i)
        out.append(parts[i])
    return out


def part_graph(g: ConvGenome, parts: List[List[int]]) -> Dict[str, dict]:
    """Entries, exits and consumers of each part (ids 'P1', 'P2', ...)."""
    pid = {n: f"P{i + 1}" for i, p in enumerate(parts) for n in p}
    out = {}
    for i, p in enumerate(parts):
        s = set(p)
        name = f"P{i + 1}"
        entries = sorted({c.src for c in g.enabled_conns() if c.dst in s and c.src not in s})
        exits = [n for n in p if any(c.src == n and c.dst not in s for c in g.enabled_conns())]
        cons = sorted({pid.get(c.dst, f"node {c.dst}") for c in g.enabled_conns()
                       if c.src in s and c.dst not in s})
        feeds_out = any(g.nodes[c.dst].kind == "output" for c in g.enabled_conns() if c.src in s)
        out[name] = {"nodes": p, "entries": entries, "exits": exits,
                     "entry_parts": sorted({pid.get(e, "input" if g.nodes[e].kind == "input" else f"node {e}")
                                            for e in entries}),
                     "consumers": cons, "feeds_output": feeds_out}
    return out


def compose_and_check(g: ConvGenome, cfg: ConvNeatConfig, parts: List[List[int]], X: np.ndarray) -> dict:
    """Evaluate the hierarchy as formulas only: the leaves' systems in part
    order, then the root (output map equation and head). Compare with the
    engine (float64). Also check coverage of the reachable hidden nodes."""
    ctx = FormulaContext(g, cfg)
    systems = [part_system(ctx, set(p)) for p in parts]
    fn = functional_nodes(g)
    reachable = {k for k in fn if g.nodes[k].kind == "hidden"}
    covered = set(n for p in parts for n in p)
    root_eqs = [(o, node_formula(ctx, o)[0]) for o in g.output_keys()]
    x = torch.as_tensor(X, dtype=torch.float64)
    env: Dict[int, torch.Tensor] = {k: x[:, -k - 1:-k] for k in g.input_keys()}
    # nodes outside every part that a formula reads (non-functional sources,
    # e.g. constant maps with no input): take them from the engine
    eng = ExplainNet(g, cfg).forward(x)
    for k in g.nodes:
        if k not in covered and g.nodes[k].kind == "hidden":
            env[k] = eng.maps[k]
    for s in systems:
        env.update(s.evaluate(env, cfg))
    for o, e in root_eqs:
        env[o] = evaluate(e, env, cfg)
    sgap = torch.cat([env[o].mean(dim=(2, 3)) for o in g.output_keys()], 1)
    logits = sgap @ torch.as_tensor(np.asarray(g.head_w, np.float64)).t() + torch.as_tensor(np.asarray(g.head_b, np.float64))
    from ..network import NaiveConvNet
    with torch.no_grad():
        ref = NaiveConvNet(g, cfg, dtype=torch.float64)(x)
    node_err = max((float((env[k] - eng.maps[k]).abs().max()) for s in systems for k in s.exits), default=0.0)
    return {"max_abs_logit_diff": float((logits - ref).abs().max()),
            "max_abs_exit_map_diff": node_err,
            "n_examples": int(len(X)),
            "coverage": len(covered & reachable) / len(reachable) if reachable else 1.0,
            "uncovered": sorted(reachable - covered),
            "n_reachable_hidden": len(reachable),
            "systems": systems, "root_eqs": root_eqs}

