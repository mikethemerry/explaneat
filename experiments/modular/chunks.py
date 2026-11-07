"""Checkpoint chunk finder (PLAN.md "Checkpoint chunk finder").

Algorithmic adaptation of the C2/D2 candidate criteria to genomes without a
human explanation. For a genome G:

1. Live nodes: the hidden nodes NeuralNeat computes (its non-input,
   non-output layers). Only enabled connections between live nodes, inputs
   and the output count.
2. Seeds: every live hidden node h with an enabled readout edge h -> output.
3. Closure: C = h plus all its live hidden ancestors. While some node in C
   has an enabled edge to a live hidden node b outside C, add b and all of
   b's live hidden ancestors. The result is the smallest node set containing
   h whose only ingress is from inputs (all-input entries; input entries
   may be split, an identity operation) and whose only egress is readout
   edges to the output (exit-only egress; the exits are pure because every
   predecessor of a C node is in C or is an input). It is therefore
   collapsible and graftable along its boundary.
4. Exits: nodes of C with a readout edge; the chunk's logit contribution is
   s_C(x) = sum over its readout edges of weight * exit activation.
5. Train restore gain (the C/D2 quantity, for a genome partition into
   chunks): with H(x) the summed logit contribution of every hidden readout
   and P the least-squares projection onto span{1, inputs} on the current
   train rows, all-linear logit = z - H + P H, restored logit =
   z - H + s_C + P (H - s_C); gain = AUC(restored) - AUC(all-linear).
6. Candidate iff |C| <= MAX_HIDDEN and gain >= MIN_GAIN.

Chunks with identical node sets are found once.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments" / "churn")):
    if p not in sys.path:
        sys.path.insert(0, p)

import genome_surgery as gs  # noqa: E402

MAX_HIDDEN = 8
MIN_GAIN = 0.02
LINEAR_TOL = 1e-7


@dataclass
class Chunk:
    """A clean-boundary subgraph of one genome, in that genome's ids."""

    hidden_nodes: List[int]
    entry_inputs: List[int]
    exit_nodes: List[int]
    # (src, dst, weight); src is an input or a chunk node, dst a chunk node
    internal: List[Tuple[int, int, float]]
    # (exit, output, weight)
    readout: List[Tuple[int, int, float]]
    # node id -> (bias, response, activation, aggregation)
    node_attrs: Dict[int, Tuple[float, float, str, str]]
    gain: float = float("nan")
    nonlinear: bool = True
    s_probe: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def readout_keys(self):
        return [(a, b) for a, b, _ in self.readout]

    def to_json(self) -> dict:
        return {"hidden_nodes": list(self.hidden_nodes), "entry_inputs": list(self.entry_inputs),
                "exit_nodes": list(self.exit_nodes),
                "internal": [[a, b, w] for a, b, w in self.internal],
                "readout": [[a, b, w] for a, b, w in self.readout],
                "node_attrs": {str(k): list(v) for k, v in self.node_attrs.items()},
                "gain": self.gain, "nonlinear": self.nonlinear, "meta": dict(self.meta)}


def live_structure(genome, config):
    """(net, live hidden set, enabled live edges {(a, b)}, readout row map)."""
    net = gs.build_net(genome, config)
    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    live = set()
    for layer in net.layers.values():
        if layer["layer_type"] in ("INPUT", "OUTPUT"):
            continue
        live.update(layer["nodes"].keys())
    live -= inputs | outputs
    ok = live | inputs | outputs
    edges = {k for k, c in genome.connections.items()
             if c.enabled and k[0] in ok and k[1] in ok}
    return net, live, edges


def _ancestors(nodes, preds, live):
    out, stack = set(), list(nodes)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        for p in preds.get(n, ()):
            if p in live and p not in out:
                stack.append(p)
    return out


def closure(h, live, edges):
    preds, succ = {}, {}
    for a, b in edges:
        preds.setdefault(b, []).append(a)
        succ.setdefault(a, []).append(b)
    C = _ancestors([h], preds, live)
    while True:
        extra = sorted({b for a in C for b in succ.get(a, ()) if b in live and b not in C})
        if not extra:
            return C
        C = _ancestors(list(C) + extra, preds, live)


def projection(X):
    """Function v -> least-squares fit of v on [1, X] (rank-safe, lstsq)."""
    A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])

    def P(v):
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return A @ coef
    return P


def readout_terms(genome, config, X, output_key=0):
    """z, and per hidden readout key its logit contribution, on X."""
    t = gs.output_preactivation_terms(genome, config, X, output_key)
    imap = t["net"].layers[t["layer"]]["input_map"]
    inputs = set(config.genome_config.input_keys)
    contrib = {}
    for k, (r, c) in imap.items():
        if c != t["col"] or k[0] in inputs:
            continue
        contrib[tuple(k)] = t["w"][r] * t["inputs"][:, r]
    return t["z"], contrib


def find_chunks(genome, config, X, y, probe_X=None, output_key=0,
                max_hidden=MAX_HIDDEN, min_gain=MIN_GAIN, keep_all=False):
    """Candidate chunks of ``genome`` (gain on (X, y); s on ``probe_X``).

    Returns the qualifying chunks sorted by (-gain, min node id); with
    ``keep_all`` every distinct closure is returned (``meta['qualifies']``).
    """
    net, live, edges = live_structure(genome, config)
    readout_seeds = sorted(a for a, b in edges if b == output_key and a in live)
    if not readout_seeds:
        return []
    z, contrib = readout_terms(genome, config, X, output_key)
    yv = np.asarray(y, dtype=np.float64).ravel()
    H = np.zeros(len(yv))
    for v in contrib.values():
        H = H + v
    P = projection(X)
    PH = P(H)
    base = z - H
    try:
        auc_lin = roc_auc_score(yv, base + PH)
    except ValueError:
        return []
    seen, out = set(), []
    for h in readout_seeds:
        C = closure(h, live, edges)
        key = frozenset(C)
        if key in seen:
            continue
        seen.add(key)
        ro = sorted((a, b) for a, b in edges if a in C and b == output_key)
        sC = np.zeros(len(yv))
        for k in ro:
            sC = sC + contrib.get(k, 0.0)
        resid = sC - P(sC)
        scale = max(1.0, float(np.max(np.abs(sC))) if len(sC) else 1.0)
        nonlinear = bool(np.max(np.abs(resid)) > LINEAR_TOL * scale)
        gain = 0.0
        if nonlinear:
            gain = float(roc_auc_score(yv, base + sC + P(H - sC)) - auc_lin)
        qualifies = len(C) <= max_hidden and gain >= min_gain
        if not (qualifies or keep_all):
            continue
        internal = sorted((a, b, float(genome.connections[(a, b)].weight))
                          for a, b in edges if b in C)
        readout = [(a, b, float(genome.connections[(a, b)].weight)) for a, b in ro]
        entries = sorted({a for a, b, _ in internal if a not in C})
        attrs = {n: (float(genome.nodes[n].bias), float(genome.nodes[n].response),
                     genome.nodes[n].activation, genome.nodes[n].aggregation)
                 for n in sorted(C)}
        ch = Chunk(sorted(C), entries, sorted({a for a, _, _ in readout}), internal, readout,
                   attrs, gain=gain, nonlinear=nonlinear,
                   meta={"qualifies": qualifies, "auc_all_linear": float(auc_lin),
                         "genome_key": genome.key})
        if probe_X is not None:
            ch.s_probe = chunk_contribution(genome, config, probe_X, ch, output_key)
        out.append(ch)
    out.sort(key=lambda c: (-c.gain, c.hidden_nodes[0]))
    return out


def chunk_contribution(genome, config, X, chunk: Chunk, output_key=0):
    """s_C on X computed with ``genome`` (whose ids the chunk uses)."""
    _, contrib = readout_terms(genome, config, X, output_key)
    s = np.zeros(len(np.asarray(X)))
    for k in chunk.readout_keys:
        s = s + contrib.get(tuple(k), 0.0)
    return s
