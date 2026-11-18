"""Explanation engine: the naive ConvNEAT semantics with hooks.

Same function as ``explaneat.conv.network.NaiveConvNet`` (per edge, in
topological order, resample-then-conv), but it also returns pre-activations
and per-edge contributions, and it can hold any node's map or any edge's
contribution at a given value (mean-ablation, removal). Float64 by default.

Edge i -> j contribution: ``z_ji = K_ji * R_{r_i -> r_j}(h_i)``.
Node j: ``pre_j = b_j + sum_i z_ji``, ``h_j = act(pre_j)`` (ReLU hidden,
identity output). Head: ``logit = head_w @ GAP(h_out) + head_b``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F

from ..genome import ConvGenome, ConvNeatConfig, topo_order
from ..network import resample


def functional_nodes(g: ConvGenome) -> Set[int]:
    """Nodes on an enabled input -> output path (the only ones that can matter
    and carry signal from the image)."""
    fwd, bwd = {}, {}
    for c in g.enabled_conns():
        fwd.setdefault(c.src, []).append(c.dst)
        bwd.setdefault(c.dst, []).append(c.src)

    def reach(starts, adj):
        seen, st = set(starts), list(starts)
        while st:
            n = st.pop()
            for m in adj.get(n, []):
                if m not in seen:
                    seen.add(m)
                    st.append(m)
        return seen

    return reach(g.input_keys(), fwd) & reach(g.output_keys(), bwd)


def depth_map(g: ConvGenome, nodes: Optional[Set[int]] = None) -> Dict[int, int]:
    """Longest path from an input over enabled edges among ``nodes`` (default: all)."""
    nodes = set(g.nodes) if nodes is None else nodes
    d: Dict[int, int] = {}
    for n in topo_order(g):
        if n not in nodes:
            continue
        ps = [c.src for c in g.enabled_conns() if c.dst == n and c.src in nodes]
        d[n] = 0 if not ps else max(d[p] + 1 for p in ps)
    return d


@dataclass
class Forward:
    maps: Dict[int, torch.Tensor]      # post-activation, (N, 1, h, w)
    pre: Dict[int, torch.Tensor]       # pre-activation (non-input nodes)
    contrib: Dict[int, torch.Tensor]   # per enabled edge innov: z_ji
    logits: torch.Tensor               # (N, n_classes)


class ExplainNet:
    """Hookable forward pass. ``relu_off`` makes chosen hidden nodes linear and
    ``zero_bias`` drops all biases (used for linear surrogates)."""

    def __init__(self, genome: ConvGenome, cfg: ConvNeatConfig, dtype=torch.float64,
                 kernel_fn=None):
        self.g, self.cfg, self.dtype = genome, cfg, dtype
        self.order = topo_order(genome)
        self.kind = {k: n.kind for k, n in genome.nodes.items()}
        self.level = {k: n.level for k, n in genome.nodes.items()}
        self.incoming: Dict[int, List] = {k: [] for k in genome.nodes}
        self.K: Dict[int, torch.Tensor] = {}
        for c in sorted(genome.enabled_conns(), key=lambda c: c.innov):
            k = c.kernel if kernel_fn is None else kernel_fn(c.kernel)
            self.K[c.innov] = torch.tensor(np.asarray(k, dtype=np.float64), dtype=dtype).view(1, 1, c.k, c.k)
            self.incoming[c.dst].append(c)
        self.bias = {k: float(n.bias) for k, n in genome.nodes.items()}
        self.head_w = torch.tensor(np.asarray(genome.head_w, np.float64), dtype=dtype)
        self.head_b = torch.tensor(np.asarray(genome.head_b, np.float64), dtype=dtype)
        self.out_keys = genome.output_keys()

    def size(self, node: int):
        return self.cfg.level_size(self.level[node])

    def forward(self, x: torch.Tensor, node_override: Optional[Dict[int, torch.Tensor]] = None,
                edge_override: Optional[Dict[int, Optional[torch.Tensor]]] = None,
                relu_off: Optional[Set[int]] = None, zero_bias: bool = False,
                keep_contrib: bool = False) -> Forward:
        node_override = node_override or {}
        edge_override = edge_override or {}
        relu_off = relu_off or set()
        x = x.to(self.dtype)
        n = x.shape[0]
        maps, pre, contrib = {}, {}, {}
        for node in self.order:
            if self.kind[node] == "input":
                ch = -node - 1
                m = x[:, ch:ch + 1]
            else:
                lvl = self.level[node]
                size = self.cfg.level_size(lvl)
                b = 0.0 if zero_bias else self.bias[node]
                p = torch.full((n, 1, *size), b, dtype=self.dtype)
                for c in self.incoming[node]:
                    if c.innov in edge_override:
                        z = edge_override[c.innov]
                        if z is None:
                            continue
                        z = z.to(self.dtype).expand(n, 1, *size)
                    else:
                        src = resample(maps[c.src], self.level[c.src], lvl, size)
                        z = F.conv2d(src, self.K[c.innov], padding=c.k // 2)
                    if keep_contrib:
                        contrib[c.innov] = z
                    p = p + z
                pre[node] = p
                m = F.relu(p) if (self.kind[node] == "hidden" and node not in relu_off) else p
            if node in node_override:
                m = node_override[node].to(self.dtype).expand_as(m)
            maps[node] = m
        s = torch.cat([maps[o].mean(dim=(2, 3)) for o in self.out_keys], 1)
        logits = s @ self.head_w.t() + (0.0 if zero_bias else self.head_b)
        return Forward(maps, pre, contrib, logits)

    @torch.no_grad()
    def run(self, X: np.ndarray, batch: int = 1000, **kw) -> Forward:
        """Batched forward on a numpy array; returns numpy-backed tensors (CPU)."""
        parts: List[Forward] = []
        for i in range(0, len(X), batch):
            parts.append(self.forward(torch.as_tensor(X[i:i + batch]), **kw))
        cat = lambda dicts: {k: torch.cat([d[k] for d in dicts]) for k in dicts[0]}  # noqa: E731
        return Forward(cat([p.maps for p in parts]), cat([p.pre for p in parts]),
                       cat([p.contrib for p in parts]) if parts[0].contrib else {},
                       torch.cat([p.logits for p in parts]))


def log_loss(logits: torch.Tensor, y: np.ndarray, n_classes: int) -> float:
    yt = torch.as_tensor(y)
    if n_classes == 1:
        return float(F.binary_cross_entropy_with_logits(logits[:, 0], yt.to(logits.dtype)))
    return float(F.cross_entropy(logits, yt.long()))


def accuracy(logits: torch.Tensor, y: np.ndarray) -> float:
    pred = (logits[:, 0] > 0).long() if logits.shape[1] == 1 else logits.argmax(1)
    return float((pred.numpy() == y).mean())
