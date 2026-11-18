"""PropNEAT genome-to-tensor mapping generalised to convolutions.

The tabular version is ``explaneat.core.neuralneat`` (Merry et al., GECCO '25):
nodes are grouped into layers by depth, each layer's inputs are the
concatenation of every shallower layer that feeds it (skip layers), and one
masked dense weight matrix per layer carries every gene. Here:

Traversal
  * forward BFS from the inputs  -> ``fwd_reach`` (recorded);
  * backward BFS from the outputs -> ``kept``. Nodes that cannot reach an
    output are dropped (exact: they cannot influence the logits). Nodes that
    reach an output but no input are *kept*: they are constant maps
    act(bias), and dropping them would change the function (the tabular
    PropNEAT drops them; we keep them so the engine is exactly the naive
    semantics).
  * depth(n) = longest path to n from a node with no kept predecessor
    (inputs have depth 0).

Layers
  Inputs form layer 0. Every other kept node goes into layer (depth, level).
  A layer is a C-channel tensor at its level's resolution, one channel per
  node (channels in ascending node id). Hidden channels get ReLU, output
  channels identity (a fixed per-channel activation mask).

Layer connections
  For target layer T, every source layer S holding a source node of a gene
  into T (including skip layers from any shallower depth) is resampled to T's
  level and the results are concatenated along the channel axis in ascending
  layer index -> C_in_total channels. One masked conv2d with weight
  [C_T, C_in_total, k_max, k_max] (k_max = largest kernel among T's genes)
  and a per-output-channel bias (the node biases) gives T's pre-activation.

Gene <-> tensor bijection (``gene_index``)
  gene innov (i -> j) <-> (layer t, group k_g, out channel = channel of j in T,
  in channel = offset(S) + channel of i in S, centre offset o = (k_g - k)//2,
  k). Its kernel lives in W[t,k_g][out, in, o:o+k, o:o+k] (centre-embedded,
  zero-padded). Bias of node j <-> bias[t][channel of j]. ``write_back``
  copies the trained slices back (centre-cropped to each gene's own k).

Masking
  The forward pass uses W * M with a fixed binary M (1 exactly on gene
  slices), so masked entries receive zero gradient and stay exactly 0 under
  Adam (no weight decay). ``assert_masks_hold`` checks this.

Mixed kernel sizes
  ``mixed_k="embed"`` (default): one conv per layer at k_max (extra FLOPs,
  reported by ``flop_report``). ``mixed_k="split"``: one conv per distinct
  kernel size in the layer, summed.

Retrain variant
  ``retrain=True`` sets every mask to 1: the minimal covering dense CNN with
  the same layer structure (PropNEAT-retrain). The gene bijection is lost:
  ``write_back`` refuses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .genome import ConvGenome, ConvNeatConfig, topo_order
from .network import head_pool, resample


def traverse(genome: ConvGenome):
    """Forward/backward reachability and depth over enabled edges."""
    enabled = genome.enabled_conns()
    succ, pred = defaultdict(list), defaultdict(list)
    for c in enabled:
        succ[c.src].append(c.dst)
        pred[c.dst].append(c.src)

    def bfs(starts, adj):
        seen, frontier = set(starts), list(starts)
        while frontier:
            nxt = []
            for n in frontier:
                for m in adj[n]:
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
            frontier = nxt
        return seen

    fwd = bfs(genome.input_keys(), succ)
    kept = bfs(genome.output_keys(), pred)
    depth: Dict[int, int] = {}
    for n in topo_order(genome):
        if n not in kept:
            continue
        ps = [p for p in pred[n] if p in kept]  # all preds of a kept node are kept
        depth[n] = 0 if (not ps or genome.nodes[n].kind == "input") else max(depth[p] + 1 for p in ps)
    return fwd, kept, depth


class PropNEATConvNet(nn.Module):
    def __init__(self, genome: ConvGenome, cfg: ConvNeatConfig, dtype=torch.float32,
                 mixed_k: str = "embed", retrain: bool = False):
        super().__init__()
        assert mixed_k in ("embed", "split")
        self.genome, self.cfg, self.mixed_k, self.retrain = genome, cfg, mixed_k, retrain
        self.fwd_reach, self.kept, self.depth = traverse(genome)
        nodes = genome.nodes

        # ---- layers
        keys: Dict[Tuple, List[int]] = defaultdict(list)
        for n in sorted(self.kept):
            key = (-1, 0) if nodes[n].kind == "input" else (self.depth[n], nodes[n].level)
            keys[key].append(n)
        if (-1, 0) not in keys:
            keys[(-1, 0)] = []
        self.layer_keys = sorted(keys)  # input layer first, then by (depth, level)
        self.layers = [sorted(keys[k], reverse=(k == (-1, 0))) for k in self.layer_keys]  # inputs: -1,-2,..
        self.layer_level = [0 if k == (-1, 0) else k[1] for k in self.layer_keys]
        self.node_loc = {n: (t, ch) for t, ns in enumerate(self.layers) for ch, n in enumerate(ns)}
        self.input_channels = [-n - 1 for n in self.layers[0]]

        # ---- layer connections, weights, masks, bijection
        by_target = defaultdict(list)
        for c in genome.enabled_conns():
            if c.dst in self.kept:
                by_target[self.node_loc[c.dst][0]].append(c)
        self.sources: Dict[int, List[int]] = {}
        self.groups: Dict[int, List[int]] = {}
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        self.gene_index: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        self.bias_index: Dict[int, Tuple[int, int]] = {}
        for t in range(1, len(self.layers)):
            genes = sorted(by_target[t], key=lambda c: c.innov)
            src_layers = sorted({self.node_loc[c.src][0] for c in genes})
            offset, off = {}, 0
            for s in src_layers:
                offset[s] = off
                off += len(self.layers[s])
            c_out, c_in = len(self.layers[t]), off
            self.sources[t] = src_layers
            if mixed_k == "embed":
                gk = {c.innov: max([g.k for g in genes], default=1) for c in genes}
            else:
                gk = {c.innov: c.k for c in genes}
            ks = sorted(set(gk.values()))
            self.groups[t] = ks
            for k_g in ks:
                W = torch.zeros(c_out, c_in, k_g, k_g, dtype=dtype)
                M = torch.zeros_like(W)
                for c in genes:
                    if gk[c.innov] != k_g:
                        continue
                    j = self.node_loc[c.dst][1]
                    s, ch = self.node_loc[c.src]
                    i = offset[s] + ch
                    o = (k_g - c.k) // 2
                    W[j, i, o:o + c.k, o:o + c.k] = torch.tensor(c.kernel, dtype=dtype)
                    M[j, i, o:o + c.k, o:o + c.k] = 1.0
                    self.gene_index[c.innov] = (t, k_g, j, i, o, c.k)
                if retrain:
                    M = torch.ones_like(M)
                self.weights[f"{t}_{k_g}"] = nn.Parameter(W)
                self.register_buffer(f"mask_{t}_{k_g}", M)
            b = torch.tensor([nodes[n].bias for n in self.layers[t]], dtype=dtype)
            self.biases[str(t)] = nn.Parameter(b)
            for ch, n in enumerate(self.layers[t]):
                self.bias_index[n] = (t, ch)
            hid = torch.tensor([nodes[n].kind == "hidden" for n in self.layers[t]]).view(1, -1, 1, 1)
            self.register_buffer(f"hidden_{t}", hid)
        self.head_w = nn.Parameter(torch.tensor(genome.head_w, dtype=dtype))
        self.head_b = nn.Parameter(torch.tensor(genome.head_b, dtype=dtype))
        self.out_keys = genome.output_keys()

    # ------------------------------------------------------------------
    def mask(self, t: int, k_g: int) -> torch.Tensor:
        return getattr(self, f"mask_{t}_{k_g}")

    def forward_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        n = x.shape[0]
        L = [x[:, self.input_channels]]
        for t in range(1, len(self.layers)):
            lvl = self.layer_level[t]
            size = self.cfg.level_size(lvl)
            pre = self.biases[str(t)].view(1, -1, 1, 1).expand(n, -1, *size)
            if self.sources[t]:
                cat = torch.cat([resample(L[s], self.layer_level[s], lvl, size) for s in self.sources[t]], 1)
                for k_g in self.groups[t]:
                    w = self.weights[f"{t}_{k_g}"] * self.mask(t, k_g)
                    pre = pre + F.conv2d(cat, w, padding=k_g // 2)
            L.append(torch.where(getattr(self, f"hidden_{t}"), F.relu(pre), pre))
        return L

    def forward_maps(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        L = self.forward_layers(x)
        return {n: L[t][:, ch:ch + 1] for n, (t, ch) in self.node_loc.items()}

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        L = self.forward_layers(x)
        s = head_pool([L[self.node_loc[o][0]][:, self.node_loc[o][1]:self.node_loc[o][1] + 1]
                       for o in self.out_keys], self.cfg)
        logits = s @ self.head_w.t() + self.head_b
        if return_maps:
            return logits, {n: L[t][:, ch:ch + 1] for n, (t, ch) in self.node_loc.items()}
        return logits

    # ------------------------------------------------------------------
    def gene_kernel(self, innov: int, grad: bool = False) -> np.ndarray:
        t, k_g, j, i, o, k = self.gene_index[innov]
        W = self.weights[f"{t}_{k_g}"]
        src = W.grad if grad else W
        return src[j, i, o:o + k, o:o + k].detach().cpu().numpy().copy()

    def kernel_grads(self) -> Dict[int, np.ndarray]:
        return {inn: self.gene_kernel(inn, grad=True) for inn in self.gene_index}

    @torch.no_grad()
    def assert_masks_hold(self) -> None:
        if self.retrain:
            return
        for t, ks in self.groups.items():
            for k_g in ks:
                W, M = self.weights[f"{t}_{k_g}"], self.mask(t, k_g)
                bad = (W[M == 0] != 0).sum().item()
                assert bad == 0, f"layer {t} k={k_g}: {bad} masked weights moved off zero"

    @torch.no_grad()
    def write_back(self) -> None:
        if self.retrain:
            raise RuntimeError("retrain mode: masks lifted, the gene bijection is lost; nothing to write back")
        for inn, _ in self.gene_index.items():
            self.genome.conns[inn].kernel = self.gene_kernel(inn).astype(np.float32)
        for n, (t, ch) in self.bias_index.items():
            self.genome.nodes[n].bias = float(self.biases[str(t)][ch].detach().cpu())
        self.genome.head_w = self.head_w.detach().cpu().numpy().astype(np.float32).copy()
        self.genome.head_b = self.head_b.detach().cpu().numpy().astype(np.float32).copy()


def flop_report(genome: ConvGenome, cfg: ConvNeatConfig, mixed_k: str = "embed") -> dict:
    """Multiply-accumulates per example: the genes' own convs (useful) vs what the
    masked dense layer convs compute (k_max embedding and empty channel slices).
    Resampling cost is ignored in both."""
    net = PropNEATConvNet(genome, cfg, mixed_k=mixed_k)
    useful, dense = 0, 0
    for c in genome.enabled_conns():
        if c.dst in net.kept:
            h, w = cfg.level_size(genome.nodes[c.dst].level)
            useful += c.k * c.k * h * w
    for t, ks in net.groups.items():
        h, w = cfg.level_size(net.layer_level[t])
        for k_g in ks:
            W = net.weights[f"{t}_{k_g}"]
            dense += W.shape[0] * W.shape[1] * k_g * k_g * h * w
    return {"useful_macs": useful, "dense_macs": dense,
            "overhead": dense / useful if useful else float("nan"),
            "n_layers": len(net.layers) - 1, "n_layer_convs": sum(len(v) for v in net.groups.values()),
            "n_kept_nodes": len(net.kept), "n_genome_nodes": len(genome.nodes)}
