"""Phenotype semantics, the naive reference engine, and shared helpers.

Semantics of one edge i -> j (resample-then-conv):

    z_ji = W_ji * resample(x_i, level_i -> level_j)

The source map is first resampled to the target's resolution (adaptive
average pooling going down, which equals avg-pool by 2**(r_j - r_i) when the
sizes divide; nearest-neighbour upsampling going up; identity at the same
level), then the k x k kernel is applied with stride 1 and "same" padding at
the target resolution. Pooling and convolution do not commute, so the order is
fixed. Node j computes ``y_j = act(b_j + sum_i z_ji)``; act = ReLU for hidden
nodes, identity for output nodes.

Head (fixed, not evolved): every output map is reduced by global average
pooling to a scalar, giving s (length n_out_maps; with ``cfg.head = "grid2"``,
a 2x2 average pool gives 4 values per map); logits = head_w @ s +
head_b. Binary tasks: one output map, sigmoid on the single logit. k-class
tasks: k output maps, softmax.

Two engines compute this function:
* ``NaiveConvNet`` (this module): per edge, in topological order. Slow but
  obviously correct; the ground truth for tests. Not used for training.
* ``PropNEATConvNet`` (propneat.py): layers of (depth, level), one masked
  conv2d per layer. Used for training.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .genome import ConvGenome, ConvNeatConfig, topo_order


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def head_pool(maps: List[torch.Tensor], cfg: ConvNeatConfig) -> torch.Tensor:
    """The fixed head's pooling: GAP (one value per output map) or, with ``cfg.head == "grid2"``,
    a 2x2 adaptive average pool (4 values per map, row-major cells, map-major order)."""
    if getattr(cfg, "head", "gap") == "grid2":
        return torch.cat([F.adaptive_avg_pool2d(m, (2, 2)).flatten(1) for m in maps], 1)
    return torch.cat([m.mean(dim=(2, 3)) for m in maps], 1)


def resample(x: torch.Tensor, src_level: int, dst_level: int, size) -> torch.Tensor:
    """Derived (never evolved) resampling of a source map to the target level."""
    if dst_level == src_level:
        return x
    if dst_level > src_level:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="nearest")


class NaiveConvNet(nn.Module):
    """Reference engine: for each node, resample each source, conv, sum, bias, activate."""

    def __init__(self, genome: ConvGenome, cfg: ConvNeatConfig, dtype=torch.float32):
        super().__init__()
        self.genome = genome
        self.cfg = cfg
        self.order = topo_order(genome)
        self.level = {k: n.level for k, n in genome.nodes.items()}
        self.kind = {k: n.kind for k, n in genome.nodes.items()}
        self.kernels = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        self.incoming: Dict[int, List] = {k: [] for k in genome.nodes}
        for c in sorted(genome.enabled_conns(), key=lambda c: c.innov):
            self.kernels[str(c.innov)] = nn.Parameter(torch.tensor(c.kernel, dtype=dtype))
            self.incoming[c.dst].append(c)
        for k, n in genome.nodes.items():
            if n.kind != "input":
                self.biases[str(k)] = nn.Parameter(torch.tensor(float(n.bias), dtype=dtype))
        self.head_w = nn.Parameter(torch.tensor(genome.head_w, dtype=dtype))
        self.head_b = nn.Parameter(torch.tensor(genome.head_b, dtype=dtype))
        self.out_keys = genome.output_keys()

    def forward_maps(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        n = x.shape[0]
        maps: Dict[int, torch.Tensor] = {}
        for node in self.order:
            if self.kind[node] == "input":
                ch = -node - 1
                maps[node] = x[:, ch:ch + 1]
                continue
            lvl = self.level[node]
            size = self.cfg.level_size(lvl)
            pre = self.biases[str(node)].expand(n, 1, *size)
            for c in self.incoming[node]:
                src = resample(maps[c.src], self.level[c.src], lvl, size)
                w = self.kernels[str(c.innov)].view(1, 1, c.k, c.k)
                pre = pre + F.conv2d(src, w, padding=c.k // 2)
            maps[node] = F.relu(pre) if self.kind[node] == "hidden" else pre
        return maps

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        maps = self.forward_maps(x)
        s = head_pool([maps[o] for o in self.out_keys], self.cfg)  # GAP (or 2x2 grid)
        logits = s @ self.head_w.t() + self.head_b
        return (logits, maps) if return_maps else logits

    @torch.no_grad()
    def write_back(self) -> None:
        for c in self.genome.enabled_conns():
            c.kernel = self.kernels[str(c.innov)].detach().cpu().numpy().astype(np.float32).copy()
        for k, n in self.genome.nodes.items():
            if n.kind != "input":
                n.bias = float(self.biases[str(k)].detach().cpu())
        self.genome.head_w = self.head_w.detach().cpu().numpy().astype(np.float32).copy()
        self.genome.head_b = self.head_b.detach().cpu().numpy().astype(np.float32).copy()

    def kernel_grads(self) -> Dict[int, np.ndarray]:
        return {int(k): p.grad.detach().cpu().numpy() for k, p in self.kernels.items() if p.grad is not None}


def loss_fn(logits: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes == 1:
        return F.binary_cross_entropy_with_logits(logits[:, 0], y.float())
    return F.cross_entropy(logits, y.long())


@torch.no_grad()
def predict(net: nn.Module, X: torch.Tensor, batch: int = 1024) -> np.ndarray:
    """Class probabilities (binary: P(y=1), shape [N]; k-class: [N, k])."""
    net.eval()
    out = []
    for i in range(0, len(X), batch):
        lo = net(X[i:i + batch])
        out.append(torch.sigmoid(lo[:, 0]) if lo.shape[1] == 1 else torch.softmax(lo, 1))
    return torch.cat(out).cpu().numpy()


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    pred = (probs > 0.5).astype(int) if probs.ndim == 1 else probs.argmax(1)
    return float((pred == y).mean())


def build_engine(genome: ConvGenome, cfg: ConvNeatConfig, engine: str = "propneat", **kw) -> nn.Module:
    if engine == "naive":
        return NaiveConvNet(genome, cfg, **kw)
    from .propneat import PropNEATConvNet
    return PropNEATConvNet(genome, cfg, **kw)


def train_genome(genome: ConvGenome, cfg: ConvNeatConfig, X: torch.Tensor, y: torch.Tensor,
                 epochs: int, rng: np.random.Generator, device: torch.device,
                 engine: str = "propneat") -> float:
    """Backprop the genome's weights (Adam, minibatch) with the PropNEAT engine,
    check the masks still hold, write the weights back (Lamarckian)."""
    net = build_engine(genome, cfg, engine).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    n = len(X)
    last = None
    net.train()
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n), device=X.device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = loss_fn(net(X[idx]), y[idx], cfg.n_classes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.detach()
    if engine == "propneat":
        net.assert_masks_hold()  # every masked entry still exactly 0
    net.write_back()
    return float(last) if last is not None else float("nan")
