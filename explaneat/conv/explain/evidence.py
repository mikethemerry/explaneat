"""Per-node evidence on an evaluation set.

* mean and SD activation maps (post-activation);
* top-k and bottom-k examples by the spatial-max pooled activation, with each
  example's activation map (for overlays);
* saliency: per-pixel |d mean-pooled activation / d input|, averaged over the
  top examples;
* feature visualisation (synthetic input, labelled as such): the input in
  [0, 1] (``x = sigmoid(u)``) that maximises the node's mean-pooled activation,
  by Adam gradient ascent from a fixed seed with L2 (towards 0.5) and
  total-variation penalties.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..genome import ConvGenome, ConvNeatConfig
from .engine import ExplainNet, Forward


def map_stats(fwd: Forward) -> Dict[int, dict]:
    out = {}
    for k, m in fwd.maps.items():
        a = m[:, 0].numpy()
        pooled_max = a.reshape(len(a), -1).max(1)
        out[k] = {"mean": a.mean(0), "sd": a.std(0), "pooled_max": pooled_max,
                  "pooled_mean": a.reshape(len(a), -1).mean(1),
                  "scale": float(np.percentile(pooled_max, 99)) if a.size else 1.0,
                  "min": float(a.min()), "max": float(a.max())}
    return out


def top_bottom(fwd: Forward, node: int, k: int = 6) -> dict:
    a = fwd.maps[node][:, 0].numpy()
    pooled = a.reshape(len(a), -1).max(1)
    order = np.argsort(-pooled, kind="stable")
    top, bot = order[:k], order[::-1][:k]
    return {"top": top, "bottom": bot, "top_val": pooled[top], "bottom_val": pooled[bot],
            "top_maps": a[top], "bottom_maps": a[bot]}


def saliency(net: ExplainNet, X: np.ndarray, node: int, idx: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(X[idx], dtype=torch.float64).clone().requires_grad_(True)
    f = net.forward(x)
    f.maps[node].mean(dim=(1, 2, 3)).sum().backward()
    return np.abs(x.grad.numpy()[:, 0]).mean(0)


def feature_vis(net: ExplainNet, cfg: ConvNeatConfig, node: int, steps: int = 300, lr: float = 0.08,
                l2: float = 0.02, tv: float = 0.05, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    u = (torch.randn(1, cfg.in_channels, *cfg.image_hw, generator=gen, dtype=torch.float64) * 0.1)
    u.requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        x = torch.sigmoid(u)
        act = net.forward(x).maps[node].mean()
        reg = l2 * ((x - 0.5) ** 2).mean() + tv * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
                                                  + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
        loss = -act + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.sigmoid(u)
        f = net.forward(x)
    return {"image": x[0, 0].numpy(), "map": f.maps[node][0, 0].numpy(),
            "pooled_mean": float(f.maps[node].mean()), "pooled_max": float(f.maps[node].max()),
            "settings": {"steps": steps, "lr": lr, "l2": l2, "tv": tv, "seed": seed,
                         "objective": "mean-pooled activation", "param": "x = sigmoid(u), u0 ~ N(0, 0.1^2)"}}
