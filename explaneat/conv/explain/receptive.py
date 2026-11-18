"""Receptive fields and effective input kernels.

* **Receptive field box** (exact, structural): the input pixels that can affect
  one pixel of a node's pre-activation. Computed by back-propagating from the
  centre pixel through a copy of the network with every kernel replaced by its
  absolute value plus one, biases zero and ReLUs removed, on a large canvas so
  the box is not clipped by the image border. With positive weights nothing
  cancels, so the non-zero gradient is exactly the set of pixels connected to
  that output pixel through the composed kernel extents and resamplings.
* **Effective input kernel** of a node's pre-activation: the composed linear
  map from the input image to one pixel (the centre pixel; with pooling the map
  is periodic with period 2^level, and the centre pixel is one phase of it).
  It is **exact** when every node strictly upstream of the node (on an
  input -> node path) is linear on the region considered, i.e. an output
  (identity) node or a ReLU node whose pre-activation is >= 0 on every pixel of
  every evaluation example. Then the node's pre-activation is
  ``bias-term + K_eff * x`` on the evaluation set, up to border effects; the
  kernel is the Jacobian row of the linear surrogate. Otherwise a
  **gradient ERF** is reported instead: the mean over examples of
  d pre_n[centre] / d x (signed; also its mean absolute value), labelled as
  approximate.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Set

import numpy as np
import torch

from ..genome import ConvGenome, ConvNeatConfig
from .engine import ExplainNet


def canvas_cfg(cfg: ConvNeatConfig, size: int = 96) -> ConvNeatConfig:
    return dataclasses.replace(cfg, image_hw=(size, size))


def upstream(g: ConvGenome, node: int) -> Set[int]:
    pred = {}
    for c in g.enabled_conns():
        pred.setdefault(c.dst, []).append(c.src)
    seen, st = set(), [node]
    while st:
        n = st.pop()
        for p in pred.get(n, []):
            if p not in seen:
                seen.add(p)
                st.append(p)
    return seen


def _centre_grad(net: ExplainNet, cfg: ConvNeatConfig, node: int, x: torch.Tensor, **kw) -> np.ndarray:
    x = x.clone().requires_grad_(True)
    f = net.forward(x, **kw)
    t = f.pre[node] if node in f.pre else f.maps[node]
    h, w = t.shape[2:]
    t[:, 0, h // 2, w // 2].sum().backward()
    return x.grad.detach().numpy()[:, 0]


def receptive_box(g: ConvGenome, cfg: ConvNeatConfig, node: int, canvas: int = 96) -> dict:
    cc = canvas_cfg(cfg, canvas)
    net = ExplainNet(g, cc, kernel_fn=lambda k: np.abs(k) + 1.0)
    relu_off = set(g.hidden_keys())
    x = torch.ones(1, cfg.in_channels, canvas, canvas, dtype=torch.float64)
    gr = _centre_grad(net, cc, node, x, relu_off=relu_off, zero_bias=True)[0]
    nz = np.argwhere(gr != 0)
    if len(nz) == 0:
        return {"h": 0, "w": 0, "rows": [0, 0], "cols": [0, 0], "frac_image": 0.0}
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    H, W = cfg.image_hw
    return {"h": h, "w": w, "rows": [int(r0), int(r1)], "cols": [int(c0), int(c1)],
            "frac_image": float(min(h, H) * min(w, W) / (H * W)), "canvas": canvas,
            "covers_image": bool(h >= 2 * H - 1 and w >= 2 * W - 1)}


def linear_on(fwd_pre: Dict[int, torch.Tensor], g: ConvGenome) -> Dict[int, bool]:
    """Per node: is its activation linear on the evaluation set?"""
    out = {}
    for k, n in g.nodes.items():
        if n.kind == "input" or n.kind == "output":
            out[k] = True
        else:
            out[k] = bool(fwd_pre[k].min() >= 0)
    return out


def effective_input_kernel(g: ConvGenome, cfg: ConvNeatConfig, node: int, linear: Dict[int, bool],
                           X_eval: np.ndarray, box: dict, n_examples: int = 256, canvas: int = 96) -> dict:
    ups = upstream(g, node)
    blockers = sorted(k for k in ups if not linear[k] and g.nodes[k].kind != "input")
    if not blockers:
        cc = canvas_cfg(cfg, canvas)
        net = ExplainNet(g, cc)
        x = torch.zeros(1, cfg.in_channels, canvas, canvas, dtype=torch.float64)
        gr = _centre_grad(net, cc, node, x, relu_off=set(g.hidden_keys()), zero_bias=True)[0]
        r0, r1 = box["rows"]
        c0, c1 = box["cols"]
        return {"exact": True, "kernel": gr[r0:r1 + 1, c0:c1 + 1],
                "method": "exact: composed linear map (every upstream node is linear on the evaluation set)"}
    net = ExplainNet(g, cfg)
    x = torch.as_tensor(X_eval[:n_examples], dtype=torch.float64)
    gr = _centre_grad(net, cfg, node, x)
    return {"exact": False, "kernel": gr.mean(0), "abs_kernel": np.abs(gr).mean(0),
            "blockers": blockers, "n_examples": int(len(x)),
            "method": (f"approximate: gradient ERF, mean d pre-activation[centre] / d input over "
                       f"{len(x)} evaluation images (nodes {blockers} are not linear on the data)")}
