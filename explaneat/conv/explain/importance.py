"""Importance by mean-ablation.

A node is held at its mean activation map (mean over the validation set, per
pixel); an edge's contribution ``z_ji`` is held at its mean map. The drop in
validation accuracy (full - ablated) and the rise in log-loss (ablated - full)
are reported. A node or edge is **signal-carrying** if
``d_acc >= ACC_T`` or ``d_logloss >= LL_T`` (both 0.01), else **marginal**.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from .engine import ExplainNet, accuracy, log_loss

ACC_T = 0.01
LL_T = 0.01


def _score(net: ExplainNet, X: np.ndarray, y: np.ndarray, n_classes: int, **kw) -> dict:
    f = net.run(X, **kw)
    return {"acc": accuracy(f.logits, y), "logloss": log_loss(f.logits, y, n_classes)}


def classify(d_acc: float, d_ll: float) -> str:
    return "signal" if (d_acc >= ACC_T or d_ll >= LL_T) else "marginal"


def ablation_importance(net: ExplainNet, X: np.ndarray, y: np.ndarray, n_classes: int,
                        nodes: Iterable[int], edges: Iterable[int],
                        groups: Dict[str, Iterable[int]] = None) -> dict:
    base_f = net.run(X, keep_contrib=True)
    base = {"acc": accuracy(base_f.logits, y), "logloss": log_loss(base_f.logits, y, n_classes)}
    mean_map = {k: m.mean(0, keepdim=True) for k, m in base_f.maps.items()}
    mean_z = {k: z.mean(0, keepdim=True) for k, z in base_f.contrib.items()}

    def row(s):
        da, dl = base["acc"] - s["acc"], s["logloss"] - base["logloss"]
        return {"acc": s["acc"], "logloss": s["logloss"], "d_acc": da, "d_logloss": dl,
                "class": classify(da, dl)}

    out = {"base": base, "nodes": {}, "edges": {}, "groups": {},
           "thresholds": {"d_acc": ACC_T, "d_logloss": LL_T},
           "method": "mean-ablation on the validation set (node map / edge contribution held at its mean map)"}
    for k in nodes:
        out["nodes"][k] = row(_score(net, X, y, n_classes, node_override={k: mean_map[k]}))
    for e in edges:
        out["edges"][e] = row(_score(net, X, y, n_classes, edge_override={e: mean_z[e]}))
    for name, ns in (groups or {}).items():
        out["groups"][name] = row(_score(net, X, y, n_classes, node_override={k: mean_map[k] for k in ns}))
    return out
