"""Per-image activation paths into each class score, and the network's most confused class pairs.

Independent of how parts are defined, so the report's class nodes, path highlighting and
confusion picker work with any decomposition.

* ``image_paths``: for each image and class c, the gradient x input of every enabled edge
  (``sum_pixels z_e * dz_c / dpre_dst``), of every non-input map (``sum h_n * dz_c / dh_n``) and the
  exact head terms ``W[c, m] * GAP(out_m)`` (whose sum plus ``b_c`` is the class score).
* ``confusion_pairs``: the most frequent unordered confusions on a labelled set, each with its
  misclassified images (both directions) and near misses (right, with the other class as a close
  runner-up).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .engine import ExplainNet


def image_paths(net: ExplainNet, X: np.ndarray, k: int) -> List[dict]:
    g = net.g
    edges = [c.innov for c in g.enabled_conns()]
    dst = {c.innov: c.dst for c in g.enabled_conns()}
    nodes = [n for n in g.nodes if g.nodes[n].kind != "input"]
    outs = g.output_keys()
    W = np.asarray(g.head_w, np.float64)
    x = torch.as_tensor(X, dtype=torch.float64).clone().requires_grad_(True)
    f = net.forward(x, keep_contrib=True)
    for t in list(f.maps.values()) + list(f.pre.values()):
        if t.requires_grad:
            t.retain_grad()
    N = len(X)
    E = {e: np.zeros((N, k)) for e in edges}
    V = {n: np.zeros((N, k)) for n in nodes}
    for c in range(k):
        for t in list(f.maps.values()) + list(f.pre.values()):
            t.grad = None
        f.logits[:, c].sum().backward(retain_graph=c < k - 1)
        for e in edges:
            gp = f.pre[dst[e]].grad
            if gp is not None:
                E[e][:, c] = (f.contrib[e] * gp).sum(dim=(1, 2, 3)).detach().numpy()
        for n in nodes:
            gh = f.maps[n].grad
            if gh is not None:
                V[n][:, c] = (f.maps[n] * gh).sum(dim=(1, 2, 3)).detach().numpy()
    gap = np.stack([f.maps[o].mean(dim=(1, 2, 3)).detach().numpy() for o in outs], 1)  # (N, m)
    head = gap[:, None, :] * W[None, :, :]  # (N, k, m): W[c, m] * GAP(out_m)
    logits = f.logits.detach().numpy()
    return [{"edge": {int(e): E[e][i].tolist() for e in edges}, "node": {int(n): V[n][i].tolist() for n in nodes},
             "head": head[i].tolist(), "logits": logits[i].tolist()} for i in range(N)]


def mean_paths(net: ExplainNet, X: np.ndarray, k: int, batch: int = 200) -> dict:
    """Average over images: signed and absolute per-edge / per-node attributions per class, and the
    mean head terms and scores."""
    acc = None
    n = 0
    for i in range(0, len(X), batch):
        P = image_paths(net, X[i:i + batch], k)
        for p in P:
            if acc is None:
                acc = {"edge_s": {e: np.zeros(k) for e in p["edge"]}, "edge_a": {e: np.zeros(k) for e in p["edge"]},
                       "node_s": {v: np.zeros(k) for v in p["node"]}, "node_a": {v: np.zeros(k) for v in p["node"]},
                       "head": np.zeros_like(np.asarray(p["head"])), "logits": np.zeros(k)}
            for e, v in p["edge"].items():
                acc["edge_s"][e] += v
                acc["edge_a"][e] += np.abs(v)
            for nn, v in p["node"].items():
                acc["node_s"][nn] += v
                acc["node_a"][nn] += np.abs(v)
            acc["head"] += np.asarray(p["head"])
            acc["logits"] += np.asarray(p["logits"])
            n += 1
    return {"edge_signed": {e: (v / n).tolist() for e, v in acc["edge_s"].items()},
            "edge_abs": {e: (v / n).tolist() for e, v in acc["edge_a"].items()},
            "node_signed": {e: (v / n).tolist() for e, v in acc["node_s"].items()},
            "node_abs": {e: (v / n).tolist() for e, v in acc["node_a"].items()},
            "head": (acc["head"] / n).tolist(), "logits": (acc["logits"] / n).tolist(), "n": n}


def confusion_pairs(logits: np.ndarray, y: np.ndarray, top: int = 4, n_wrong: int = 3, n_near: int = 2) -> List[dict]:
    pred = logits.argmax(1)
    k = logits.shape[1]
    cnt: Dict[tuple, int] = {}
    for t, p in zip(y, pred):
        if t != p:
            key = (min(t, p), max(t, p))
            cnt[key] = cnt.get(key, 0) + 1
    pairs = sorted(cnt.items(), key=lambda kv: -kv[1])[:top]
    out = []
    for (a, b), n in pairs:
        wrong = [int(i) for i in np.flatnonzero(((y == a) & (pred == b)) | ((y == b) & (pred == a)))]
        wrong.sort(key=lambda i: -(logits[i, pred[i]] - logits[i, y[i]]))
        srt = np.sort(logits, 1)
        runner = np.argsort(-logits, 1)[:, 1]
        near = [int(i) for i in np.flatnonzero(((y == a) & (pred == a) & (runner == b)) | ((y == b) & (pred == b) & (runner == a)))]
        near.sort(key=lambda i: srt[i, -1] - srt[i, -2])
        out.append({"a": int(a), "b": int(b), "count": int(n),
                    "a_as_b": int(((y == a) & (pred == b)).sum()), "b_as_a": int(((y == b) & (pred == a)).sum()),
                    "wrong": wrong[:n_wrong], "near": near[:n_near]})
    return out
