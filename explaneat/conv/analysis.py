"""Analysis for evolved ConvNEAT genomes: graph rendering, per-node feature
evidence (kernels, mean activation maps, top-activating inputs) and the
concept-concentration probe for the synthetic task.

Concept probe
-------------
For each non-input node, pool its post-activation map over space (spatial max
by default: concepts are planted "anywhere", so presence shows as a peak; the
global average used by the head is reported too) and compute, for each planted
concept c, the ROC AUC of the pooled activation for "c present" vs "c absent".
Selectivity s_c = |2*AUC_c - 1| in [0, 1] (0 = no information; the sign of
2*AUC-1 says excitatory or inhibitory). Concepts are planted independently, so
these are not confounded with each other.

Per node:
  strong(c)    := s_c >= STRONG (0.5, i.e. AUC >= 0.75 or <= 0.25)
  none         : no concept strong
  mono         : exactly one concept strong
  conjunctive  : exactly {hbar, blob} strong and the node is at least as
                 selective for the label (hbar AND blob) as for either alone,
                 i.e. it computes the task conjunction (not superposition)
  poly         : two or more concepts strong otherwise (incl. any distractor)
  concentration: max_c s_c / sum_c s_c over the four base concepts
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .genome import ConvGenome, ConvNeatConfig  # noqa: E402
from .network import build_engine  # noqa: E402

STRONG = 0.5
LEVEL_COLOURS = ["#cfe8ff", "#ffe2b8", "#d9f2d0", "#f3d0f0"]


def functional_nodes(g: ConvGenome) -> set:
    """Nodes on some enabled input -> output path (the rest cannot matter)."""
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


def render_graph(g: ConvGenome, cfg: ConvNeatConfig, path: str, probe: Optional[dict] = None) -> str:
    import graphviz
    live = functional_nodes(g)
    dot = graphviz.Digraph(format="png", graph_attr={"rankdir": "LR", "dpi": "150"})
    for k, n in sorted(g.nodes.items()):
        h, w = cfg.level_size(n.level)
        label = f"{'in' if n.kind == 'input' else 'out' if n.kind == 'output' else 'h'}{k}\\nr={n.level} ({h}x{w})"
        if probe and k in probe:
            label += f"\\n{probe[k]['category']}"
            if probe[k]["category"] in ("mono",):
                label += f":{probe[k]['top_concept']}"
        dot.node(str(k), label, style="filled" if k in live else "dashed",
                 fillcolor=LEVEL_COLOURS[min(n.level, 3)] if k in live else "white",
                 shape="box" if n.kind != "hidden" else "ellipse",
                 fontcolor="black" if k in live else "grey")
    for c in g.enabled_conns():
        norm = float(np.abs(c.kernel).sum())
        dot.edge(str(c.src), str(c.dst), label=f"k={c.k}",
                 penwidth=str(0.5 + min(3.0, norm)),
                 color="black" if (c.src in live and c.dst in live) else "grey")
    base, _ = os.path.splitext(path)
    dot.save(base + ".gv")
    dot.render(base, cleanup=True)
    return base + ".png"


def render_kernels(g: ConvGenome, path: str) -> None:
    conns = sorted(g.enabled_conns(), key=lambda c: (c.src, c.dst))
    if not conns:
        return
    n = len(conns)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, c in zip(axes.flat, conns):
        v = float(np.abs(c.kernel).max()) or 1.0
        ax.imshow(c.kernel, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{c.src}->{c.dst} k={c.k}\n|max|={v:.2f}", fontsize=7)
    fig.suptitle("Edge kernels (red +, blue -; each panel own scale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def node_maps(g: ConvGenome, cfg: ConvNeatConfig, X: np.ndarray, batch: int = 512) -> Dict[int, np.ndarray]:
    net = build_engine(g, cfg).eval()  # PropNEAT engine: maps of kept nodes
    out: Dict[int, List[np.ndarray]] = {}
    for i in range(0, len(X), batch):
        maps = net.forward_maps(torch.tensor(X[i:i + batch]))
        for k, m in maps.items():
            out.setdefault(k, []).append(m[:, 0].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def render_mean_maps(g: ConvGenome, maps: Dict[int, np.ndarray], path: str) -> None:
    keys = [k for k in sorted(maps) if g.nodes[k].kind != "input"]
    keys = [k for k in g.input_keys() if k in maps] + keys
    cols = min(8, len(keys))
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, keys):
        m = maps[k].mean(0)
        ax.imshow(m, cmap="viridis")
        ax.set_title(f"node {k} (r={g.nodes[k].level})\nmean={m.mean():.2f}", fontsize=7)
    fig.suptitle("Mean activation map per node (evaluation set)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_top_activating(g: ConvGenome, maps: Dict[int, np.ndarray], X: np.ndarray, path: str,
                          n_top: int = 6, keys: Optional[List[int]] = None) -> None:
    keys = keys or [k for k in sorted(maps) if g.nodes[k].kind != "input"]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), n_top, figsize=(1.2 * n_top, 1.3 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        pooled = maps[k].reshape(len(X), -1).max(1)
        top = np.argsort(-pooled)[:n_top]
        for j, idx in enumerate(top):
            ax = axes[r, j]
            ax.imshow(X[idx, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"node {k}", fontsize=7)
            ax.set_title(f"{pooled[idx]:.2f}", fontsize=6)
    fig.suptitle("Top-activating inputs per node (spatial-max pooled activation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _auc(a: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(a) < 1e-9 or y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, a))


def categorise(signed: Dict[str, float], label_sel: float, tau: float = STRONG):
    """Category of one node from its signed selectivities (see module docstring)."""
    sel = {c: abs(v) for c, v in signed.items()}
    strong = [c for c in signed if sel[c] >= tau]
    if not strong:
        return "none", strong
    if len(strong) == 1:
        return "mono", strong
    if set(strong) == {"hbar", "blob"} and label_sel >= max(sel["hbar"], sel["blob"]):
        return "conjunctive", strong
    return "poly", strong


def concept_probe(g: ConvGenome, maps: Dict[int, np.ndarray], C: np.ndarray, label: np.ndarray,
                  names: List[str], pooling: str = "max") -> Dict[int, dict]:
    live = functional_nodes(g)
    res = {}
    for k in sorted(maps):
        if g.nodes[k].kind == "input":
            continue
        m = maps[k].reshape(len(C), -1)
        a = m.max(1) if pooling == "max" else m.mean(1)
        signed = {c: 2 * _auc(a, C[:, j]) - 1 for j, c in enumerate(names)}
        sel = {c: abs(v) for c, v in signed.items()}
        lab = abs(2 * _auc(a, label) - 1)
        cat, strong = categorise(signed, lab, STRONG)
        tot = sum(sel.values())
        res[k] = {"kind": g.nodes[k].kind, "level": g.nodes[k].level, "functional": k in live,
                  "signed_selectivity": signed, "label_selectivity": lab,
                  "strong": strong, "category": cat,
                  "top_concept": max(sel, key=sel.get),
                  "concentration": (max(sel.values()) / tot) if tot > 1e-9 else 0.0,
                  "constant": bool(np.ptp(a) < 1e-9)}
    return res


def render_probe(probe: Dict[int, dict], names: List[str], path: str, title: str) -> None:
    keys = [k for k in probe if probe[k]["functional"]]
    if not keys:
        return
    cols = names + ["label"]
    M = np.array([[probe[k]["signed_selectivity"][c] for c in names] + [probe[k]["label_selectivity"]]
                  for k in keys])
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 0.38 * len(keys) + 1.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols[:-1] + ["label |s|"], fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{probe[k]['kind'][0]}{k} r{probe[k]['level']}: {probe[k]['category']}" for k in keys],
                       fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="2*AUC-1 (selectivity, signed)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
