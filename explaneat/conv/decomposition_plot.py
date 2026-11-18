"""Static check graphs for a conv decomposition (matplotlib, PNG).

* ``plot_collapsed``: every top-level structure collapsed to one function node; edges between
  them are the junction edges (counted), from the image to the structures and from the
  structures to the score maps.
* ``plot_expanded``: one structure opened: its entries (ports, copies of outside maps) on the
  left, sub-parts as coloured boxes, exits outlined, outside sources and consumers in grey.
  Edge colours: inside a sub-part (black), between sub-parts (blue), into an entry (green), out of
  an exit (orange), anything else crossing the boundary (red: there should be none).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .model_state import ConvModelStateEngine, _id_key  # noqa: E402

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D", "#BAB0AC"]


def _depths(nodes: List[str], edges: List[Tuple[str, str]]) -> Dict[str, int]:
    pred = {n: [] for n in nodes}
    for a, b in edges:
        if a in pred and b in pred:
            pred[b].append(a)
    d: Dict[str, int] = {}

    def depth(n, stack=()):
        if n in d:
            return d[n]
        d[n] = 0 if not pred[n] else 1 + max(depth(p) for p in pred[n])
        return d[n]
    for n in nodes:
        depth(n)
    return d


def _layout(nodes: List[str], edges, depth: Optional[Dict[str, int]] = None, xgap=3.0, ygap=1.2):
    depth = depth or _depths(nodes, edges)
    cols: Dict[int, List[str]] = {}
    for n in nodes:
        cols.setdefault(depth[n], []).append(n)
    pos = {}
    for c, ns in cols.items():
        ns = sorted(ns, key=_id_key)
        for i, n in enumerate(ns):
            pos[n] = (c * xgap, (i - (len(ns) - 1) / 2) * ygap)
    return pos


def _arrow(ax, p, q, color, lw=1.0, alpha=0.9, label=None, rad=0.08):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                                     shrinkA=14, shrinkB=14, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, label, fontsize=7, color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


def top_owner(eng: ConvModelStateEngine) -> Dict[str, str]:
    """node -> its top-level structure (child of the root); inputs -> 'image', outputs -> 'scores'."""
    st = eng.state
    own = {n: "image" for n in st.structure.input_node_ids}
    own.update({n: "scores" for n in st.structure.output_node_ids})
    for t in eng.children_of("root"):
        for n in eng.expanded_nodes(t):
            own[n] = t
    return own


def plot_collapsed(eng: ConvModelStateEngine, path: str, title: str = "") -> dict:
    st = eng.state
    own = top_owner(eng)
    tops = eng.children_of("root")
    nodes = ["image"] + tops + ["scores"]
    q: Dict[Tuple[str, str], int] = {}
    for a, b in st.edges():
        oa, ob = own[a], own[b]
        if oa != ob:
            q[(oa, ob)] = q.get((oa, ob), 0) + 1
    # structures in topological order, staggered above and below the image -> scores axis
    pos = {"image": (0.0, 0.0), "scores": (3.2 * (len(tops) + 1), 0.0)}
    for i, t in enumerate(tops):
        pos[t] = (3.2 * (i + 1), (2.3 if i % 2 == 0 else -2.3))
    fig, ax = plt.subplots(figsize=(17, 7))
    for (a, b), n in q.items():
        col = "#555555" if a != "image" else "#9a9a9a"
        _arrow(ax, pos[a], pos[b], col, lw=0.6 + 0.25 * n, label=str(n), rad=0.18 if a == "image" or b == "scores" else 0.08)
    for i, nm in enumerate(nodes):
        x, y = pos[nm]
        if nm in ("image", "scores"):
            lab = "image x" if nm == "image" else f"{len(st.structure.output_node_ids)} score maps\n→ GAP → head"
            fc, ec = "#f0f0f0", "#666666"
        else:
            a = eng.annotation(nm)
            n_maps = len({n.split("_")[0] for n in eng.expanded_nodes(nm) if st.node(n).activation == "relu"})
            t = (a.display_name or nm)
            t = t if len(t) <= 28 else t[:26] + "…"
            lab = f"{nm}  ({n_maps} maps)\n{t}\n{len(a.entry_nodes)} in · {len(a.exit_nodes)} out"
            fc, ec = PALETTE[tops.index(nm) % len(PALETTE)] + "33", PALETTE[tops.index(nm) % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.42), 2.6, 0.84, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.6,
                                    zorder=2))
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.5, zorder=3)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 2.0, max(ys) + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or "Collapsed structures (each a function node; numbers = kernel edges between them)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"nodes": nodes, "quotient_edges": {f"{a}->{b}": n for (a, b), n in q.items()}}


def plot_expanded(eng: ConvModelStateEngine, name: str, path: str, title: str = "") -> dict:
    st = eng.state
    a = eng.annotation(name)
    inside = eng.expanded_nodes(name)
    kids = eng.children_of(name) or [name]
    sub_of = {}
    for k in kids:
        for n in eng.expanded_nodes(k):
            sub_of[n] = k
    entries, exits = set(a.entry_nodes), set(a.exit_nodes)
    own = top_owner(eng)
    edges = st.edges()
    ext_src = sorted({u for u, v in edges if v in inside and u not in inside}, key=_id_key)
    ext_dst_raw = [(u, v) for u, v in edges if u in inside and v not in inside]
    # outside consumers grouped by their owner (another structure or the score maps)
    dst_group = {v: f"→ {own.get(v, v)}" for _, v in ext_dst_raw}
    groups = sorted(set(dst_group.values()))
    nodes = ext_src + sorted(inside, key=_id_key) + groups
    e2 = [(u, v) for u, v in edges if u in nodes and v in nodes] + [(u, dst_group[v]) for u, v in ext_dst_raw]
    d = _depths(nodes, e2)
    # one horizontal band per sub-part (so the boxes do not overlap); x = depth in the structure
    pos, y0 = {}, 0.0
    maxd = max(d.values())
    for k in kids:
        mem = [n for n in sorted(inside, key=_id_key) if sub_of.get(n) == k]
        cols: Dict[int, List[str]] = {}
        for n in mem:
            cols.setdefault(d[n], []).append(n)
        h = max((len(v) for v in cols.values()), default=1)
        for c, ns in cols.items():
            for i, n in enumerate(ns):
                pos[n] = (c * 2.2, -(y0 + i * 0.9 + (h - len(ns)) * 0.45))
        y0 += h * 0.9 + 1.0
    mid = -(y0 - 1.0) / 2
    for i, n in enumerate(ext_src):
        pos[n] = (-2.2, mid + (i - (len(ext_src) - 1) / 2) * 1.2)
    for i, n in enumerate(groups):
        pos[n] = ((maxd + 1) * 2.2, mid + (i - (len(groups) - 1) / 2) * 1.2)
    fig, ax = plt.subplots(figsize=(15, 9))
    # sub-part boxes
    for i, k in enumerate(kids):
        ps = [pos[n] for n in inside if sub_of.get(n) == k]
        if not ps:
            continue
        x0, x1 = min(p[0] for p in ps) - 0.7, max(p[0] for p in ps) + 0.7
        y0, y1 = min(p[1] for p in ps) - 0.45, max(p[1] for p in ps) + 0.45
        col = PALETTE[i % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.05", fc=col + "1a", ec=col, lw=1.2))
        ka = eng.annotation(k)
        t = ka.display_name or k
        ax.text(x0 + 0.05, y1 + 0.08, f"{k}: {t if len(t) <= 34 else t[:32] + '…'}", fontsize=7, color=col, va="bottom")
    counts = {"inside": 0, "between_subparts": 0, "into_entry": 0, "out_of_exit": 0, "unclean": 0}
    for u, v in e2:
        if u in inside and v in inside:
            if sub_of.get(u) == sub_of.get(v):
                col, key = "#222222", "inside"
            else:
                col, key = "#1f5fbf", "between_subparts"
        elif v in inside:
            col, key = ("#2e8b57", "into_entry") if v in entries else ("#d62728", "unclean")
        else:
            col, key = ("#e67e22", "out_of_exit") if u in exits else ("#d62728", "unclean")
        counts[key] += 1
        _arrow(ax, pos[u], pos[v], col, lw=1.0 if key != "unclean" else 2.0, rad=0.06)
    for n in nodes:
        x, y = pos[n]
        if n in inside:
            nd = st.node(n)
            meta = st.meta.get(n, {})
            if meta.get("identity") == "source":
                lab, marker, fc = f"{n}\n={meta['of']}", "D", "#c7e9c0"
            else:
                lab, marker, fc = n, "o", "#ffffff" if nd.activation == "relu" else "#eeeeee"
            ax.scatter([x], [y], s=520, marker=marker, c=fc, edgecolors="#c0392b" if n in exits else "#333333",
                       linewidths=2.4 if n in exits else 1.0, zorder=3)
            ax.text(x, y, lab, ha="center", va="center", fontsize=6.2, zorder=4)
        else:
            ax.scatter([x], [y], s=420, marker="s", c="#e5e5e5", edgecolors="#999999", zorder=3)
            ax.text(x, y, n if not n.startswith("→") else n, ha="center", va="center", fontsize=6.2, color="#555555",
                    zorder=4)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.2)
    ax.axis("off")
    ax.set_title(title or (f"{name} expanded: diamonds = entries (ports), red outline = exits, boxes = sub-parts; "
                           f"green into entries, orange out of exits, blue between sub-parts, red = unclean "
                           f"({counts['unclean']})"), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return counts
