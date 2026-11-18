"""PRELIMINARY first look at the selected 3 vs 8 vs noise genome with the *current* explanation
pipeline (explaneat.conv.explain.multiclass, the 10-class MNIST pipeline run with k = 3), plus
per-class and per-noise-type profiles and the shape concepts (explain/shape_concepts.py).

  uv run python experiments/conv_neat/three8noise/explain.py [results/run_s0]

Everything is computed on VALIDATION images (the test split was read once by run.py; it is not
touched here). The pipeline's own one-line readings name classes as digits "0", "1", "2"; here
class 0 = '3', 1 = '8', 2 = 'noise', so this script writes its own table and does not reuse them.
The full structure-first explanation will be rerun with the new pipeline when it lands.

Outputs in the run directory: first_look.json, graph.png, profiles.png.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from data import CLASS_NAMES, NOISE_TYPES, load  # noqa: E402
from explaneat.conv.explain import importance as IM  # noqa: E402
from explaneat.conv.explain import shape_concepts as SC  # noqa: E402
from explaneat.conv.explain.engine import ExplainNet, depth_map, functional_nodes  # noqa: E402
from explaneat.conv.explain.multiclass import explain_network_multi  # noqa: E402
from explaneat.conv.genome import ConvNeatConfig, topo_order  # noqa: E402
from explaneat.conv.population import genome_from_dict  # noqa: E402
from explaneat.conv.explain.strokes import STROKES10, stroke_concepts  # noqa: E402

GROUPS = ["3", "8"] + NOISE_TYPES


def auc_s(v, c):
    from sklearn.metrics import roc_auc_score
    c = np.asarray(c).astype(bool)
    if c.all() or (~c).all():
        return None
    return float(2 * roc_auc_score(c, v) - 1)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "run_s0")
    conf = json.load(open(os.path.join(run, "config.json")))
    cd = dict(conf["config"])
    cd["image_hw"], cd["kernel_sizes"] = tuple(cd["image_hw"]), tuple(cd["kernel_sizes"])
    cfg = ConvNeatConfig(**cd)
    g = genome_from_dict(json.load(open(os.path.join(run, "selected_genome.json"))))
    data = load(conf["args"]["seed"])
    data_v = {k: v for k, v in data.items() if not k.endswith("_test")}  # the pipeline never sees test
    data_v["X_test"], data_v["y_test"] = data["X_val"][:0], data["y_val"][:0]
    n_ex = int(os.environ.get("N_EXPLAIN", 1500))
    E = explain_network_multi(g, cfg, data_v, n_explain=n_ex, n_imp=len(data["y_val"]) - n_ex, seed=0,
                              eval_split="val")
    rows = np.asarray(E["explain_rows"])
    X, y, t = data["X_val"][rows], data["y_val"][rows], data["type_val"][rows]
    grp = np.where(y == 0, "3", np.where(y == 1, "8", np.array(NOISE_TYPES, dtype=object)[np.maximum(t, 0)]))
    net = ExplainNet(g, cfg)
    fwd = net.run(X)
    pred = fwd.logits.numpy().argmax(1)
    fn = functional_nodes(g)
    depth = depth_map(g, fn)
    order = [n for n in topo_order(g) if n in fn and g.nodes[n].kind != "input"]
    shapes = SC.shape_table(X)
    strokes = stroke_concepts(X, STROKES10)
    is_dig = y < 2

    # per-subgroup ablation: mean-ablate each node (means over all explain rows), score on digits / noise
    imp_all = IM.ablation_importance(net, X, y, 3, order, [])
    base_mean = {k: m.mean(0, keepdim=True) for k, m in fwd.maps.items()}
    sub_eff = {}
    for n in order:
        p2 = net.run(X, node_override={n: base_mean[n]}).logits.numpy().argmax(1)
        sub_eff[n] = {"d_digit_acc": float((p2[is_dig] == y[is_dig]).mean() - (pred[is_dig] == y[is_dig]).mean()),
                      "d_noise_recall": float((p2[~is_dig] == 2).mean() - (pred[~is_dig] == 2).mean()),
                      **{f"d_recall_{nt}": float((p2[t == i] == 2).mean() - (pred[t == i] == 2).mean())
                         for i, nt in enumerate(NOISE_TYPES)}}

    HW = np.asarray(g.head_w, np.float64)
    outs = g.output_keys()
    nodes = {}
    for n in order:
        m = fwd.maps[n][:, 0].numpy().reshape(len(y), -1)
        pool = m.max(1) if g.nodes[n].kind == "hidden" else m.mean(1)
        prof = {gname: float(pool[grp == gname].mean()) for gname in GROUPS}
        r = {"kind": g.nodes[n].kind, "level": int(g.nodes[n].level), "depth": depth.get(n),
             "pooling": "max" if g.nodes[n].kind == "hidden" else "mean",
             "profile_mean_pooled": prof,
             "s_noise_vs_digit": auc_s(pool, ~is_dig),
             "s_8_vs_3_within_digits": auc_s(pool[is_dig], y[is_dig] == 1),
             "s_shuffle_vs_digits": auc_s(pool[(t == 0) | is_dig], t[(t == 0) | is_dig] == 0),
             "s_patch_vs_digits": auc_s(pool[(t == 3) | is_dig], t[(t == 3) | is_dig] == 3),
             "shape_s_all": {c: auc_s(pool, shapes[c]) for c in SC.BINARY},
             "shape_s_within_digits": {c: auc_s(pool[is_dig], shapes[c][is_dig]) for c in SC.BINARY},
             "stroke_s_within_digits": {c: auc_s(pool[is_dig], strokes[is_dig, j]) for j, c in enumerate(STROKES10)},
             "importance_all": imp_all["nodes"].get(n), "ablation_by_subgroup": sub_eff[n],
             "in_edges": [{"src": c.src, "k": c.k, "sum": float(np.sum(c.kernel))} for c in g.enabled_conns()
                          if c.dst == n and c.src in fn]}
        if g.nodes[n].kind == "output":
            r["head_weights"] = dict(zip(CLASS_NAMES, HW[:, outs.index(n)].round(3).tolist()))
        # one-line plain reading (computed, labelled preliminary)
        hi = max(prof, key=prof.get)
        lo = min(prof, key=prof.get)
        r["reading"] = (f"pooled value highest on {hi}, lowest on {lo}; noise vs digit s = {r['s_noise_vs_digit']:+.2f}, "
                        f"8 vs 3 s = {r['s_8_vs_3_within_digits']:+.2f}")
        nodes[n] = r

    from sklearn.metrics import confusion_matrix
    res = {"label": "PRELIMINARY first look (current pipeline; validation images only)",
           "n_rows": int(len(y)), "val_acc_explain_rows": float((pred == y).mean()),
           "confusion_val_explain_rows": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
           "noise_recall_by_type_val": {nt: float((pred[t == i] == 2).mean()) for i, nt in enumerate(NOISE_TYPES)},
           "shape_concept_rates_by_group": {gname: {c: float(shapes[c][grp == gname].mean()) for c in SC.BINARY}
                                            for gname in GROUPS},
           "shape_numeric_means_by_group": {gname: {c: float(shapes[c][grp == gname].mean()) for c in SC.NUMERIC}
                                            for gname in GROUPS},
           "pipeline": {"size": E["size"], "check": E["check"], "importance_base": E["importance_base"],
                        "parts": [{"name": p["name"], "nodes": p["nodes"], "entries": p["entries"], "exits": p["exits"],
                                   "importance": p["importance"], "formula_text": p["formula_text"]} for p in E["parts"]],
                        "root_text": E["root"]["text"]},
           "nodes": {str(n): r for n, r in nodes.items()}}
    with open(os.path.join(run, "first_look.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    figures(g, cfg, fn, depth, order, nodes, run)
    print(json.dumps({k: res[k] for k in ("val_acc_explain_rows", "confusion_val_explain_rows",
                                          "noise_recall_by_type_val")}))
    for n, r in nodes.items():
        print(f"h{n}" if r["kind"] == "hidden" else f"out{n}", r["level"], r["reading"],
              {k: round(v, 3) for k, v in r["ablation_by_subgroup"].items()})


def figures(g, cfg, fn, depth, order, nodes, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # profiles heatmap: rows = nodes, columns = groups, values min-max scaled per row
    labels = [(f"h{n}" if nodes[n]["kind"] == "hidden" else f"out{n}") + f" L{nodes[n]['level']}" for n in order]
    P = np.array([[nodes[n]["profile_mean_pooled"][gname] for gname in GROUPS] for n in order])
    lo, hi = P.min(1, keepdims=True), P.max(1, keepdims=True)
    Pn = (P - lo) / np.maximum(hi - lo, 1e-12)
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(order) + 1.2))
    ax.imshow(Pn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(GROUPS)), GROUPS)
    ax.set_yticks(range(len(order)), labels, fontsize=7)
    ax.set_title("PRELIMINARY: mean pooled value per group (row-scaled; val)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(run, "profiles.png"), dpi=120)
    plt.close(fig)
    # graph: x = depth, colour = group with the highest mean pooled value
    cols = {"3": "#1f77b4", "8": "#ff7f0e", "shuffle": "#2ca02c", "scribble": "#d62728", "blob": "#9467bd",
            "patch": "#8c564b"}
    nodes_all = [n for n in topo_order(g) if n in fn]
    by_d = {}
    for n in nodes_all:
        by_d.setdefault(depth.get(n, 0), []).append(n)
    pos = {}
    for d_, ns in by_d.items():
        for i, n in enumerate(sorted(ns, key=lambda n: (g.nodes[n].kind, n))):
            pos[n] = (d_, i - (len(ns) - 1) / 2)
    fig, ax = plt.subplots(figsize=(2.2 * (max(by_d) + 1) + 2, 0.7 * max(len(v) for v in by_d.values()) + 2))
    for c in g.enabled_conns():
        if c.src in pos and c.dst in pos:
            (x0, y0), (x1, y1) = pos[c.src], pos[c.dst]
            s = float(np.sum(c.kernel))
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.4 + 1.5 * min(abs(s), 2),
                                                                 color="#c0392b" if s > 0 else "#2471a3", alpha=0.6))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{c.k}", fontsize=5, color="gray")
    for n, (x_, y_) in pos.items():
        nd = g.nodes[n]
        if nd.kind == "input":
            fc, lab = "#dddddd", "image"
        else:
            prof = nodes[n]["profile_mean_pooled"]
            fc = cols[max(prof, key=prof.get)]
            lab = (f"h{n}" if nd.kind == "hidden" else f"out{n}") + f"\nL{nd.level}"
        ax.scatter([x_], [y_], s=520, c=fc, edgecolors="k", zorder=3)
        ax.text(x_, y_, lab, ha="center", va="center", fontsize=6, zorder=4, color="white" if nd.kind != "input" else "k")
    for k_, v in cols.items():
        ax.scatter([], [], c=v, label=f"highest on {k_}")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("PRELIMINARY graph (functional nodes; edge label = kernel size; red = positive kernel sum)", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(run, "graph.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
