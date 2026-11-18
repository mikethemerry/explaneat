"""Structural check of the preliminary 3 vs 8 vs noise reading (validation images only).

Reading under test (three8noise/RESULTS.md): a full-resolution "scattered ink" map (h3) and a coarse
7x7 "ink spread" map (h31) both vote for noise against a shared "digit strokes" map (h5), and a
closed-loop map (h14) separates 3 from 8. For every hidden map: its level, its annotation in the
record, the mechanism test on the map itself (shape + mechanism concepts), and its exact direct
term into each class score (head weight x GAP of each kernel edge into the score maps), as the
mean margin towards each class on validation images.
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "conv_neat", "three8noise"))
import data as D  # noqa: E402

from explaneat.conv.explain.multiclass import stratified_subset  # noqa: E402
from explaneat.conv.explain.structured import within_test  # noqa: E402
from explaneat.conv.hypotheses import HypothesisContext  # noqa: E402
from explaneat.conv.model_state import load_record  # noqa: E402
from explaneat.conv.network import resample  # noqa: E402

eng, g, cfg, rec = load_record(os.path.join(ROOT, "experiments", "conv_neat", "explanations", "three8noise_run_s0.json"))
st = eng.state
d = D.load(0)
te = stratified_subset(d["y_val"], 2000, 0)
X, y = d["X_val"][te], d["y_val"][te]
ctx = HypothesisContext(st, X, y, 3, {}, 0, class_names=["3", "8", "noise"], shape=True)
lib = ctx.lib
maps = sorted({int(n.split("_")[0]) for n in st.node_ids() if st.node(n).activation == "relu"})
leaf_of = {}
for a in eng.annotations:
    if a.name != "root" and not eng.children_of(a.name):
        for n in a.subgraph_nodes:
            leaf_of[n.split("_")[0]] = a.name
outs = st.structure.output_node_ids
HW = np.asarray(st.head_w)
rows = []
for m in maps:
    node = next(n for n in st.node_ids() if n.split("_")[0] == str(m) and st.node(n).activation == "relu")
    p = ctx.pooled(node)
    best = None
    for pool, v in p.items():
        for c in lib["strokes"]:
            t = within_test(v, lib["C"][:, lib["names"].index(c)], y, lib["counts"]["ink"], 3)
            if best is None or abs(t["s"]) > abs(best[2]["s"]):
                best = (pool, c, t)
    # direct terms into the class scores from this map's copies
    T = np.zeros((len(X), 3))
    for c in st.structure.connections:
        if c.to_node in outs and c.from_node.split("_")[0] == str(m):
            lvl = st.level(c.to_node)
            r = resample(ctx.maps[c.from_node] if c.from_node in ctx.maps else st.run(X, keep={c.from_node})[0][c.from_node],
                         st.level(c.from_node), lvl, cfg.level_size(lvl))
            k = st.kernels[c.innovation]
            z = F.conv2d(r, torch.as_tensor(k).view(1, 1, *k.shape), padding=k.shape[0] // 2).mean(dim=(1, 2, 3)).numpy()
            T += z[:, None] * HW[:, outs.index(c.to_node)][None, :]
    marg = T - T.mean(1, keepdims=True)
    push = [float(marg[y == j, j].mean()) for j in range(3)]
    srcs = sorted({u.split("_")[0] for n in st.node_ids() if n.split("_")[0] == str(m) for u in st.pred()[n]})
    srcs = sorted({(st.pred()[u][0].split("_")[0] if st.meta.get(u, {}).get("identity") == "source" else u.split("_")[0]) for u in
                   [q for n in st.node_ids() if n.split("_")[0] == str(m) for q in st.pred()[n]]})
    rows.append({"map": f"h{m}", "level": st.level(str(m)) if str(m) in st.levels else st.level(node),
                 "size": "×".join(map(str, cfg.level_size(st.level(node)))), "leaf": leaf_of.get(str(m)),
                 "inputs": ["x" if s_ == "-1" else "h" + s_ for s_ in srcs],
                 "best_concept": f"{best[1]} ({best[0]})", "s": round(best[2]["s"], 2),
                 "within": None if best[2]["within_mean"] is None else round(best[2]["within_mean"], 2),
                 "push_3_8_noise": [round(v, 2) for v in push]})
for r in rows:
    print(r)
json.dump(rows, open(os.path.join(HERE, "three8noise_maps.json"), "w"), indent=1)
