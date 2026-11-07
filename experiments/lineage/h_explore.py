"""Study H exploratory descriptives (NOT pre-registered; labelled as such in RESULTS_H.md).

E1: share of the total lineage train-AUC gain contributed by jump steps (tau95), other
    child (structural) steps, and elite (backprop-only) steps; test-AUC alongside.
E2: among child-structural steps, P(jump) when the step adds a node vs when it does not,
    with a within-lineage permutation of the node-added flag (10,000, seed 42), and the
    same restricted to generations > 10.
Writes h_explore.csv.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
st = pd.read_csv(HERE / "h_steps.csv")
rows = []
for s, d in st.groupby("set"):
    tot = d.groupby("seed").delta.sum()
    tot_test = d.groupby("seed").delta_test.sum()
    for name, m in [("jump_q95", d.jump_q95 == 1),
                    ("child_nonjump", (d.elite == 0) & (d.jump_q95 == 0)),
                    ("elite", d.elite == 1)]:
        share = d[m].groupby("seed").delta.sum().reindex(tot.index, fill_value=0) / tot
        share_t = d[m].groupby("seed").delta_test.sum().reindex(tot.index, fill_value=0) / tot_test
        rows.append(dict(set=s, analysis="E1_gain_share", group=name, n_steps=int(m.sum()),
                         median_share_train=float(share.median()),
                         min_share_train=float(share.min()), max_share_train=float(share.max()),
                         median_share_test=float(share_t.median())))
    rng = np.random.default_rng(42)
    for lab, dd in [("all", d), ("gen11plus", d[d.generation > 10])]:
        c = dd[dd.step_type == "child-structural"].copy()
        c["node_added"] = (c.n_added_nodes > 0).astype(int)
        def rd(x):
            a = x[x.node_added == 1].jump_q95.mean(); b = x[x.node_added == 0].jump_q95.mean()
            return a - b, a, b
        obs, a, b = rd(c)
        perm = []
        grp = [g.index for _, g in c.groupby("seed")]
        for _ in range(10_000):
            cc = c.copy()
            for idx in grp:
                cc.loc[idx, "node_added"] = rng.permutation(cc.loc[idx, "node_added"].values)
            perm.append(rd(cc)[0])
        perm = np.array(perm)
        rows.append(dict(set=s, analysis=f"E2_node_added_{lab}", group="child-structural",
                         n_steps=len(c), n_node_added=int(c.node_added.sum()),
                         p_jump_node_added=a, p_jump_no_node=b, RD=obs,
                         p_perm_one_sided=float((1 + (perm >= obs).sum()) / 10_001)))
pd.DataFrame(rows).to_csv(HERE / "h_explore.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
