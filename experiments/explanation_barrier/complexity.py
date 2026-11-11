"""Complexity measures of each ladder network, independent of the explanation method.

    uv run python experiments/explanation_barrier/complexity.py <task> <arch> [...pairs]

All on the training rows (z-scored inputs, the network's logit g(x)):
  * n_params, hidden units, layers;
  * hidden units that switch (active on some rows, inactive on others): strict, and with a 1%
    tolerance (a unit active on <1% or >99% of rows counts as dead / always on);
  * distinct activation patterns of all hidden units on the data (linear regions visited), and the
    number of patterns that hold 99% of the rows;
  * share of the logit's variance explained by: a linear model (OLS), an additive model (boosted
    trees with one feature per tree, i.e. a GAM) and a pairwise model (one pair of features per tree,
    a GA2M). Interaction strength = 1 - R2_additive; order-3+ residual = 1 - R2_pairwise;
  * AUC of the network and of the OLS-linear approximation of its logit (the linear gap).
For synthetic tasks also the ground truth's own R2s (the same fits to f), as a check of the dial.
Writes logs/<label>.complexity.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
from import_mlp import label_of  # noqa: E402
from train import MLP, parse  # noqa: E402


def r2_fits(X, t, seed=0):
    """Nested fits of t (in-sample, training rows), each stage boosting the previous stage's residual,
    so R2_linear <= R2_additive <= R2_pairwise <= R2_any:
      linear   OLS on the inputs;
      additive OLS plus one-hot dummies of every input with <= 5 values, then boosted trees that each
               use one input (a GAM);
      pairwise then trees that each use one pair of inputs (a GA2M);
      any      then unconstrained trees (the ceiling the tree fits reach)."""
    n, d = X.shape
    A = np.column_stack([X, np.ones(n)])
    lin = A @ np.linalg.lstsq(A, t, rcond=None)[0]
    out = {"r2_linear": float(r2_score(t, lin))}
    cols = [A]
    for i in range(d):
        vals = np.unique(X[:, i])
        if 2 < len(vals) <= 5:
            cols.append(np.column_stack([(X[:, i] == v).astype(float) for v in vals[1:]]))
    B = np.column_stack(cols)
    pred = B @ np.linalg.lstsq(B, t, rcond=None)[0]
    kw = dict(learning_rate=0.1, max_leaf_nodes=15, early_stopping=False, random_state=seed)
    stages = [("r2_additive", [[i] for i in range(d)], 600),
              ("r2_pairwise", [list(p) for p in itertools.combinations(range(d), 2)], 1000),
              ("r2_any", None, 600)]
    for name, cst, iters in stages:
        m = HistGradientBoostingRegressor(interaction_cst=cst, max_iter=iters, **kw).fit(X, t - pred)
        pred = pred + m.predict(X)
        out[name] = float(r2_score(t, pred))
    out["interaction_strength"] = 1 - out["r2_additive"]
    out["order3plus_residual"] = 1 - out["r2_pairwise"]
    return out, lin


def measure(task, arch):
    t0 = time.time()
    d = data.prepare(task)
    X, y = d["X_train"], d["y_train"]
    net = MLP(X.shape[1], parse(arch))
    net.load_state_dict(torch.load(HERE / "models" / f"{task}-{arch}.pt"))
    net.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X)
        acts = [a.numpy() for a in net.hidden_acts(Xt)]
        g = net.logit(Xt).numpy()
    H = np.concatenate(acts, axis=1)
    on = H > 0
    frac = on.mean(axis=0)
    pats, counts = np.unique(on, axis=0, return_counts=True)
    cs = np.sort(counts)[::-1]
    n99 = int(np.searchsorted(np.cumsum(cs), 0.99 * len(X)) + 1)
    used = sorted({i for i in range(X.shape[1])})
    m = dict(task=task, arch=arch, label=label_of(task, arch), n_inputs=X.shape[1], n_train=len(X),
             n_params=int(sum(p.numel() for p in net.parameters())), hidden_units=int(H.shape[1]),
             layers=len(acts),
             switching_strict=int(((frac > 0) & (frac < 1)).sum()), dead_strict=int((frac == 0).sum()),
             always_on_strict=int((frac == 1).sum()),
             switching_1pct=int(((frac > 0.01) & (frac < 0.99)).sum()),
             activation_patterns=int(len(pats)), patterns_99pct_rows=n99,
             unit_active_fraction=[float(f) for f in frac])
    fits, lin = r2_fits(X, g)
    m.update(fits)
    m["auc_network_train"] = float(roc_auc_score(y, g))
    m["auc_linear_approx_train"] = float(roc_auc_score(y, lin))
    m["linear_auc_gap"] = m["auc_network_train"] - m["auc_linear_approx_train"]
    if d["f"] is not None:
        f = d["f"][d["train_rows"]]
        gt, _ = r2_fits(X, f)
        m["ground_truth"] = dict(description=data.DESCRIPTIONS[task], logit_scale=data.SCALE[task], **gt)
        m["logit_corr_f_train"] = float(np.corrcoef(g, f)[0, 1])
    m["wall_s"] = time.time() - t0
    (HERE / "logs" / f"{m['label']}.complexity.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


if __name__ == "__main__":
    torch.set_num_threads(1)
    a = sys.argv[1:]
    for task, arch in zip(a[::2], a[1::2]):
        m = measure(task, arch)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                          if k not in ("unit_active_fraction", "ground_truth")}), flush=True)
