"""Does the learnt non-linearity match threats, parity or lines? (training split only)

For one run's best genome (in memory: phenotype -> StructureNetwork, the same function the DB
tools see), on the 54,045 training rows:

1. linear part: AUC of the network logit vs the logit with every hidden ReLU replaced by the
   identity (the network made linear), and vs an ordinal LR refit to the network's own logit;
2. the non-linear residual r = logit - (best linear fit of the logit on the 42 cells): how much
   of r the c4_features (line, threat, parity, centre counts) explain (R^2 of a linear fit), and
   its correlation with each feature;
3. per hidden unit: active share, the correlation of its output with each feature, and the
   feature that best explains it (R^2 of cells alone vs cells + features for that unit);
4. tactical blind spot check (as complex_real): positions where o / x has a threat
   (three in an open line), outcome rate vs mean predicted.

    uv run python experiments/c4_capacity/domain_check.py <run name>   # -> logs/<name>.domain.json
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import neat
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "complex_real"))
sys.path.insert(0, str(HERE))
from c4_features import features  # noqa: E402
from data import prepare  # noqa: E402
from explaneat.core.explaneat import ExplaNEAT  # noqa: E402
from explaneat.core.structure_network import StructureNetwork  # noqa: E402

name = sys.argv[1]
RUNS = Path.home() / ".explaneat" / "c4_capacity_runs"
with gzip.open(RUNS / name / "best_genome.pkl.gz", "rb") as f:
    g = pickle.load(f)["genome"]
cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                  neat.DefaultStagnation, str(HERE / "configs" / f"{name}.ini"))
d = prepare("c4")
Xz, y = d["X_train"], d["y_train"]
F_all, fn_all = features(d["X_raw"][d["train_rows"]])
keep = F_all.std(0) > 0
F, fnames = F_all[:, keep], [n for n, k in zip(fn_all, keep) if k]

pheno = ExplaNEAT(g, cfg).get_phenotype_network()
ins, outs = list(pheno.input_node_ids), list(pheno.output_node_ids)
out_id = outs[0]
W = {(c.from_node, c.to_node): c.weight for c in pheno.connections if c.enabled}
bias = {n.id: (n.bias or 0.0) for n in pheno.nodes}
sn = StructureNetwork(pheno)
p = sn.forward(torch.as_tensor(Xz, dtype=torch.float64)).detach().numpy().ravel()
logit = np.log(p / (1 - p))
hidden = [h for h in sn.node_info if h not in set(ins) | set(outs)]
H = {h: np.asarray(sn.get_node_activation(h)).ravel() for h in hidden}


def preact(h):
    # pre-activation of a hidden node from its inputs (inputs: z column; hidden: its post-ReLU value)
    col = {nid: i for i, nid in enumerate(pheno.input_node_ids)}
    s = np.full(len(y), bias[h])
    for (a, b), w in W.items():
        if b == h:
            s += w * (Xz[:, col[a]] if a in col else H[a])
    return s


# 1. the network made linear: every hidden unit's ReLU replaced by identity (depth-ordered)
order = sorted(hidden, key=lambda h: sn.node_info[h]["depth"])
col = {nid: i for i, nid in enumerate(pheno.input_node_ids)}
Lin = {}
for h in order:
    s = np.full(len(y), bias[h])
    for (a, b), w in W.items():
        if b == h:
            s += w * (Xz[:, col[a]] if a in col else Lin[a])
    Lin[h] = s
lin_logit = np.full(len(y), bias[out_id])
for (a, b), w in W.items():
    if b == out_id:
        lin_logit += w * (Xz[:, col[a]] if a in col else Lin[a])
lr_cells = LinearRegression().fit(Xz, logit)
lin_fit = lr_cells.predict(Xz)
resid = logit - lin_fit
lr_feat = LinearRegression().fit(F, resid)
res = dict(
    name=name, n=int(len(y)),
    auc_network=float(roc_auc_score(y, logit)),
    auc_network_made_linear=float(roc_auc_score(y, lin_logit)),
    auc_best_linear_fit_of_logit=float(roc_auc_score(y, lin_fit)),
    r2_logit_on_cells=float(lr_cells.score(Xz, logit)),
    residual_sd=float(resid.std()), logit_sd=float(logit.std()),
    r2_residual_on_features=float(lr_feat.score(F, resid)),
    r2_logit_on_cells_plus_features=float(LinearRegression().fit(np.c_[Xz, F], logit).score(np.c_[Xz, F], logit)),
    residual_corr={n: float(np.corrcoef(F[:, i], resid)[0, 1]) for i, n in enumerate(fnames)},
)
# outcome vs features beyond cells (what the true function needs): LR on cells + features
lrc = LogisticRegression(max_iter=5000).fit(Xz, y)
res["auc_lr_cells"] = float(roc_auc_score(y, lrc.decision_function(Xz)))
Zf = (F - F.mean(0)) / F.std(0)
lrf = LogisticRegression(max_iter=5000).fit(np.c_[Xz, Zf], y)
res["auc_lr_cells_features"] = float(roc_auc_score(y, lrf.decision_function(np.c_[Xz, Zf])))
res["auc_network_plus_features"] = float(roc_auc_score(
    y, LogisticRegression(max_iter=5000).fit(np.c_[logit, Zf], y).decision_function(np.c_[logit, Zf])))

# 3. hidden units
units = {}
for h in hidden:
    a = H[h]
    on = a > 0
    w_out = W.get((h, out_id), 0.0)
    r2c = LinearRegression().fit(Xz, a).score(Xz, a)
    r2cf = LinearRegression().fit(np.c_[Xz, F], a).score(np.c_[Xz, F], a)
    corr = {n: float(np.corrcoef(F[:, i], a)[0, 1]) if a.std() > 0 else None for i, n in enumerate(fnames)}
    top = sorted(((abs(v), k, v) for k, v in corr.items() if v is not None), reverse=True)[:4]
    pre = preact(h)
    units[str(h)] = dict(active_share=float(on.mean()), w_to_output=float(w_out),
                         sd_contribution=float((w_out * a).std()),
                         r2_on_cells=float(r2c), r2_on_cells_plus_features=float(r2cf),
                         top_feature_corr=[(k, round(v, 3)) for _, k, v in top],
                         outcome_rate_active=float(y[on].mean()) if on.any() else None,
                         outcome_rate_inactive=float(y[~on].mean()) if (~on).any() else None,
                         preact_sd=float(pre.std()))
res["hidden_units"] = dict(sorted(units.items(), key=lambda kv: -kv[1]["sd_contribution"]))

# 4. threats
fi = {n: i for i, n in enumerate(fnames)}
for k in ("o_thr", "x_thr", "o_thr_even", "x_thr_odd"):
    if k in fi:
        m = F[:, fi[k]] > 0
        res[f"{k}>0"] = dict(n=int(m.sum()), outcome_rate=float(y[m].mean()), mean_pred=float(p[m].mean()))
out = HERE / "logs" / f"{name}.domain.json"
out.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps({k: v for k, v in res.items() if k not in ("hidden_units", "residual_corr")}, indent=2))
for h, u in res["hidden_units"].items():
    print(h, {k: u[k] for k in ("active_share", "w_to_output", "sd_contribution", "r2_on_cells",
                                "r2_on_cells_plus_features", "top_feature_corr")})
