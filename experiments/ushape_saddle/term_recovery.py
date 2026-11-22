"""How much of each true non-linear term did a model recover? (synthetic U + saddle, layer 1, read-only DB)

    PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_dev \\
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 .claude/worktrees/conv-neat/.venv/bin/python experiments/ushape_saddle/term_recovery.py
    (any venv with `interpret` for the EBM and this checkout's `explaneat`; about 10 minutes)

Method (functional ANOVA under the training rows' empirical distribution; Hooker 2004, Friedman & Popescu 2008):
every model's logit F is decomposed the same way as the true logit:
  * main effect of input k at value x: mean over the 1,500 training rows j of F(x_k = x, rest of row j), centred
    over the training rows (partial dependence);
  * the heart rate x systolic BP interaction, purified: g_hs(h, s) - g_h(h) - g_s(s), centred, at each training
    row's (h, s).
The true terms are the same components of the true logit (experiments/ushape_saddle/data.py, task "both"):
the sodium main effect (the U) and the heart-rate x SBP interaction (the saddle). Per model and term, on the
training rows: recovered share 1 - E[(model - truth)^2] / Var(truth); size ratio, the least-squares slope of model
on truth (1 = the right size); correlation; and for the U the minimum of the sodium main effect. Extra structure:
the share of the model's logit variance outside its main effects and the heart-rate x SBP interaction (the truth
has none beyond the artefact of correlated inputs, reported for the truth itself).

Models (all fitted on the same 1,500 training rows, z-scored as the benchmarks):
  * truth: the generating logit (a check of the method: recovered share 1);
  * oracle_form: logistic regression on the true basis (backbone terms, (zn + 0.5)^2, zh*zp), unpenalised: the
    model that recovers the terms if the data allow -- the ceiling for 1,500 rows;
  * propneat: the explained network (explanations/shape-C.json), read from the database;
  * lr_l2, gb, ebm, mlp: the comparators with the settings their cross-validation chose
    (experiments/benchmarks/benchmarks.csv, dataset shape_both), seed 42.
Writes results/term_recovery.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import data  # noqa: E402

SEED = 42
OUT = HERE / "results" / "term_recovery.json"
BENCH = REPO / "experiments" / "benchmarks" / "benchmarks.csv"
CHUNK = 200_000
PAIR = ("heart_rate", "sbp")
U_INPUT = "sodium"


def bench_params(key):
    with open(BENCH) as f:
        (row,) = [r for r in csv.DictReader(f) if r["dataset"] == "shape_both" and r["model_key"] == key]
    return json.loads(row["params"])


def logit_p(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def batched(F, A):
    return np.concatenate([F(A[i:i + CHUNK]) for i in range(0, len(A), CHUNK)])


def components(F, Xtr, col):
    """Main effects of every input and the purified heart-rate x SBP interaction, at the training rows."""
    n = len(Xtr)
    full = batched(F, Xtr)
    main = {}
    for k, name in enumerate(data.FEATURES):
        vals, inv = np.unique(Xtr[:, k], return_inverse=True)
        A = np.repeat(Xtr[None, :, :], len(vals), axis=0)
        A[:, :, k] = vals[:, None]
        pd_ = batched(F, A.reshape(-1, Xtr.shape[1])).reshape(len(vals), n).mean(axis=1)
        g = pd_[inv]
        main[name] = (g - g.mean(), vals, pd_ - g.mean())
    ih, ip = col[PAIR[0]], col[PAIR[1]]
    pairs, inv = np.unique(Xtr[:, [ih, ip]], axis=0, return_inverse=True)
    inv = inv.ravel()
    g2 = np.empty(len(pairs))
    for a in range(0, len(pairs), 100):
        P = pairs[a:a + 100]
        A = np.repeat(Xtr[None, :, :], len(P), axis=0)
        A[:, :, ih] = P[:, 0:1]
        A[:, :, ip] = P[:, 1:2]
        g2[a:a + len(P)] = batched(F, A.reshape(-1, Xtr.shape[1])).reshape(len(P), n).mean(axis=1)
    g2 = g2[inv]
    inter = g2 - g2.mean() - main[PAIR[0]][0] - main[PAIR[1]][0]
    inter = inter - inter.mean()
    approx = full.mean() + sum(m[0] for m in main.values()) + inter
    extra = float(np.var(full - approx) / np.var(full))
    return full, main, inter, extra


def score(est, true):
    v = float(np.var(true))
    return dict(recovered_share=float(1 - np.mean((est - true) ** 2) / v),
                size_ratio=float(np.cov(est, true, bias=True)[0, 1] / v),
                correlation=float(np.corrcoef(est, true)[0, 1]) if np.var(est) > 0 else None,
                rmse_logit=float(np.sqrt(np.mean((est - true) ** 2))), truth_sd=float(np.sqrt(v)))


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    P = data.prepare("both")
    tr = P["train_rows"]
    Xtr = P["X_raw"][tr]
    col = {n: i for i, n in enumerate(data.FEATURES)}
    mean, scale = P["mean"], P["scale"]
    z = lambda A: (A - mean) / scale  # noqa: E731
    ytr = P["y_train"]

    models = {"truth": lambda A: data.ground_truth("both", A)[0]}

    # the correctly specified model: logistic regression on the generating basis
    def basis(A):
        t = data._z(A)
        return np.column_stack([t["za"], t["zc"], t["zb"], t["sm"], t["dm"], t["ma"], (t["zn"] - data.C_Z) ** 2,
                                t["zh"] * t["zp"]])
    from sklearn.linear_model import LogisticRegression
    of = LogisticRegression(penalty=None, max_iter=10_000).fit(basis(Xtr), ytr)
    models["oracle_form"] = lambda A: of.decision_function(basis(A))

    # the explained network
    from explaneat.verify.model import load
    from explaneat.verify.probe import to_z
    rec = json.loads((HERE / "explanations" / "shape-C.json").read_text())
    vm = load(rec["genome_id"])
    names = list(vm.base.feature_names)
    perm = [col[nm] for nm in names]
    assert np.array_equal(vm.m.raw_train, Xtr[:, perm]), "the network's training rows differ from data.prepare('both')"
    models["propneat"] = lambda A: vm.m.logit(to_z(vm, A[:, perm]))

    # comparators with their cross-validated settings
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    lr = LogisticRegression(max_iter=10_000, random_state=SEED, **bench_params("lr_l2")).fit(z(Xtr), ytr)
    models["lr_l2"] = lambda A: lr.decision_function(z(A))
    gb = HistGradientBoostingClassifier(random_state=SEED, **bench_params("gb")).fit(z(Xtr), ytr)
    models["gb"] = lambda A: gb.decision_function(z(A))
    mp = bench_params("mlp")
    mp["hidden_layer_sizes"] = tuple(mp["hidden_layer_sizes"])
    mlp = MLPClassifier(max_iter=2000, random_state=SEED, **mp).fit(z(Xtr), ytr)
    models["mlp"] = lambda A: logit_p(mlp.predict_proba(z(A))[:, 1])
    from interpret.glassbox import ExplainableBoostingClassifier
    ebm = ExplainableBoostingClassifier(random_state=SEED, n_jobs=1, **bench_params("ebm")).fit(z(Xtr), ytr)
    models["ebm"] = lambda A: ebm.decision_function(z(A))

    out = dict(method=__doc__.split("Method")[1].split("Models")[0].strip(), n_train=int(len(tr)),
               oracle_form_coef=dict(zip(["za", "zc", "zb", "smoker", "diabetes", "male", "(zn+0.5)^2", "zh*zp"],
                                         map(float, of.coef_[0]))), models={})
    comps = {}
    for name, F in models.items():
        full, main_, inter, extra = components(F, Xtr, col)
        comps[name] = (main_, inter)
        print(name, flush=True)
        out["models"][name] = dict(extra_structure_share=extra, logit_sd=float(np.std(full)))
    tm, ti = comps["truth"]
    for name, (m, i) in comps.items():
        vals, curve = m[U_INPUT][1], m[U_INPUT][2]
        lo, hi = np.percentile(Xtr[:, col[U_INPUT]], [1, 99])
        inside = (vals >= lo) & (vals <= hi)
        out["models"][name].update(
            u=dict(score(m[U_INPUT][0], tm[U_INPUT][0]), minimum_at=float(vals[inside][np.argmin(curve[inside])])),
            saddle=score(i, ti),
            backbone={k: score(m[k][0], tm[k][0]) for k in ("age", "creatinine", "bmi", "smoker", "diabetes", "male")})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for name, r in out["models"].items():
        print(f"{name:12s} U: rec {r['u']['recovered_share']:.3f} size {r['u']['size_ratio']:.2f} min {r['u']['minimum_at']:.0f} | "
              f"saddle: rec {r['saddle']['recovered_share']:.3f} size {r['saddle']['size_ratio']:.2f} | extra {r['extra_structure_share']:.3f}")


if __name__ == "__main__":
    main()
