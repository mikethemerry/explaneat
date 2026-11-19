"""Adult signal catalogue for Experiment 2's ten Adult networks (read-only, light).

Modelled on ``experiments/churn_signals/catalogue.py`` (same terms, same thresholds), lighter:
no disagreement, consolidation or decodability analyses.

- **signal**: an outcome-relevant function of the inputs, defined on the data alone: one block of a
  fixed design matrix ``Phi`` built from the training rows (``build_basis`` below): linear terms
  per variable (a one-hot category is one block of offsets), one-variable bends (hinges at fixed
  knots, residualised on every linear column) and a few interactions that the explanations' units
  read together (residualised on every linear and bend column).
- a network's **realisation** of signal ``s``: the component ``Phi_s beta_s`` of the least-squares
  projection of its logit on ``[1, Phi]`` (training rows); ``r2`` says how much of the logit the
  basis holds. The data's own version: the same component of a ridge logistic fit of the labels on
  ``Phi`` (C chosen by 5-fold CV log-loss); relevance = the CV log-loss gain of the block.
- **used**: the network's component has SD >= 0.10 logit on the training rows, as churn, after
  clipping it to its 1st-99th percentiles (Adult has offsets of tens of logits on countries or
  work classes with a few dozen rows, which would otherwise dominate the SD; such seeds are
  marked `rare_only`).
- **realisation code** of a used non-linear signal: the switching units carrying it (unit fixed at
  its training mean; the change in the logit projected on the block; carriers have component SD
  >= 0.05 and >= 20% of the top carrier, as churn). H: one carrier reading only the signal's
  variables (a hinge); G: one carrier that also reads another variable (a gate); S<k>: split
  across k units; /<n>p: in n explanation parts (leaf annotations). Linear signals: ''.
- **binary one-step rule** (``explaneat/analysis/plain_language_linearity.py``): a ReLU that reads
  one two-valued input (a yes/no input or one level of a one-hot category) is exactly linear on
  the data, so such units count as linear (``unit_kinds(m, binary_step=True)``) and never as
  carriers of a non-linear signal; ``matrix.csv`` notes how many a linear signal's columns feed.

Outcome: the dataset ``adult_ohe_v2`` stores class 1 as ``<=50k`` (asserted below), so every
logit here is the log-odds of an income of $50,000 a year or less.

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/adult_signals/catalogue.py [--refresh]

Every database session is READ ONLY (``plain_language.load_model`` issues SET TRANSACTION READ
ONLY). The forward passes are cached in ``cache/`` (``--refresh`` recomputes them).
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/churn_signals")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastboot import Boot  # noqa: E402  (experiments/churn_signals/fastboot.py)
from explaneat.analysis.signal_presence import consensus, consensus_direction, presence  # noqa: E402  (shared with churn, heart)

CACHE = HERE / "cache"
LABELS = [f"adult-s{i}" for i in range(10)]
CARRY_SD, CARRY_REL = 0.05, 0.20
N_BOOT, BOOT_SEED = 2000, 42      # paired bootstrap on the test rows (9,769 rows: 2,000 resamples)

# signal id -> (short name, one-line description, data variables, direction check)
# direction check: (variable, sign of the correlation of the DATA's component with it) or None.
SIGNALS = {
    "lin:household": ("Marital status and household role",
                      "low income likelier outside marriage: husbands and wives sit well below the unmarried",
                      ["marital-status", "relationship"], ("married", -1)),
    "lin:capgain": ("Capital gain (slope)", "capital gains lower the chance of a low income",
                    ["capital-gain"], ("capital-gain", -1)),
    "shape:capgain": ("Capital-gain threshold",
                      "past a few thousand dollars of capital gain the chance of a low income drops steeply",
                      ["capital-gain"], None),
    "lin:edu": ("Education (years)", "each extra year of schooling lowers the chance of a low income",
                ["education-num"], ("education-num", -1)),
    "lin:edulevel": ("Education levels", "offsets for single education levels beyond the years-of-schooling line",
                     ["education"], None),
    "lin:age": ("Age (slope)", "older people are less likely to have a low income",
                ["age"], ("age", -1)),
    "shape:age": ("Age bend", "the fall with age is steep for the young and flattens or turns back later in life",
                  ["age"], None),
    "lin:hours": ("Hours per week (slope)", "longer working weeks lower the chance of a low income",
                  ["hours-per-week"], ("hours-per-week", -1)),
    "shape:hours": ("Hours bend", "short weeks raise the chance of a low income more than long weeks lower it",
                    ["hours-per-week"], None),
    "lin:work": ("Occupation and work class",
                 "executive, managerial and professional work sit below; service and handling work and missing job records above",
                 ["occupation", "workclass"], None),
    "lin:sex": ("Sex", "women are more likely to have a low income than otherwise similar men",
                ["sex"], ("male", -1)),
    "lin:caploss": ("Capital loss (slope)", "recorded capital losses lower the chance of a low income",
                    ["capital-loss"], ("capital-loss", -1)),
    "shape:caploss": ("Capital-loss bend", "bends in the capital-loss effect across loss amounts",
                      ["capital-loss"], None),
    "lin:country": ("Country of birth", "offsets for single countries of birth, mostly small groups",
                    ["native-country"], None),
    "lin:race": ("Race", "small offsets by recorded race", ["race"], None),
    "lin:fnlwgt": ("Census weight (slope)", "the survey sampling weight, not a personal attribute",
                   ["fnlwgt"], None),
    "shape:fnlwgt": ("Census weight bend", "bends in the survey sampling weight", ["fnlwgt"], None),
    "int:sex*married": ("Sex × marriage", "the gap between women and men differs inside and outside marriage",
                        ["sex", "marital-status", "relationship"], None),
    "int:capgain*married": ("Capital gain × marriage", "the capital-gain threshold acts differently for the married",
                            ["capital-gain", "marital-status", "relationship"], None),
    "int:age*married": ("Age × marriage", "the age effect differs inside and outside marriage",
                        ["age", "marital-status", "relationship"], None),
    "int:age*hours": ("Age × hours", "the hours effect changes with age",
                      ["age", "hours-per-week"], None),
}
BLOCK_VARS = {b: set(v[2]) for b, v in SIGNALS.items()}
# a unit reading the other household variable is still reading "marriage"
ALIAS = {"marital-status": "household", "relationship": "household",
         "education": "education", "education-num": "education",
         "occupation": "work", "workclass": "work"}

CAPGAIN_KNOTS = [2000.0, 4000.0, 5000.0, 7000.0, 10000.0, 15000.0]
CAPLOSS_KNOTS = [1500.0, 1800.0, 2000.0, 2400.0]
HOURS_KNOTS = [20.0, 30.0, 40.0, 45.0, 50.0, 60.0]
AGE_Q = [0.10, 0.25, 0.50, 0.75, 0.90]
FNL_Q = [0.10, 0.25, 0.50, 0.75, 0.90]


def relu(x):
    return np.maximum(x, 0.0)


# ============================================================================ load (cached)

def genome_ids():
    with open(REPO / "experiments/10seed/runs.csv") as f:
        return {f"adult-s{r['seed']}": r["genome_id"] for r in csv.DictReader(f) if r["dataset"] == "adult"}


def load_network(lab, gid):
    """Forward passes of one network: logit, switching-unit knock-outs, unit -> leaf part."""
    from explaneat.analysis.plain_language import build_parts, load_model
    from explaneat.analysis.plain_language_linearity import unit_kinds
    from explaneat.core.genome_network import NodeType
    from explaneat.core.split_ids import base_id
    m, meta = load_model(gid)
    assert meta["dataset"]["class_names"] == [">50k", "<=50k"], meta["dataset"]["class_names"]
    assert m.codebook["outcome"]["positive"].startswith("an income of $50,000 a year or less")
    kinds = unit_kinds(m, binary_step=True)
    parts = [p for p in build_parts(m) if not p.is_root]
    leaves = [p for p in parts if p.kind == "leaf"]

    def leaf_of(n):
        c = [p for p in leaves if n in {str(x) for x in p.unit.get("subgraph_nodes", [])} | set(p.exits)]
        c.sort(key=lambda p: len(p.unit.get("subgraph_nodes", [])))
        return c[0].name if c else ""

    anc_cache = {}

    def anc_vars(n):
        if n in anc_cache:
            return anc_cache[n]
        node = m.nodes[n]
        if node.type == NodeType.INPUT:
            from explaneat.analysis.explanation_tools import input_column
            out = {m.base_of_col[input_column(n)]}
        else:
            out = set()
            for p, w in m.ev.preds.get(n, []):
                if w != 0:
                    out |= anc_vars(p)
        anc_cache[n] = out
        return out

    vals = m.ev.forward(m.X)
    groups = {}
    for n, k in kinds.items():
        if k == "switching":
            groups.setdefault(base_id(n) or n, []).append(n)
    units = []
    for b, ns in sorted(groups.items()):
        mu = {n: float(np.mean(vals[n])) for n in ns}
        units.append(dict(unit=b, nodes=ns, part=leaf_of(ns[0]),
                          reads=sorted(set().union(*[anc_vars(n) for n in ns])),
                          rows_on=int(np.sum(vals[ns[0]] > 0)),
                          d_tr=m.base_logit - m.logit(m.X, ablate=mu),
                          d_te=m.base_logit_test - m.logit(m.Xt, ablate=mu)))
    # how each variable reaches the output: a direct edge (through identity nodes only) and/or
    # through ReLU units (dead units excluded)
    succ = {}
    for n, ps in m.ev.preds.items():
        for p, w in ps:
            if w != 0:
                succ.setdefault(p, []).append(n)

    def linear_to_out(n):
        if n in m.out_ids:
            return True
        nd = m.nodes[n]
        return nd.activation == "identity" and any(linear_to_out(s) for s in succ.get(n, []))

    route = {}
    for v in m.variables:
        nds = [nd for c in v.columns for nd in m.input_nodes_by_col.get(c, [])]
        direct = any(linear_to_out(s) for nd in nds for s in succ.get(nd, []))
        via = set()

        def walk(n, seen):
            for s2 in succ.get(n, []):
                if s2 in seen:
                    continue
                seen.add(s2)
                if s2 in kinds and kinds[s2] not in ("linear", "dead"):
                    via.add(kinds[s2])
                elif s2 in kinds and kinds[s2] == "linear" and s2 not in m.out_ids:
                    walk(s2, seen)
        for nd in nds:
            walk(nd, set())
        route[v.base] = dict(direct=bool(direct), via=sorted(via))
    # binary one-step units per variable (the linearity rule): counted for the notes
    bstep = {}
    for n, k in kinds.items():
        if k == "binary_step":
            for v in anc_vars(n):
                bstep[v] = bstep.get(v, 0) + 1
    return m, dict(label=lab, genome_id=gid, eta_tr=m.base_logit, eta_te=m.base_logit_test,
                   units=units, bstep=bstep, route=route, n_switching=len(units),
                   n_binary_step=sum(k == "binary_step" for k in kinds.values()),
                   parts=[p.name for p in leaves])


def load_all(refresh=False):
    CACHE.mkdir(exist_ok=True)
    ids = genome_ids()
    nets, data = {}, None
    dp = CACHE / "data.pkl"
    for lab in LABELS:
        p = CACHE / f"{lab}.pkl"
        if p.exists() and dp.exists() and not refresh:
            nets[lab] = pickle.loads(p.read_bytes())
            continue
        if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
            sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
        print("loading", lab, flush=True)
        m, net = load_network(lab, ids[lab])
        p.write_bytes(pickle.dumps(net))
        nets[lab] = net
        if data is None:
            data = dict(X=m.X, Xt=m.Xt, y=m.y, yt=m.yt, raw=m.raw_train, raw_te=m.raw_test,
                        names=m.feature_names, mean=m.mean, scale=m.scale,
                        class_names=[">50k", "<=50k"],
                        labels={k: {c: lv["label"] for c, lv in v["levels"].items()}
                                for k, v in m.codebook["variables"].items() if isinstance(v, dict) and "levels" in v})
            dp.write_bytes(pickle.dumps(data))
    if data is None:
        data = pickle.loads(dp.read_bytes())
    return data, nets


# ============================================================================ basis

def build_basis(D):
    names = D["names"]
    col = {n: i for i, n in enumerate(names)}
    pref = lambda p: [i for i, n in enumerate(names) if n.startswith(p + ":")]

    def zq(z, qs):
        return [float(np.quantile(z, q)) for q in qs]

    Ztr, Rtr = D["X"], D["raw"]

    def rawknots(c, ks):   # raw -> z
        return [(k - D["mean"][col[c]]) / D["scale"][col[c]] for k in ks]

    kn = {"age": zq(Ztr[:, col["age"]], AGE_Q), "fnlwgt": zq(Ztr[:, col["fnlwgt"]], FNL_Q),
          "capital-gain": rawknots("capital-gain", CAPGAIN_KNOTS),
          "capital-loss": rawknots("capital-loss", CAPLOSS_KNOTS),
          "hours-per-week": rawknots("hours-per-week", HOURS_KNOTS)}
    m2 = col["marital-status:2"]
    age30 = rawknots("age", [30.0])[0]
    cg5 = rawknots("capital-gain", [5000.0])[0]

    def build(X):
        cols, nm, blocks = [], [], {}

        def add(b, name, a):
            blocks.setdefault(b, []).append(len(nm))
            nm.append(name)
            cols.append(np.asarray(a, float))

        z = lambda c: X[:, col[c]]
        lin_one = {"lin:age": "age", "lin:fnlwgt": "fnlwgt", "lin:edu": "education-num", "lin:sex": "sex",
                   "lin:capgain": "capital-gain", "lin:caploss": "capital-loss", "lin:hours": "hours-per-week"}
        for b, c in lin_one.items():
            add(b, c, z(c))
        for b, ps in {"lin:household": ["marital-status", "relationship"], "lin:work": ["occupation", "workclass"], "lin:race": ["race"], "lin:country": ["native-country"]}.items():
            for p in ps:
                for i in pref(p)[:-1]:          # drop one level per group (collinear with the intercept)
                    add(b, names[i], X[:, i])
        build.n_lin = len(nm)
        for b, c in {"shape:age": "age", "shape:fnlwgt": "fnlwgt", "shape:capgain": "capital-gain",
                     "shape:caploss": "capital-loss", "shape:hours": "hours-per-week"}.items():
            for j, t in enumerate(kn[c]):
                add(b, f"{c}>{j}", relu(z(c) - t))
        for i in pref("education"):            # level offsets beyond the years line
            add("lin:edulevel", names[i], X[:, i])
        build.n_main = len(nm)
        mar = X[:, m2]
        add("int:sex*married", "sex*married", z("sex") * mar)
        for j, a in enumerate([z("capital-gain"), relu(z("capital-gain") - cg5)]):
            add("int:capgain*married", f"cg{j}*married", a * mar)
        for j, a in enumerate([z("age"), relu(age30 - z("age"))]):
            add("int:age*married", f"age{j}*married", a * mar)
        h40 = rawknots("hours-per-week", [40.0])[0]
        for i, a in enumerate([z("age"), relu(z("age") - kn["age"][2])]):
            for j, h in enumerate([z("hours-per-week"), relu(z("hours-per-week") - h40)]):
                add("int:age*hours", f"age{i}*hours{j}", a * h)
        build.names, build.blocks = nm, blocks
        return np.column_stack(cols)

    P = build(Ztr)
    nm, blocks, n_lin, n_main = build.names, build.blocks, build.n_lin, build.n_main
    steps = []
    one = np.ones((len(P), 1))
    for lo, hi in [(n_lin, n_main), (n_main, len(nm))]:
        A = np.hstack([one, P[:, :lo]])
        coef, *_ = np.linalg.lstsq(A, P[:, lo:hi], rcond=None)
        P[:, lo:hi] -= A @ coef
        steps.append((lo, hi, coef))
    mu = P[:, :n_lin].mean(axis=0)
    P[:, :n_lin] -= mu
    keep = [i for i in range(P.shape[1]) if P[:, i].std() > 1e-8]

    def phi(X):
        Q = build(X)
        o = np.ones((len(Q), 1))
        for lo, hi, coef in steps:
            Q[:, lo:hi] -= np.hstack([o, Q[:, :lo]]) @ coef
        Q[:, :n_lin] -= mu
        return Q[:, keep]

    remap = {old: new for new, old in enumerate(keep)}
    blocks = {b: [remap[i] for i in idx if i in remap] for b, idx in blocks.items()}
    blocks = {b: idx for b, idx in blocks.items() if idx}
    Ptr = P[:, keep]
    assert np.max(np.abs(phi(Ztr) - Ptr)) < 1e-8
    knots_raw = {"capital-gain": CAPGAIN_KNOTS, "capital-loss": CAPLOSS_KNOTS, "hours-per-week": HOURS_KNOTS,
                 "age": [float(D["mean"][col["age"]] + D["scale"][col["age"]] * k) for k in kn["age"]]}
    return dict(Phi_tr=Ptr, Phi_te=phi(D["Xt"]), blocks=blocks, names=[nm[i] for i in keep], knots=knots_raw,
                phi=phi)


def comp(B, beta, b, test=False):
    P = B["Phi_te"] if test else B["Phi_tr"]
    idx = B["blocks"][b]
    return P[:, idx] @ beta[idx]


# ============================================================================ data's own fit

def fit_logit(P, y, C):
    sd = P.std(axis=0)
    sd[sd == 0] = 1
    lr = LogisticRegression(C=C, max_iter=5000, tol=1e-6).fit(P / sd, y)
    return lr.intercept_[0], lr.coef_[0] / sd


def cv_score(P, y, C, folds):
    ll = []
    for tr, va in folds:
        a, b = fit_logit(P[tr], y[tr], C)
        ll.append(log_loss(y[va], 1 / (1 + np.exp(-(a + P[va] @ b)))))
    return float(np.mean(ll))


def data_fit(B, y, yt, boot):
    P = B["Phi_tr"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(P, y))
    grid = [0.01, 0.1, 1.0]
    scores = {C: cv_score(P, y, C, folds) for C in grid}
    C = min(scores, key=scores.get)
    a, beta = fit_logit(P, y, C)
    z_te = a + B["Phi_te"] @ beta
    rows = {}
    for b, idx in B["blocks"].items():
        keep = [i for i in range(P.shape[1]) if i not in set(idx)]
        c_te = comp(B, beta, b, test=True)
        lo, hi = boot.delta_ci(z_te, z_te - c_te)
        rows[b] = dict(data_cv_logloss_gain=cv_score(P[:, keep], y, C, folds) - scores[C],
                       data_component_sd=float(np.std(comp(B, beta, b))),
                       data_ablation_gain_test=roc_auc_score(yt, z_te) - roc_auc_score(yt, z_te - c_te),
                       data_ablation_gain_test_lo=lo, data_ablation_gain_test_hi=hi)
        print("  data block", b, f"{rows[b]['data_cv_logloss_gain']:+.4f}", flush=True)
    return a, beta, rows, dict(C=C, cv_logloss=scores, auc_train=roc_auc_score(y, a + P @ beta),
                               auc_test=roc_auc_score(yt, z_te))


# ============================================================================ main

def project(B, e):
    A = np.hstack([np.ones((len(B["Phi_tr"]), 1)), B["Phi_tr"]])
    c, *_ = np.linalg.lstsq(A, e, rcond=None)
    return float(c[0]), c[1:]


def r2(y, f):
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


def realisation_code(b, used, carriers):
    if not used or b.startswith("lin:"):
        return ""
    if not carriers:
        return "--"   # used, but no single switching unit carries it (spread thinly)
    parts = sorted({u["part"] for u in carriers})
    n = f"/{len(parts)}p"
    if len(carriers) == 1:
        others = {ALIAS.get(v, v) for v in carriers[0]["reads"]} - {ALIAS.get(v, v) for v in BLOCK_VARS[b]}
        return ("G" if others else "H") + n
    return f"S{len(carriers)}{n}"


def check_descriptions(B, beta, D):
    """Each description's direction, checked on the data's own fit (training rows). Returns the
    check text per signal; fails loudly if a description contradicts the data."""
    names = D["names"]
    col = {n: i for i, n in enumerate(names)}
    R = D["raw"]
    out = {}
    for b, (_, _, _, chk) in SIGNALS.items():
        if chk:
            v = {"married": R[:, col["marital-status:2"]], "male": R[:, col["sex"]]}.get(chk[0])
            v = R[:, col[chk[0]]] if v is None else v
            r = float(np.corrcoef(comp(B, beta, b), v)[0, 1])
            assert np.sign(r) == chk[1], (b, r)
            out[b] = f"corr of the data's component with {chk[0]}: {r:+.2f}"

    def mean_by(b, onehot):
        c = comp(B, beta, b)
        return {k: float(c[R[:, col[k]] > 0.5].mean()) for k in onehot}

    h = mean_by("lin:household", ["relationship:0", "relationship:5", "relationship:4", "relationship:1",
                                  "relationship:3"])
    assert max(h["relationship:0"], h["relationship:5"]) < min(h["relationship:4"], h["relationship:1"],
                                                               h["relationship:3"]), h
    out["lin:household"] += "; husband/wife means " + ", ".join(f"{h[k]:+.2f}" for k in ("relationship:0", "relationship:5")) + \
        " vs unmarried / not in family / own child " + ", ".join(f"{h[k]:+.2f}" for k in ("relationship:4", "relationship:1", "relationship:3"))
    w = mean_by("lin:work", ["occupation:4", "occupation:10", "occupation:8", "occupation:6", "occupation:0"])
    assert max(w["occupation:4"], w["occupation:10"]) < min(w["occupation:8"], w["occupation:6"], w["occupation:0"]), w
    out["lin:work"] = "occupation means (exec, prof | other service, handlers, missing): " + \
        ", ".join(f"{w[k]:+.2f}" for k in w)

    def slopes(var, blocks_, ranges):
        x = R[:, col[var]]
        t = sum(comp(B, beta, bb) for bb in blocks_)
        res = []
        for lo, hi in ranges:
            m_ = (x >= lo) & (x <= hi)
            res.append(float(np.polyfit(x[m_], t[m_], 1)[0]))
        return res

    lo_, hi_ = slopes("capital-gain", ["lin:capgain", "shape:capgain"], [(0, 3000), (5000, 10000)])
    assert hi_ < lo_ <= 0 or hi_ < lo_, (lo_, hi_)
    out["shape:capgain"] = f"slope of lin+shape per $1000: {1000 * lo_:+.3f} (0-3000) vs {1000 * hi_:+.3f} (5000-10000)"
    a1, a2 = slopes("age", ["lin:age", "shape:age"], [(20, 35), (50, 70)])
    assert a1 < 0 and a1 < a2, (a1, a2)
    out["shape:age"] = f"slope of lin+shape per year: {a1:+.3f} (20-35) vs {a2:+.3f} (50-70)"
    h1, h2 = slopes("hours-per-week", ["lin:hours", "shape:hours"], [(10, 40), (40, 70)])
    assert h1 < 0 and abs(h1) > abs(h2), (h1, h2)
    out["shape:hours"] = f"slope of lin+shape per hour: {h1:+.3f} (10-40) vs {h2:+.3f} (40-70)"
    print("description checks:", json.dumps(out, indent=1), flush=True)
    return out


def main():
    t0 = time.time()
    D, nets = load_all(refresh="--refresh" in sys.argv)
    y, yt = D["y"], D["yt"]
    names = D["names"]
    rng = np.random.RandomState(BOOT_SEED)
    boot = Boot(np.stack([rng.randint(0, len(yt), size=len(yt)) for _ in range(N_BOOT)]), yt)
    B = build_basis(D)
    blocks = list(SIGNALS)
    assert set(blocks) == set(B["blocks"]), set(blocks) ^ set(B["blocks"])
    print("basis", B["Phi_tr"].shape, flush=True)
    fp = CACHE / "datafit.pkl"
    key = (tuple(blocks), B["Phi_tr"].shape)
    cached = pickle.loads(fp.read_bytes()) if fp.exists() else None
    if cached and cached[0] == key and "--refresh" not in sys.argv:
        a_d, beta_d, rel, info = cached[1]
    else:
        a_d, beta_d, rel, info = data_fit(B, y, yt, boot)
        fp.write_bytes(pickle.dumps((key, (a_d, beta_d, rel, info))))
    checks = check_descriptions(B, beta_d, D)

    proj_rows, mat_rows, comps = [], [], {}
    for lab, nt in nets.items():
        seed = int(lab.split("-s")[1])
        e_tr, e_te = nt["eta_tr"], nt["eta_te"]
        a, beta = project(B, e_tr)
        f_tr, f_te = a + B["Phi_tr"] @ beta, a + B["Phi_te"] @ beta
        proj_rows.append(dict(network=lab, seed=seed, r2_train=r2(e_tr, f_tr), r2_test=r2(e_te, f_te),
                              auc_train=roc_auc_score(y, e_tr), auc_test=roc_auc_score(yt, e_te),
                              auc_projection_train=roc_auc_score(y, f_tr),
                              auc_projection_test=roc_auc_score(yt, f_te),
                              n_switching_units=nt["n_switching"], n_binary_step_units=nt["n_binary_step"]))
        # carriers of the non-linear blocks
        car = {}
        for u in nt["units"]:
            au, bu = project(B, u["d_tr"])
            sds = {b: float(np.std(comp(B, bu, b))) for b in blocks if not b.startswith("lin:")}
            for b, s in sds.items():
                if s >= CARRY_SD:
                    car.setdefault(b, []).append(dict(u, sd=s))
        for b in blocks:
            c, c_te = comp(B, beta, b), comp(B, beta, b, test=True)
            # explaneat/analysis/signal_presence.py (one rule for all three sets); a network that
            # reads none of the block's variables carries no weight on it
            reads = any(nt["route"].get(v, dict(direct=False, via=[]))["direct"] or
                        nt["route"].get(v, dict(direct=False, via=[]))["via"] for v in BLOCK_VARS[b])
            pr = presence(c, reads=reads)
            comps[(lab, b)] = c
            sd, sd_rob = pr["sd_raw"], pr["sd"]
            used = pr["state"] == "used"
            rare_only = pr["rare_only"]
            cs = car.get(b, [])
            mx = max([x["sd"] for x in cs], default=0.0)
            cs = [x for x in cs if x["sd"] >= CARRY_REL * mx]
            code = realisation_code(b, used, cs)
            lo, hi = boot.delta_ci(e_te, e_te - c_te)
            route = ""
            if b.startswith("lin:"):
                rt = [nt["route"].get(v, dict(direct=False, via=[])) for v in BLOCK_VARS[b]]
                e_, u_ = any(r["direct"] for r in rt), any(r["via"] for r in rt)
                route = "edge+units" if e_ and u_ else "edge" if e_ else "units" if u_ else "none"
                nb = sum(nt["bstep"].get(v, 0) for v in BLOCK_VARS[b])
                note = (f"{nb} one-level unit(s), linear by the binary one-step rule" if nb else "") if used else (
                    "large values on under 1% of rows only" if rare_only else "")
            else:
                note = "; ".join(f"unit {x['unit']} in {x['part']} reads {', '.join(x['reads'])}"
                                 for x in sorted(cs, key=lambda x: -x["sd"])) if used else ""
            mat_rows.append(dict(network=lab, seed=seed, signal=b, sd_train=sd, sd_robust=sd_rob, used=used,
                                 presence=pr["state"], reads=reads,
                                 rare_only=rare_only, code=code, route=route,
                                 ablation_gain_train=roc_auc_score(y, e_tr) - roc_auc_score(y, e_tr - c),
                                 ablation_gain_test=roc_auc_score(yt, e_te) - roc_auc_score(yt, e_te - c_te),
                                 ablation_gain_test_lo=lo, ablation_gain_test_hi=hi,
                                 corr_with_data=(float(np.corrcoef(c, comp(B, beta_d, b))[0, 1])
                                                 if sd > 1e-9 else float("nan")),
                                 n_carriers=len(cs) if used and not b.startswith("lin:") else 0,
                                 parts=" | ".join(sorted({x["part"] for x in cs})) if used else "",
                                 note=note))
        print(lab, f"R2 train {proj_rows[-1]['r2_train']:.3f}", flush=True)

    # direction of each used signal against the consensus of the seeds that use it
    for r in mat_rows:
        used_c = [comps[(l, r["signal"])] for l in nets
                  if any(x["network"] == l and x["signal"] == r["signal"] and x["used"] for x in mat_rows)]
        r["direction"] = (consensus_direction(comps[(r["network"], r["signal"])], consensus(used_c))
                          if r["used"] else "")
    M = pd.DataFrame(mat_rows)
    sig_rows = []
    for b, (name, desc, vars_, _) in SIGNALS.items():
        mb = M[M.signal == b]
        u = mb[mb.used]
        codes = [c for c in u.code if isinstance(c, str) and c]
        if b.startswith("lin:"):
            rc = u.route.value_counts()
            lab_ = {"edge": "edge", "edge+units": "edge+units", "units": "units only", "none": "none"}
            notes = ", ".join(f"{lab_[k]} {int(rc[k])}" for k in ("edge", "edge+units", "units", "none") if k in rc)
            nb = int(u.note.fillna("").str.contains("one-level").sum())
            if nb:
                notes += f"; one-level units in {nb}"
            nr = int(mb.rare_only.sum())
            if nr:
                what = "extreme rows only" if b in ("lin:fnlwgt", "lin:age", "lin:hours", "lin:capgain", "lin:caploss",
                                                    "lin:edu") else "small-group offsets only"
                notes += ("; " if notes else "") + f"{what} in {nr}"
        else:
            kinds_ = pd.Series([c.split("/")[0] if c != "--" else "spread" for c in codes]).value_counts()
            notes = ", ".join(f"{k} {int(v)}" for k, v in kinds_.items())
            np_ = sorted({int(c.split("/")[1][:-1]) for c in codes if "/" in c})
            if np_:
                notes += f"; {np_[0]}" + (f"-{np_[-1]}" if np_[-1] != np_[0] else "") + (" part" if np_ == [1] else " parts")
        r = rel[b]
        sig_rows.append(dict(signal=b, name=name, description=desc, kind="lin" if b.startswith("lin:") else "nonlin",
                             variables=" + ".join(vars_), data_cv_logloss_gain=r["data_cv_logloss_gain"],
                             data_relevant=bool(r["data_cv_logloss_gain"] > 0),
                             data_component_sd=r["data_component_sd"],
                             data_ablation_gain_test=r["data_ablation_gain_test"],
                             data_ablation_gain_test_lo=r["data_ablation_gain_test_lo"],
                             data_ablation_gain_test_hi=r["data_ablation_gain_test_hi"],
                             n_seeds_used=int(len(u)), description_check=checks.get(b, ""),
                             n_seeds_test_ci_above_0=int((u.ablation_gain_test_lo > 0).sum()),
                             median_sd_used=float(u.sd_train.median()) if len(u) else float("nan"),
                             median_ablation_gain_train=float(u.ablation_gain_train.median()) if len(u) else float("nan"),
                             median_ablation_gain_test=float(u.ablation_gain_test.median()) if len(u) else float("nan"),
                             notes=notes))
    S = pd.DataFrame(sig_rows)
    S.to_csv(HERE / "signals.csv", index=False)
    M.to_csv(HERE / "matrix.csv", index=False)
    pd.DataFrame(proj_rows).to_csv(HERE / "projection.csv", index=False)
    (CACHE / "meta.json").write_text(json.dumps(dict(
        outcome="class 1 = <=50k (income of $50,000 a year or less); logits are its log-odds",
        class_names=D["class_names"], data_fit=info, description_checks=checks, knots=B["knots"],
        blocks={b: len(i) for b, i in B["blocks"].items()}, n_boot=N_BOOT, boot_seed=BOOT_SEED,
        n_train=int(len(y)), n_test=int(len(yt))), indent=1, default=float))
    np.savez_compressed(CACHE / "betas.npz", beta_data=beta_d, a_data=a_d)
    print(S[["signal", "n_seeds_used", "n_seeds_test_ci_above_0", "data_cv_logloss_gain", "notes"]].to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
