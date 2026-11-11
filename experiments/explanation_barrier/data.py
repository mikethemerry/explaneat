"""Data for the explanation complexity ladder (exploratory, not pre-registered).

Synthetic family: 10 tabular-like inputs with a known ground-truth logit ``f(x)``; the label is
``y ~ Bernoulli(sigmoid(f(x)))``. Every task uses the same inputs (same seed), so tasks differ only
in ``f``. Inputs (raw units):

    age      normal(50, 12) clipped to 18-90, integer years
    income   lognormal, median about 45 (thousands), mildly correlated with age
    bmi      normal(27, 5), clipped 15-50, one decimal
    visits   Poisson(2) count, mildly raised by age
    score    uniform 0-100, one decimal
    smoker   binary, p = 0.30
    female   binary, p = 0.50
    urban    binary, p = 0.60
    insured  binary, p = 0.80
    noise    normal(0, 1): never used by any ground truth

Ground truths use the standardised quantities defined in ``_z`` (fixed constants, not data
estimates): za = (age - 50)/12, zi = (log income - 3.8)/0.5, zb = (bmi - 27)/5,
zv = (visits - 2)/1.5, zs = (score - 50)/29.

Real family: Connect-4 from ``experiments/complex_real/data.py`` (same split, same scaler, same
database dataset ``connect4_win_ordinal``).

Split for synthetic tasks: stratified 80:20, ``random_state`` 42; inputs z-scored with a
StandardScaler fit on the training rows (as in the 10-seed study).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def complex_real_data():
    """experiments/complex_real/data.py, loaded under another module name (this file is data.py too)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("complex_real_data", HERE.parent / "complex_real" / "data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

N_ROWS = 30_000
DATA_SEED = 20261111
SPLIT_SEED = 42
FEATURES = ["age", "income", "bmi", "visits", "score", "smoker", "female", "urban", "insured", "noise"]
TYPES = {"age": "integer", "income": "numeric", "bmi": "numeric", "visits": "integer", "score": "numeric",
         "smoker": "binary", "female": "binary", "urban": "binary", "insured": "binary", "noise": "numeric"}


def inputs(n=N_ROWS, seed=DATA_SEED):
    r = np.random.default_rng(seed)
    age = np.clip(np.round(r.normal(50, 12, n)), 18, 90)
    za = (age - 50) / 12
    income = np.round(np.exp(3.8 + 0.5 * (0.3 * za + np.sqrt(1 - 0.09) * r.normal(size=n))), 1)
    bmi = np.round(np.clip(r.normal(27, 5, n), 15, 50), 1)
    visits = r.poisson(np.exp(np.log(2.0) + 0.15 * za)).astype(float)
    score = np.round(r.uniform(0, 100, n), 1)
    smoker = (r.random(n) < 0.30).astype(float)
    female = (r.random(n) < 0.50).astype(float)
    urban = (r.random(n) < 0.60).astype(float)
    insured = (r.random(n) < 0.80).astype(float)
    noise = r.normal(size=n)
    return np.column_stack([age, income, bmi, visits, score, smoker, female, urban, insured, noise])


def _z(X):
    c = {k: X[:, i] for i, k in enumerate(FEATURES)}
    return dict(za=(c["age"] - 50) / 12, zi=(np.log(c["income"]) - 3.8) / 0.5, zb=(c["bmi"] - 27) / 5,
                zv=(c["visits"] - 2) / 1.5, zs=(c["score"] - 50) / 29, sm=c["smoker"], fe=c["female"],
                ur=c["urban"], ins=c["insured"], age=c["age"], bmi=c["bmi"], visits=c["visits"],
                score=c["score"])


def relu(v):
    return np.maximum(v, 0.0)


# ------------------------------------------------------------------ ground truths (logits)
# Each returns (logit, dict of named terms whose sum is the logit). The terms are the
# structure an explanation should recover.

def gt_additive(X):
    z = _z(X)
    t = {"const": np.full(len(X), -1.0), "age (linear)": 1.2 * z["za"],
         "bmi hinge above 29.5": 1.5 * relu(z["zb"] - 0.5), "income (linear)": -0.8 * z["zi"],
         "smoker step": 1.0 * z["sm"]}
    return sum(t.values()), t


def gt_pairwise(X):
    z = _z(X)
    t = {"const": np.full(len(X), -0.8), "age (linear)": 0.8 * z["za"], "income (linear)": -0.6 * z["zi"],
         "smoker x bmi above 27": 2.0 * z["sm"] * relu(z["zb"]),
         "age x income product": 1.0 * z["za"] * z["zi"]}
    return sum(t.values()), t


def gt_threeway(X):
    z = _z(X)
    t = {"const": np.full(len(X), -0.6), "age (linear)": 0.6 * z["za"], "income (linear)": -0.5 * z["zi"],
         "smoker x urban x age": 2.0 * z["sm"] * z["ur"] * z["za"],
         "female x bmi": 1.0 * z["fe"] * z["zb"]}
    return sum(t.values()), t


def gt_rule_exceptions(X):
    z = _z(X)
    base = ((z["age"] > 60) & (z["visits"] >= 3)) | (z["bmi"] > 32)
    exc_off = base & (z["ins"] == 0) & (z["score"] > 80)          # uninsured high scorers: negative
    exc_on = ~base & (z["sm"] == 1) & (z["fe"] == 1) & (z["age"] < 35)   # young female smokers: positive
    rule = (base & ~exc_off) | exc_on
    t = {"rule: (age>60 and visits>=3) or bmi>32, except uninsured with score>80; "
         "plus female smokers under 35": 5.0 * rule.astype(float) - 2.5}
    return sum(t.values()), t


def gt_parity(X, k=3):
    z = _z(X)
    bits = [z["sm"], z["fe"], z["ur"], z["ins"]][:k]
    par = np.mod(sum(bits), 2)
    t = {f"parity of {k} bits ({', '.join(['smoker', 'female', 'urban', 'insured'][:k])})": 4.0 * par - 2.0,
         "age (linear)": 0.5 * z["za"]}
    return sum(t.values()), t


HINGES = [  # (scale, weights on za, zi, zb, zv, zs, smoker, female, urban, insured, bias)
    (1.5, dict(za=1.0, zb=0.8), -0.5),
    (-1.2, dict(zi=1.0, zs=-0.7), -0.3),
    (1.0, dict(zv=1.0, sm=1.0), -1.0),
    (-1.5, dict(za=-0.8, fe=1.2), -0.4),
    (1.2, dict(zs=1.0, ur=-1.0, zb=-0.5), 0.0),
    (1.0, dict(zi=0.7, ins=-1.5, za=0.5), 0.2),
]


def gt_hinges(X, m=6):
    z = _z(X)
    t = {"const": np.full(len(X), -0.5)}
    for j, (a, w, b) in enumerate(HINGES[:m]):
        pre = b + sum(v * z[k] for k, v in w.items())
        t[f"hinge {j + 1}: {a:+.1f} * relu({b:+.1f} " + " ".join(f"{v:+.1f}*{k}" for k, v in w.items()) + ")"] = a * relu(pre)
    return sum(t.values()), t


TASKS = {
    "additive": gt_additive,
    "pairwise": gt_pairwise,
    "threeway": gt_threeway,
    "rule": gt_rule_exceptions,
    "parity2": lambda X: gt_parity(X, 2),
    "parity3": lambda X: gt_parity(X, 3),
    "parity4": lambda X: gt_parity(X, 4),
    "hinges3": lambda X: gt_hinges(X, 3),
    "hinges6": lambda X: gt_hinges(X, 6),
}

DESCRIPTIONS = {
    "additive": "f = -1 + 1.2 za + 1.5 relu(zb - 0.5) - 0.8 zi + 1.0 smoker (additive, k = 1)",
    "pairwise": "f = -0.8 + 0.8 za - 0.6 zi + 2.0 smoker*relu(zb) + 1.0 za*zi (two 2-way interactions)",
    "threeway": "f = -0.6 + 0.6 za - 0.5 zi + 2.0 smoker*urban*za + 1.0 female*zb (a 3-way and a 2-way)",
    "rule": "f = 5*rule - 2.5; rule = ((age>60 & visits>=3) | bmi>32) & not(uninsured & score>80), "
            "or (smoker & female & age<35) (crisp rule with two exceptions)",
    "parity2": "f = 4*xor(smoker, female) - 2 + 0.5 za",
    "parity3": "f = 4*parity(smoker, female, urban) - 2 + 0.5 za",
    "parity4": "f = 4*parity(smoker, female, urban, insured) - 2 + 0.5 za",
    "hinges3": "f = -0.5 + sum of 3 hinges relu(w.x + b) over 2 inputs each (data.HINGES[:3])",
    "hinges6": "f = -0.5 + sum of 6 hinges relu(w.x + b) over 2-3 inputs each (data.HINGES)",
}


# Signal strength: the logit used for the labels is SCALE * f (and the terms are scaled with it),
# so the networks can fit the structure, not the label noise.
SCALE = {"additive": 2.0, "pairwise": 2.0, "threeway": 2.0, "rule": 1.0, "parity2": 1.5, "parity3": 1.5,
         "parity4": 1.5, "hinges3": 2.0, "hinges6": 2.0}


def make_synthetic(task):
    X = inputs()
    f, terms = TASKS[task](X)
    f = SCALE[task] * f
    terms = {k: SCALE[task] * v for k, v in terms.items()}
    r = np.random.default_rng(DATA_SEED + 1 + sorted(TASKS).index(task))
    y = (r.random(len(X)) < 1 / (1 + np.exp(-f))).astype(np.int64)
    return X, y, f, terms


def split_indices(y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    return np.sort(tr), np.sort(te)


def prepare(task):
    """Raw and z-scored arrays; for synthetic tasks also the ground-truth logit and terms."""
    if task == "c4":
        d = complex_real_data().prepare("c4")
        d.update(f=None, terms=None, types={n: "ordinal" for n in d["names"]})
        return d
    X, y, f, terms = make_synthetic(task)
    tr, te = split_indices(y)
    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0)
    scale[scale == 0.0] = 1.0
    z = lambda A: (A - mean) / scale  # noqa: E731
    meta = dict(task=task, ground_truth=DESCRIPTIONS[task], logit_scale=SCALE[task], n_rows=int(len(y)), n_train=int(len(tr)),
                n_test=int(len(te)), positive_share=float(y.mean()), split_seed=SPLIT_SEED, data_seed=DATA_SEED,
                scaler="StandardScaler fit on the training rows", class_names=["0", "1"],
                encoding="raw units; binaries 0/1")
    return dict(X_raw=X, y_all=y, names=list(FEATURES), types=dict(TYPES), train_rows=tr, test_rows=te,
                X_train=z(X[tr]), y_train=y[tr], X_test=z(X[te]), y_test=y[te], mean=mean, scale=scale,
                f=f, terms=terms, meta=meta)
