"""Data for the synthetic U-shape and saddle study (exploratory, not pre-registered).

Three tabular-like datasets share the same 10 inputs (same rows, same seed) and the same linear
backbone; they differ only in the nonlinear term added to the logit. The label is
``y ~ Bernoulli(sigmoid(f(x)))``. Inputs (raw units; clinical-style names and distributions, in the
style of ``experiments/explanation_barrier/data.py``):

    age          normal(62, 12) clipped 30-95, integer years
    bmi          normal(27.5, 5) clipped 15-50, one decimal
    sbp          systolic blood pressure, mmHg: 132 + 18 * (0.25 za + sqrt(1-0.0625) N), clipped 85-210, integer
    heart_rate   beats/min, normal(76, 13) clipped 40-140, integer
    sodium       serum sodium, mmol/L, normal(139, 4) clipped 120-158, integer
    creatinine   serum creatinine, umol/L, exp(log 85 + 0.3 (0.2 za + sqrt(1-0.04) N)), clipped 40-400, integer
    smoker       binary, p = 0.25
    diabetes     binary, p = 0.30
    male         binary, p = 0.55
    noise        normal(0, 1): never used by any ground truth

Standardised quantities (fixed constants, not data estimates): za = (age-62)/12, zb = (bmi-27.5)/5,
zp = (sbp-132)/18, zh = (heart_rate-76)/13, zn = (sodium-139)/4, zc = (log creatinine - log 85)/0.3.

Backbone (all three datasets): -1.2 + 0.7 za + 0.5 zc + 0.3 zb + 0.8 smoker + 0.6 diabetes + 0.3 male.
Nonlinear terms:
    U (shape-A):       BETA_U * ((zn - C_Z)^2 - 1), a U-shaped risk curve on sodium with its minimum at
                       C_Z = -0.5 in z units, i.e. 137 mmol/L (the "sweet spot"); both low and high
                       sodium raise risk. The -1 keeps the mean logit near the backbone's.
    saddle (shape-B):  -BETA_S * zh * zp, a hyperbolic paraboloid between heart rate and systolic
                       pressure: risk rises when one is high and the other low, falls when both are
                       high or both low. Neither input is in the backbone, so the interaction is
                       the whole of their signal.
    both (shape-C):    the backbone plus both terms.

BETA_U and BETA_S were set by ``calibrate.py`` before any network was trained (RESULTS.md section 2):
the smallest value on a grid at which the oracle (the true logit) beats L2 logistic regression by
at least 0.10 test AUC (twice the 0.05 floor of the brief) and gradient boosting beats it by at least
0.05 (the signal is learnable at n = 1,500, so a PropNEAT null is not down to a weak signal).

Rows: 2,000; split stratified 75:25 with random_state 42 (1,500 train, 500 test); inputs z-scored
with a StandardScaler fit on the training rows (as in the 10-seed study).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

N_ROWS = 2_000
DATA_SEED = 20261114
SPLIT_SEED = 42
TEST_SIZE = 0.25
FEATURES = ["age", "bmi", "sbp", "heart_rate", "sodium", "creatinine", "smoker", "diabetes", "male", "noise"]
TYPES = {"age": "integer", "bmi": "numeric", "sbp": "integer", "heart_rate": "integer", "sodium": "integer",
         "creatinine": "integer", "smoker": "binary", "diabetes": "binary", "male": "binary", "noise": "numeric"}

# Nonlinear strengths: fixed by calibrate.py (see its output in RESULTS.md section 2).
BETA_U = 1.0   # calibrate.py: oracle - LR = 0.120 (floor 0.10)
BETA_S = 1.5   # calibrate.py: oracle - LR = 0.115 (floor 0.10)
C_Z = -0.5                     # minimum of the U in z units of sodium -> 137 mmol/L
SODIUM_SWEET_SPOT = 139 + 4 * C_Z

# Opaque labels: the explaining agent sees the label and the dataset name, not the shape.
LABELS = {"u": "shape-A", "saddle": "shape-B", "both": "shape-C"}


def inputs(n=N_ROWS, seed=DATA_SEED):
    r = np.random.default_rng(seed)
    age = np.clip(np.round(r.normal(62, 12, n)), 30, 95)
    za = (age - 62) / 12
    bmi = np.round(np.clip(r.normal(27.5, 5, n), 15, 50), 1)
    sbp = np.clip(np.round(132 + 18 * (0.25 * za + np.sqrt(1 - 0.0625) * r.normal(size=n))), 85, 210)
    heart_rate = np.clip(np.round(r.normal(76, 13, n)), 40, 140)
    sodium = np.clip(np.round(r.normal(139, 4, n)), 120, 158)
    creatinine = np.clip(np.round(np.exp(np.log(85) + 0.3 * (0.2 * za + np.sqrt(1 - 0.04) * r.normal(size=n)))), 40, 400)
    smoker = (r.random(n) < 0.25).astype(float)
    diabetes = (r.random(n) < 0.30).astype(float)
    male = (r.random(n) < 0.55).astype(float)
    noise = r.normal(size=n)
    return np.column_stack([age, bmi, sbp, heart_rate, sodium, creatinine, smoker, diabetes, male, noise])


def _z(X):
    c = {k: X[:, i] for i, k in enumerate(FEATURES)}
    return dict(za=(c["age"] - 62) / 12, zb=(c["bmi"] - 27.5) / 5, zp=(c["sbp"] - 132) / 18,
                zh=(c["heart_rate"] - 76) / 13, zn=(c["sodium"] - 139) / 4,
                zc=(np.log(c["creatinine"]) - np.log(85)) / 0.3,
                sm=c["smoker"], dm=c["diabetes"], ma=c["male"])


def backbone_terms(X):
    z = _z(X)
    return {"const": np.full(len(X), -1.2), "age (linear)": 0.7 * z["za"], "creatinine (linear in log)": 0.5 * z["zc"],
            "bmi (linear)": 0.3 * z["zb"], "smoker step": 0.8 * z["sm"], "diabetes step": 0.6 * z["dm"],
            "male step": 0.3 * z["ma"]}


def u_term(X, beta=None):
    z = _z(X)
    b = BETA_U if beta is None else beta
    return {f"sodium U: {b:g} * ((zn - ({C_Z:g}))^2 - 1), minimum at {SODIUM_SWEET_SPOT:g} mmol/L":
            b * ((z["zn"] - C_Z) ** 2 - 1.0)}


def saddle_term(X, beta=None):
    z = _z(X)
    b = BETA_S if beta is None else beta
    return {f"heart rate x systolic BP saddle: -{b:g} * zh * zp": -b * z["zh"] * z["zp"]}


def ground_truth(task, X, beta_u=None, beta_s=None):
    t = backbone_terms(X)
    if task in ("u", "both"):
        t.update(u_term(X, beta_u))
    if task in ("saddle", "both"):
        t.update(saddle_term(X, beta_s))
    return sum(t.values()), t


TASKS = ["u", "saddle", "both"]
DESCRIPTIONS = {
    "u": "f = backbone + BETA_U * ((zn + 0.5)^2 - 1): U-shaped risk on sodium, minimum 137 mmol/L",
    "saddle": "f = backbone - BETA_S * zh * zp: saddle between heart rate and systolic BP (opposing directions raise risk)",
    "both": "f = backbone + the sodium U + the heart rate x systolic BP saddle",
}


def make_synthetic(task, beta_u=None, beta_s=None):
    X = inputs()
    f, terms = ground_truth(task, X, beta_u, beta_s)
    r = np.random.default_rng(DATA_SEED + 1 + TASKS.index(task))
    y = (r.random(len(X)) < 1 / (1 + np.exp(-f))).astype(np.int64)
    return X, y, f, terms


def split_indices(y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
    return np.sort(tr), np.sort(te)


def prepare(task, beta_u=None, beta_s=None):
    """Raw and z-scored arrays, the ground-truth logit and its named terms."""
    X, y, f, terms = make_synthetic(task, beta_u, beta_s)
    tr, te = split_indices(y)
    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0)
    scale[scale == 0.0] = 1.0
    z = lambda A: (A - mean) / scale  # noqa: E731
    meta = dict(task=task, label=LABELS[task], ground_truth=DESCRIPTIONS[task], beta_u=BETA_U if beta_u is None else beta_u,
                beta_s=BETA_S if beta_s is None else beta_s, n_rows=int(len(y)), n_train=int(len(tr)), n_test=int(len(te)),
                positive_share=float(y.mean()), split_seed=SPLIT_SEED, data_seed=DATA_SEED,
                scaler="StandardScaler fit on the training rows", class_names=["0", "1"], encoding="raw units; binaries 0/1")
    return dict(X_raw=X, y_all=y, names=list(FEATURES), types=dict(TYPES), train_rows=tr, test_rows=te,
                X_train=z(X[tr]), y_train=y[tr], X_test=z(X[te]), y_test=y[te], mean=mean, scale=scale,
                f=f, terms=terms, meta=meta)


def dataset_name(task):
    """Opaque database name (the explaining agent sees it)."""
    return "shape_syn_" + hashlib.sha1(f"ushape_saddle/{task}".encode()).hexdigest()[:5]


def registry_path():
    return HERE / "datasets.json"


def registry():
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {}
