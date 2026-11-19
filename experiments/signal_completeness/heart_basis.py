"""A data basis for the Heart-100 signal list, for the completeness check only.

The Heart-100 tick table's non-linear rows are *families of switching units* (the inputs that switch
a unit, ``experiments/heart100_audit/features.csv``), not data-defined blocks as for Adult and
Churn. To project a network's logit on "the signals the table lists", each row is given a block of
data functions here, chosen from the drivers and modifiers its units read (design choices, in the
style of ``experiments/adult_signals/catalogue.py``):

- linear core (one block per linear-core term): ``heart100_audit.audit.linear_basis``;
- bends (hinges at the 10/25/50/75/90% training quantiles; residualised on every linear column):
  ``shape:thalach`` (Maximum-heart-rate bend), ``shape:trestbps`` (Resting blood pressure
  threshold), ``shape:age`` (Age brake), ``shape:oldpeak`` (ST-depression threshold; knots at
  0.5, 1, 2, 3 mm since most patients have 0), ``shape:chol`` (Other: cholesterol);
- interactions (residualised on every linear and bend column), "defect" = a fixed or reversible
  thallium defect, "flat" = a flat ST segment:
  ``int:thal*oldpeak`` (Thallium-scan gate, with ST depression), ``int:thal*slope`` (the same
  gate with the ST slope), ``int:thal*age`` (Thallium scan x age), ``int:exang*sex`` (Exercise
  angina gate with sex), ``int:bp*other`` (resting BP threshold with slope, heart rate, angina or
  scan), ``int:slope*cp`` and ``int:slope*chol`` (Flat ST step with chest pain or cholesterol),
  ``int:restecg*oldpeak`` (Resting-ECG gate with its modifiers ST depression and heart rate),
  ``int:slope*age`` and ``int:ca*age`` (Other: ST slope x age, vessels x age).

A single category read through a gate (chest pain, fasting sugar, sex alone) is linear on the data
and is held by the linear core.
"""
from __future__ import annotations

import numpy as np

Q = [0.10, 0.25, 0.50, 0.75, 0.90]
OLDPEAK_KNOTS = [0.5, 1.0, 2.0, 3.0]

# block -> (tick-table row, data variables)
BLOCKS = {
    "shape:thalach": ("Maximum-heart-rate bend", ["thalach"]),
    "shape:trestbps": ("Resting blood pressure", ["trestbps"]),
    "shape:age": ("Age brake", ["age"]),
    "shape:oldpeak": ("ST-depression threshold", ["oldpeak"]),
    "shape:chol": ("Other: cholesterol", ["chol"]),
    "int:thal*oldpeak": ("Thallium-scan gate", ["thal", "oldpeak"]),
    "int:thal*slope": ("Thallium-scan gate", ["thal", "slope"]),
    "int:thal*age": ("Thallium scan x age", ["thal", "age"]),
    "int:exang*sex": ("Exercise angina", ["exang", "sex"]),
    "int:bp*other": ("Resting blood pressure", ["trestbps", "slope", "thalach", "exang", "thal"]),
    "int:slope*cp": ("Flat ST segment step", ["slope", "cp"]),
    "int:slope*chol": ("Flat ST segment step", ["slope", "chol"]),
    "int:restecg*oldpeak": ("Resting-ECG gate", ["restecg", "oldpeak", "thalach"]),
    "int:slope*age": ("Other: ST slope x age", ["slope", "age"]),
    "int:ca*age": ("Other: vessels x age", ["ca", "age"]),
}


def relu(x):
    return np.maximum(x, 0.0)


def build(m, linear_basis):
    """Phi (train, test), block -> column indices, block -> variables. ``linear_basis`` is
    ``heart100_audit.audit.linear_basis``."""
    V = m.var_by_base
    L_tr, terms = linear_basis(m, m.X)
    L_te, _ = linear_basis(m, m.Xt)

    def num(b, test):
        return np.asarray(V[b].raw_test if test else V[b].raw_train, float)

    def code(b, test):
        return np.asarray(V[b].raw_test if test else V[b].raw_train).astype(str)

    knots = {b: [float(np.quantile(num(b, False), q)) for q in Q] for b in ("thalach", "trestbps", "age", "chol")}
    knots["oldpeak"] = OLDPEAK_KNOTS
    med = {b: float(np.median(num(b, False))) for b in ("age", "trestbps", "oldpeak", "thalach", "chol")}

    def cols(test):
        out, names, blocks = [], [], {}

        def add(blk, name, a):
            blocks.setdefault(blk, []).append(len(names))
            names.append(name)
            out.append(np.asarray(a, float))

        for b in ("thalach", "trestbps", "age", "oldpeak", "chol"):
            for j, t in enumerate(sorted(set(knots[b]))):
                add(f"shape:{b}", f"{b}>{t:g}", relu(num(b, test) - t))
        n_main = len(names)
        defect = np.isin(code("thal", test), ["6", "7"]).astype(float)
        flat = (code("slope", test) == "2").astype(float)
        female = (code("sex", test) == "0").astype(float)
        exang = (code("exang", test) == "1").astype(float)
        asym = (code("cp", test) == "4").astype(float)
        lvh = (code("restecg", test) == "2").astype(float)
        op = num("oldpeak", test)
        age = num("age", test)
        add("int:thal*oldpeak", "defect*oldpeak", defect * op)
        add("int:thal*oldpeak", "defect*oldpeak>1", defect * relu(op - 1.0))
        add("int:thal*slope", "defect*flat", defect * flat)
        add("int:thal*age", "defect*age", defect * age)
        add("int:thal*age", "defect*age>med", defect * relu(age - med["age"]))
        add("int:exang*sex", "exang*female", exang * female)
        bp = relu(num("trestbps", test) - med["trestbps"])
        for nm, a in (("flat", flat), ("thalach", num("thalach", test)), ("exang", exang), ("defect", defect)):
            add("int:bp*other", f"bp>med*{nm}", bp * a)
        add("int:slope*cp", "flat*asymptomatic", flat * asym)
        add("int:slope*chol", "flat*chol", flat * num("chol", test))
        add("int:restecg*oldpeak", "lvh*oldpeak", lvh * op)
        add("int:restecg*oldpeak", "lvh*thalach", lvh * num("thalach", test))
        add("int:slope*age", "flat*age", flat * age)
        add("int:ca*age", "ca*age", num("ca", test) * age)
        return np.column_stack(out), names, blocks, n_main

    N_tr, names, nblocks, n_main = cols(False)
    N_te, _, _, _ = cols(True)
    # hierarchical residualisation on the training rows: bends on [1, linear], interactions on
    # [1, linear, bends]; the same coefficients applied to the test rows
    one_tr, one_te = np.ones((len(L_tr), 1)), np.ones((len(L_te), 1))
    A_tr = np.hstack([one_tr, L_tr])
    A_te = np.hstack([one_te, L_te])
    c1, *_ = np.linalg.lstsq(A_tr, N_tr[:, :n_main], rcond=None)
    S_tr = N_tr[:, :n_main] - A_tr @ c1
    S_te = N_te[:, :n_main] - A_te @ c1
    A2_tr, A2_te = np.hstack([A_tr, S_tr]), np.hstack([A_te, S_te])
    c2, *_ = np.linalg.lstsq(A2_tr, N_tr[:, n_main:], rcond=None)
    I_tr = N_tr[:, n_main:] - A2_tr @ c2
    I_te = N_te[:, n_main:] - A2_te @ c2
    mu = L_tr.mean(axis=0)
    P_tr = np.hstack([L_tr - mu, S_tr, I_tr])
    P_te = np.hstack([L_te - mu, S_te, I_te])
    blocks = {f"lin:{t}": [j] for t, j in terms}
    k = L_tr.shape[1]
    for b, idx in nblocks.items():
        blocks[b] = [k + i for i in idx]
    keep = [i for i in range(P_tr.shape[1]) if P_tr[:, i].std() > 1e-8]
    remap = {o: n for n, o in enumerate(keep)}
    blocks = {b: [remap[i] for i in idx if i in remap] for b, idx in blocks.items()}
    blocks = {b: idx for b, idx in blocks.items() if idx}
    block_vars = {f"lin:{t}": [t.split("=")[0]] for t, _ in terms}
    block_vars.update({b: v[1] for b, v in BLOCKS.items()})
    return P_tr[:, keep], P_te[:, keep], blocks, block_vars
