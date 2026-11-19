"""Linear-term betas of Experiment 2's thirty networks (adult, churn100, heart100; v4), read-only.

Definition (one for all three datasets). Each network's logit (training rows) is projected by
least squares on a fixed basis of data functions ``[1, Phi]``: a linear block (one column per
number, one offset column per category level except the reference, aliases merged), then bends
(hinges) residualised on every linear column, then the interactions the explanations' units read,
residualised on every linear and bend column. A linear term's **beta** is its coefficient on the
linear block. Because every non-linear column is orthogonal to the linear block, this equals the
least-squares coefficient of the logit on the linear block alone (Frisch-Waugh), so it is defined
even when the same input also feeds a hinge: the hinge goes to its non-linear block. Churn and
Adult use the signal catalogues' bases (``experiments/churn_signals/basis.py``,
``experiments/adult_signals/catalogue.py``); Heart-100 gets the same construction here
(``heart_basis``). ``projection_<set>.csv`` gives the R^2 of each network's projection.

Units: change in the log-odds of the model's positive class per real unit of the input (per year,
per call, per minute, per US$1,000, per mm Hg, ...); for a yes/no input or a category level, the
change for that level against its reference (the most common level on the training rows).
Adult's positive class is an income of $50,000 a year or less (``adult_ohe_v2``: class 1 =
``<=50k``), so a positive beta raises the chance of a LOW income.

Aliases: churn's ``total X charge`` is a fixed multiple of ``total X minutes``; the two columns
enter as one usage term (the mean of their z-columns, the catalogue's merge), and its beta is per
minute with the charge moving with it (beta_minutes + rate * beta_charge in the explanation's
terms). Voice-mail plan and voice-mail messages stay separate; ``voicemail_net`` gives the net
change for a typical plan holder (plan + median messages among holders) per seed.

Adult education levels (``lin:edulevel``): the catalogue gives the 16 levels their own block next
to the 1-16 years-of-schooling line. Their betas are the levels' offsets from the straight line in
years: the projection's coefficients on the level indicators (with the linear block and the bends,
whose span is the catalogue's), minus their training-count-weighted least-squares line in years.

Reference: an L2 logistic regression (C = 1, sklearn's default) of the labels on the same linear
block (the networks' standardised inputs), training rows; ``lr_se`` from the inverse penalised
Hessian. Linear-path check: for an input that is purely linear in a seed (``exact_linear`` or
``support_linear`` in ``experiments/10seed/linearity_v2/<label>.json``), the explanation's own
linear-path coefficient (``per_unit`` ``score_per_unit``) is set beside the beta; ``agree`` when
they differ by at most 20% of the larger or by at most 0.05 logit over the input's training SD
(or over the level change).

Which terms and seeds: exactly the linear rows of the signal tick tables
(``thesis/ch7/exp2/signal_ticks.py``): a seed "uses" a term when its tick is set (churn / adult:
``matrix.csv`` ``used``; heart: a non-zero average effect in ``linear_core.csv``).

    DATABASE_URL=postgresql://localhost/explaneat_v4 \\
    EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \\
    uv run python experiments/linear_betas/betas.py [--refresh]

Every database session is read-only (``plain_language.load_model`` issues SET TRANSACTION READ
ONLY); the forward passes are cached in ``cache/`` (pickles, not committed).
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = HERE / "cache"
LIN_V2 = REPO / "experiments/10seed/linearity_v2"
SETS = {"adult": "adult", "churn": "churn100", "heart100": "heart100"}
REL_TOL, ABS_TOL = 0.20, 0.05
MIN_SHARE = 0.01           # main-table candidates: levels with at least 1% of the training rows
HB_MODIFIERS, HB_MIN_LEVEL, HB_MIN_ROWS = True, 10, 10   # heart basis options (see heart_basis)
TOP_K = {"lin:household": 4, "lin:work": 4, "lin:edulevel": 3, "lin:country": 3, "lin:race": 2,
         "lin:state": 4}

for p in (str(REPO), str(REPO / "experiments/churn_signals"), str(REPO / "experiments/adult_signals")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================================ load (cached)

def genome_id(label):
    from explaneat.export.plain_cli import genome_for_label
    with open(REPO / "experiments/10seed/runs.csv") as f:
        ids = {f"{r['dataset']}-s{r['seed']}": r["genome_id"] for r in csv.DictReader(f)}
    return ids.get(label) or genome_for_label(label)


def var_info(m):
    out = []
    for v in m.variables:
        d = dict(base=v.base, columns=list(v.columns), codes=list(v.codes), cat=v.is_category(),
                 unit=v.unit, short=v.short, mean=np.asarray(v.mean, float), scale=np.asarray(v.scale, float))
        if v.is_category():
            codes, cnt = np.unique(np.asarray(v.raw_train).astype(str), return_counts=True)
            d["counts"] = {str(c): int(n) for c, n in zip(codes, cnt)}
            d["labels"] = {str(c): v.level(c)["label"] for c in set(map(str, v.codes)) | set(d["counts"])}
            d["ref"] = str(codes[np.argmax(cnt)])
        else:
            d["median"] = float(np.median(np.asarray(v.raw_train, float)))
            d["sd"] = float(np.std(np.asarray(v.raw_train, float)))
        out.append(d)
    return out


def load_set(key, refresh=False):
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{key}.pkl"
    if fp.exists() and not refresh:
        return pickle.loads(fp.read_bytes())
    if "explaneat_v4" not in os.environ.get("DATABASE_URL", ""):
        sys.exit("set DATABASE_URL=postgresql://localhost/explaneat_v4 (read-only use)")
    from explaneat.analysis.plain_language import load_model
    D = dict(eta_tr={}, eta_te={})
    for s in range(10):
        lab = f"{SETS[key]}-s{s}"
        print("loading", lab, flush=True)
        m, meta = load_model(genome_id(lab))
        if key == "adult":
            assert meta["dataset"]["class_names"] == [">50k", "<=50k"]
        D["eta_tr"][s], D["eta_te"][s] = np.asarray(m.base_logit), np.asarray(m.base_logit_test)
        if s == 0:
            D.update(X=m.X, Xt=m.Xt, y=m.y, yt=m.yt, raw=m.raw_train, raw_te=m.raw_test,
                     names=list(m.feature_names), mean=m.mean, scale=m.scale, vars=var_info(m),
                     class_names=meta["dataset"]["class_names"])
            if key == "churn":
                from basis import build_basis
                B = build_basis(m)
                D["Phi_tr"], D["Phi_te"], D["blocks"] = B.Phi_tr, B.Phi_te, B.blocks
        else:
            assert np.max(np.abs(m.X - D["X"])) < 1e-12
    if key == "adult":
        from catalogue import build_basis as adult_basis
        B = adult_basis(dict(X=D["X"], Xt=D["Xt"], raw=D["raw"], names=D["names"], mean=D["mean"],
                             scale=D["scale"]))
        D["Phi_tr"], D["Phi_te"], D["blocks"] = B["Phi_tr"], B["Phi_te"], B["blocks"]
    fp.write_bytes(pickle.dumps(D))
    return D


# ============================================================================ linear block

def vars_by_base(D):
    return {v["base"]: v for v in D["vars"]}


CHURN_ALIAS = {"total day minutes": "total day charge", "total eve minutes": "total eve charge",
               "total night minutes": "total night charge", "total intl minutes": "total intl charge"}


def linear_spec(key, D):
    """Columns of the linear block as (name, variable bases, function Z -> column)."""
    V = vars_by_base(D)
    spec = []
    skip = {"education"} if key == "adult" else set()
    charges = set(CHURN_ALIAS.values()) if key == "churn" else set()
    for v in D["vars"]:
        b = v["base"]
        if b in skip or b in charges:
            continue
        if key == "churn" and b in CHURN_ALIAS:
            c1, c2 = v["columns"][0], V[CHURN_ALIAS[b]]["columns"][0]
            spec.append((b, (b, CHURN_ALIAS[b]), lambda Z, c1=c1, c2=c2: 0.5 * (Z[:, c1] + Z[:, c2])))
            continue
        if v["cat"] and len(v["columns"]) > 1:
            for c, code in zip(v["columns"], v["codes"]):
                if code != v["ref"]:
                    spec.append((f"{b}={code}", (b,), lambda Z, c=c: Z[:, c]))
        else:
            c = v["columns"][0]
            spec.append((b, (b,), lambda Z, c=c: Z[:, c]))
    return spec


def L_of(spec, Z):
    return np.column_stack([f(Z) for _, _, f in spec])


def ols(A, e):
    """Least squares on [1, A]; columns scaled to unit SD first (conditioning), minimum-norm
    solution with a relative singular-value cut of 1e-10 (rank-deficient bases)."""
    sd = A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A1 = np.hstack([np.ones((len(A), 1)), A / sd])
    c, *_ = np.linalg.lstsq(A1, e, rcond=1e-10)
    c[1:] = c[1:] / sd
    return c


def r2(y, f):
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


# ============================================================================ heart basis

def heart_basis(D, L, lin_names=None, use_modifiers=HB_MODIFIERS, min_level=HB_MIN_LEVEL, min_rows=HB_MIN_ROWS):
    """Phi for Heart-100 in the catalogues' form: the linear block, bends (hinges at the training
    quartiles; vessels at 1 and 2), and the interactions of every pair of variables that one of the
    ten networks' switching units reads together (drivers and modifiers in
    ``experiments/heart100_audit/features.csv``): products of coarse bases (a number: z and its hinge
    at the median; a category: its 0/1 level indicators, levels with >= 10 rows; a product with a
    level kept when it is non-zero on >= 10 training rows). Bends are residualised on
    [1, linear], interactions on [1, linear, bends]."""
    V = vars_by_base(D)
    Z, Zt = D["X"], D["Xt"]

    def numcol(Zm, b):
        return Zm[:, V[b]["columns"][0]]

    num = [v["base"] for v in D["vars"] if not v["cat"]]
    shape = []   # (name, fn)
    for b in num:
        z = numcol(Z, b)
        if b == "ca":
            v = V[b]
            ks = [(k - v["mean"][0]) / v["scale"][0] for k in (1.0, 2.0)]
        else:
            ks = [float(np.quantile(z, q)) for q in (0.25, 0.5, 0.75)]
        for j, k in enumerate(ks):
            shape.append((f"shape:{b}", lambda Zm, b=b, k=k: np.maximum(numcol(Zm, b) - k, 0.0)))

    def coarse(b):
        v = V[b]
        if not v["cat"]:
            med = float(np.median(numcol(Z, b)))
            return [(b, lambda Zm, b=b: numcol(Zm, b)),
                    (f"{b}>med", lambda Zm, b=b, med=med: np.maximum(numcol(Zm, b) - med, 0.0))]
        out = []
        for j, (c, code) in enumerate(zip(v["columns"], v["codes"])):
            if code != v["ref"] and v["counts"].get(code, 0) >= min_level:      # the 0/1 level indicator
                m_, s_ = v["mean"][j], v["scale"][j]
                out.append((f"{b}={code}", lambda Zm, c=c, m_=m_, s_=s_: np.round(Zm[:, c] * s_ + m_)))
        if len(v["columns"]) == 1:      # not the case for heart (two-column yes/no), kept general
            out = [(b, lambda Zm, c=v["columns"][0]: Zm[:, c])]
        return out

    F = pd.read_csv(REPO / "experiments/heart100_audit/features.csv")
    F = F[F.level == "unit"]
    pairs = set()
    for _, r in F.iterrows():
        vs = set(str(r["drivers"]).split())
        if use_modifiers and isinstance(r["modifiers"], str):
            vs |= set(r["modifiers"].split())
        vs = {x for x in vs if x in V}
        for a, b in combinations(sorted(vs), 2):
            pairs.add((a, b))
    inter = []
    for a, b in sorted(pairs):
        for na, fa in coarse(a):
            for nb, fb in coarse(b):
                if V[a]["cat"] or V[b]["cat"]:
                    if np.sum(np.abs(fa(Z) * fb(Z)) > 1e-9) < min_rows:
                        continue    # a level product present on too few training rows
                inter.append((f"int:{a}*{b}", lambda Zm, fa=fa, fb=fb: fa(Zm) * fb(Zm)))

    def raw_cols(Zm):
        return (np.column_stack([f(Zm) for _, f in shape]), np.column_stack([f(Zm) for _, f in inter]))

    S, I = raw_cols(Z)
    St, It = raw_cols(Zt)
    Lt = L["test"]
    one, onet = np.ones((len(Z), 1)), np.ones((len(Zt), 1))
    A = np.hstack([one, L["train"]])
    cs, *_ = np.linalg.lstsq(A, S, rcond=None)
    S_r, St_r = S - A @ cs, St - np.hstack([onet, Lt]) @ cs
    A2 = np.hstack([A, S])
    ci, *_ = np.linalg.lstsq(A2, I, rcond=None)
    I_r, It_r = I - A2 @ ci, It - np.hstack([onet, Lt, St]) @ ci
    mu = L["train"].mean(axis=0)
    P_tr = np.hstack([L["train"] - mu, S_r, I_r])
    P_te = np.hstack([Lt - mu, St_r, It_r])
    keep = [i for i in range(P_tr.shape[1]) if P_tr[:, i].std() > 1e-8]
    nL = L["train"].shape[1]
    nS = S.shape[1]
    blocks = {"lin": [], "shape": [], "int": []}
    for j, i in enumerate(keep):
        blocks["lin" if i < nL else "shape" if i < nL + nS else "int"].append(j)
    names = list(lin_names or [f"lin:{i}" for i in range(nL)]) + [n for n, _ in shape] + [n for n, _ in inter]
    signals = {}
    for j, i in enumerate(keep):
        signals.setdefault(names[i], []).append(j)
    return dict(Phi_tr=P_tr[:, keep], Phi_te=P_te[:, keep], signals=signals,
                blocks=blocks, pairs=sorted(pairs),
                n_cols=dict(linear=nL, bends=S.shape[1], interactions=I.shape[1]))


# ============================================================================ terms

def churn_rate(D, minutes, charge):
    V = vars_by_base(D)
    a = D["raw"][:, V[minutes]["columns"][0]]
    b = D["raw"][:, V[charge]["columns"][0]]
    return float(np.polyfit(a, b, 1)[0])


def point(D, base_row, sets):
    """A z-row: ``base_row`` with variables set to raw values / category codes."""
    V = vars_by_base(D)
    x = base_row.copy()
    for b, val in sets.items():
        v = V[b]
        if v["cat"]:
            for c, code, m, s in zip(v["columns"], v["codes"], v["mean"], v["scale"]):
                x[c] = ((1.0 if code == str(val) else 0.0) - m) / s
        else:
            x[v["columns"][0]] = (float(val) - v["mean"][0]) / v["scale"][0]
    return x


UNIT_WORD = {"years": "per year", "minutes": "per minute", "calls": "per call", "messages": "per message",
             "hours": "per hour", "mm Hg": "per mm Hg", "mg/dL": "per mg/dL", "beats per minute": "per beat/min",
             "mm": "per mm", "vessels": "per vessel", "on the 1-16 scale": "per step (1--16 scale)"}

# signal id -> list of (variable base, kind) for the terms of the tick tables' linear rows
SIG_VARS = {
    "churn": {"lin:csc": ["number customer service calls"], "lin:day": ["total day minutes"],
              "lin:eve": ["total eve minutes"], "lin:night": ["total night minutes"],
              "lin:intlplan": ["international plan"], "lin:vmail": ["voice mail plan", "number vmail messages"],
              "lin:state": ["state"], "lin:intl": ["total intl minutes"], "lin:daycalls": ["total day calls"],
              "lin:acct": ["account length"], "lin:intlcalls": ["total intl calls"],
              "lin:area": ["area code"], "lin:evecalls": ["total eve calls"], "lin:nightcalls": ["total night calls"]},
    "adult": {"lin:household": ["marital-status", "relationship"], "lin:capgain": ["capital-gain"],
              "lin:edu": ["education-num"], "lin:age": ["age"], "lin:hours": ["hours-per-week"],
              "lin:work": ["occupation", "workclass"], "lin:sex": ["sex"], "lin:caploss": ["capital-loss"],
              "lin:edulevel": ["education"], "lin:country": ["native-country"], "lin:race": ["race"],
              "lin:fnlwgt": ["fnlwgt"]},
}
NAMES = {"lin:csc": "Service calls", "lin:day": "Day usage", "lin:eve": "Evening usage",
         "lin:night": "Night usage", "lin:intlplan": "International plan", "lin:vmail": "Voice mail",
         "lin:state": "State", "lin:intl": "International minutes", "lin:daycalls": "Day calls",
         "lin:acct": "Account length", "lin:intlcalls": "International calls",
         "lin:household": "Marital status and household role", "lin:capgain": "Capital gain",
         "lin:edu": "Education (years)", "lin:age": "Age", "lin:hours": "Hours per week",
         "lin:work": "Occupation and work class", "lin:sex": "Sex", "lin:caploss": "Capital loss",
         "lin:edulevel": "Education levels", "lin:country": "Country of birth", "lin:race": "Race"}
HEART_SHORT = {"age": "Age", "ca": "Vessels coloured", "chol": "Cholesterol", "cp=1": "Typical angina",
               "cp=2": "Atypical angina", "cp=3": "Non-anginal pain", "exang=1": "Exercise angina",
               "fbs=1": "Fasting sugar above 120", "oldpeak": "ST depression", "restecg=1": "ECG ST-T abnormality",
               "restecg=2": "ECG LV hypertrophy", "sex=0": "Female (vs male)", "slope=2": "Flat ST slope",
               "slope=3": "Downsloping ST slope", "thal=4": "Thallium not recorded",
               "thal=6": "Thallium fixed defect", "thal=7": "Thallium reversible defect",
               "thalach": "Maximum heart rate", "trestbps": "Resting blood pressure"}


def numeric_term(D, b, signal, name):
    V = vars_by_base(D)
    v = V[b]
    step, unit = 1.0, UNIT_WORD.get(v["unit"], f"per {v['unit']}" if v["unit"] else "per unit (none recorded)")
    if v["unit"] == "US dollars":
        step, unit = 1000.0, "per US\\$1,000"
    sets_a, sets_b = {b: v["median"] + step}, {b: v["median"]}
    if b in CHURN_ALIAS:
        ch = CHURN_ALIAS[b]
        rate = churn_rate(D, b, ch)
        cm = V[ch]["median"]
        sets_a[ch], sets_b[ch] = cm + rate * step, cm
    return dict(signal=signal, term=b, level="", name=name, unit=unit, sets_a=sets_a, sets_b=sets_b,
                step=step, sd_x=v["sd"], vars=[b] + ([CHURN_ALIAS[b]] if b in CHURN_ALIAS else []))


def level_terms(D, b, signal, name, include_ref_note=True):
    v = vars_by_base(D)[b]
    out = []
    ref = v["ref"]
    ref_lab = v["labels"][ref]
    n = sum(v["counts"].values())
    for code in v["codes"]:
        if code == ref or v["counts"].get(code, 0) == 0:
            continue
        lab = v["labels"][code]
        out.append(dict(signal=signal, term=f"{b}={code}", level=lab, name=name, unit=f"vs {ref_lab}",
                        sets_a={b: code}, sets_b={b: ref}, step=1.0, sd_x=None, vars=[b],
                        share=v["counts"].get(code, 0) / n, variable=b))
    return out


def binary_term(D, b, signal, name, level_name=None):
    v = vars_by_base(D)[b]
    ref = v["ref"]
    other = [c for c in v["codes"] if c != ref] or [c for c in ["0", "1"] if c != ref]
    if len(v["columns"]) == 1:
        codes = ["1", "0"] if v["codes"][0] == "1" else [v["codes"][0], "__none__"]
        other = [c for c in codes if c != ref]
    code = other[0]
    lab, ref_lab = v["labels"][code], v["labels"][ref]
    return dict(signal=signal, term=f"{b}={code}", level="", name=level_name or name,
                unit=f"{lab} vs {ref_lab}", sets_a={b: code}, sets_b={b: ref}, step=1.0, sd_x=None, vars=[b])


def build_terms(key, D):
    terms = []
    if key == "heart100":
        lc = pd.read_csv(REPO / "experiments/heart100_audit/linear_core.csv")
        V = vars_by_base(D)
        for t in dict.fromkeys(lc.term):
            if "=" in t:
                b, code = t.split("=")
                v = V[b]
                assert code != v["ref"]
                ref_lab = v["labels"][v["ref"]]
                terms.append(dict(signal=t, term=t, level="", name=HEART_SHORT[t], unit=f"vs {ref_lab}",
                                  sets_a={b: code}, sets_b={b: v["ref"]}, step=1.0, sd_x=None, vars=[b],
                                  share=v["counts"].get(code, 0) / sum(v["counts"].values())))
            else:
                terms.append(numeric_term(D, t, t, HEART_SHORT[t]))
        return terms
    for sig, bases in SIG_VARS[key].items():
        name = NAMES.get(sig, sig)
        if key == "adult" and sig in ("lin:household", "lin:work"):
            terms += adult_combined(D, sig, name)
            continue
        for b in bases:
            v = vars_by_base(D)[b]
            if b == "education":
                lt = level_terms(D, b, sig, name)
                for t in lt:
                    t["unit"] = "offset from the straight line in years of schooling"
                terms += lt
            elif not v["cat"]:
                t = numeric_term(D, b, sig, name)
                if b == "number vmail messages":
                    t["name"] = "Voice-mail messages"
                terms.append(t)
            elif len(v["columns"]) == 1 or len(v["codes"]) == 2 and b in ("sex",):
                nm = {"voice mail plan": "Voice-mail plan", "international plan": "International plan",
                      "sex": "Sex"}.get(b, name)
                terms.append(binary_term(D, b, sig, nm))
            else:
                terms += level_terms(D, b, sig, name)
    return terms


def adult_combined(D, sig, name):
    """Adult's aliased one-hot groups. Household: marital status and household role are read
    together (a husband or wife is married), so a term is one observed (marital status, role)
    pair against the most common pair. Work: a missing occupation and a missing work class are
    the same people (2,222 of 2,231 rows), so "job record missing" sets both; "never worked" is
    work class never-worked with occupation missing; other levels are single offsets."""
    V = vars_by_base(D)
    n = len(D["raw"])
    out = []
    if sig == "lin:household":
        ms, rl = V["marital-status"], V["relationship"]
        cm = [ms["columns"][ms["codes"].index(c)] for c in ms["codes"]]
        cr = [rl["columns"][rl["codes"].index(c)] for c in rl["codes"]]
        cnt = {}
        for i, a in enumerate(ms["codes"]):
            for j, b in enumerate(rl["codes"]):
                k = int(np.sum((D["raw"][:, cm[i]] > 0.5) & (D["raw"][:, cr[j]] > 0.5)))
                if k:
                    cnt[(a, b)] = k
        ref = max(cnt, key=cnt.get)
        ref_lab = f"{ms['labels'][ref[0]]}, {rl['labels'][ref[1]]}"
        for (a, b), k in sorted(cnt.items(), key=lambda kv: -kv[1]):
            if (a, b) == ref:
                continue
            out.append(dict(signal=sig, term=f"marital-status={a}&relationship={b}",
                            level=f"{ms['labels'][a]}, {rl['labels'][b]}", name=name, unit=f"vs {ref_lab}",
                            sets_a={"marital-status": a, "relationship": b},
                            sets_b={"marital-status": ref[0], "relationship": ref[1]}, step=1.0, sd_x=None,
                            vars=["marital-status", "relationship"], share=k / n))
        return out
    oc, wc = V["occupation"], V["workclass"]
    ref = {"occupation": oc["ref"], "workclass": wc["ref"]}
    ref_lab = f"{wc['labels'][wc['ref']]} work class, {oc['labels'][oc['ref']]}"

    def rows(**kw):
        m_ = np.ones(n, bool)
        for b, code in kw.items():
            v = V[b]
            m_ &= D["raw"][:, v["columns"][v["codes"].index(code)]] > 0.5
        return int(m_.sum())

    special = [("occupation=0&workclass=0", "job record missing (occupation and work class)", {"occupation": "0", "workclass": "0"}),
               ("occupation=0&workclass=3", "never worked (occupation missing)", {"occupation": "0", "workclass": "3"})]
    for term, lab, sets in special:
        out.append(dict(signal=sig, term=term, level=lab, name=name, unit=f"vs {ref_lab}", sets_a=sets,
                        sets_b=dict(ref), step=1.0, sd_x=None, vars=["occupation", "workclass"],
                        share=rows(**sets) / n))
    for b, v, skip in (("occupation", oc, {"0"}), ("workclass", wc, {"0", "3"})):
        for code in v["codes"]:
            if code == v["ref"] or code in skip or v["counts"].get(code, 0) == 0:
                continue
            out.append(dict(signal=sig, term=f"{b}={code}", level=f"{v['labels'][code]} ({'occupation' if b == 'occupation' else 'work class'})",
                            name=name, unit=f"vs {ref_lab}", sets_a={b: code}, sets_b={b: v["ref"]}, step=1.0,
                            sd_x=None, vars=[b], share=v["counts"][code] / n))
    return out


def term_vector(D, spec, t, base_row):
    xa = point(D, base_row, t["sets_a"])[None, :]
    xb = point(D, base_row, t["sets_b"])[None, :]
    return (L_of(spec, xa) - L_of(spec, xb))[0]          # per step (1 unit, or US$1,000)


# ============================================================================ explanation's linear path

def path_coef(label, t, D):
    """The explanation's linear-path coefficient for a term when all its inputs are purely linear in
    that seed (exact_linear / support_linear, or not read), else None."""
    J = json.loads((LIN_V2 / f"{label}.json").read_text())
    st = {}
    for c in J["linearity_claims_after"]:
        s = c["statement"]
        if s.get("kind") == "variable":
            st[s["variable"]] = s
    if not any(b in st for b in t["vars"]):
        return None, "not read"
    for b in t["vars"]:
        if b in st and st[b]["class"] not in ("exact_linear", "support_linear"):
            return None, st[b]["class"]
    V = vars_by_base(D)
    tot = 0.0
    for b in t["vars"]:
        if b not in st:
            continue
        v = V[b]
        s_col = {int(p["column"]): float(p["score_per_unit"]) for p in st[b]["per_unit"]}
        if v["cat"]:
            for c, code in zip(v["columns"], v["codes"]):
                ia = 1.0 if code == str(t["sets_a"][b]) else 0.0
                ib = 1.0 if code == str(t["sets_b"][b]) else 0.0
                tot += s_col.get(c, 0.0) * (ia - ib)
        else:
            tot += s_col.get(v["columns"][0], 0.0) * (t["sets_a"][b] - t["sets_b"][b])
    return tot, "linear"


def agree(beta, pc, t, D):
    if pc is None:
        return ""
    d = abs(beta - pc)
    scale = t["sd_x"] / t["step"] if t["sd_x"] else 1.0     # logit change over one training SD, or one level change
    if d <= REL_TOL * max(abs(beta), abs(pc)) or d * scale <= ABS_TOL:
        return "agree"
    return "differs"


# ============================================================================ use (tick) rule

def used_matrix(key, terms):
    """seed -> set of term signals ticked (the tick tables' rule)."""
    out = {}
    if key in ("churn", "adult"):
        base = {"churn": "experiments/churn_signals", "adult": "experiments/adult_signals"}[key]
        M = pd.read_csv(REPO / base / "matrix.csv")
        M = M[M.signal.str.startswith("lin:")]
        for _, r in M.iterrows():
            out.setdefault(int(r.seed), {})[r.signal] = (
                "opp" if (str(r.used) == "True" and float(r.corr_with_data) <= 0) else
                "ok" if str(r.used) == "True" else None)
    else:
        lc = pd.read_csv(REPO / "experiments/heart100_audit/linear_core.csv")
        eff = {}
        for _, r in lc.iterrows():
            eff.setdefault(r.term, [0.0] * 10)[int(r.network.rsplit("-s", 1)[1])] = float(r.avg_effect)
        from collections import Counter
        for term, vals in eff.items():
            sg = [(1 if v > 1e-9 else -1 if v < -1e-9 else 0) for v in vals]
            nz = [x for x in sg if x]
            maj = Counter(nz).most_common(1)[0][0] if nz else 0
            for s, x in enumerate(sg):
                out.setdefault(s, {})[term] = None if x == 0 else ("ok" if x == maj else "opp")
    return out


# ============================================================================ main per set

def lr_reference(L, y):
    lr = LogisticRegression(C=1.0, max_iter=20000, tol=1e-8).fit(L, y)
    coef = np.concatenate([lr.intercept_, lr.coef_[0]])
    A = np.hstack([np.ones((len(L), 1)), L])
    p = 1 / (1 + np.exp(-(A @ coef)))
    H = A.T @ (A * (p * (1 - p))[:, None])
    H[1:, 1:] += np.eye(L.shape[1])           # penalty 1/(2C) |w|^2 with C = 1
    cov = np.linalg.pinv(H)
    return coef, cov


def edulevel(D, key, spec, Phi_shape, eta, y=None):
    """Education level offsets from the straight line in years (adult only)."""
    V = vars_by_base(D)
    v = V["education"]
    Z = D["X"]
    En = [c for c, code in zip(v["columns"], v["codes"]) if code != v["ref"]]
    codes = [code for code in v["codes"] if code != v["ref"]]
    L = L_of(spec, Z)
    yrs = {}
    en_col = V["education-num"]["columns"][0]
    raw_en = D["raw"][:, en_col]
    for c, code in zip(v["columns"], v["codes"]):
        rows = D["raw"][:, c] > 0.5
        if rows.any():
            yrs[code] = float(np.median(raw_en[rows]))
            assert np.all(raw_en[rows] == yrs[code])
    cnt = v["counts"]

    def offsets(f):          # f: code -> value; remove weighted LS line in years
        cs = [c for c in v["codes"] if cnt.get(c, 0) > 0]
        x = np.array([yrs[c] for c in cs]); w = np.array([cnt[c] for c in cs], float)
        fv = np.array([f[c] for c in cs])
        W = np.diag(w); A = np.column_stack([np.ones(len(x)), x])
        ab = np.linalg.solve(A.T @ W @ A, A.T @ W @ fv)
        return {c: float(fv[i] - A[i] @ ab) for i, c in enumerate(cs)}

    if y is None:
        A = np.hstack([L, Phi_shape, Z[:, En]])
        c = ols(A, eta)
        cl = c[1 + L.shape[1] + Phi_shape.shape[1]:]
    else:
        A = np.hstack([L, Z[:, En]])
        lr = LogisticRegression(C=1.0, max_iter=20000, tol=1e-8).fit(A, y)
        cl = lr.coef_[0][L.shape[1]:]
    # per raw indicator: z-column coefficient / scale; the reference level is 0
    f = {v["ref"]: 0.0}
    for code, col, b in zip(codes, En, cl):
        f[code] = float(b / v["scale"][v["columns"].index(col)])
    return offsets(f)


def run_set(key, refresh=False):
    D = load_set(key, refresh)
    spec = linear_spec(key, D)
    Ltr, Lte = L_of(spec, D["X"]), L_of(spec, D["Xt"])
    terms = build_terms(key, D)
    base_row = D["X"].mean(axis=0)
    G = np.array([term_vector(D, spec, t, base_row) for t in terms])   # terms x linear cols
    # full projection basis
    if key == "heart100":
        HB = heart_basis(D, dict(train=Ltr, test=Lte), [f"lin:{vs[0]}" for _, vs, _ in spec])
        Phi_tr, Phi_te, blocks = HB["Phi_tr"], HB["Phi_te"], HB["blocks"]
        lin_idx = blocks["lin"]
    else:
        Phi_tr, Phi_te, blocks = D["Phi_tr"], D["Phi_te"], D["blocks"]
        lin_idx = [i for b, idx in blocks.items() if b.startswith("lin:") and b != "lin:edulevel" for i in idx]
    used = used_matrix(key, terms)
    lr_c, lr_cov = lr_reference(Ltr, D["y"])
    proj_rows, rows = [], []
    for s in range(10):
        lab = f"{SETS[key]}-s{s}"
        e_tr, e_te = D["eta_tr"][s], D["eta_te"][s]
        c_full = ols(Phi_tr, e_tr)
        f_tr = c_full[0] + Phi_tr @ c_full[1:]
        f_te = c_full[0] + Phi_te @ c_full[1:]
        c_lin = ols(Ltr, e_tr)
        lin_fit = c_lin[0] + Ltr @ c_lin[1:]
        # Frisch-Waugh check: the linear block's component in the full projection = the linear-only fit
        comp = Phi_tr[:, lin_idx] @ c_full[1:][lin_idx]
        fw = float(np.max(np.abs((comp - comp.mean()) - (lin_fit - lin_fit.mean()))))
        assert fw < 1e-6 * max(1.0, float(np.std(e_tr))), (lab, fw)
        proj_rows.append(dict(network=lab, seed=s, r2_train=r2(e_tr, f_tr), r2_test=r2(e_te, f_te),
                              r2_linear_train=r2(e_tr, lin_fit),
                              r2_linear_test=r2(e_te, c_lin[0] + Lte @ c_lin[1:]),
                              n_basis_columns=Phi_tr.shape[1], frisch_waugh_max_diff=fw))
        betas = G @ c_lin[1:]
        if key == "adult":
            shp = [i for b, idx in blocks.items() if b.startswith("shape:") for i in idx]
            edu = edulevel(D, key, spec, Phi_tr[:, shp], e_tr)
        for t, bt in zip(terms, betas):
            if t["signal"] == "lin:edulevel":
                code = t["term"].split("=")[1]
                bt = edu.get(code, float("nan"))
            pc, why = path_coef(lab, t, D) if t["signal"] != "lin:edulevel" else (None, "n/a")
            state = used.get(s, {}).get(t["signal"])
            rows.append(dict(seed=s, network=lab, signal=t["signal"], term=t["term"], name=t["name"],
                             level=t["level"], beta=float(bt), unit=t["unit"], used=state is not None,
                             direction_vs_most=state or "", path_coef=pc, path_status=why,
                             agreement=agree(float(bt), pc, t, D) if state is not None else ""))
    # reference
    lr_beta = G @ lr_c[1:]
    lr_se = np.sqrt(np.einsum("ij,jk,ik->i", G, lr_cov[1:, 1:], G))
    if key == "adult":
        edu_lr = edulevel(D, key, spec, None, None, y=D["y"])
    ref = {}
    for t, b, se in zip(terms, lr_beta, lr_se):
        if t["signal"] == "lin:edulevel":
            b, se = edu_lr.get(t["term"].split("=")[1], float("nan")), float("nan")
        ref[t["term"]] = (float(b), float(se))
    R = pd.DataFrame(rows)
    R["lr_beta"] = R.term.map(lambda x: ref[x][0])
    R["lr_se"] = R.term.map(lambda x: ref[x][1])
    R["share_train"] = R.term.map({t["term"]: t.get("share") for t in terms})
    P = pd.DataFrame(proj_rows)
    extra = {}
    if key == "churn":
        V = vars_by_base(D)
        plan = D["raw"][:, V["voice mail plan"]["columns"][0]] > 0.5
        msg = D["raw"][:, V["number vmail messages"]["columns"][0]]
        med = float(np.median(msg[plan]))
        net = []
        for s in range(10):
            rp = R[(R.seed == s) & (R.term == "voice mail plan=1")].iloc[0]
            rm = R[(R.seed == s) & (R.term == "number vmail messages")].iloc[0]
            bp, bm = rp.beta, rm.beta
            pn = (rp.path_coef + med * rm.path_coef) if (pd.notna(rp.path_coef) and pd.notna(rm.path_coef)) else None
            nt = bp + med * bm
            ag = "" if pn is None else ("agree" if abs(nt - pn) <= max(REL_TOL * max(abs(nt), abs(pn)), ABS_TOL) else "differs")
            net.append(dict(seed=s, plan_beta=bp, messages_beta=bm, median_messages_holders=med,
                            net_typical_holder=nt, path_net=pn, agreement=ag))
        lp = ref["voice mail plan=1"][0]; lm = ref["number vmail messages"][0]
        extra["voicemail"] = dict(rows=net, median_messages=med, lr_net=lp + med * lm,
                                  corr=float(np.corrcoef(plan, msg)[0, 1]))
        pd.DataFrame(net).to_csv(HERE / "voicemail_net_churn.csv", index=False)
    if key == "heart100":
        extra["heart_basis"] = dict(pairs=HB["pairs"], n_cols=HB["n_cols"])
    return R, P, extra


def summarise(key, R):
    out = []
    for term, g in R.groupby("term", sort=False):
        u = g[g.used]
        b = u.beta.values
        row = dict(signal=g.signal.iloc[0], term=term, name=g.name.iloc[0], level=g.level.iloc[0],
                   unit=g.unit.iloc[0], n_used=len(u), share_train=g.share_train.iloc[0],
                   median=float(np.median(b)) if len(b) else float("nan"),
                   min=float(b.min()) if len(b) else float("nan"), max=float(b.max()) if len(b) else float("nan"),
                   lr_beta=g.lr_beta.iloc[0], lr_se=g.lr_se.iloc[0],
                   n_path_checked=int((u.agreement != "").sum()), n_path_differs=int((u.agreement == "differs").sum()))
        a = np.abs(b[np.abs(b) > 0])
        row["sign_differs"] = bool(len(b) and (b.min() < 0 < b.max()))
        row["size_ratio"] = float(a.max() / a.min()) if len(a) >= 2 else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def main():
    refresh = "--refresh" in sys.argv
    meta = {}
    for key in SETS:
        R, P, extra = run_set(key, refresh)
        R.to_csv(HERE / f"betas_{key}.csv", index=False)
        P.to_csv(HERE / f"projection_{key}.csv", index=False)
        S = summarise(key, R)
        S.to_csv(HERE / f"summary_{key}.csv", index=False)
        meta[key] = dict(r2_train_min=float(P.r2_train.min()), r2_train_median=float(P.r2_train.median()),
                         r2_test_min=float(P.r2_test.min()), r2_test_median=float(P.r2_test.median()),
                         r2_linear_train_median=float(P.r2_linear_train.median()),
                         n_basis_columns=int(P.n_basis_columns.iloc[0]), **{k: v for k, v in extra.items()})
        print(f"== {key}: R2 train min {P.r2_train.min():.4f} median {P.r2_train.median():.4f}; "
              f"test min {P.r2_test.min():.4f}; linear-only median {P.r2_linear_train.median():.3f}")
        u = R[R.used]
        print(f"   path checks: {int((u.agreement != '').sum())}, differ: {int((u.agreement == 'differs').sum())}")
    (HERE / "meta.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
