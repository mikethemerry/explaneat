"""Every row description of the Experiment 2 signal tables, checked by code.

Each table row (a signal) has a one-line plain description. This script splits every description
into its factual claims, tests each claim, and writes the text the tables print. Numbers and counts
in the text are filled from computed values.

- A claim about the *signal* is checked on the data's own fit: the ridge logistic fit of the labels
  on the catalogue basis (``experiments/{adult,churn}_signals/cache/betas.npz``), read through
  partial dependence. One or more inputs are set to grid values on 3,000 fixed training rows
  (``RandomState(0)``); the mean of the blocks' component is the curve (a one-variable signal:
  its own linear and bend blocks, as the tables' shape figures; an interaction: the whole fit). The same claim is then
  evaluated on each seed that uses the signal (its own projection on the basis) and reported as
  "k of n seeds"; it is not asserted.
- A claim about the *networks* (Heart-100 unit families; Churn "read together") is checked on the
  realisations: ``experiments/heart100_audit/features.csv`` (drivers, modifiers, on/off
  profile), ``family_presence.csv``, ``experiments/churn_signals/realisations.csv``.

A claim that fails stops the script (``AssertionError``) unless the description is generated from
the computed value. ``original`` keeps the text the tables printed before these checks (a record
of what changed, not used for output). Outputs, next to this file:

- ``descriptions.json``: {data set: {row key: text}}, read by ``thesis/ch7/exp2/signal_tables.py``
  for all three data sets (the Adult catalogue's own ``signals.csv`` description column is its
  unchecked draft and is no longer printed).
- ``claims.csv``: one row per claim (data set, row, claim, test, value, holds, seeds agreeing).
- ``rewrites.csv``: rows whose text differs from ``original``, with the evidence.

Needs ``experiments/signal_completeness/cache/{adult,churn}.pkl`` (``load.py``, read-only DB) and the
committed catalogue outputs.

    OMP_NUM_THREADS=1 uv run python experiments/signal_descriptions/describe.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO), str(REPO / "experiments/signal_completeness")):
    if p not in sys.path:
        sys.path.insert(0, p)
import load  # noqa: E402  (experiments/signal_completeness/load.py)
from explaneat.analysis.signal_presence import USED_SD, WEAK_SD  # noqa: E402

N_PD = 3000            # rows for partial dependence (fixed subsample, RandomState(0))
SMALL_SD = 0.25        # a "small" slope: the data's component SD below a quarter logit (design choice)
CLAIMS, REWRITES = [], []


def claim(ds, row, text, test, value, holds, seeds=""):
    CLAIMS.append(dict(dataset=ds, row=row, claim=text, test=test, value=value, holds=bool(holds),
                       seeds_agreeing=seeds))
    return bool(holds)


def must(ds, row, text, test, value, holds, seeds=""):
    ok = claim(ds, row, text, test, value, holds, seeds)
    assert ok, (ds, row, text, value)


def finish(ds, row, original, text, evidence=""):
    if original != text:
        REWRITES.append(dict(dataset=ds, row=row, original=original, new=text, evidence=evidence))
    return text


def project(Phi, e):
    A = np.hstack([np.ones((len(Phi), 1)), Phi])
    c, *_ = np.linalg.lstsq(A, e, rcond=None)
    return c[1:]


class PD:
    """Partial dependence of a set of basis blocks under the data's fit or a seed's projection."""

    def __init__(self, d, phi, blocks, colz):
        self.d, self.phi, self.blocks, self.colz = d, phi, blocks, colz
        rng = np.random.RandomState(0)
        self.rows = np.sort(rng.choice(len(d["X"]), min(N_PD, len(d["X"])), replace=False))

    def __call__(self, beta, blks, settings, rows=None):
        """Mean over the rows of the blocks' component with the inputs set (settings: {var: raw})."""
        X = self.d["X"][self.rows if rows is None else rows].copy()
        for var, v in settings.items():
            for c, z in self.colz(var, v).items():
                X[:, c] = z
        P = self.phi(X)
        idx = [i for b in blks for i in self.blocks[b]]
        return float(np.mean(P[:, idx] @ beta[idx]))

    def curve(self, beta, blks, var, grid, extra=None, rows=None):
        return np.array([self(beta, blks, {var: g, **(extra or {})}, rows) for g in grid])


def seeds_using(path):
    M = pd.read_csv(REPO / path)
    return {s: list(g[g.used].network) for s, g in M.groupby("signal")}


def agree(fn, betas, labs):
    """'k of n' seeds (of those using the signal) for which fn(beta) holds."""
    if not labs:
        return "no seed uses it"
    k = sum(bool(fn(betas[l])) for l in labs)
    return f"{k} of {len(labs)}"


# =============================================================================== Adult

def adult():
    ds = "adult"
    d = load.get(ds)
    cat = load.adult_catalogue()
    D = dict(X=d["X"], Xt=d["Xt"], raw=d["raw_train"], names=d["feature_names"], mean=d["mean"], scale=d["scale"])
    B = cat.build_basis(D)
    assert np.allclose(B["Phi_tr"], d["Phi_tr"])
    bz = np.load(REPO / "experiments/adult_signals/cache/betas.npz")
    beta_d = bz["beta_data"]
    assert len(beta_d) == B["Phi_tr"].shape[1]
    col = {n: i for i, n in enumerate(d["feature_names"])}
    R = d["raw_train"]

    def colz(var, v):
        c = col[var]
        return {c: (v - d["mean"][c]) / d["scale"][c]}

    pd_ = PD(d, B["phi"], B["blocks"], colz)
    ALL = list(B["blocks"])   # interaction claims read the whole fitted logit; one-variable bends their own blocks
    betas = {l: project(d["Phi_tr"], nt["eta_tr"]) for l, nt in d["nets"].items()}
    using = seeds_using("experiments/adult_signals/matrix.csv")
    comp = lambda beta, b: d["Phi_tr"][:, B["blocks"][b]] @ beta[B["blocks"][b]]
    married = R[:, col["marital-status:2"]] > 0.5
    rows_m = pd_.rows[married[pd_.rows]]
    rows_u = pd_.rows[~married[pd_.rows]]
    out = {}

    def down(b, var, lo, hi):
        f = lambda beta: pd_(beta, [b], {var: hi}) < pd_(beta, [b], {var: lo})
        must(ds, b, f"falls with {var}", f"PD of {b}: {var}={hi:g} below {var}={lo:g}",
             f"{pd_(beta_d, [b], {var: hi}) - pd_(beta_d, [b], {var: lo}):+.3f}", f(beta_d), agree(f, betas, using.get(b, [])))

    # ---- household (category): the existing checks, on group means of the data's component
    c = comp(beta_d, "lin:household")
    h = {k: float(c[R[:, col[k]] > 0.5].mean()) for k in
         ("relationship:0", "relationship:5", "relationship:4", "relationship:1", "relationship:3")}
    f = lambda beta: (lambda cc: max(cc[R[:, col["relationship:0"]] > .5].mean(), cc[R[:, col["relationship:5"]] > .5].mean())
                      < min(cc[R[:, col[k]] > .5].mean() for k in ("relationship:4", "relationship:1", "relationship:3")))(comp(beta, "lin:household"))
    must(ds, "lin:household", "husbands and wives sit below the unmarried, not in family and own child",
         "group means of the component", ", ".join(f"{k} {v:+.2f}" for k, v in h.items()), f(beta_d),
         agree(f, betas, using["lin:household"]))
    must(ds, "lin:household", "low income likelier outside marriage", "corr(component, married) < 0",
         f"{np.corrcoef(c, married)[0, 1]:+.2f}", np.corrcoef(c, married)[0, 1] < 0)
    out["lin:household"] = cat.SIGNALS["lin:household"][1]
    # ---- slopes
    down("lin:capgain", "capital-gain", 0, 10000)
    out["lin:capgain"] = cat.SIGNALS["lin:capgain"][1]
    down("lin:edu", "education-num", 9, 13)
    out["lin:edu"] = cat.SIGNALS["lin:edu"][1]
    down("lin:age", "age", 25, 55)
    out["lin:age"] = cat.SIGNALS["lin:age"][1]
    down("lin:hours", "hours-per-week", 30, 50)
    out["lin:hours"] = cat.SIGNALS["lin:hours"][1]
    down("lin:caploss", "capital-loss", 0, 1900)
    out["lin:caploss"] = cat.SIGNALS["lin:caploss"][1]
    # ---- capital-gain threshold: where the steepest fall starts
    grid = np.arange(0, 20001, 250.0)
    blks = ["lin:capgain", "shape:capgain"]   # a one-variable signal: its own blocks

    def steep(beta):
        g = pd_.curve(beta, blks, "capital-gain", grid)
        s = np.diff(g) / np.diff(grid)
        j = int(np.argmin(s))
        return grid[j], s[j], float(np.mean(s[grid[1:] <= 2000]))

    x0, s0, slo = steep(beta_d)
    must(ds, "shape:capgain", "the fall is steep past a threshold", "steepest slope at least 3x the slope below $2,000",
         f"{1000 * s0:+.3f} vs {1000 * slo:+.3f} per $1000", s0 <= 3 * slo and s0 < 0)
    must(ds, "shape:capgain", "the threshold is a few thousand dollars", "steepest segment starts between $1,000 and $10,000",
         f"${x0:,.0f}", 1000 <= x0 <= 10000,
         agree(lambda b: 1000 <= steep(b)[0] <= 10000, betas, using.get("shape:capgain", [])))
    out["shape:capgain"] = finish(ds, "shape:capgain", cat.SIGNALS["shape:capgain"][1],
                                  f"past about ${round(x0, -3):,.0f} of capital gain the chance of a low income drops steeply",
                                  f"steepest fall of the data's curve starts at ${x0:,.0f}")
    # ---- age bend
    def age_slopes(beta):
        g = pd_.curve(beta, ["lin:age", "shape:age"], "age", [20, 35, 50, 70])
        return (g[1] - g[0]) / 15, (g[3] - g[2]) / 20

    a1, a2 = age_slopes(beta_d)
    must(ds, "shape:age", "the fall with age is steep for the young", "slope 20-35 < 0 and below slope 50-70",
         f"{a1:+.3f} vs {a2:+.3f} per year", a1 < 0 and a1 < a2,
         agree(lambda b: (lambda s: s[0] < 0 and s[0] < s[1])(age_slopes(b)), betas, using.get("shape:age", [])))
    later = "turns back later in life" if a2 > 0 else "flattens later in life"
    claim(ds, "shape:age", later, "sign of the slope 50-70", f"{a2:+.3f} per year", True)
    out["shape:age"] = finish(ds, "shape:age", cat.SIGNALS["shape:age"][1],
                              f"the fall with age is steep for the young and {later}",
                              f"data's slope {a1:+.3f}/year at 20-35, {a2:+.3f}/year at 50-70")
    # ---- hours bend
    def hrs(beta):
        g = pd_.curve(beta, ["lin:hours", "shape:hours"], "hours-per-week", [10, 40, 70])
        return g[0] - g[1], g[1] - g[2]

    up, dn = hrs(beta_d)
    must(ds, "shape:hours", "short weeks raise it more than long weeks lower it",
         "PD(10 h) - PD(40 h) > PD(40 h) - PD(70 h) > 0", f"{up:+.3f} vs {dn:+.3f}", up > dn > 0,
         agree(lambda b: (lambda u: u[0] > u[1] > 0)(hrs(b)), betas, using.get("shape:hours", [])))
    out["shape:hours"] = cat.SIGNALS["shape:hours"][1]
    # ---- work
    c = comp(beta_d, "lin:work")
    w = {k: float(c[R[:, col[k]] > 0.5].mean()) for k in ("occupation:4", "occupation:10", "occupation:8", "occupation:6", "occupation:0")}
    must(ds, "lin:work", "executive/managerial and professional below; service, handling and missing above",
         "occupation means of the component", ", ".join(f"{k} {v:+.2f}" for k, v in w.items()),
         max(w["occupation:4"], w["occupation:10"]) < min(w["occupation:8"], w["occupation:6"], w["occupation:0"]))
    out["lin:work"] = cat.SIGNALS["lin:work"][1]
    # ---- sex
    c = comp(beta_d, "lin:sex")
    male = R[:, col["sex"]] > 0.5
    must(ds, "lin:sex", "women more likely to have a low income", "mean component women - men > 0",
         f"{c[~male].mean() - c[male].mean():+.3f}", c[~male].mean() > c[male].mean(),
         agree(lambda b: (lambda cc: cc[~male].mean() > cc[male].mean())(comp(b, "lin:sex")), betas, using["lin:sex"]))
    out["lin:sex"] = cat.SIGNALS["lin:sex"][1]
    # ---- bends whose text only claims a bend exists
    for b, base in (("shape:caploss", "bends in the capital-loss effect across loss amounts"),
                    ("shape:fnlwgt", "bends in the survey sampling weight")):
        sd = float(np.std(comp(beta_d, b)))
        ok = claim(ds, b, "the data's fit has a bend", "data component SD >= 0.03", f"{sd:.3f}", sd >= WEAK_SD)
        out[b] = finish(ds, b, base, base if ok else base.replace("bends in", "slight bends in"),
                        f"data component SD {sd:.3f}")
    out["lin:edulevel"] = cat.SIGNALS["lin:edulevel"][1]
    claim(ds, "lin:edulevel", "(describes the block, no factual claim)", "-", "-", True)
    out["lin:fnlwgt"] = cat.SIGNALS["lin:fnlwgt"][1]
    claim(ds, "lin:fnlwgt", "(describes the variable, no factual claim)", "-", "-", True)
    # ---- country: mostly small groups
    c = comp(beta_d, "lin:country")
    lv = [n for n in d["feature_names"] if n.startswith("native-country:") and (R[:, col[n]] > .5).any()]
    big = [(n, float(c[R[:, col[n]] > .5].mean() - np.median(c)), float(np.mean(R[:, col[n]] > .5))) for n in lv]
    big = [x for x in big if abs(x[1]) >= USED_SD]
    small = sum(x[2] < 0.01 for x in big)
    must(ds, "lin:country", "offsets mostly on small groups", "of the countries with an offset >= 0.10 logit, most have < 1% of rows",
         f"{small} of {len(big)}", small * 2 > len(big))
    out["lin:country"] = cat.SIGNALS["lin:country"][1]
    # ---- race: small
    sd = float(np.std(comp(beta_d, "lin:race")))
    must(ds, "lin:race", "small offsets", "data component SD < 0.10", f"{sd:.3f}", sd < USED_SD)
    out["lin:race"] = cat.SIGNALS["lin:race"][1]
    # ---- interactions with marriage / age x hours: the difference and its direction
    def gap(beta, blks, var, a, b_, rows):
        return pd_(beta, blks, {var: b_}, rows) - pd_(beta, blks, {var: a}, rows)

    sx = ALL
    gm, gu = gap(beta_d, sx, "sex", 1, 0, rows_m), gap(beta_d, sx, "sex", 1, 0, rows_u)
    diff_ok = claim(ds, "int:sex*married", "the women-men gap differs inside and outside marriage",
                    "|gap married - gap unmarried| >= 0.10 logit", f"{gm:+.3f} vs {gu:+.3f}", abs(gm - gu) >= USED_SD)
    where = "inside" if abs(gm) > abs(gu) else "outside"
    out["int:sex*married"] = finish(
        ds, "int:sex*married", cat.SIGNALS["int:sex*married"][1],
        (f"the gap between women and men is larger {where} marriage" if diff_ok else
         f"the gap between women and men barely differs inside and outside marriage ({gm:+.2f} vs {gu:+.2f} logit)"),
        f"data's women-men gap {gm:+.3f} (married) vs {gu:+.3f} (not)")
    cg = ALL
    dm, du = gap(beta_d, cg, "capital-gain", 0, 10000, rows_m), gap(beta_d, cg, "capital-gain", 0, 10000, rows_u)
    diff_ok = claim(ds, "int:capgain*married", "the capital-gain effect differs for the married",
                    "|drop married - drop unmarried| (0 -> $10,000) >= 0.10 logit", f"{dm:+.3f} vs {du:+.3f}", abs(dm - du) >= USED_SD)
    out["int:capgain*married"] = finish(
        ds, "int:capgain*married", cat.SIGNALS["int:capgain*married"][1],
        (f"the capital-gain threshold acts {'more' if abs(dm) > abs(du) else 'less'} strongly for the married" if diff_ok else
         "the capital-gain threshold acts about the same for the married"),
        f"data's change 0 -> $10,000: {dm:+.3f} (married) vs {du:+.3f} (not)")
    ag = ALL
    am, au = gap(beta_d, ag, "age", 25, 50, rows_m), gap(beta_d, ag, "age", 25, 50, rows_u)
    diff_ok = claim(ds, "int:age*married", "the age effect differs inside and outside marriage",
                    "|change 25 -> 50 married - unmarried| >= 0.10 logit", f"{am:+.3f} vs {au:+.3f}", abs(am - au) >= USED_SD)
    out["int:age*married"] = finish(
        ds, "int:age*married", cat.SIGNALS["int:age*married"][1],
        (f"the age effect is {'stronger' if abs(am) > abs(au) else 'weaker'} inside marriage" if diff_ok else
         "the age effect is about the same inside and outside marriage"),
        f"data's change from age 25 to 50: {am:+.3f} (married) vs {au:+.3f} (not)")
    hb = ALL
    hy = pd_(beta_d, hb, {"hours-per-week": 50, "age": 25}) - pd_(beta_d, hb, {"hours-per-week": 30, "age": 25})
    ho = pd_(beta_d, hb, {"hours-per-week": 50, "age": 50}) - pd_(beta_d, hb, {"hours-per-week": 30, "age": 50})
    diff_ok = claim(ds, "int:age*hours", "the hours effect changes with age",
                    "|change 30 -> 50 h at age 25 - at age 50| >= 0.10 logit", f"{hy:+.3f} vs {ho:+.3f}", abs(hy - ho) >= USED_SD)
    out["int:age*hours"] = finish(
        ds, "int:age*hours", cat.SIGNALS["int:age*hours"][1],
        (f"the hours effect is {'stronger' if abs(ho) > abs(hy) else 'weaker'} for older people" if diff_ok else
         "the hours effect is about the same at every age"),
        f"data's change 30 -> 50 hours: {hy:+.3f} at age 25 vs {ho:+.3f} at 50")
    assert set(out) == set(cat.SIGNALS), set(cat.SIGNALS) ^ set(out)
    return out


# =============================================================================== Churn

CHURN_ORIGINAL = {   # the texts thesis/ch7/exp2/signal_tables.churn_desc printed before these checks
    "lin:csc": "churn rises with each call to customer service",
    "shape:csc": "churn flat over the first calls, steep from {k} calls",
    "int:csc*day": "the service-call rise is strongest for light daytime users",
    "lin:day": "churn rises with daytime minutes (a charge is a fixed multiple of its minutes)",
    "shape:day": "heavy daytime use: churn climbs steeply above about {up} minutes",
    "int:day*eve": "the combined day and evening bill",
    "int:day*vmail": "heavy daytime use matters mostly without a voice-mail plan",
    "lin:eve": "churn rises with evening minutes",
    "lin:night": "a small slope in night minutes",
    "lin:intlplan": "an international plan raises churn",
    "lin:vmail": "a voice-mail plan (and its messages) lowers churn",
    "lin:state": "a separate offset for some states",
    "lin:intl": "a slope in international minutes",
    "lin:intlcalls": "a slope in the number of international calls",
    "lin:acct": "a small slope in account length",
    "lin:daycalls": "a small slope in the number of day calls",
    "shape:eve": "a bend in evening use",
    "shape:intl": "a bend in international minutes",
    "int:day*night": "day and night use read together",
    "int:eve*csc": "evening use read with service calls",
    "int:csc*vmail": "service calls read with the voice-mail plan",
    "int:intlplan*intl": "plan holders churn when they use the plan heavily or barely",
    "int:csc*intlplan": "service calls read with the international plan",
    "shape:intlcalls": "a bend in international calls",
    "shape:acct": "a bend in account length",
}
CHURN_COLS = {"service calls": ["number customer service calls"], "day usage": ["total day minutes", "total day charge"],
              "evening usage": ["total eve minutes", "total eve charge"], "night usage": ["total night minutes", "total night charge"],
              "international usage": ["total intl minutes", "total intl charge"], "international plan": ["international plan"],
              "voice mail plan": ["voice mail plan"], "voice mail messages": ["number vmail messages"],
              "international calls": ["total intl calls"], "account length": ["account length"], "day calls": ["total day calls"]}
WORDS = {"lin:night": "night minutes", "lin:intl": "international minutes", "lin:intlcalls": "the number of international calls",
         "lin:acct": "account length", "lin:daycalls": "the number of day calls"}
LIN_VAR = {"lin:csc": "service calls", "lin:day": "day usage", "lin:eve": "evening usage", "lin:night": "night usage",
           "lin:intl": "international usage", "lin:intlcalls": "international calls", "lin:acct": "account length",
           "lin:daycalls": "day calls"}
SHORT = {"day usage": "day", "evening usage": "eve", "night usage": "night", "international usage": "intl",
         "voice mail": "vmail", "international plan": "intlplan", "service calls": "csc"}


def churn():
    ds = "churn"
    d = load.get(ds)
    _, bas, _ = load.churn_modules()
    m = SimpleNamespace(X=d["X"], Xt=d["Xt"], raw_train=d["raw_train"], raw_test=d["raw_test"],
                        feature_names=d["feature_names"])
    B = bas.build_basis(m)
    assert np.allclose(B.Phi_tr, d["Phi_tr"])
    bz = np.load(REPO / "experiments/churn_signals/cache/betas.npz")
    beta_d = bz["beta_data"]
    betas = {l: bz[f"beta_{l}"] for l in d["nets"]}
    col = {n: i for i, n in enumerate(d["feature_names"])}
    R = d["raw_train"]
    ratio = {}
    for g, cs in CHURN_COLS.items():
        if len(cs) == 2:
            a, b = R[:, col[cs[0]]], R[:, col[cs[1]]]
            r = b[a > 0] / a[a > 0]
            ratio[g] = float(np.mean(r))
            if g == "day usage":
                must(ds, "lin:day", "a charge is a fixed multiple of its minutes", "SD / mean of charge/minutes < 1%",
                     f"{np.std(r) / np.mean(r):.5f}", np.std(r) / np.mean(r) < 0.01)

    def colz(var, v):
        out = {}
        for i, c in enumerate(CHURN_COLS[var]):
            raw = v * (ratio[var] if i == 1 else 1.0)
            out[col[c]] = (raw - d["mean"][col[c]]) / d["scale"][col[c]]
        return out

    pd_ = PD(d, lambda X: B.phi(X), B.blocks, colz)
    ALL = list(B.blocks)   # interaction claims read the whole fitted logit; one-variable bends their own blocks
    using = seeds_using("experiments/churn_signals/matrix.csv")
    comp = lambda beta, b: d["Phi_tr"][:, B.blocks[b]] @ beta[B.blocks[b]]
    q = lambda var, p: float(np.quantile(R[:, col[CHURN_COLS[var][0]]], p))
    out = {}
    # ---- linear slopes: direction from the data's fit; "small" by its size
    for b, var in LIN_VAR.items():
        lo, hi = q(var, 0.25), q(var, 0.75)
        f = lambda beta, b=b, var=var, lo=lo, hi=hi: pd_(beta, [b], {var: hi}) > pd_(beta, [b], {var: lo})
        rises = f(beta_d)
        sd = float(np.std(comp(beta_d, b)))
        claim(ds, b, "direction of the slope", f"PD of {b} at the 75th vs the 25th percentile",
              f"{pd_(beta_d, [b], {var: hi}) - pd_(beta_d, [b], {var: lo}):+.3f} ({'rises' if rises else 'falls'})", True,
              agree(lambda beta: f(beta) == rises, betas, using.get(b, [])))
        orig = CHURN_ORIGINAL[b]
        if orig.startswith("churn rises"):
            must(ds, b, "churn rises with it", "PD rises", f"{rises}", rises, agree(f, betas, using.get(b, [])))
            out[b] = orig
            continue
        small = sd < SMALL_SD
        if "small" in orig:
            claim(ds, b, "the slope is small", f"data component SD < {SMALL_SD}", f"{sd:.3f}", small)
        word = "rises" if rises else "falls"
        out[b] = finish(ds, b, orig, f"churn {word} {'slightly ' if small else ''}with {WORDS[b]}",
                        f"data's slope {word}; component SD {sd:.3f}")
    # ---- offsets
    f = lambda beta: pd_(beta, ["lin:intlplan"], {"international plan": 1}) > pd_(beta, ["lin:intlplan"], {"international plan": 0})
    must(ds, "lin:intlplan", "an international plan raises churn", "PD(plan) > PD(no plan)", "", f(beta_d),
         agree(f, betas, using["lin:intlplan"]))
    out["lin:intlplan"] = CHURN_ORIGINAL["lin:intlplan"]
    msg = float(np.median(R[R[:, col["voice mail plan"]] > .5, col["number vmail messages"]]))
    f = lambda beta: (pd_(beta, ["lin:vmail"], {"voice mail plan": 1, "voice mail messages": msg})
                      < pd_(beta, ["lin:vmail"], {"voice mail plan": 0, "voice mail messages": 0}))
    must(ds, "lin:vmail", "a voice-mail plan (and its messages) lowers churn",
         f"PD(plan, {msg:g} messages) < PD(no plan)", "", f(beta_d), agree(f, betas, using["lin:vmail"]))
    out["lin:vmail"] = CHURN_ORIGINAL["lin:vmail"]
    c = comp(beta_d, "lin:state")
    st = [n for n in d["feature_names"] if n.startswith("state:")]
    offs = [abs(float(c[R[:, col[n]] > .5].mean() - np.median(c))) for n in st]
    nbig = sum(o >= USED_SD for o in offs)
    must(ds, "lin:state", "a separate offset for some states", "states whose mean offset is >= 0.10 logit from the median",
         f"{nbig} of {len(st)}", 0 < nbig < len(st))
    out["lin:state"] = CHURN_ORIGINAL["lin:state"]
    # ---- service-call threshold (the rule churn_desc used on the same curve)
    calls = np.arange(0, 10, dtype=float)

    def kcalls(beta):
        g = pd_.curve(beta, ["lin:csc", "shape:csc"], "service calls", calls)
        inc = np.diff(g)
        k = int(calls[1:][np.argmax(inc >= 0.5 * inc.max())])
        return k, inc

    k, inc = kcalls(beta_d)
    before = float(np.sum(inc[: k - 1]))
    must(ds, "shape:csc", "flat over the first calls", f"the rise from 0 to {k - 1} calls is below the single step to {k}",
         f"{before:+.3f} vs {inc[k - 1]:+.3f}", before < inc[k - 1],
         agree(lambda b: kcalls(b)[0] == k, betas, using["shape:csc"]))
    out["shape:csc"] = finish(ds, "shape:csc", CHURN_ORIGINAL["shape:csc"].format(k=k),
                              f"churn flat over the first calls, steep from {k} calls", "")
    # ---- calls x light day use
    days = [q("day usage", p) for p in (0.10, 0.25, 0.50, 0.75, 0.90)]

    def rise_by_day(beta):
        blks = ALL
        return [pd_(beta, blks, {"service calls": 5, "day usage": dv}) - pd_(beta, blks, {"service calls": 1, "day usage": dv})
                for dv in days]

    rr = rise_by_day(beta_d)
    f = lambda beta: int(np.argmax(rise_by_day(beta))) <= 1
    must(ds, "int:csc*day", "the service-call rise is strongest for light daytime users",
         "rise 1 -> 5 calls is largest at the 10th or 25th percentile of day minutes",
         " / ".join(f"{dv:.0f} min {x:+.2f}" for dv, x in zip(days, rr)), f(beta_d), agree(f, betas, using["int:csc*day"]))
    out["int:csc*day"] = CHURN_ORIGINAL["int:csc*day"]
    # ---- heavy-day hinge (the rule churn_desc used)
    xd = np.linspace(40, 340, 61)

    def upday(beta):
        yd = pd_.curve(beta, ["lin:day", "shape:day"], "day usage", xd)
        sd_ = np.diff(yd) / np.diff(xd)
        lo = xd[int(np.argmin(yd))]
        return int(round(xd[1:][np.argmax((sd_ >= 0.5 * sd_.max()) & (xd[1:] > lo))] / 10) * 10), sd_

    up, sdd = upday(beta_d)
    s_lo = float(np.mean(sdd[xd[1:] <= up - 50]))
    s_hi = float(np.mean(sdd[xd[1:] > up]))
    must(ds, "shape:day", "churn climbs steeply above the threshold", f"mean slope above {up} min at least 3x the slope below {up - 50}",
         f"{s_hi:+.4f} vs {s_lo:+.4f} per minute", s_hi >= 3 * max(s_lo, 1e-9),
         agree(lambda b: abs(upday(b)[0] - up) <= 30, betas, using["shape:day"]))
    out["shape:day"] = finish(ds, "shape:day", CHURN_ORIGINAL["shape:day"].format(up=up),
                              f"heavy daytime use: churn climbs steeply above about {up} minutes", "")
    # ---- day + evening bill: evening use matters more for heavy day users
    def eve_by_day(beta):
        blks = ALL
        e1, e2 = q("evening usage", 0.25), q("evening usage", 0.90)
        return [pd_(beta, blks, {"evening usage": e2, "day usage": dv}) - pd_(beta, blks, {"evening usage": e1, "day usage": dv})
                for dv in (q("day usage", 0.25), q("day usage", 0.90))]

    el, eh = eve_by_day(beta_d)
    f = lambda beta: (lambda x: x[1] > x[0])(eve_by_day(beta))
    ok = claim(ds, "int:day*eve", "it is the combined bill: evening use raises churn more for heavy day users",
               "rise 25th -> 90th percentile of evening minutes, at the 90th vs the 25th percentile of day minutes",
               f"{eh:+.3f} vs {el:+.3f}", eh > el, agree(f, betas, using["int:day*eve"]))
    out["int:day*eve"] = finish(ds, "int:day*eve", CHURN_ORIGINAL["int:day*eve"],
                                ("the combined bill: evening minutes raise churn more for heavy day users" if ok else
                                 "day and evening use read together"),
                                f"evening rise {eh:+.3f} (heavy day) vs {el:+.3f} (light day)")
    # ---- heavy day use without voice mail
    def day_by_vm(beta):
        blks = ALL
        d1, d2 = q("day usage", 0.50), q("day usage", 0.95)
        return [pd_(beta, blks, {"day usage": d2, "voice mail plan": p, "voice mail messages": mm})
                - pd_(beta, blks, {"day usage": d1, "voice mail plan": p, "voice mail messages": mm})
                for p, mm in ((0, 0), (1, msg))]

    r0, r1 = day_by_vm(beta_d)
    f = lambda beta: (lambda x: x[0] > 0 and x[1] < 0.5 * x[0])(day_by_vm(beta))
    must(ds, "int:day*vmail", "heavy daytime use matters mostly without a voice-mail plan",
         "rise from the median to the 95th percentile of day minutes: with a plan below half of without",
         f"{r1:+.3f} (plan) vs {r0:+.3f} (no plan)", f(beta_d), agree(f, betas, using["int:day*vmail"]))
    out["int:day*vmail"] = CHURN_ORIGINAL["int:day*vmail"]
    # ---- plan x international use: U shape among plan holders
    imin = np.linspace(2, 18, 33)

    def ushape(beta):
        g = pd_.curve(beta, ALL, "international usage", imin,
                      extra={"international plan": 1})
        mid = g[(imin >= 8) & (imin <= 12)].min()
        return g[imin <= 4].max() - mid, g[imin >= 14].max() - mid

    lo_, hi_ = ushape(beta_d)
    claim(ds, "int:intlplan*intl", "plan holders churn when they barely use the plan (few minutes)",
          "among plan holders, PD at <= 4 min above the minimum at 8-12 min by >= 0.10", f"{lo_:+.3f}", lo_ >= USED_SD)
    must(ds, "int:intlplan*intl", "plan holders churn when they use the plan heavily",
         "among plan holders, PD at >= 14 min above the minimum at 8-12 min by >= 0.10", f"{hi_:+.3f}", hi_ >= USED_SD)
    ncalls = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    gc = pd_.curve(beta_d, ALL, "international calls", ncalls, extra={"international plan": 1})
    few = float(gc[:2].mean() - gc[3:].mean())
    rare = claim(ds, "int:intlplan*intl", "... or barely: few international calls",
                 "among plan holders, PD at 1-2 calls above PD at 4-6 calls by >= 0.10", f"{few:+.3f}", few >= USED_SD)
    new = ("plan holders churn when they use the plan heavily (many minutes) or rarely (few calls)" if rare else
           "plan holders churn when they use the plan heavily")
    out["int:intlplan*intl"] = finish(ds, "int:intlplan*intl", CHURN_ORIGINAL["int:intlplan*intl"], new,
                                      f"plan holders: {lo_:+.3f} at <= 4 min and {hi_:+.3f} at >= 14 min over the 8-12 min "
                                      f"minimum; 1-2 calls {few:+.3f} over 4-6 calls")
    # ---- bends whose text claims only that a bend exists: on the data's fit
    for b in ("shape:eve", "shape:intl", "shape:intlcalls", "shape:acct"):
        sd = float(np.std(comp(beta_d, b)))
        ok = claim(ds, b, "the data's fit has a bend", "data component SD >= 0.03", f"{sd:.3f}", sd >= WEAK_SD)
        out[b] = CHURN_ORIGINAL[b]
    # ---- "read together": the seeds carrying the interaction have a unit reading both variables
    Rz = pd.read_csv(REPO / "experiments/churn_signals/realisations.csv")
    pairs = {"int:day*night": ("day", "night"), "int:eve*csc": ("eve", "csc"),
             "int:csc*vmail": ("csc", "vmail"), "int:csc*intlplan": ("csc", "intlplan")}
    for b, (u, v) in pairs.items():
        labs = using.get(b, [])
        hits = [l for l in labs if any({u, v} <= set(str(x).split(",")) for x in Rz[(Rz.network == l) & (Rz.signal == b)].upstream)]
        ok = claim(ds, b, f"a unit reads {u} and {v} together", "in each seed using it, a carrier unit reads both",
                   f"{len(hits)} of {len(labs)} seeds", len(hits) == len(labs) and labs)
        out[b] = finish(ds, b, CHURN_ORIGINAL[b],
                        CHURN_ORIGINAL[b] if ok else CHURN_ORIGINAL[b].replace(" read with ", " with ").replace(" read together", " together"),
                        f"{len(hits)} of {len(labs)} using seeds have a unit reading both")
    assert set(out) == set(CHURN_ORIGINAL), set(out) ^ set(CHURN_ORIGINAL)
    return out, {"k_calls": k, "up_day": up}


# =============================================================================== Heart-100

HEART_ORIGINAL = {   # FAM_ROWS descriptions in thesis/ch7/exp2/signal_tables.py before these checks
    "thal": "a scan defect read through a gate, mostly with ST depression",
    "max HR": "a bend or gate in peak heart rate",
    "BP": "a threshold, alone or with slope, heart rate, angina or scan",
    "exang": "a gate, read with sex in some seeds (for women in one)",
    "age": "risk stops rising, or falls back, past an age",
    "ST depression": "counted only past a threshold",
    "ST slope": "a step, read with chest pain or cholesterol",
    "resting ECG": "the resting ECG through a switching unit",
    "thal × age": "a scan defect read with age",
}
TAGW = {"slope": "ST slope", "thalach": "heart rate", "exang": "exercise angina", "thal": "thallium scan",
        "cp": "chest pain", "chol": "cholesterol", "sex": "sex", "age": "age", "oldpeak": "ST depression",
        "trestbps": "blood pressure", "restecg": "resting ECG", "fbs": "fasting sugar", "ca": "vessels"}
ROWVAR = {"thal": "thal", "max HR": "thalach", "BP": "trestbps", "exang": "exang", "age": "age",
          "ST depression": "oldpeak", "ST slope": "slope", "resting ECG": "restecg", "thal × age": "thal"}


def words(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def heart():
    ds = "heart100"
    F = pd.read_csv(REPO / "experiments/heart100_audit/features.csv")
    F = F[F.level == "unit"].copy()
    F["row_family"] = F.family_group.map(lambda g: "BP" if g.startswith("BP") else g)
    FP = pd.read_csv(REPO / "experiments/heart100_audit/family_presence.csv")
    present = {(r.network, r.row_family) for r in FP.itertuples() if r.presence == "used"}
    U = F[[(r.network, r.row_family) in present for r in F.itertuples()]]
    vs = lambda r: set(str(r.drivers).split()) | (set(str(r.modifiers).split()) if isinstance(r.modifiers, str) else set())
    out = {}
    for key, orig in HEART_ORIGINAL.items():
        G = U[U.row_family == key]
        n, ns = len(G), G.network.nunique()
        assert n, key
        v0 = ROWVAR[key]
        drv = sum(v0 in str(r.drivers).split() for r in G.itertuples())
        must(ds, key, f"every unit is switched by {TAGW[v0]}", f"{v0} among the drivers", f"{drv} of {n} units", drv == n)
        if key == "thal":
            k = sum("oldpeak" in vs(r) for r in G.itertuples())
            ok = claim(ds, key, "mostly with ST depression", "units with ST depression as a driver or modifier, more than half",
                       f"{k} of {n} units", k * 2 > n)
            text = (f"the thallium scan read through a gate, with ST depression in {k} of {n} units")
            out[key] = finish(ds, key, orig, text, f"{k} of {n} units in the seeds that carry it read ST depression")
        elif key == "BP":
            others = Counter(x for r in G.itertuples() for x in vs(r) - {"trestbps"})
            alone = sum(not (vs(r) - {"trestbps"}) for r in G.itertuples())
            claim(ds, key, "alone or with other inputs", "units with no other driver or modifier; the others",
                  f"alone {alone} of {n}; with " + ", ".join(f"{TAGW[x]} {c}" for x, c in sorted(others.items())), True)
            common = [TAGW[x] for x, c in sorted(others.items()) if c >= 2]
            rare = [x for x, c in others.items() if c < 2]
            lst = words(common) + (" (others in single units)" if rare else "")
            text = (f"a threshold, alone ({alone} of {n} units) or with {lst}" if alone else f"a threshold read with {lst}")
            out[key] = finish(ds, key, orig, text, f"{alone} of {n} units alone; co-inputs {dict(sorted(others.items()))}")
        elif key == "exang":
            sx = sorted({r.network for r in G.itertuples() if "sex" in vs(r)})
            women = sorted({r.network for r in G.itertuples() if "sex" in vs(r) and
                            json.loads(r.on_profile).get("sex", {}).get("on", {}).get("0", 0) > 0.5})
            claim(ds, key, "read with sex in some seeds", "seeds with an exercise-angina unit reading sex",
                  f"{len(sx)} of {ns} seeds", len(sx) > 0)
            claim(ds, key, "for women in one", "seeds with such a unit on mostly (> 50% of its on rows) for women",
                  f"{len(women)} ({' '.join(s.rsplit('-', 1)[1] for s in women)})", len(women) == 1)
            text = (f"a gate, read with sex in {len(sx)} of {ns} seeds" +
                    (f" (on mostly for women in {len(women)})" if women else ""))
            out[key] = finish(ds, key, orig, text,
                              f"sex read in {', '.join(s.rsplit('-', 1)[1] for s in sx)}; on mostly for women in "
                              f"{', '.join(s.rsplit('-', 1)[1] for s in women) or 'none'}")
        elif key == "age":
            # a ReLU monotone in age that lowers risk when on bends the age curve down past its
            # knot (on for the older) or up before it (on for the younger): the rise slows past an age
            brake = sum(r.on_direction == "lowers" for r in G.itertuples())
            claim(ds, key, "risk stops rising, or falls back, past an age",
                  "not checked as stated: the units' slopes beyond their knot are not computed", "-", False)
            claim(ds, key, "the rise of risk with age slows past an age", "units that lower risk when on (a brake)",
                  f"{brake} of {n} units", brake * 2 > n)
            text = f"the rise of risk with age slows past an age ({brake} of {n} units)"
            out[key] = finish(ds, key, orig, text, f"{brake} of {n} units lower risk when on; "
                              "whether risk stops rising or falls back is not computed")
        elif key == "ST depression":
            thr = sum((lambda p: p["on"] and p["off"] and p["on"]["min"] > p["off"]["max"] - 0.2 and p["on"]["mean"] > p["off"]["mean"])(
                json.loads(r.on_profile).get("oldpeak", {"on": {}, "off": {}})) for r in G.itertuples())
            ok = claim(ds, key, "counted only past a threshold",
                       "units on only above an ST-depression value (on rows' minimum >= off rows' maximum - 0.2 mm)",
                       f"{thr} of {n} units", thr == n)
            text = orig if ok else f"counted only past a threshold in {thr} of {n} units"
            out[key] = finish(ds, key, orig, text, f"{thr} of {n} units switch on above an ST-depression threshold")
        elif key == "ST slope":
            others = Counter(x for r in G.itertuples() for x in vs(r) - {"slope"})
            claim(ds, key, "read with chest pain or cholesterol", "other drivers and modifiers of the units",
                  ", ".join(f"{TAGW[x]} {c}" for x, c in sorted(others.items())), set(others) <= {"cp", "chol"})
            text = f"a step, read with {words(TAGW[x] for x in sorted(others))}"
            out[key] = finish(ds, key, orig, text, f"co-inputs {dict(sorted(others.items()))}")
        else:
            if key == "max HR":
                multi = sum(bool(vs(r) - {"thalach"}) for r in G.itertuples())
                claim(ds, key, "a bend or gate", "units reading heart rate alone (bend) or with others (gate)",
                      f"bend {n - multi}, gate {multi}", True)
                text = ("a bend in peak heart rate" if not multi else
                        "a gate in peak heart rate" if multi == n else orig)
                out[key] = finish(ds, key, orig, text, f"bend {n - multi}, gate {multi} of {n} units")
            elif key == "thal × age":
                k = sum("age" in vs(r) for r in G.itertuples())
                must(ds, key, "read with age", "units with age as a driver or modifier", f"{k} of {n}", k == n)
                out[key] = finish(ds, key, orig, "the thallium scan read with age", "the units read thal and age")
            else:
                out[key] = orig
    # linear rows: the direction most seeds give the term, against the data's own fit (reported)
    C = pd.read_csv(REPO / "experiments/heart100_audit/linear_core.csv")
    for t, g in C.groupby("term", sort=False):
        u = g[g.presence == "used"]
        if not len(u):
            continue
        ok = u[u.direction == "ok"]
        s = np.sign(np.sign(ok.proj_coef).sum())
        claim(ds, f"lin:{t}", "direction most seeds give it", "sign of the weights of the seeds in the majority",
              f"{'+' if s > 0 else '-'} ({len(ok)} of {len(u)} seeds)", True,
              f"data's own fit {'agrees' if np.sign(g.data_coef.iloc[0]) == s else 'disagrees'} ({g.data_coef.iloc[0]:+.3f})")
    return out


def main():
    a = adult()
    c, nums = churn()
    h = heart()
    res = {"adult": a, "churn": c, "heart100": h, "numbers": {"churn": nums}}
    (HERE / "descriptions.json").write_text(json.dumps(res, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    pd.DataFrame(CLAIMS).to_csv(HERE / "claims.csv", index=False)
    pd.DataFrame(REWRITES, columns=["dataset", "row", "original", "new", "evidence"]).to_csv(HERE / "rewrites.csv", index=False)
    print(f"{len(CLAIMS)} claims, {sum(not c['holds'] for c in CLAIMS)} do not hold as originally worded; "
          f"{len(REWRITES)} rewrites")


if __name__ == "__main__":
    main()
