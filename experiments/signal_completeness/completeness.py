"""Is each data set's signal list complete? Networks -> catalogue, at the output and part level,
plus a scan of what the catalogue leaves over.

Reads ``cache/<set>.pkl`` (``load.py``; read-only DB). Deterministic: every fold split and
permutation is seeded (``SEED``), single-threaded. Writes, next to this file:

- ``output_level.csv``: per network, how much of its logit (variance over the training rows; the
  test rows as a secondary check) its least-squares projection on the catalogue basis reproduces:
  R^2, residual SD (raw and clipped as in the presence rule), and R^2 of the linear blocks alone.
- ``part_level.csv``: per explanation part (non-root, not a duplicate), the R^2 of the projection
  of its logit contribution on the catalogue, its residual SD, and the catalogue signals it maps to
  (blocks whose component of the part's projection is *used* by the shared presence rule).
- ``residual_scan.csv``: per network, per data variable and per variable pair, the out-of-fold
  structure in the residual logit (binned means; pairs after removing both one-variable effects),
  its permutation baseline and the logit SD of the structure; plus a gradient-boosting check on
  all variables together (``feature = "(all, GB)"``).
- ``candidates.csv``: residual structures detected in >= 2 networks, or with SD >= 0.10 logit in
  any network. Detected: out-of-fold R^2 gain above the network's family-wise permutation
  threshold (the largest gain any variable or pair reaches on 19 permuted residuals: a
  max-statistic test, family-wise error about 5% per network) and structure SD >= 0.03 logit.
- ``summary.json``: the numbers RESULTS.md quotes; ``RESULTS.md`` is written from it.

**Completeness criterion** (fixed before running; thresholds from ``signal_presence``): a data
set's signal list is complete when (1) in every one of its ten networks the residual logit (logit
minus its projection on the catalogue) has a clipped SD below 0.10 on the training rows, and
(2) the residual scan finds no candidate signal.

    uv run python experiments/signal_completeness/completeness.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

threadpool_limits(1)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from explaneat.analysis.signal_presence import USED_SD, WEAK_SD, contribution_sd, presence  # noqa: E402
import load  # noqa: E402

SETS = ["adult", "churn", "heart100"]
SEED = 0
N_PERM = 19            # permutations for the family-wise (max-statistic) threshold
BINS_1D, BINS_2D = 10, 4
SHRINK = 20.0          # cell means shrunk towards 0 with weight n / (n + 20)
GB_ROWS = 10000        # rows for the gradient-boosting check (a fixed subsample when larger)
GB_PERM = 3


def project(Phi, e):
    A = np.hstack([np.ones((len(Phi), 1)), Phi])
    c, *_ = np.linalg.lstsq(A, e, rcond=None)
    return float(c[0]), c[1:]


def r2(y, f):
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - np.mean(y)) ** 2))


# ------------------------------------------------------------------ binned structure

def codes(kind, x, bins):
    """Integer cell codes: categories by level, numbers by training quantile bins."""
    if kind == "cat":
        _, inv = np.unique(np.asarray(x).astype(str), return_inverse=True)
        return inv
    x = np.asarray(x, float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, x, side="right")


def cell_fit(c, t, idx):
    """Shrunk cell means of t over rows idx -> per-cell values."""
    k = int(c.max()) + 1
    s = np.bincount(c[idx], weights=t[idx], minlength=k)
    n = np.bincount(c[idx], minlength=k)
    return s / (n + SHRINK)


def oof(c, t, folds):
    p = np.zeros(len(t))
    for tr, va in folds:
        p[va] = cell_fit(c, t, tr)[c[va]]
    return p


def cv_gain(c, t, folds, sst):
    p = oof(c, t, folds)
    return float((np.sum(t ** 2) - np.sum((t - p) ** 2)) / sst)


def scan_one(c, t, folds, perms, sst):
    """(out-of-fold gain, gains on the permuted targets, SD of the in-sample structure)."""
    g = cv_gain(c, t, folds, sst)
    gp = np.array([cv_gain(c, t[pi], folds, sst) for pi in perms])
    fit = cell_fit(c, t, np.arange(len(t)))[c]
    return g, gp, contribution_sd(fit)


# ------------------------------------------------------------------ main

def output_level(d):
    rows, fits = [], {}
    lin = [i for b, idx in d["blocks"].items() if b.startswith("lin:") for i in idx]
    for lab, nt in d["nets"].items():
        e, et = nt["eta_tr"], nt["eta_te"]
        a, beta = project(d["Phi_tr"], e)
        f, ft = a + d["Phi_tr"] @ beta, a + d["Phi_te"] @ beta
        al, bl = project(d["Phi_tr"][:, lin], e)
        res, res_t = e - f, et - ft
        fits[lab] = (a, beta, res, res_t)
        rows.append(dict(dataset=d["dataset"], network=lab, seed=int(lab.rsplit("-s", 1)[1]),
                         logit_sd_train=float(np.std(e)), r2_train=r2(e, f), r2_test=r2(et, ft),
                         r2_linear_train=r2(e, al + d["Phi_tr"][:, lin] @ bl),
                         resid_sd_train=contribution_sd(res), resid_sd_train_raw=contribution_sd(res, clip=False),
                         resid_sd_test=contribution_sd(res_t), resid_sd_test_raw=contribution_sd(res_t, clip=False),
                         n_basis_columns=int(d["Phi_tr"].shape[1])))
    return rows, fits


def part_level(d):
    rows = []
    for lab, nt in d["nets"].items():
        for p in nt["parts"]:
            if p["duplicate_of"]:
                continue
            c, ct = p["c_tr"], p["c_te"]
            own = presence(c)
            a, beta = project(d["Phi_tr"], c)
            f, ft = a + d["Phi_tr"] @ beta, a + d["Phi_te"] @ beta
            comp = {b: presence(d["Phi_tr"][:, idx] @ beta[idx]) for b, idx in d["blocks"].items()}
            mapped = sorted((b for b, pr in comp.items() if pr["state"] == "used"), key=lambda b: -comp[b]["sd"])
            top = sorted(comp, key=lambda b: -comp[b]["sd"])[:3]
            sst = np.sum((c - c.mean()) ** 2)
            rp = presence(c - f)
            status = ("small (own SD < 0.10)" if own["state"] != "used" else
                      "unmapped" if not mapped else
                      "mapped, residual >= 0.10" if rp["state"] == "used" else "mapped")
            rows.append(dict(dataset=d["dataset"], network=lab, part=p["name"], part_id=p["id"], kind=p["kind"],
                             variables=" ".join(p["variables"]), own_sd=own["sd"], own_presence=own["state"],
                             r2_train=r2(c, f) if sst > 1e-12 else float("nan"),
                             r2_test=r2(ct, ft) if np.sum((ct - ct.mean()) ** 2) > 1e-12 else float("nan"),
                             resid_sd=rp["sd"], n_mapped=len(mapped), mapped=" ".join(mapped),
                             top_blocks=" ".join(f"{b}:{comp[b]['sd']:.3f}" for b in top), status=status))
    return rows


def residual_scan(d, fits):
    rows = []
    V = d["variables"]
    names = list(V)
    cats = {v: codes(V[v][0], V[v][1], BINS_1D) for v in names}
    cats2 = {v: codes(V[v][0], V[v][1], BINS_2D) for v in names}
    n = len(d["y"])
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)))
    rng = np.random.RandomState(SEED)
    perms = [rng.permutation(n) for _ in range(N_PERM)]
    # gradient-boosting check on all variables together (a fixed subsample on large sets)
    gb_idx = np.arange(n) if n <= GB_ROWS else np.sort(np.random.RandomState(SEED).choice(n, GB_ROWS, replace=False))
    Xgb = np.column_stack([cats[v] if V[v][0] == "cat" else np.asarray(V[v][1], float) for v in names])[gb_idx]
    catmask = np.array([V[v][0] == "cat" for v in names])
    gfolds = list(KFold(5, shuffle=True, random_state=SEED).split(gb_idx))
    gperm_rng = np.random.RandomState(SEED + 1)
    gperms = [gperm_rng.permutation(len(gb_idx)) for _ in range(GB_PERM)]

    def gb_oof(t):
        p = np.zeros(len(t))
        for tr, va in gfolds:
            mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                categorical_features=catmask if catmask.any() else None,
                                                random_state=SEED)
            p[va] = mdl.fit(Xgb[tr], t[tr]).predict(Xgb[va])
        return p

    for lab, (_, _, res, _) in fits.items():
        sst = float(np.sum((res - res.mean()) ** 2))
        r = res - res.mean()
        base = dict(dataset=d["dataset"], network=lab, seed=int(lab.rsplit("-s", 1)[1]))
        net_rows, pg = [], []
        for v in names:
            g, gp, sd = scan_one(cats[v], r, folds, perms, sst)
            net_rows.append(dict(base, feature=v, kind="one variable", cv_gain=g, perm_max=float(gp.max()), sd=sd))
            pg.append(gp)
        main = {v: cell_fit(cats[v], r, np.arange(n))[cats[v]] for v in names}
        for a_, b_ in combinations(names, 2):
            t = r - main[a_]
            t = t - cell_fit(cats[b_], t, np.arange(n))[cats[b_]]
            c = cats2[a_] * (int(cats2[b_].max()) + 1) + cats2[b_]
            _, c = np.unique(c, return_inverse=True)
            g, gp, sd = scan_one(c, t, folds, perms, sst)
            net_rows.append(dict(base, feature=f"{a_} x {b_}", kind="pair (beyond both one-variable effects)",
                                 cv_gain=g, perm_max=float(gp.max()), sd=sd))
            pg.append(gp)
        # family-wise threshold for this network (max-statistic permutation test over every
        # variable and pair): the largest gain any feature reaches on any of the permuted residuals
        thr = float(np.max(np.stack(pg)))
        for x in net_rows:
            x["fw_threshold"] = thr
        rows += net_rows
        rg = r[gb_idx]
        sg = float(np.sum((rg - rg.mean()) ** 2))
        p = gb_oof(rg)
        g = float(1 - np.sum((rg - p) ** 2) / sg)
        gp = max(float(1 - np.sum((rg[pi] - gb_oof(rg[pi])) ** 2) / sg) for pi in gperms)
        rows.append(dict(base, feature="(all, GB)", kind="all variables, gradient boosting (depth 3)",
                         cv_gain=g, perm_max=gp, sd=contribution_sd(p), fw_threshold=gp))
        print(" ", d["dataset"], lab, "scanned", flush=True)
    R = pd.DataFrame(rows)
    R["detected"] = (R.cv_gain > R.fw_threshold) & (R.cv_gain > 0) & (R.sd >= WEAK_SD)
    return R


def candidates(R, d):
    out = []
    covered = {}
    for b, vs in d["block_vars"].items():
        if not b.startswith("lin:"):
            covered.setdefault(frozenset(vs if len(vs) <= 2 else vs[:2]), []).append(b)
    for (ds, feat), g in R[R.feature != "(all, GB)"].groupby(["dataset", "feature"], sort=False):
        det = g[g.detected]
        if len(det) >= 2 or (det.sd >= USED_SD).any():
            vs = feat.split(" x ")
            blocks = sorted({b for b, bvs in d["block_vars"].items() if not b.startswith("lin:")
                             and set(vs) <= set(bvs)})
            out.append(dict(dataset=ds, feature=feat, kind=g.kind.iloc[0], n_networks_detected=len(det),
                            networks=" ".join(f"s{s}" for s in det.seed), max_sd=float(det.sd.max()),
                            median_sd=float(det.sd.median()), max_cv_gain=float(det.cv_gain.max()),
                            catalogue_blocks_on_these_variables=" ".join(blocks)))
    return out


def main():
    allo, allp, allr, allc, summ = [], [], [], [], {}
    for ds in SETS:
        d = load.get(ds)
        print(ds, "basis", d["Phi_tr"].shape, flush=True)
        o, fits = output_level(d)
        p = part_level(d)
        R = residual_scan(d, fits)
        c = candidates(R, d)
        allo += o
        allp += p
        allr.append(R)
        allc += c
        O, P = pd.DataFrame(o), pd.DataFrame(p)
        gb = R[R.feature == "(all, GB)"]
        complete = bool((O.resid_sd_train < USED_SD).all() and not c)
        summ[ds] = dict(
            n_networks=len(O), n_basis_columns=int(O.n_basis_columns.iloc[0]),
            r2_train_median=float(O.r2_train.median()), r2_train_min=float(O.r2_train.min()),
            r2_test_median=float(O.r2_test.median()), r2_test_min=float(O.r2_test.min()),
            r2_linear_train_median=float(O.r2_linear_train.median()),
            resid_sd_train_median=float(O.resid_sd_train.median()), resid_sd_train_max=float(O.resid_sd_train.max()),
            resid_sd_test_median=float(O.resid_sd_test.median()), resid_sd_test_max=float(O.resid_sd_test.max()),
            networks_resid_sd_ge_0_10=[s for s in O[O.resid_sd_train >= USED_SD].network],
            n_parts=len(P), n_parts_used=int((P.own_presence == "used").sum()),
            n_parts_unmapped=int((P.status == "unmapped").sum()),
            n_parts_mapped_with_residual=int((P.status == "mapped, residual >= 0.10").sum()),
            parts_unmapped=[f"{r.network}:{r.part}" for r in P[P.status == "unmapped"].itertuples()],
            part_r2_train_median=float(P[P.own_presence == "used"].r2_train.median()),
            gb_detected=int(gb.detected.sum()), gb_sd_max=float(gb.sd.max()),
            n_candidates=len(c), candidates=[x["feature"] for x in c], complete=complete)
        print(json.dumps(summ[ds], indent=1), flush=True)
    pd.DataFrame(allo).to_csv(HERE / "output_level.csv", index=False, float_format="%.6g")
    pd.DataFrame(allp).to_csv(HERE / "part_level.csv", index=False, float_format="%.6g")
    pd.concat(allr).to_csv(HERE / "residual_scan.csv", index=False, float_format="%.6g")
    pd.DataFrame(allc, columns=["dataset", "feature", "kind", "n_networks_detected", "networks", "max_sd",
                                "median_sd", "max_cv_gain", "catalogue_blocks_on_these_variables"]).to_csv(
        HERE / "candidates.csv", index=False, float_format="%.6g")
    (HERE / "summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    import report
    report.write()


if __name__ == "__main__":
    main()
