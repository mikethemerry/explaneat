"""Reference explanation for the Cox survival case: what the comparators say, per clinical unit.

    PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/reference_explanation.py

Reads only committed files and the scikit-survival bundled data (no database). On each
dataset's own split (``splits/<ds>.splits.json``, training rows only):

1. **Cox PH side by side with the network.** lifelines ``CoxPHFitter(penalizer=0)`` (Efron) on
   the standardised training rows, exactly the fit of ``references.py`` (its coefficients are
   checked against ``references.json``). Per-unit hazard ratios are the standardised fit
   converted: beta_unit = beta_sd / sd_train, the CI ends likewise, p unchanged (a Cox model is
   equivariant under rescaling an input). A refit on raw units confirms the conversion (the
   largest coefficient difference is stored). The Schoenfeld p is read from ``references.json``
   (rank transform, training rows). The network's hazard ratios are read from
   ``hazard_readings/<tag>.json`` (the explanation's own readings), not retyped.
2. **The forest's reference reading.** The random survival forest of ``references.py``, refitted
   with the same parameters (its test Harrell C is checked against ``references.json``):
   permutation importance on the training rows (Harrell C of the risk score), partial dependence
   of the log cumulative hazard at the horizon (5 y GBSG2, 3 y WHAS500) for the top inputs,
   centred at the training median, so that a Cox model draws a straight line; Friedman's H for
   the pairs among the top five inputs (on the risk at the horizon); for GBSG2, grade III's
   log cumulative hazard ratio over time.
3. **Where the forest's GBSG2 edge comes from.** Cox variants on the training rows, each adding
   one thing the forest can do: spline main effects (one input at a time, then all), the
   forest's strongest pair as a product term, and grade III stratified (no PH for grade III).
   Each is scored by a likelihood-ratio test on the training rows (where nested) and by the
   test-fold Harrell C, Uno C and IBS of its survival curves.

Outputs: ``reference_explanation.json``, ``figures/ref_forest_<ds>.png`` (Cox HR with CI per unit,
the network's HR as a second marker), ``figures/ref_pd_<ds>.png`` (forest PD against the Cox and
network lines), ``figures/ref_h2d_<ds>.png`` (2-D PD of the strongest pair),
``figures/ref_grade3_time_gbsg2.png``, and ``thesis_exports/ch7/ch7_cox_reference_table.tex``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SURV = HERE.parent
REPO = SURV.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SURV))
sys.path.insert(0, str(HERE))

from references import RSF  # noqa: E402
from run_case import DATASETS, HORIZONS  # noqa: E402

from explaneat.survival import BreslowBaseline  # noqa: E402
from explaneat.survival.data import load_builtin, load_split_indices, standardise  # noqa: E402
from explaneat.survival.metrics import harrell_c, integrated_brier, uno_c, brier_at  # noqa: E402

FIG = HERE / "figures"
TEX = REPO / "thesis_exports" / "ch7" / "ch7_cox_reference_table.tex"
TAGS = {"gbsg2": "gbsg2-s1", "whas500": "whas500-s0"}
YEARS = {"gbsg2": 5, "whas500": 3}

# (label, unit for the per-unit hazard ratio); binary inputs read as "yes vs no" contrasts
UNITS = {
    "gbsg2": {
        "age": ("Age", "per year"),
        "estrec": ("Oestrogen receptor", "per fmol/l"),
        "horTh=yes": ("Hormone therapy", "yes vs no"),
        "menostat=Post": ("Postmenopausal", "post vs pre"),
        "pnodes": ("Positive lymph nodes", "per node"),
        "progrec": ("Progesterone receptor", "per fmol/l"),
        "tgrade=II": ("Tumour grade II", "II vs I"),
        "tgrade=III": ("Tumour grade III", "III vs I"),
        "tsize": ("Tumour size", "per mm"),
    },
    "whas500": {
        "afb=1": ("Atrial fibrillation", "yes vs no"),
        "age": ("Age", "per year"),
        "av3=1": ("Complete heart block", "yes vs no"),
        "bmi": ("Body mass index", "per kg/m²"),
        "chf=1": ("Congestive heart complications", "yes vs no"),
        "cvd=1": ("History of CVD", "yes vs no"),
        "diasbp": ("Diastolic BP", "per mmHg"),
        "gender=1": ("Female sex", "female vs male"),
        "hr": ("Heart rate", "per beat/min"),
        "los": ("Length of stay", "per day"),
        "miord=1": ("Recurrent MI", "recurrent vs first"),
        "mitype=1": ("Q-wave MI", "Q-wave vs non-Q"),
        "sho=1": ("Cardiogenic shock", "yes vs no"),
        "sysbp": ("Systolic BP", "per mmHg"),
    },
}

# reference palette slots 1-3 (dataviz skill): Cox, network, forest
C_COX, C_NET, C_RSF = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def agreement(cox_lo, cox_hi, cox_hr, net_hr):
    if net_hr is None:
        return "dropped"
    if np.sign(np.log(net_hr)) != np.sign(np.log(cox_hr)):
        return "opposite"
    return "within" if cox_lo <= net_hr <= cox_hi else "outside"


def fit_cph(df, **kw):
    from lifelines import CoxPHFitter
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CoxPHFitter(penalizer=0.0).fit(df, "T", "E", **kw)


def frame(X, y, cols):
    df = pd.DataFrame(X, columns=cols)
    df["T"], df["E"] = y[:, 0], y[:, 1]
    return df


def surv_metrics(risk_te, surv_fn, ytr, yte, h):
    """Test-fold Harrell C, Uno C (on risk_te), IBS and Brier at the horizon from surv_fn(grid)."""
    ttr, etr, tte, ete = ytr[:, 0], ytr[:, 1], yte[:, 0], yte[:, 1]
    br = brier_at(surv_fn([h]), ttr, etr, tte, ete, [h])
    return {"harrell_c": harrell_c(risk_te, tte, ete), "uno_c": uno_c(risk_te, ttr, etr, tte, ete),
            "ibs": integrated_brier(surv_fn, ttr, etr, tte, ete), "brier_h": float(br["brier"][0])}


def cph_surv_fn(cph, df_te):
    def f(grid):
        sf = cph.predict_survival_function(df_te, times=np.asarray(grid, float))
        # lifelines groups a stratified model's columns by stratum: restore the row order
        return sf.loc[:, df_te.index].to_numpy().T
    return f


# --------------------------------------------------------------------------- part 1


def side_by_side(ds, names, Xtr_raw, Xtr, ytr, sd, refs, readings):
    cols = [f"x{i}" for i in range(len(names))]
    cph = fit_cph(frame(Xtr, ytr, cols))
    s = cph.summary
    beta = s["coef"].to_numpy()
    ref_coef = np.array([refs["models"]["cox_ph"]["coef"][n] for n in names])
    assert np.allclose(beta, ref_coef, atol=1e-6), "Cox PH refit differs from references.json"
    lo_sd, hi_sd = s["coef lower 95%"].to_numpy(), s["coef upper 95%"].to_numpy()
    raw = fit_cph(frame(Xtr_raw, ytr, cols))
    beta_raw = raw.params_.to_numpy()
    conv_err = float(np.max(np.abs(beta_raw - beta / sd)))
    sch = refs["models"]["cox_ph"]["schoenfeld"]["p"]
    net = {r["feature"]: r for r in readings["readings"]}
    rows = []
    for i, n in enumerate(names):
        r = net.get(n)
        row = {"input": n, "label": UNITS[ds][n][0], "unit": UNITS[ds][n][1], "sd_train": float(sd[i]),
               "cox_hr_unit": float(np.exp(beta[i] / sd[i])), "cox_lo_unit": float(np.exp(lo_sd[i] / sd[i])),
               "cox_hi_unit": float(np.exp(hi_sd[i] / sd[i])),
               "cox_hr_sd": float(np.exp(beta[i])), "cox_lo_sd": float(np.exp(lo_sd[i])),
               "cox_hi_sd": float(np.exp(hi_sd[i])), "cox_p": float(s["p"].iloc[i]),
               "schoenfeld_p": float(sch[n]),
               "net_hr_unit": float(r["hr_per_unit"]) if r else None,
               "net_hr_sd": float(r["hr_per_sd"]) if r else None,
               "kept": r is not None}
        row["agreement"] = agreement(row["cox_lo_sd"], row["cox_hi_sd"], row["cox_hr_sd"], row["net_hr_sd"])
        rows.append(row)
    return rows, conv_err


# --------------------------------------------------------------------------- part 2


class ForestReader:
    def __init__(self, rsf, h):
        self.rsf, self.h = rsf, h
        self.idx = int(np.searchsorted(rsf.unique_times_, h, side="right") - 1)

    def surv_h(self, X):
        sf = self.rsf.predict_survival_function(X, return_array=True)
        return np.clip(sf[:, self.idx], 1e-9, 1 - 1e-9)

    def risk(self, X):
        return 1.0 - self.surv_h(X)

    def logH(self, X):
        return np.log(-np.log(self.surv_h(X)))


def pd_curve(fn, Xbase, j, grid_std):
    out = []
    for v in grid_std:
        Xm = Xbase.copy()
        Xm[:, j] = v
        out.append(float(np.mean(fn(Xm))))
    return np.array(out)


def friedman_h(fn, Xs, j, k):
    """Friedman's H^2 (normalised) and the unnormalised H (root mean square of the residual)."""
    n = len(Xs)
    rep = np.repeat(Xs, n, axis=0)                      # row a*n+b: background row b
    tile_i = np.tile(np.arange(n), n)
    pt = np.repeat(np.arange(n), n)                     # the point x_a whose values are fixed

    def pd_at(feats):
        Xm = Xs[tile_i].copy()
        for f in feats:
            Xm[:, f] = Xs[pt, f]
        return fn(Xm).reshape(n, n).mean(1)
    pj, pk, pjk = pd_at([j]), pd_at([k]), pd_at([j, k])
    pj, pk, pjk = pj - pj.mean(), pk - pk.mean(), pjk - pjk.mean()
    num = np.sum((pjk - pj - pk) ** 2)
    den = np.sum(pjk ** 2)
    del rep
    return float(num / den) if den > 0 else 0.0, float(np.sqrt(num / n))


def forest_reading(ds, names, Xtr, ytr, Xte, yte, mu, sd, refs, h):
    from sklearn.inspection import permutation_importance
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    rsf = RandomSurvivalForest(**RSF, n_jobs=1).fit(Xtr, Surv.from_arrays(ytr[:, 1].astype(bool), ytr[:, 0]))
    c_te = harrell_c(rsf.predict(Xte), yte[:, 0], yte[:, 1])
    assert abs(c_te - refs["models"]["rsf"]["harrell_c"]) < 1e-9, "forest refit differs from references.json"
    pi = permutation_importance(rsf, Xtr, Surv.from_arrays(ytr[:, 1].astype(bool), ytr[:, 0]),
                                n_repeats=10, random_state=0, n_jobs=1)
    imp = sorted(({"input": n, "mean": float(m), "sd": float(s)}
                  for n, m, s in zip(names, pi.importances_mean, pi.importances_std)),
                 key=lambda d: -d["mean"])
    reader = ForestReader(rsf, h)
    top = [d["input"] for d in imp[:4]]
    top5 = [d["input"] for d in imp[:5]]
    rng = np.random.default_rng(0)
    sub = Xtr[rng.choice(len(Xtr), size=min(120, len(Xtr)), replace=False)]
    pairs = []
    for a in range(len(top5)):
        for b in range(a + 1, len(top5)):
            j, k = names.index(top5[a]), names.index(top5[b])
            h2, hraw = friedman_h(reader.risk, sub, j, k)
            pairs.append({"pair": [top5[a], top5[b]], "H2": h2, "H_abs_risk": hraw})
    pairs.sort(key=lambda d: -d["H_abs_risk"])
    return rsf, reader, imp, top, pairs, c_te


def grid_for(x_raw):
    u = np.unique(x_raw)
    if len(u) <= 2:
        return u
    return np.unique(np.quantile(x_raw, np.linspace(0.05, 0.95, 19)))


# --------------------------------------------------------------------------- part 3


def gbsg2_diagnosis(names, Xtr, ytr, Xte, yte, reader, h, best_pair):
    from scipy.stats import chi2
    from sklearn.preprocessing import SplineTransformer

    cont = ["age", "estrec", "pnodes", "progrec", "tsize"]
    cols = [f"x{i}" for i in range(len(names))]

    def design(Xa, Xb, spline=(), product=None):
        A, B, cn = [Xa], [Xb], list(cols)
        for n in spline:
            j = names.index(n)
            st = SplineTransformer(n_knots=4, degree=3, knots="quantile", include_bias=False).fit(Xa[:, [j]])
            sa, sb = st.transform(Xa[:, [j]]), st.transform(Xb[:, [j]])
            # drop the column most collinear with the linear term's span: keep df = 4 extra
            sa, sb = sa[:, 1:], sb[:, 1:]
            A.append(sa), B.append(sb)
            cn += [f"s_{n}_{m}" for m in range(sa.shape[1])]
        if product:
            j, k = names.index(product[0]), names.index(product[1])
            A.append((Xa[:, j] * Xa[:, k])[:, None]), B.append((Xb[:, j] * Xb[:, k])[:, None])
            cn.append("prod")
        return np.hstack(A), np.hstack(B), cn

    def run(label, spline=(), product=None, strata=False):
        A, B, cn = design(Xtr, Xte, spline, product)
        dtr, dte = frame(A, ytr, cn), frame(B, yte, cn)
        kw = {}
        if strata:
            g3 = cols[names.index("tgrade=III")]
            kw = {"strata": [g3]}
        cph = fit_cph(dtr, **kw)
        risk_te = 1.0 - cph_surv_fn(cph, dte)([h])[:, 0]
        m = surv_metrics(risk_te, cph_surv_fn(cph, dte), ytr, yte, h)
        return {"model": label, "loglik_train": float(cph.log_likelihood_), "n_params": int(len(cph.params_)), **m,
                "strata": strata}

    base = run("Cox PH, linear")
    out = [base]
    for n in cont:
        out.append(run(f"+ spline {n}", spline=[n]))
    out.append(run("+ splines age and pnodes", spline=["age", "pnodes"]))
    out.append(run("+ splines, all five continuous", spline=cont))
    out.append(run(f"+ product {best_pair[0]} x {best_pair[1]}", product=best_pair))
    out.append(run("grade III stratified (no PH for grade III)", strata=True))
    out.append(run("splines + product + grade III stratified", spline=cont, product=best_pair, strata=True))
    for r in out:
        if not r["strata"] and r is not base:
            df = r["n_params"] - base["n_params"]
            lr = 2 * (r["loglik_train"] - base["loglik_train"])
            r.update(lr=float(lr), df=int(df), lr_p=float(chi2.sf(lr, df)))
    # forest scored the same way (risk at the horizon for C; its own curves for IBS)
    fns = reader.rsf.predict_survival_function(Xte)

    def fsurv(grid):
        g = np.clip(np.asarray(grid, float), fns[0].x.min(), fns[0].x.max())
        return np.array([np.atleast_1d(f(g)) for f in fns])
    out.append({"model": "Random survival forest", **surv_metrics(reader.risk(Xte), fsurv, ytr, yte, h),
                "strata": False})
    return out


def grade3_over_time(names, Xtr, mu, sd, reader, cox_beta_sd, net_beta_sd):
    j = names.index("tgrade=III")
    sd_j, mu_j = sd[j], mu[j]   # raw-unit scaler: Xtr is standardised
    on, off = (1 - mu_j) / sd_j, (0 - mu_j) / sd_j
    rsf = reader.rsf
    times = np.arange(180, 2400, 60, dtype=float)
    times = times[times <= rsf.unique_times_.max()]
    res = []
    X1, X0 = Xtr.copy(), Xtr.copy()
    X1[:, j], X0[:, j] = on, off
    # grade II stays as observed: the contrast is III vs not-III (I or II)
    s1 = rsf.predict_survival_function(X1, return_array=True)
    s0 = rsf.predict_survival_function(X0, return_array=True)
    for t in times:
        i = int(np.searchsorted(rsf.unique_times_, t, side="right") - 1)
        a, b = np.clip(s1[:, i], 1e-9, 1 - 1e-9), np.clip(s0[:, i], 1e-9, 1 - 1e-9)
        res.append({"t_days": float(t), "rsf_log_HR": float(np.mean(np.log(-np.log(a)) - np.log(-np.log(b))))})
    return {"times": res, "cox_log_HR": float(cox_beta_sd * (on - off)),
            "net_log_HR": float(net_beta_sd * (on - off))}


# --------------------------------------------------------------------------- figures


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_forest(ds, rows, path):
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["cox_hr_sd"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(rows) + 1.4), sharey=True)
    for ax, scale in zip(axes, ("unit", "sd")):
        style(ax)
        y = np.arange(len(rows))
        hr = np.array([r[f"cox_hr_{scale}"] for r in rows])
        lo = np.array([r[f"cox_lo_{scale}"] for r in rows])
        hi = np.array([r[f"cox_hi_{scale}"] for r in rows])
        ax.hlines(y, lo, hi, color=C_COX, lw=2)
        ax.plot(hr, y, "o", color=C_COX, ms=7, mec="white", mew=1.5, label="Cox PH (95% CI)")
        for yi, r in zip(y, rows):
            v = r[f"net_hr_{scale}"]
            if v is None:
                ax.plot(1.0, yi, "x", color=INK2, ms=6, mew=1.2)
            else:
                ax.plot(v, yi, "D", color=C_NET, ms=7, mec="white", mew=1.5)
        ax.plot([], [], "D", color=C_NET, ms=7, label="PropNEAT-Cox")
        ax.plot([], [], "x", color=INK2, ms=6, label="dropped by the network")
        ax.axvline(1.0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("hazard ratio per clinical unit (log scale)" if scale == "unit"
                      else "hazard ratio per training SD (log scale)", color=INK, fontsize=9)
    axes[0].set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels([f"{r['label']} ({r['unit']})" for r in rows], fontsize=8, color=INK)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"{ds.upper()}: Cox PH and the network, training rows", color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def fig_pd(ds, panels, years, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.2))
    for ax, p in zip(np.atleast_1d(axes), panels):
        style(ax)
        x = np.array(p["grid_raw"])
        kind = "o-" if len(x) <= 2 else "-"
        ax.plot(x, p["rsf"], kind, color=C_RSF, lw=2, ms=6, label="forest (PD)")
        ax.plot(x, p["cox"], kind, color=C_COX, lw=2, ms=6, label="Cox PH")
        ax.plot(x, p["net"], kind.replace("-", "--") if len(x) > 2 else "D--", color=C_NET, lw=2, ms=6,
                label="PropNEAT-Cox")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"{p['label']}", fontsize=9, color=INK)
        ax.set_xlabel(p["unit"].replace("per ", ""), fontsize=8, color=INK2)
        if len(x) <= 2:
            ax.set_xticks(x)
    np.atleast_1d(axes)[0].set_ylabel(f"log cumulative hazard at {years} y,\nrelative to the median patient",
                                      fontsize=8, color=INK)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{ds.upper()}: the forest's top inputs by permutation importance (training rows)",
                 color=INK, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def fig_h2d(ds, h2d, years, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    z = np.array(h2d["risk"])
    im = ax.imshow(z, origin="lower", aspect="auto", cmap="Blues",
                   extent=[0, z.shape[1], 0, z.shape[0]])
    ax.set_xticks(np.arange(z.shape[1]) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in h2d["grid_b"]], fontsize=6, rotation=90)
    ax.set_yticks(np.arange(z.shape[0]) + 0.5)
    ax.set_yticklabels([f"{v:g}" for v in h2d["grid_a"]], fontsize=6)
    ax.set_xlabel(h2d["b"], fontsize=8, color=INK)
    ax.set_ylabel(h2d["a"], fontsize=8, color=INK)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"forest {years}-year risk (PD)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"{ds.upper()}: strongest pair, {h2d['a']} x {h2d['b']}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def fig_grade3(g3, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    style(ax)
    t = np.array([r["t_days"] for r in g3["times"]]) / 365.25
    ax.plot(t, [r["rsf_log_HR"] for r in g3["times"]], color=C_RSF, lw=2, label="forest (PD)")
    ax.axhline(g3["cox_log_HR"], color=C_COX, lw=2, label="Cox PH (constant by assumption)")
    ax.axhline(g3["net_log_HR"], color=C_NET, lw=2, ls="--", label="PropNEAT-Cox")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("years since surgery", fontsize=8, color=INK)
    ax.set_ylabel("grade III vs not: log cumulative\nhazard ratio at t", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("GBSG2: is grade III's effect constant over time?", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- tables


def f3(v, d=3):
    return "--" if v is None else f"{v:.{d}f}"


def dig(r, scale):
    """Per-unit HRs of continuous inputs sit close to 1: four decimals for the whole row."""
    return 4 if scale == "unit" and r["unit"].startswith("per ") else 3


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(rows, scale):
    u = "per unit" if scale == "unit" else "per SD"
    out = [f"| Input | Unit | Cox PH HR {u} (95% CI) | p | PH test p | Network HR {u} | Network | Agreement |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        unit = r["unit"] if scale == "unit" else f"SD = {r['sd_train']:.3g}"
        d = dig(r, scale)
        out.append(f"| {r['label']} | {unit} | {f3(r[f'cox_hr_{scale}'], d)} ({f3(r[f'cox_lo_{scale}'], d)}–"
                   f"{f3(r[f'cox_hi_{scale}'], d)}) | {fp(r['cox_p'])} | {fp(r['schoenfeld_p'])} | "
                   f"{f3(r[f'net_hr_{scale}'], d)} | {'kept' if r['kept'] else 'dropped'} | {r['agreement']} |")
    return "\n".join(out)


def tex_escape(s):
    return s.replace("²", "$^2$").replace("%", r"\%").replace("&", r"\&")


def tex_table(ds, rows, scale, caption, label):
    u = "per unit" if scale == "unit" else "per SD"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\begin{tabular}{llrrrrll}", r"\toprule",
             rf"Input & Unit & Cox PH HR {u} (95\% CI) & $p$ & PH $p$ & Network HR & Network & Agreement \\",
             r"\midrule"]
    for r in rows:
        unit = r["unit"] if scale == "unit" else f"SD {r['sd_train']:.3g}"
        d = dig(r, scale)
        lines.append(rf"{tex_escape(r['label'])} & {tex_escape(unit)} & {f3(r[f'cox_hr_{scale}'], d)} "
                     rf"({f3(r[f'cox_lo_{scale}'], d)}--{f3(r[f'cox_hi_{scale}'], d)}) & {fp(r['cox_p']).replace('<', '$<$')} & "
                     rf"{fp(r['schoenfeld_p']).replace('<', '$<$')} & {f3(r[f'net_hr_{scale}'], d)} & "
                     rf"{'kept' if r['kept'] else 'dropped'} & {r['agreement']} \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def main():
    FIG.mkdir(exist_ok=True)
    refs_all = json.loads((HERE / "references.json").read_text())
    out = {"_doc": (__doc__ or "").strip().splitlines()[0], "datasets": {}}
    tex = ["% Generated by experiments/survival/cox_case/reference_explanation.py; do not edit.",
           "% Cox PH (lifelines, unpenalised, Efron) on the training rows of each case split;",
           "% per-unit HRs converted from the standardised fit; network HRs from hazard_readings/.", ""]
    for ds, spec in DATASETS.items():
        h = spec["calibration_horizon"]
        X, y, names = load_builtin(ds)
        sp = load_split_indices(HERE / "splits" / f"{ds}.splits.json")
        tr, te = sp["train"], sp["test"]
        mu, sd, Xtr, Xte = standardise(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        refs = refs_all[ds]
        readings = json.loads((HERE / "hazard_readings" / f"{TAGS[ds]}.json").read_text())

        rows, conv_err = side_by_side(ds, names, X[tr], Xtr, ytr, sd, refs, readings)

        # network log-risk from its readings (linear, standardised inputs); check training C
        w_net = np.array([next((r["log_hr_per_sd"] for r in readings["readings"] if r["feature"] == n), 0.0)
                          for n in names])
        g_net_tr = Xtr @ w_net
        net_train_c = harrell_c(g_net_tr, ytr[:, 0], ytr[:, 1])
        beta_cox = np.array([refs["models"]["cox_ph"]["coef"][n] for n in names])

        rsf, reader, imp, top, pairs, rsf_c = forest_reading(ds, names, Xtr, ytr, Xte, yte, mu, sd, refs, h)

        # PD on the log cumulative hazard at the horizon, centred at the training median
        panels = []
        for n in top:
            j = names.index(n)
            graw = grid_for(X[tr][:, j])
            gstd = (graw - mu[j]) / sd[j]
            med_std = (np.median(X[tr][:, j]) - mu[j]) / sd[j]
            ref_std = gstd[0] if len(graw) <= 2 else med_std
            rsf_pd = pd_curve(reader.logH, Xtr, j, gstd) - pd_curve(reader.logH, Xtr, j, [ref_std])[0]
            risk_pd = pd_curve(reader.risk, Xtr, j, gstd)
            panels.append({"input": n, "label": UNITS[ds][n][0], "unit": UNITS[ds][n][1],
                           "grid_raw": graw.tolist(), "rsf": rsf_pd.tolist(),
                           "rsf_risk_at_h": risk_pd.tolist(),
                           "cox": (beta_cox[j] * (gstd - ref_std)).tolist(),
                           "net": (w_net[j] * (gstd - ref_std)).tolist(),
                           "reference": "the lower level" if len(graw) <= 2 else "the training median"})
        # 2-D PD of the strongest pair
        a, b = pairs[0]["pair"]
        ja, jb = names.index(a), names.index(b)
        ga, gb = grid_for(X[tr][:, ja])[::2] if len(np.unique(X[tr][:, ja])) > 2 else grid_for(X[tr][:, ja]), \
            grid_for(X[tr][:, jb])[::2] if len(np.unique(X[tr][:, jb])) > 2 else grid_for(X[tr][:, jb])
        z = np.zeros((len(ga), len(gb)))
        for p, va in enumerate(ga):
            for q, vb in enumerate(gb):
                Xm = Xtr.copy()
                Xm[:, ja], Xm[:, jb] = (va - mu[ja]) / sd[ja], (vb - mu[jb]) / sd[jb]
                z[p, q] = np.mean(reader.risk(Xm))
        h2d = {"a": a, "b": b, "grid_a": [float(v) for v in ga], "grid_b": [float(v) for v in gb], "risk": z.tolist()}

        res = {"horizon_days": h, "years": YEARS[ds], "n_train": int(len(tr)),
               "cox_per_unit_conversion": "standardised fit converted (beta / sd_train); raw-unit refit "
                                          f"max |coef diff| {conv_err:.2e}",
               "network_train_harrell_c_from_readings": net_train_c,
               "rsf_test_harrell_c_check": rsf_c,
               "table": rows, "rsf_permutation_importance": imp, "rsf_top_inputs": top,
               "rsf_friedman_h": pairs, "pd": panels, "pd2d": h2d}
        if ds == "gbsg2":
            res["diagnosis"] = gbsg2_diagnosis(names, Xtr, ytr, Xte, yte, reader, h, pairs[0]["pair"])
            j3 = names.index("tgrade=III")
            res["grade3_over_time"] = grade3_over_time(names, Xtr, mu, sd, reader, beta_cox[j3], w_net[j3])
            fig_grade3(res["grade3_over_time"], FIG / "ref_grade3_time_gbsg2.png")
        fig_forest(ds, rows, FIG / f"ref_forest_{ds}.png")
        fig_pd(ds, panels, YEARS[ds], FIG / f"ref_pd_{ds}.png")
        fig_h2d(ds, h2d, YEARS[ds], FIG / f"ref_h2d_{ds}.png")
        out["datasets"][ds] = res

        print(f"\n## {ds}  (conversion check {conv_err:.1e}; network train C {net_train_c:.3f}; forest test C {rsf_c:.3f})")
        print(md_table(rows, "unit"))
        print()
        print(md_table(rows, "sd"))
        print("\nforest permutation importance:", [(d["input"], round(d["mean"], 4)) for d in imp])
        print("Friedman H (top 5):", [(p["pair"], round(p["H2"], 3), round(p["H_abs_risk"], 4)) for p in pairs])
        for p in panels:
            print(p["input"], "risk PD", np.round(p["rsf_risk_at_h"], 3).tolist())
        if ds == "gbsg2":
            for r in res["diagnosis"]:
                print(f"  {r['model']:45s} C {r['harrell_c']:.3f} Uno {r['uno_c']:.3f} IBS {r['ibs']:.4f} "
                      f"Brier5y {r['brier_h']:.4f}" + (f" LR {r['lr']:.2f} df {r['df']} p {r['lr_p']:.3g}" if "lr" in r else ""))
            print("grade III log-HR over time:", [(round(r["t_days"] / 365.25, 1), round(r["rsf_log_HR"], 3))
                                                  for r in res["grade3_over_time"]["times"][::4]],
                  "Cox", round(res["grade3_over_time"]["cox_log_HR"], 3))

        name = "GBSG2 (recurrence-free survival)" if ds == "gbsg2" else "WHAS500 (all-cause death)"
        for scale in ("unit", "sd"):
            what = "per clinical unit" if scale == "unit" else "per training standard deviation"
            tex.append(tex_table(
                ds, rows, scale,
                f"{name}: unpenalised Cox PH hazard ratios {what} with 95\\% CI and Wald $p$, the Schoenfeld "
                f"proportional-hazards test $p$ (rank transform), and the PropNEAT-Cox network's hazard ratio, on the "
                f"{len(tr)} training rows of the case split. Agreement: same direction and inside the Cox CI "
                f"(within), same direction outside it (outside), opposite sign (opposite), input not used by the "
                f"network (dropped).",
                f"tab:cox-ref-{ds}-{scale}"))
    (HERE / "reference_explanation.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    TEX.write_text("\n".join(tex))


if __name__ == "__main__":
    main()
