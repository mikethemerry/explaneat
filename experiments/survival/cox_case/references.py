"""Reference models for the Cox survival case, on the case's own split and test fold.

    uv run python experiments/survival/cox_case/references.py

Per dataset, on the standardised features (scaler fitted on the training rows, as the runs):

* ridge Cox: scikit-survival ``CoxPHSurvivalAnalysis`` (Efron ties), alpha tuned on the
  validation fold by validation Cox loss (``run_seed.fit_cox``, the existing comparator);
* Cox PH, unpenalised: lifelines ``CoxPHFitter(penalizer=0)`` (Efron ties) on the training
  rows, plus its Schoenfeld proportional-hazards test (rank transform) for the PH caveat;
* random survival forest: scikit-survival ``RandomSurvivalForest`` (500 trees,
  min_samples_leaf 10, random_state 42) on the training rows; the nonlinear reference.

The Cox models' survival curves use the same Breslow baseline as the networks
(``BreslowBaseline`` on the training log-risks); the forest's come from its own
``predict_survival_function``. Metrics are ``explaneat.survival.metrics`` on the test fold at
the case's horizons. Output: ``references.json`` and ``references.csv`` (one row per model,
including the selected network's own test row copied from its run JSON).
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SURV = HERE.parent
REPO = SURV.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SURV))

from run_case import DATASETS, HORIZONS  # noqa: E402
from run_seed import TIES, fit_cox  # noqa: E402

from explaneat.survival import BreslowBaseline, CoxRiskSets, cox_ph_loss  # noqa: E402
from explaneat.survival.data import load_builtin, load_split_indices, standardise  # noqa: E402
from explaneat.survival.metrics import (  # noqa: E402
    brier_at, calibration_at_horizon, harrell_c, integrated_brier, survival_metrics,
    time_dependent_auc, uno_c, valid_horizons,
)

RSF = {"n_estimators": 500, "min_samples_leaf": 10, "max_features": "sqrt", "random_state": 42}


def closs(g, yy):
    return float(cox_ph_loss(torch.as_tensor(np.asarray(g, dtype=np.float64)),
                             CoxRiskSets(yy[:, 0], yy[:, 1]), ties=TIES))


def flat(m, horizons, cal_h):
    cal = m.get("calibration") or {}
    auc = dict(zip([int(t) for t in m.get("td_auc", {}).get("times", [])], m.get("td_auc", {}).get("auc", [])))
    br = dict(zip([int(t) for t in m.get("brier", {}).get("times", [])], m.get("brier", {}).get("brier", [])))
    return {"harrell_c": m["harrell_c"], "uno_c": m["uno_c"],
            "td_auc_1y": auc.get(365), "td_auc_3y": auc.get(1095), "td_auc_5y": auc.get(1825),
            "td_auc_mean": m.get("td_auc", {}).get("mean_auc"),
            "brier_at_cal": br.get(int(cal_h)), "ibs": m.get("ibs"),
            "cal_horizon": cal.get("horizon"), "cal_eo": cal.get("expected_over_observed"),
            "cal_max_abs_group_diff": cal.get("max_abs_group_diff"),
            "test_cox_loss": m.get("test_cox_loss")}


def rsf_metrics(rsf, Xte, ytr, yte, horizons, cal_h):
    """The same bundle for a model without a log-risk: risk = the forest's ensemble risk score."""
    risk = rsf.predict(Xte)
    ttr, etr, tte, ete = ytr[:, 0], ytr[:, 1], yte[:, 0], yte[:, 1]
    fns = rsf.predict_survival_function(Xte)

    def surv(grid):
        g = np.clip(np.asarray(grid, dtype=float), fns[0].x.min(), fns[0].x.max())
        return np.array([np.atleast_1d(f(g)) for f in fns])

    out = {"n_test": int(len(tte)), "events_test": int(ete.sum()),
           "harrell_c": harrell_c(risk, tte, ete), "uno_c": uno_c(risk, ttr, etr, tte, ete),
           "td_auc": time_dependent_auc(risk, ttr, etr, tte, ete, horizons),
           "ibs": integrated_brier(surv, ttr, etr, tte, ete)}
    h = valid_horizons(horizons, ttr, etr, tte)
    out["brier"] = brier_at(surv(h), ttr, etr, tte, ete, h) if len(h) else {"times": [], "brier": []}
    out["calibration"] = calibration_at_horizon(1.0 - surv([cal_h])[:, 0], tte, ete, cal_h)
    return out


def main():
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    selected = json.loads((HERE / "selected.json").read_text())
    horizons = [float(h) for h in HORIZONS.split(",")]
    results, rows = {}, []
    for ds, spec in DATASETS.items():
        cal_h = spec["calibration_horizon"]
        X, y, names = load_builtin(ds)
        splits = load_split_indices(HERE / "splits" / f"{ds}.splits.json")
        tr, va, te = splits["train"], splits["val"], splits["test"]
        mu, sd, Xtr, Xte, Xva = standardise(X[tr], X[te], X[va])
        ytr, yva, yte = y[tr], y[va], y[te]

        def evaluate(g_tr, g_te, g_va):
            bl = BreslowBaseline().fit(g_tr, ytr[:, 0], ytr[:, 1])
            m = survival_metrics(g_tr, g_te, ytr[:, 0], ytr[:, 1], yte[:, 0], yte[:, 1], baseline=bl,
                                 horizons=horizons, calibration_horizon=cal_h)
            m["test_cox_loss"], m["train_cox_loss"], m["val_cox_loss"] = closs(g_te, yte), closs(g_tr, ytr), closs(g_va, yva)
            return m

        res = {}
        # 1. ridge Cox (the existing comparator)
        cox, alpha, scores = fit_cox(Xtr, ytr, Xva, yva)
        res["ridge_cox"] = evaluate(cox.predict(Xtr), cox.predict(Xte), cox.predict(Xva))
        res["ridge_cox"].update(alpha=alpha, alpha_val_loss={str(k): v for k, v in scores.items()},
                                coef=dict(zip(names, map(float, cox.coef_))), n_params=len(names))
        # 2. unpenalised Cox PH (lifelines, Efron) + Schoenfeld test on the training rows
        cols = [f"x{i}" for i in range(len(names))]
        df_tr = pd.DataFrame(Xtr, columns=cols)
        df_tr["T"], df_tr["E"] = ytr[:, 0], ytr[:, 1]
        cph = CoxPHFitter(penalizer=0.0).fit(df_tr, "T", "E")
        g = lambda Xs: cph.predict_log_partial_hazard(pd.DataFrame(Xs, columns=cols)).to_numpy(float)  # noqa: E731
        res["cox_ph"] = evaluate(g(Xtr), g(Xte), g(Xva))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph = proportional_hazard_test(cph, df_tr, time_transform="rank")
        ph_tab = ph.summary
        res["cox_ph"].update(
            coef=dict(zip(names, map(float, cph.params_.to_numpy()))),
            hr_per_sd=dict(zip(names, map(float, np.exp(cph.params_.to_numpy())))),
            p_values=dict(zip(names, map(float, cph.summary["p"].to_numpy()))),
            n_params=len(names), log_likelihood=float(cph.log_likelihood_),
            schoenfeld={"transform": "rank", "rows": "train",
                        "p": {names[cols.index(str(i[0]) if isinstance(i, tuple) else str(i))]: float(p)
                              for i, p in zip(ph_tab.index, ph_tab["p"].to_numpy())},
                        "violations_at_0.05": [names[cols.index(str(i[0]) if isinstance(i, tuple) else str(i))]
                                               for i, p in zip(ph_tab.index, ph_tab["p"].to_numpy()) if p < 0.05]})
        # 3. random survival forest (nonlinear reference)
        rsf = RandomSurvivalForest(**RSF, n_jobs=1).fit(Xtr, Surv.from_arrays(ytr[:, 1].astype(bool), ytr[:, 0]))
        res["rsf"] = rsf_metrics(rsf, Xte, ytr, yte, horizons, cal_h)
        res["rsf"].update(params=RSF, val_harrell_c=harrell_c(rsf.predict(Xva), yva[:, 0], yva[:, 1]))
        # the selected network's test row, from its run JSON
        run = json.loads((HERE / "runs" / f"{selected[ds]['tag']}.json").read_text())
        res["propneat_cox"] = {k: v for k, v in run["results"]["neat"].items()}
        res["propneat_cox"]["tag"] = selected[ds]["tag"]
        results[ds] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "events_test": int(yte[:, 1].sum()),
                       "horizons": horizons, "calibration_horizon": cal_h, "models": res}
        for model in ("propneat_cox", "ridge_cox", "cox_ph", "rsf"):
            m = res[model]
            rows.append({"dataset": ds, "model": model, **flat(m, horizons, cal_h),
                         "k": m.get("genome", {}).get("complexity_k") if model == "propneat_cox" else m.get("n_params")})
            r = rows[-1]
            print(f"{ds:8s} {model:12s} C {r['harrell_c']:.3f} Uno {r['uno_c']:.3f} AUC1/3/5 "
                  f"{r['td_auc_1y']:.3f}/{r['td_auc_3y']:.3f}/{(r['td_auc_5y'] if r['td_auc_5y'] is not None else float('nan')):.3f} "
                  f"IBS {r['ibs']:.3f} E/O {r['cal_eo']:.3f}", flush=True)
        print(ds, "Schoenfeld p:", {k: round(v, 3) for k, v in res["cox_ph"]["schoenfeld"]["p"].items()})
    (HERE / "references.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    with open(HERE / "references.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
