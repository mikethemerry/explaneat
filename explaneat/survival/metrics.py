"""Survival evaluation metrics.

Pre-specify these before running anything (brief section 4):

* **Harrell's C**: report for comparability; biased upward under heavy
  censoring.
* **Uno's IPCW C** (truncated at tau): lead with this one.
* **Time-dependent (cumulative/dynamic) AUC** at chosen horizons, e.g. 5 years
  for PREDICT-style CVD risk.
* **IPCW Brier score** at horizons and the **integrated Brier score**:
  discrimination and calibration together; needs S(t|x), i.e. the Breslow
  baseline.
* **Calibration at a horizon**: predicted vs observed (Kaplan-Meier) risk by
  decile of predicted risk (``calibration_at_horizon``).

Concordance is always computed on the scalar log-risk g(x) for every model
(network and Cox alike), so there is no distribution-to-risk transformation to
choose (the "C-hacking" problem of Sonabend et al. 2022).

Harrell's C has a dependency-free implementation (``concordance``).  The IPCW
metrics use scikit-survival (``pip install explaneat[survival]``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=np.float64).ravel()


def _surv(time, event):
    from sksurv.util import Surv
    return Surv.from_arrays(event=_arr(event).astype(bool), time=_arr(time))


def concordance(risk, time, event) -> float:
    """Harrell's C-index (higher risk = shorter survival), no dependencies.

    Comparability follows scikit-survival: an event at t_i is comparable with j
    if t_j > t_i, or t_j == t_i and j is censored.  Two events at the same time
    are not comparable.  Tied risk scores score 0.5.  Matches
    ``sksurv.metrics.concordance_index_censored`` exactly.
    """
    r, t, e = _arr(risk), _arr(time), _arr(event)
    conc = disc = tied = 0.0
    for i in np.flatnonzero(e == 1):
        comparable = (t > t[i]) | ((t == t[i]) & (e == 0))
        if not comparable.any():
            continue
        rj = r[comparable]
        conc += float((rj < r[i]).sum())
        disc += float((rj > r[i]).sum())
        tied += float((rj == r[i]).sum())
    denom = conc + disc + tied
    return (conc + 0.5 * tied) / denom if denom else float("nan")


def harrell_c(risk, time, event) -> float:
    """Harrell's C via scikit-survival when installed, else ``concordance``."""
    try:
        from sksurv.metrics import concordance_index_censored
    except ImportError:  # pragma: no cover - optional dependency
        return concordance(risk, time, event)
    return float(concordance_index_censored(_arr(event).astype(bool), _arr(time), _arr(risk))[0])


def default_tau(time_train, event_train, time_test, event_test) -> float:
    """A truncation time inside both folds' event support."""
    tt, et, ts, es = _arr(time_train), _arr(event_train), _arr(time_test), _arr(event_test)
    return float(min(ts[es == 1].max(), tt[et == 1].max()))


def uno_c(risk_test, time_train, event_train, time_test, event_test,
          tau: Optional[float] = None) -> float:
    """Uno's IPCW concordance, censoring distribution estimated on the train fold."""
    from sksurv.metrics import concordance_index_ipcw
    if tau is None:
        tau = default_tau(time_train, event_train, time_test, event_test)
    return float(concordance_index_ipcw(_surv(time_train, event_train),
                                        _surv(time_test, event_test),
                                        _arr(risk_test), tau=tau)[0])


def valid_horizons(times: Sequence[float], time_train, event_train, time_test) -> np.ndarray:
    """Keep the horizons at which the IPCW metrics are defined for these folds."""
    tt, ts = _arr(time_train), _arr(time_test)
    h = np.asarray(times, dtype=np.float64)
    hi = min(ts.max(), tt.max())
    return h[(h > ts.min()) & (h < hi)]


def time_dependent_auc(risk_test, time_train, event_train, time_test, event_test,
                       times: Sequence[float]) -> Dict[str, object]:
    """Cumulative/dynamic AUC at each horizon plus the integrated mean AUC."""
    from sksurv.metrics import cumulative_dynamic_auc
    h = valid_horizons(times, time_train, event_train, time_test)
    if len(h) == 0:
        return {"times": [], "auc": [], "mean_auc": float("nan")}
    auc, mean_auc = cumulative_dynamic_auc(_surv(time_train, event_train),
                                           _surv(time_test, event_test),
                                           _arr(risk_test), h)
    return {"times": h.tolist(), "auc": [float(a) for a in auc], "mean_auc": float(mean_auc)}


def brier_at(surv_test, time_train, event_train, time_test, event_test,
             times: Sequence[float]) -> Dict[str, object]:
    """IPCW Brier score at each horizon.  ``surv_test`` is [n_test, len(times)]."""
    from sksurv.metrics import brier_score
    t_out, scores = brier_score(_surv(time_train, event_train), _surv(time_test, event_test),
                                np.asarray(surv_test, dtype=np.float64),
                                np.asarray(times, dtype=np.float64))
    return {"times": [float(x) for x in t_out], "brier": [float(s) for s in scores]}


def default_time_grid(time_train, event_train, time_test, event_test, n: int = 25) -> np.ndarray:
    """IBS grid: inside the follow-up of both folds, up to the 80th event percentile."""
    tt, et, ts, es = _arr(time_train), _arr(event_train), _arr(time_test), _arr(event_test)
    lo = max(tt.min(), ts.min())
    hi = min(np.percentile(ts[es == 1], 80), tt[et == 1].max() * 0.99)
    if not hi > lo:
        return np.array([])
    grid = np.linspace(lo, hi, n)
    return grid[(grid > ts.min()) & (grid < ts.max())]


def integrated_brier(surv_fn, time_train, event_train, time_test, event_test,
                     grid: Optional[np.ndarray] = None) -> float:
    """Integrated IPCW Brier score.  ``surv_fn(grid) -> [n_test, len(grid)]``."""
    from sksurv.metrics import integrated_brier_score
    if grid is None:
        grid = default_time_grid(time_train, event_train, time_test, event_test)
    if len(grid) < 2:
        return float("nan")
    return float(integrated_brier_score(_surv(time_train, event_train),
                                        _surv(time_test, event_test),
                                        surv_fn(grid), grid))


def kaplan_meier(time, event, at: float) -> float:
    """Kaplan-Meier survival estimate at ``at`` (dependency-free)."""
    t, e = _arr(time), _arr(event)
    s = 1.0
    for tau in np.unique(t[(e == 1) & (t <= at)]):
        n_risk = float((t >= tau).sum())
        d = float(((t == tau) & (e == 1)).sum())
        s *= 1.0 - d / n_risk
    return s


def calibration_at_horizon(predicted_risk, time, event, horizon: float,
                           n_groups: int = 10) -> Dict[str, object]:
    """Predicted vs observed risk at ``horizon`` by group of predicted risk.

    This is the PREDICT-style calibration check: "of the people we said had a
    10% 5-year risk, did ~10% have an event within 5 years?".

    ``predicted_risk`` is 1 - S(horizon | x) (``BreslowBaseline.risk_at``).
    Rows are cut into ``n_groups`` quantile groups (deciles by default); the
    observed risk in each is 1 - KM(horizon) within the group, which handles
    censoring before the horizon correctly (a naive event proportion does not).

    Returns per-group rows and summaries: mean predicted and observed risk
    (calibration-in-the-large), the expected/observed ratio, and the maximum
    absolute group difference.  This is a hook: recalibration and smoothed
    calibration curves can be layered on these rows.
    """
    p, t, e = _arr(predicted_risk), _arr(time), _arr(event)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, n_groups)
    rows = []
    for k, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        obs = 1.0 - kaplan_meier(t[idx], e[idx], horizon)
        rows.append({
            "group": k + 1,
            "n": int(len(idx)),
            "events_by_horizon": int(((e[idx] == 1) & (t[idx] <= horizon)).sum()),
            "mean_predicted": float(p[idx].mean()),
            "observed_km": float(obs),
        })
    obs_all = 1.0 - kaplan_meier(t, e, horizon)
    mean_pred = float(p.mean())
    return {
        "horizon": float(horizon),
        "n_groups": len(rows),
        "groups": rows,
        "mean_predicted": mean_pred,
        "observed_km": float(obs_all),
        "expected_over_observed": float(mean_pred / obs_all) if obs_all > 0 else float("nan"),
        "max_abs_group_diff": float(max(abs(r["mean_predicted"] - r["observed_km"]) for r in rows)),
    }


def survival_metrics(risk_train, risk_test, time_train, event_train, time_test, event_test,
                     baseline=None, horizons: Optional[Sequence[float]] = None,
                     calibration_horizon: Optional[float] = None,
                     tau: Optional[float] = None) -> Dict[str, object]:
    """The pre-specified metric bundle for one fold.

    ``risk_*`` are log-risks g(x).  ``baseline`` is a fitted ``BreslowBaseline``
    for the same model (fitted on the training rows); without it the
    curve-based metrics (Brier, IBS, calibration) are omitted.  Every metric is
    wrapped so a degenerate fold reports NaN rather than raising.
    """
    out: Dict[str, object] = {
        "n_test": int(len(_arr(time_test))),
        "events_test": int(_arr(event_test).sum()),
        "harrell_c": harrell_c(risk_test, time_test, event_test),
    }
    try:
        out["uno_c"] = uno_c(risk_test, time_train, event_train, time_test, event_test, tau=tau)
    except Exception:
        out["uno_c"] = float("nan")
    if horizons is not None and len(horizons):
        try:
            out["td_auc"] = time_dependent_auc(risk_test, time_train, event_train,
                                               time_test, event_test, horizons)
        except Exception:
            out["td_auc"] = {"times": [], "auc": [], "mean_auc": float("nan")}
    if baseline is not None:
        r_te = _arr(risk_test)
        try:
            out["ibs"] = integrated_brier(lambda grid: baseline.survival(r_te, grid),
                                          time_train, event_train, time_test, event_test)
        except Exception:
            out["ibs"] = float("nan")
        if horizons is not None and len(horizons):
            h = valid_horizons(horizons, time_train, event_train, time_test)
            try:
                out["brier"] = brier_at(baseline.survival(r_te, h), time_train, event_train,
                                        time_test, event_test, h) if len(h) else {"times": [], "brier": []}
            except Exception:
                out["brier"] = {"times": [], "brier": []}
        if calibration_horizon is not None:
            out["calibration"] = calibration_at_horizon(
                baseline.risk_at(r_te, calibration_horizon), time_test, event_test,
                calibration_horizon)
    return out
