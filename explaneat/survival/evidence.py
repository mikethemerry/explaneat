"""Model-level survival performance for the evidence tools (MCP / API).

``compute_performance`` on a dataset tagged ``task_type: "survival"`` reports
survival metrics instead of AUC / accuracy: Harrell's and Uno's C on the
log-risk g(x), time-dependent AUC and IPCW Brier at horizons, the integrated
Brier score, the Cox loss against its null, and calibration at a horizon
(predicted vs Kaplan-Meier observed risk by decile).  The Breslow baseline is
always fitted on the split's TRAINING rows, whatever split is evaluated.

Horizons come from, in order: the call's ``horizons`` argument, the dataset's
``additional_metadata["horizons"]``, else the 25/50/75% quantiles of the
training event times.  The calibration horizon likewise: argument,
``additional_metadata["calibration_horizon"]`` (e.g. 5 years for
PREDICT-style CVD risk), else the middle horizon.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from .breslow import BreslowBaseline
from .cox import CoxRiskSets, cox_ph_loss, null_cox_loss
from .metrics import survival_metrics

PH_CAVEAT = (
    "Proportional hazards is inherited, not chosen: g(x) has no time input, so "
    "hazard ratios between subjects are constant over follow-up and survival "
    "curves cannot cross. Check PH on the comparator Cox model (Schoenfeld "
    "residuals) before reading these as time-varying effects."
)


def _clean(obj):
    """NaN/inf -> None, numpy scalars -> python, recursively (JSON-safe)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def default_horizons(y_train) -> list:
    t, e = np.asarray(y_train)[:, 0], np.asarray(y_train)[:, 1]
    return [float(q) for q in np.quantile(t[e == 1], [0.25, 0.5, 0.75])]


def survival_performance(predict_g: Callable, X_train, y_train, X_eval, y_eval,
                         meta: Optional[dict] = None,
                         horizons: Optional[Sequence[float]] = None,
                         calibration_horizon: Optional[float] = None,
                         ties: str = "efron") -> Dict[str, object]:
    """Survival metric bundle for a model whose output is the log-risk g(x)."""
    meta = meta or {}
    y_train = np.asarray(y_train, dtype=np.float64)
    y_eval = np.asarray(y_eval, dtype=np.float64)
    g_tr = np.asarray(predict_g(X_train), dtype=np.float64).ravel()
    g_ev = np.asarray(predict_g(X_eval), dtype=np.float64).ravel()
    t_tr, e_tr = y_train[:, 0], y_train[:, 1]
    t_ev, e_ev = y_eval[:, 0], y_eval[:, 1]

    if horizons is None:
        horizons = meta.get("horizons") or default_horizons(y_train)
    horizons = [float(h) for h in horizons]
    if calibration_horizon is None:
        calibration_horizon = meta.get("calibration_horizon") or horizons[len(horizons) // 2]

    baseline = BreslowBaseline().fit(g_tr, t_tr, e_tr)
    out = survival_metrics(g_tr, g_ev, t_tr, e_tr, t_ev, e_ev, baseline=baseline,
                           horizons=horizons, calibration_horizon=float(calibration_horizon))
    rs = CoxRiskSets(t_ev, e_ev)
    out["cox_loss"] = float(cox_ph_loss(torch.as_tensor(g_ev), rs, ties=ties))
    out["null_cox_loss"] = null_cox_loss(rs, ties=ties)
    out["censored_fraction"] = float(1.0 - e_ev.mean())
    out["log_risk"] = {"mean": float(g_ev.mean()), "sd": float(g_ev.std()),
                       "p5": float(np.percentile(g_ev, 5)), "p95": float(np.percentile(g_ev, 95)),
                       "hr_p95_vs_p5": float(np.exp(np.percentile(g_ev, 95) - np.percentile(g_ev, 5)))}
    out["task_type"] = "survival"
    out["risk_score"] = "log-risk g(x) (model output, identity activation); C computed on g"
    out["horizons"] = horizons
    if meta.get("time_unit"):
        out["time_unit"] = meta["time_unit"]
    out["ph_caveat"] = PH_CAVEAT
    return _clean(out)


def structure_log_risk_fn(model_state) -> Callable:
    """g(x) from an annotated model state (StructureNetwork, trained activations)."""
    from explaneat.core.structure_network import StructureNetwork

    sn = StructureNetwork(model_state)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")

    def g(X):
        out = sn.forward(torch.as_tensor(np.asarray(X), dtype=torch.float64)).detach().numpy()
        return out.reshape(len(out), -1)[:, 0]

    return g
