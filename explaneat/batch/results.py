"""Results table across seeds, with a linear comparator on the same split.

Per experiment (seed): the selected genome (``config_json["batch"]["selected"]``,
the validation-selected champion; else the best training-fitness genome) is
evaluated on the test fold from its phenotype (``StructureNetwork`` with the
trained output activation).

* survival: Harrell's C with a bootstrap 95% CI, Uno's C, mean td-AUC, IBS,
  test Cox loss, calibration E/O at the calibration horizon
  (``explaneat.survival.evidence.survival_performance``);
* classification: test AUC with a bootstrap 95% CI, Brier, log loss.

Comparator, fitted by the portal on the same split's train fold (tuned on the
validation fold when there is one): ridge Cox (Efron; alpha by validation Cox
loss, as ``experiments/survival/run_seed.py``) or L2 logistic regression (C by
validation AUC). Paired bootstrap CIs of NEAT − comparator use the same
resamples. The summary reports the across-seed mean with a t-interval.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .common import json_safe, load_split_arrays

N_BOOT = 500
COX_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LR_CS = (0.01, 0.1, 1.0, 10.0, 100.0)

_cache: Dict[tuple, Dict[str, Any]] = {}
_comp_cache: Dict[str, Dict[str, Any]] = {}  # split id -> fitted comparator (it depends on the split only)


def _t975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
             9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}
    if df > 29:
        return 1.96
    return table.get(df, table[min(table, key=lambda k: abs(k - df))])


def mean_ci(values) -> Dict[str, Optional[float]]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    n = len(v)
    if not n:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "sd": None}
    m = sum(v) / n
    if n == 1:
        return {"n": 1, "mean": m, "lo": None, "hi": None, "sd": None}
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    h = _t975(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "lo": m - h, "hi": m + h, "sd": sd}


def _boot_idx(n: int, seed: int = 42) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, n, size=(N_BOOT, n))


def _ci(vals) -> List[Optional[float]]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) < 10:
        return [None, None]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


# ----------------------------------------------------------------------------- metrics


def _harrell(g, y):
    from ..survival.metrics import harrell_c

    if y[:, 1].sum() == 0:
        return float("nan")
    return harrell_c(g, y[:, 0], y[:, 1])


def _auc(p, y):
    from sklearn.metrics import roc_auc_score

    y = y.ravel()
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def _primary(task):
    return (_harrell, "harrell_c") if task == "survival" else (_auc, "auc")


def _boot_metric(task, score, y, idx):
    f, _ = _primary(task)
    return [f(score[ix], y[ix]) for ix in idx]


def _survival_metrics(g_fn, data):
    from ..survival.evidence import survival_performance

    return survival_performance(g_fn, data["X_train"], data["y_train"], data["X_test"], data["y_test"],
                                meta=data["meta"])


def _classification_metrics(p_fn, data):
    from sklearn.metrics import brier_score_loss, log_loss

    p = np.clip(np.asarray(p_fn(data["X_test"])).ravel(), 1e-12, 1 - 1e-12)
    y = data["y_test"].ravel()
    return {"auc": _auc(p, y), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5))), "n_test": int(len(y)), "positives_test": int(y.sum())}


def _detail(task, m):
    """Per-horizon metrics and the calibration table (aggregates only, no rows)."""
    if task != "survival":
        return {}
    return json_safe({"horizons": m.get("horizons"), "td_auc": m.get("td_auc"), "brier": m.get("brier"),
                      "calibration": m.get("calibration"), "null_cox_loss": m.get("null_cox_loss"),
                      "censored_fraction": m.get("censored_fraction"), "time_unit": m.get("time_unit")})


def _summarise(task, m):
    if task == "survival":
        cal = m.get("calibration") or {}
        return {"harrell_c": m.get("harrell_c"), "uno_c": m.get("uno_c"),
                "td_auc_mean": (m.get("td_auc") or {}).get("mean_auc"), "ibs": m.get("ibs"),
                "cox_loss": m.get("cox_loss"), "calibration_eo": cal.get("expected_over_observed"),
                "calibration_horizon": cal.get("horizon"), "n_test": m.get("n_test"),
                "events_test": m.get("events_test")}
    return m


# ----------------------------------------------------------------------------- models


def _genome_model(session, experiment):
    """(predict_fn on scaled X, genome summary, selection) for the selected genome."""
    import torch

    from ..core.config_utils import load_neat_config
    from ..core.explaneat import ExplaNEAT
    from ..core.structure_network import StructureNetwork
    from ..db.models import Genome, Population

    sel = ((experiment.config_json or {}).get("batch") or {}).get("selected") or {}
    g_row = None
    if sel.get("genome_id"):
        g_row = session.get(Genome, uuid.UUID(sel["genome_id"]))
    if g_row is None:
        g_row = (session.query(Genome).join(Population, Population.id == Genome.population_id)
                 .filter(Population.experiment_id == experiment.id, Genome.fitness.isnot(None))
                 .order_by(Genome.fitness.desc()).first())
        sel = {"rule": "best training fitness", "genome_id": str(g_row.id) if g_row else None}
    if g_row is None:
        raise ValueError("experiment has no evaluated genomes")
    config = load_neat_config(experiment.neat_config_text or "", experiment.config_json)
    genome = g_row.to_neat_genome(config)
    ph = ExplaNEAT(genome, config).get_phenotype_network()
    sn = StructureNetwork(ph)
    sn.override_output_activation(sn.model_output_activation)
    sn.override_hidden_activation("relu")

    def predict(X):
        return sn.forward(torch.as_tensor(np.asarray(X), dtype=torch.float64)).detach().numpy().reshape(len(X), -1)[:, 0]

    out_keys = set(config.genome_config.output_keys)
    hidden = [k for k in genome.nodes if k not in out_keys]
    enabled = [c for c in genome.connections.values() if c.enabled]
    used = sorted({c.from_node for c in ph.connections if c.enabled and str(c.from_node).startswith("-")})
    struct = {"hidden_nodes": len(hidden), "enabled_connections": len(enabled),
              "complexity_k": len(hidden) + len(enabled), "inputs_used": len(used),
              "phenotype_nodes": len(ph.nodes),
              "phenotype_connections": sum(1 for c in ph.connections if c.enabled)}
    sel = {**sel, "genome_id": str(g_row.id), "train_fitness": g_row.fitness}
    return predict, struct, sel


def _comparator(task, data):
    """Fitted linear comparator: (predict_fn, info)."""
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    if task == "survival":
        import torch
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        from ..survival.cox import CoxRiskSets, cox_ph_loss

        s_tr = Surv.from_arrays(event=ytr[:, 1].astype(bool), time=ytr[:, 0])
        scores = {}
        if Xva is not None:
            rs = CoxRiskSets(yva[:, 0], yva[:, 1])
            for a in COX_ALPHAS:
                try:
                    m = CoxPHSurvivalAnalysis(alpha=a, ties="efron").fit(Xtr, s_tr)
                    scores[a] = float(cox_ph_loss(torch.as_tensor(m.predict(Xva)), rs, ties="efron"))
                except Exception:
                    continue
        alpha = min(scores, key=scores.get) if scores else 1e-2
        m = CoxPHSurvivalAnalysis(alpha=alpha, ties="efron").fit(Xtr, s_tr)
        return m.predict, {"model": "ridge Cox (Efron)", "alpha": alpha, "n_params": int(Xtr.shape[1]),
                           "tuned_on": "validation Cox loss" if scores else "fixed alpha (no validation fold)"}
    from sklearn.linear_model import LogisticRegression

    scores = {}
    if Xva is not None:
        for c in LR_CS:
            m = LogisticRegression(C=c, max_iter=2000).fit(Xtr, ytr.ravel())
            scores[c] = _auc(m.predict_proba(Xva)[:, 1], yva)
    c = max(scores, key=scores.get) if scores else 1.0
    m = LogisticRegression(C=c, max_iter=2000).fit(Xtr, ytr.ravel())
    return (lambda X: m.predict_proba(X)[:, 1]), {
        "model": "L2 logistic regression", "C": c, "n_params": int(Xtr.shape[1]) + 1,
        "tuned_on": "validation AUC" if scores else "fixed C (no validation fold)"}


# ----------------------------------------------------------------------------- table


def batch_results(experiment_ids: List[str], refresh: bool = False) -> Dict[str, Any]:
    from ..db import db
    from ..db.models import Experiment

    rows, comparators, errors = [], {}, []
    task = None
    with db.session_scope() as s:
        for eid in experiment_ids:
            exp = s.get(Experiment, uuid.UUID(str(eid)))
            if exp is None or exp.split_id is None:
                errors.append({"experiment_id": eid, "error": "experiment not found or has no split"})
                continue
            key = (str(exp.id), exp.updated_at.isoformat() if exp.updated_at else "")
            if key in _cache and not refresh:
                rows.append(_cache[key])
                task = task or _cache[key]["task"]
                sk = _cache[key]["split_id"]
                if sk not in comparators and sk in _comp_cache:
                    comparators[sk] = _comp_cache[sk]
                continue
            try:
                data = load_split_arrays(s, exp.split_id)
                t = "survival" if data["meta"].get("task_type") == "survival" else "classification"
                task = task or t
                predict, struct, sel = _genome_model(s, exp)
                m = _survival_metrics(predict, data) if t == "survival" else _classification_metrics(predict, data)
                split_key = str(exp.split_id)
                if split_key not in comparators and split_key in _comp_cache and not refresh:
                    comparators[split_key] = _comp_cache[split_key]
                if split_key not in comparators:
                    cpred, cinfo = _comparator(t, data)
                    cm = _survival_metrics(cpred, data) if t == "survival" else _classification_metrics(cpred, data)
                    comparators[split_key] = {"predict": cpred, "info": cinfo, "metrics": _summarise(t, cm),
                                              "detail": _detail(t, cm)}
                    _comp_cache[split_key] = comparators[split_key]
                comp = comparators[split_key]
                idx = _boot_idx(len(data["y_test"]))
                s_neat = np.asarray(predict(data["X_test"])).ravel()
                s_comp = np.asarray(comp["predict"](data["X_test"])).ravel()
                b_neat = _boot_metric(t, s_neat, data["y_test"], idx)
                b_comp = _boot_metric(t, s_comp, data["y_test"], idx)
                _, pname = _primary(t)
                summ = _summarise(t, m)
                events = int(data["y_train"][:, 1].sum()) if t == "survival" else int(data["y_train"].sum())
                row = {
                    "experiment_id": str(exp.id), "experiment_name": exp.name, "status": exp.status,
                    "seed": exp.random_seed, "task": t, "split_id": split_key,
                    "selected": sel, "structure": {**struct, "events_per_parameter":
                                                   round(events / struct["complexity_k"], 2) if struct["complexity_k"] else None},
                    "metrics": summ, "primary": pname,
                    "primary_ci": _ci(b_neat),
                    "comparator_primary": comp["metrics"].get(pname),
                    "comparator_primary_ci": _ci(b_comp),
                    "delta_primary": (summ.get(pname) - comp["metrics"].get(pname))
                    if summ.get(pname) is not None and comp["metrics"].get(pname) is not None else None,
                    "delta_primary_ci": _ci([a - b for a, b in zip(b_neat, b_comp)
                                             if a is not None and b is not None]),
                }
                row["detail"] = _detail(t, m)
                row = json_safe(row)
                _cache[key] = row
                rows.append(row)
            except Exception as e:  # one bad run must not hide the table
                errors.append({"experiment_id": str(eid), "error": f"{type(e).__name__}: {e}"})
    comp_out = {k: json_safe({"info": v["info"], "metrics": v["metrics"], "detail": v.get("detail")})
                for k, v in comparators.items()}
    # cached rows may have a comparator computed earlier
    pname = "harrell_c" if task == "survival" else "auc"
    summary = {}
    if rows:
        keys = list(rows[0]["metrics"].keys())
        summary = {k: mean_ci([r["metrics"].get(k) for r in rows]) for k in keys
                   if isinstance(rows[0]["metrics"].get(k), (int, float))}
        summary["delta_primary"] = mean_ci([r.get("delta_primary") for r in rows])
        summary["comparator_primary"] = mean_ci([r.get("comparator_primary") for r in rows])
        for k in ("hidden_nodes", "enabled_connections", "complexity_k", "inputs_used", "events_per_parameter"):
            summary[f"structure.{k}"] = mean_ci([r["structure"].get(k) for r in rows])
    return json_safe({"task": task, "primary": pname, "rows": rows, "comparators": comp_out,
                      "summary": summary, "errors": errors, "n_boot": N_BOOT})


def results_csv(res: Dict[str, Any]) -> str:
    import csv
    import io

    buf = io.StringIO()
    rows = res.get("rows", [])
    if not rows:
        return ""
    mkeys = list(rows[0]["metrics"].keys())
    skeys = list(rows[0]["structure"].keys())
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["experiment_name", "experiment_id", "seed", "selected_genome_id"] + mkeys
               + ["primary_ci_lo", "primary_ci_hi", "comparator_primary", "delta_primary",
                  "delta_ci_lo", "delta_ci_hi"] + [f"structure.{k}" for k in skeys])
    for r in rows:
        w.writerow([r["experiment_name"], r["experiment_id"], r.get("seed"), (r.get("selected") or {}).get("genome_id")]
                   + [r["metrics"].get(k) for k in mkeys]
                   + list(r.get("primary_ci") or [None, None]) + [r.get("comparator_primary"), r.get("delta_primary")]
                   + list(r.get("delta_primary_ci") or [None, None]) + [r["structure"].get(k) for k in skeys])
    return buf.getvalue()


# ----------------------------------------------------------------------------- report


def _f(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return str(int(x))
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{nd}f}" if math.isfinite(x) else "-"


def _ci_text(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" [{_f(ci[0])}, {_f(ci[1])}]"


def results_report_md(res: Dict[str, Any], batch: Optional[Dict[str, Any]] = None,
                      dataset: Optional[Dict[str, Any]] = None,
                      governance: Optional[Dict[str, Any]] = None) -> str:
    """A one-file Markdown report of a batch: protocol, per-seed test metrics against
    the comparator on the same split, per-horizon metrics, calibration tables and
    the across-seed summary. Aggregates and model settings only, never rows."""
    import time as _time

    rows = res.get("rows") or []
    task = res.get("task") or (batch or {}).get("task")
    pname = res.get("primary") or ("harrell_c" if task == "survival" else "auc")
    out: List[str] = []
    w = out.append
    w(f"# ExplaNEAT batch report: {(batch or {}).get('name_prefix', 'batch')}")
    w("")
    w(f"Generated {_time.strftime('%Y-%m-%d %H:%M:%S %z')} by `GET /api/portal/batches/"
      f"{(batch or {}).get('batch_id', '?')}/report.md`. Aggregate metrics and model settings only: "
      "no row-level data. Release under the data agreement's output rules.")
    w("")
    w("## Setup")
    w("")
    w("| | |")
    w("|---|---|")
    if batch:
        w(f"| Batch | `{batch.get('batch_id')}` |")
        w(f"| Template | `{batch.get('template')}` |")
        w(f"| Seeds | {', '.join(str(x) for x in batch.get('seeds', []))} |")
        w(f"| Generations / population / epochs | {batch.get('generations')} / {batch.get('pop_size')} / "
          f"{batch.get('epochs')} |")
        w(f"| Complexity penalty (lambda) | {batch.get('complexity_penalty')} |")
        if batch.get("config_overrides"):
            w(f"| Config overrides | `{batch.get('config_overrides')}` |")
    if dataset:
        w(f"| Dataset | {dataset.get('name')} (source file `{dataset.get('source_file') or '-'}`, "
          f"restricted: {dataset.get('restricted')}) |")
        w(f"| Rows / features / events | {dataset.get('n_rows')} / {dataset.get('n_features')} / "
          f"{dataset.get('n_events', '-')} |")
        if dataset.get("time_unit"):
            w(f"| Time unit | {dataset.get('time_unit')} |")
        if dataset.get("split"):
            sp = dataset["split"]
            w(f"| Split | `{sp.get('id')}`: train {sp.get('train')}, validation {sp.get('val')}, "
              f"test {sp.get('test')} (split seed {sp.get('seed')}) |")
    if governance:
        w(f"| Restricted server mode | {governance.get('restricted_server')} |")
        w(f"| Assisted (MCP) path for restricted data | {governance.get('assisted_restricted_allowed')} |")
    w("")
    comps = res.get("comparators") or {}
    for sid, c in comps.items():
        info = c.get("info") or {}
        tuned = ", ".join(f"{k} = {v}" for k, v in info.items() if k in ("alpha", "C"))
        w(f"Comparator on split `{sid}`: {info.get('model')} ({tuned}; tuned on {info.get('tuned_on')}; "
          f"{info.get('n_params')} parameters), fitted on the same training fold and evaluated on the same "
          "test fold.")
        w("")
    if not rows:
        w("No completed runs.")
        return "\n".join(out) + "\n"
    mkeys = [k for k in rows[0]["metrics"] if isinstance(rows[0]["metrics"].get(k), (int, float))
             and k not in ("n_test", "events_test", "calibration_horizon")]
    w("## Test-fold metrics per seed")
    w("")
    w("| Seed | " + " | ".join(mkeys) + f" | {pname} 95% CI | comparator {pname} | delta {pname} [95% CI] | "
      "hidden | connections | k | events/param |")
    w("|" + "---|" * (len(mkeys) + 8))
    for r in rows:
        st = r.get("structure") or {}
        w(f"| {r.get('seed')} | " + " | ".join(_f(r["metrics"].get(k)) for k in mkeys)
          + f" | {_ci_text(r.get('primary_ci')).strip() or '-'} | {_f(r.get('comparator_primary'))} | "
          f"{_f(r.get('delta_primary'))}{_ci_text(r.get('delta_primary_ci'))} | {st.get('hidden_nodes')} | "
          f"{st.get('enabled_connections')} | {st.get('complexity_k')} | {_f(st.get('events_per_parameter'), 1)} |")
    for sid, c in comps.items():
        cm = c.get("metrics") or {}
        w(f"| comparator | " + " | ".join(_f(cm.get(k)) for k in mkeys) + " | | | | | | | |")
    w("")
    summ = res.get("summary") or {}
    if summ:
        w("## Across seeds (mean, 95% t-interval)")
        w("")
        w("| Quantity | n | mean | 95% interval | sd |")
        w("|---|---|---|---|---|")
        for k, v in summ.items():
            if k in ("calibration_horizon", "n_test", "events_test"):
                continue  # constants of the split, already in Setup
            if isinstance(v, dict) and v.get("n"):
                iv = f"[{_f(v.get('lo'))}, {_f(v.get('hi'))}]" if v.get("lo") is not None else "-"
                w(f"| {k} | {v['n']} | {_f(v.get('mean'))} | {iv} | {_f(v.get('sd'))} |")
        w("")
    if task == "survival":
        w("## Per-horizon metrics")
        w("")
        w("Time-dependent (cumulative/dynamic) AUC and IPCW Brier score on the test fold.")
        w("")
        w("| Model | horizon | td-AUC | Brier |")
        w("|---|---|---|---|")

        def _hz(label, d):
            td = (d or {}).get("td_auc") or {}
            br = (d or {}).get("brier") or {}
            bt = {round(float(t), 6): b for t, b in zip(br.get("times") or [], br.get("brier") or [])}
            for t, a in zip(td.get("times") or [], td.get("auc") or []):
                w(f"| {label} | {_f(t, 2)} | {_f(a)} | {_f(bt.get(round(float(t), 6)))} |")

        for r in rows:
            _hz(f"seed {r.get('seed')}", r.get("detail"))
        for sid, c in comps.items():
            _hz("comparator", c.get("detail"))
        w("")
        w("## Calibration at the horizon")
        w("")
        w("Predicted 1 - S(h | x) against the Kaplan-Meier observed risk, by decile of predicted risk.")
        w("")

        def _cal(label, d):
            cal = (d or {}).get("calibration") or {}
            if not cal:
                return
            w(f"**{label}**: horizon {_f(cal.get('horizon'), 2)}, mean predicted {_f(cal.get('mean_predicted'))}, "
              f"observed {_f(cal.get('observed_km'))}, E/O {_f(cal.get('expected_over_observed'))}, "
              f"max decile difference {_f(cal.get('max_abs_group_diff'))}.")
            w("")
            w("| decile | n | events by h | mean predicted | observed (KM) |")
            w("|---|---|---|---|---|")
            for g in cal.get("groups") or []:
                w(f"| {g.get('group')} | {g.get('n')} | {g.get('events_by_horizon')} | "
                  f"{_f(g.get('mean_predicted'))} | {_f(g.get('observed_km'))} |")
            w("")

        for r in rows:
            _cal(f"Seed {r.get('seed')}", r.get("detail"))
        for sid, c in comps.items():
            _cal("Comparator", c.get("detail"))
        w("## Caveats")
        w("")
        w("- Proportional hazards is inherited from the Cox loss, not tested by it: check it on the "
          "comparator (Schoenfeld residuals) before interpreting hazard ratios.")
        w("- Concordance is computed on the log-risk g(x) for both models.")
        w("- One split: the intervals are bootstrap intervals over the test rows of this split "
          "(and t-intervals over seeds), not over repeated splits.")
    if res.get("errors"):
        w("")
        w("## Errors")
        w("")
        for e in res["errors"]:
            w(f"- `{e.get('experiment_id')}`: {e.get('error')}")
    return "\n".join(out) + "\n"
