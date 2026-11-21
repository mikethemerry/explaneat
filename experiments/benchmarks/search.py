"""Hyperparameter search for the comparator suite: CV on the training rows only, in stages.

A model's search is a list of stages. Each stage is a grid; the parameters a stage does not tune are
the estimator's own settings (`base`) or the choices of the stages before it (a coarse-then-fine,
one-pass coordinate search, used only where the full product grid is too large to fit). Inside a
stage the search is exhaustive over the product grid on the same folds, as GridSearchCV:

- an ordinary stage is sklearn's GridSearchCV (refit off; mean fold score; ties go to the first
  combination in ParameterGrid order);
- a *staged* stage tunes the number of boosting iterations without refitting for each value: for
  every combination of the other parameters one model per fold is fitted with the largest iteration
  count and scored after each listed iteration count (staged predictions). With no early stopping
  and no subsampling, a boosting model after k iterations is the model fitted with k iterations, so
  this is the same grid search, only cheaper (checked in `check_staged` and in the README).

The final model is refitted on all training rows with the chosen values. Nothing here sees test rows.
"""
from __future__ import annotations

import itertools

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid


def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v.item() if hasattr(v, "item") else v


def _staged_fold(est, params, stage_param, values, X, y, tr, va, staged_scores):
    m = clone(est).set_params(**params, **{stage_param: max(values)})
    m.fit(X[tr], y[tr])
    return staged_scores(m, X[va], y[va], values)


def _key(p):
    return repr(sorted(jsonable(p).items()))


def run_stage(est, stage, fixed, X, y, folds, n_jobs, scoring=None, staged_scores=None, known=None):
    """One stage: returns (chosen params of this stage, best mean CV score, table of every combination).
    `known`: CV scores of combinations already evaluated on the same folds with the same fixed values (a
    stored stage whose grid was then widened); only the other combinations are fitted. The choice is the
    first best combination in ParameterGrid order, as GridSearchCV's."""
    grid, sp = stage["grid"], stage.get("staged")
    est = clone(est).set_params(**fixed)
    if sp is None and known:
        allp = list(ParameterGrid(grid))
        todo = [p for p in allp if _key(p) not in known]
        new = {}
        if todo:
            g = GridSearchCV(est, [{k: [v] for k, v in p.items()} for p in todo], scoring=scoring, cv=folds,
                             n_jobs=n_jobs, refit=False, error_score=np.nan).fit(X, y)
            new = {_key(p): float(s) for p, s in zip(g.cv_results_["params"], g.cv_results_["mean_test_score"])}
        table, best, best_p = [], -np.inf, None
        for p in allp:
            sc = known.get(_key(p), new.get(_key(p)))
            table.append({"params": jsonable(p), "score": sc})
            if sc > best:
                best, best_p = sc, p
        return dict(best_p), float(best), table
    if sp is None:
        g = GridSearchCV(est, grid, scoring=scoring, cv=folds, n_jobs=n_jobs, refit=False,
                         error_score=np.nan).fit(X, y)
        table = [{"params": jsonable(p), "score": float(s)} for p, s in
                 zip(g.cv_results_["params"], g.cv_results_["mean_test_score"])]
        return dict(g.best_params_), float(g.best_score_), table
    values = list(grid[sp])
    others = {k: v for k, v in grid.items() if k != sp}
    combos = list(ParameterGrid(others)) if others else [{}]
    jobs = list(itertools.product(range(len(combos)), range(len(folds))))
    res = Parallel(n_jobs=n_jobs)(delayed(_staged_fold)(est, combos[c], sp, values, X, y, folds[f][0], folds[f][1],
                                                        staged_scores) for c, f in jobs)
    scores = np.zeros((len(combos), len(folds), len(values)))
    for (c, f), s in zip(jobs, res):
        scores[c, f] = s
    mean = scores.mean(axis=1)                       # combos x values
    table, best, best_p = [], -np.inf, None
    for c, p in enumerate(combos):                   # first maximum in (combination, iteration count) order
        for j, v in enumerate(values):
            q = {**p, sp: v}
            table.append({"params": jsonable(q), "score": float(mean[c, j])})
            if mean[c, j] > best:
                best, best_p = mean[c, j], q
    return best_p, float(best), table


def search(est, stages, X, y, folds, n_jobs, scoring=None, staged_scores=None, previous=None):
    """All stages in order, then the refit on all training rows. Returns (fitted model, params, cv score, record).
    `previous`: a stored record of the same model on the same data; a stage whose grid, staged parameter and
    fixed values equal the stored stage's is taken from it instead of refitted (every fit is seeded, so the
    stored stage is what a refit would give)."""
    chosen, record, score = {}, [], None
    for i, st in enumerate(stages):
        fixed = {**st.get("base", {}), **chosen}
        old = previous[i] if previous and i < len(previous) else None
        if old and old["grid"] == jsonable(st["grid"]) and old["staged"] == st.get("staged") \
                and old["fixed"] == jsonable(fixed):
            p = {k: (tuple(v) if k == "hidden_layer_sizes" else v) for k, v in old["chosen"].items()}
            score, table = old["cv_score"], old["table"]
        else:
            known = ({_key(r["params"]): r["score"] for r in old["table"]}
                     if old and old["staged"] is None and st.get("staged") is None and old["fixed"] == jsonable(fixed)
                     else None)
            p, score, table = run_stage(est, st, fixed, X, y, folds, n_jobs, scoring, staged_scores, known)
        record.append({"grid": jsonable(st["grid"]), "staged": st.get("staged"), "fixed": jsonable(fixed),
                       "chosen": jsonable(p), "cv_score": score, "table": table})
        chosen.update(st.get("base", {}))
        chosen.update(p)
    model = clone(est).set_params(**chosen).fit(X, y)
    return model, chosen, score, record


# ------------------------------------------------------------------ staged scorers

def staged_auc(m, X, y, values):
    """ROC AUC after each listed iteration count (decision function, as the roc_auc scorer uses)."""
    from sklearn.metrics import roc_auc_score
    want, out = set(values), {}
    for k, d in enumerate(m.staged_decision_function(X), start=1):
        if k in want:
            out[k] = roc_auc_score(y, d)
    return [out[v] for v in values]                  # every count is reached: no early stopping


def staged_cindex(m, X, y, values):
    """Harrell's C (the estimator's own score) after each listed number of stages."""
    from sksurv.metrics import concordance_index_censored
    want, out = set(values), {}
    for k, r in enumerate(m.staged_predict(X), start=1):
        if k in want:
            out[k] = concordance_index_censored(y["event"], y["time"], r)[0]
    return [out[v] for v in values]


# ------------------------------------------------------------------ boundary check

def edge_hits(record, domain_ends):
    """Every tuned value that sits at either end of its stage's grid, unless it is the end of the
    parameter's range (`domain_ends[param]`, e.g. min_samples_leaf 1 or max_depth None), where the grid
    cannot be widened. A parameter tuned in several stages is judged in the last one (its final value).
    MLP architectures: the widest option of the chosen depth, or the narrowest one-layer
    option (unless one unit), counts as an end. Returns a list of (param, value, grid)."""
    hits = []
    last = {k: i for i, st in enumerate(record) for k in st["grid"]}   # a later stage that re-tunes k decides it
    for i, st in enumerate(record):
        for k, vals in st["grid"].items():
            if len(vals) < 2 or last[k] != i:
                continue
            v = st["chosen"][k]
            if k == "hidden_layer_sizes":            # widths are ordered within each depth; depth is a choice
                same = [w for w in vals if len(w) == len(v)]
                edge = v == same[-1] or (v == vals[0] and v[0] > 1)
            else:
                edge = (v == vals[0] or v == vals[-1]) and v not in domain_ends.get(k, [])
            if edge and not interior_tie(st, k, vals):
                hits.append((k, v, vals))
    return hits


def interior_tie(stage, k, vals):
    """True when some combination of the stage's grid with an interior value of `k` has exactly the
    stage's best CV score (a flat optimum, e.g. CV AUC 1.0 on MONK-2): the tie-break (the first best
    combination in grid order), not the grid, put the choice at the edge."""
    return any(row["params"][k] not in (vals[0], vals[-1]) and row["score"] >= stage["cv_score"]
               for row in stage["table"])
