"""Every input combination of a small discrete input space (e.g. MONK-2's 432), predicted.

All variables must be categories, or numbers with few distinct training values (each value is
then a level). Each combination is predicted; ``rule`` (a pandas expression over the combination
table, e.g. ``"a1_1 + a2_1 + a3_1 == 2"``, using the 0/1 indicator columns ``<var>_<code>``;
note that ``(a1 == '1') + (a2 == '1')`` adds booleans, which pandas treats as "or") is compared with the prediction
("yes" when the predicted risk is at least ``cut``), with the counterexamples listed; ``group_by``
(an expression) gives the share predicted yes per value; ``where`` restricts the combinations.
The table has a column per variable (codes as text), ``<var>_label``, ``risk``, ``yes`` and
``n_train`` (how often the combination occurs in the training rows).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import numpy as np

MAX_DISTINCT = 12
DEFAULT_LIMIT = 200_000


def levels(v, bins: int = 0) -> List[Any]:
    """The values a variable takes in the enumeration: every category, every distinct value of
    a number with at most ``MAX_DISTINCT`` of them, else (``bins`` > 0) ``bins`` training
    quantiles from the 5th to the 95th percentile (a continuous input is then binned, and the
    cells' ``n_train`` counts no longer apply to it); else an error."""
    if v.is_category():
        return list(v.all_codes)
    vals = np.unique(v.raw_train)
    if len(vals) > MAX_DISTINCT:
        if bins and bins > 0:
            qs = np.percentile(v.raw_train, np.linspace(5, 95, int(bins)))
            return [float(x) for x in np.unique(np.round(qs, v.decimals))]
        raise ValueError(f"{v.base} has {len(vals)} distinct values: not a small discrete input space "
                         f"(enumerate handles categories and numbers with at most {MAX_DISTINCT} values; "
                         f"pass bins=N to bin continuous inputs at N quantiles)")
    return [float(x) for x in vals]


def enumerate_space(vm, rule: Optional[str] = None, group_by: Optional[str] = None,
                    where: Optional[str] = None, cut: float = 0.5, limit: int = DEFAULT_LIMIT,
                    max_examples: int = 20, parts: bool = False, bins: int = 0) -> Dict[str, Any]:
    import pandas as pd
    from .probe import evaluate, hold_row, set_raw
    m = vm.m
    var_levels = [(v, levels(v, bins)) for v in m.variables]
    binned = [v.base for v, _ in var_levels if not v.is_category() and len(np.unique(v.raw_train)) > MAX_DISTINCT]
    size = int(np.prod([len(l) for _, l in var_levels]))
    if size > limit:
        raise ValueError(f"{size} combinations (more than --limit {limit})")
    base = hold_row(vm, "typical")
    combos = list(itertools.product(*[l for _, l in var_levels]))
    raws = np.repeat(base[None, :], len(combos), axis=0)
    for i, combo in enumerate(combos):
        for (v, _), val in zip(var_levels, combo):
            set_raw(raws[i], v, val)
    Z = (raws - m.mean) / m.scale
    logit = m.logit(Z)
    risk = m.risk(logit)
    cols: Dict[str, Any] = {}
    for j, (v, _) in enumerate(var_levels):
        if v.is_category():
            cols[v.base] = [str(c[j]) for c in combos]
            cols[v.base + "_label"] = [v.level(c[j])["label"] for c in combos]
            for code in v.all_codes:     # 0/1 indicators, so that counts add up: a1_1 + a2_1 == 2
                cols[f"{v.base}_{code}"] = [int(str(c[j]) == str(code)) for c in combos]
        else:
            cols[v.base] = [float(c[j]) for c in combos]
    df = pd.DataFrame(cols)
    df["risk"] = risk
    df["yes"] = risk >= cut
    # how often each combination occurs in the training rows
    keys_train = []
    for v, _ in var_levels:
        keys_train.append(np.array([str(x) for x in v.raw_train]) if v.is_category()
                          else np.array([float(x) for x in v.raw_train]))
    train_keys = {}
    for t in zip(*keys_train):
        train_keys[t] = train_keys.get(t, 0) + 1
    df["n_train"] = [train_keys.get(tuple(str(x) if v.is_category() else float(x) for (v, _), x in zip(var_levels, c)), 0)
                     for c in combos]
    if parts:
        rows = evaluate(vm, raws, parts=True)
        for p in vm.parts:
            if p.is_root:
                continue
            df[f"acts_{p.id}"] = [r["parts"][p.id]["acts"] for r in rows]
    if where:
        df = df[np.asarray(df.eval(where, engine="python"), dtype=bool)]
    res: Dict[str, Any] = {"genome_id": vm.genome_id, "n_combinations": int(len(df)), "space_size": size,
                           "cut": cut, "share_yes": float(df["yes"].mean()) if len(df) else None,
                           "n_in_training": int((df["n_train"] > 0).sum()), "where": where}
    if binned:
        lv = {v.base: l for v, l in var_levels}
        res["binned"] = {b: [float(x) for x in lv[b]] for b in binned}
        res["binned_note"] = (f"{', '.join(binned)} binned at {bins} training quantiles (5th-95th percentile): "
                              "cells are constructed cases, so n_train is 0 for them")
    if parts:
        res["parts_act_on_share"] = {c[5:]: float(df[c].mean()) for c in df.columns if c.startswith("acts_")}
    if rule:
        want = np.asarray(df.eval(rule, engine="python"), dtype=bool)
        got = df["yes"].to_numpy()
        bad = df[want != got]
        ind = {f"{v.base}_{code}" for v, _ in var_levels if v.is_category() for code in v.all_codes}
        show = [c for c in df.columns if not c.endswith("_label") and not c.startswith("acts_") and c not in ind]
        res["rule"] = {"expr": rule, "agree": int((want == got).sum()), "n": int(len(df)),
                       "agreement": float((want == got).mean()) if len(df) else None,
                       "rule_yes_model_no": int(np.sum(want & ~got)), "rule_no_model_yes": int(np.sum(~want & got)),
                       "counterexamples_in_training": int((bad["n_train"] > 0).sum()),
                       "counterexamples": bad[show].sort_values("n_train", ascending=False)
                       .head(max_examples).to_dict(orient="records")}
    if group_by:
        key = df.eval(group_by, engine="python")
        g = df.assign(_key=np.asarray(key)).groupby("_key")
        res["groups"] = [{"value": (k.item() if hasattr(k, "item") else k), "n": int(len(x)),
                          "share_yes": float(x["yes"].mean()), "mean_risk": float(x["risk"].mean()),
                          "n_train": int(x["n_train"].sum())} for k, x in g]
    ind = [f"{v.base}_{code}" for v, _ in var_levels if v.is_category() for code in v.all_codes]
    res["table"] = df.drop(columns=[c for c in df.columns if c.endswith("_label") or c in ind]).to_dict(orient="records") \
        if len(df) <= 5000 else None
    return res
