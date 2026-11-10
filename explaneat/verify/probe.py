"""L2/L3 tools: the model on constructed inputs, with a part switched off, and in a subgroup.

Inputs are in **real units** (the dataset's raw values: years, mm Hg, a category's code or its
plain label), converted with the split's scaler exactly as training did. Every function works on
a :class:`~explaneat.verify.model.VerifyModel`, i.e. the genome from the database with the
explanation's operations (possibly a copied list) replayed in memory.

* :func:`probe`: predicted risk, model score and each part's outputs and contribution for one
  or more constructed cases (``--set``), a 1-D sweep or a 2-D grid (``--sweep`` once or twice).
* :func:`ablate`: the change in prediction when a part is switched off (its exits fixed at the
  baseline the plain-language layer uses: 0 for a gated ReLU part, else the training mean), for
  the typical case and over the data.
* :func:`subgroup`: n, outcome rate and each part's activity for the rows matching an
  expression on real-unit variables.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ON_TOL = 1e-9


# ============================================================================ variables and values
def variable(vm, name: str):
    """A variable by its base name, label or short label (case-insensitive)."""
    key = str(name).strip().lower()
    for v in vm.m.variables:
        if key in {v.base.lower(), str(v.label).lower(), str(v.short).lower()}:
            return v
    raise KeyError(f"no variable {name!r}; variables: " + ", ".join(v.base for v in vm.m.variables))


def category_code(v, value: str) -> str:
    """A category's code from its code or its plain label (case-insensitive)."""
    s = str(value).strip()
    codes = [c for c in v.all_codes]
    if s in codes:
        return s
    low = s.lower()
    for c in codes:
        li = v.level(c)
        if low in {str(li.get("label", "")).lower(), str(li.get("group", "")).lower()}:
            return c
    for c in codes:      # unique prefix / substring of the label
        if low and low in str(v.level(c).get("label", "")).lower():
            return c
    if low in ("yes", "true") and "1" in codes:
        return "1"
    if low in ("no", "false") and "0" in codes:
        return "0"
    raise ValueError(f"{v.base}: no category {value!r}; codes: " +
                     "; ".join(f"{c} = {v.level(c)['label']}" for c in codes))


def set_raw(raw: np.ndarray, v, value) -> None:
    """Put ``value`` (real units, or a category code/label) into a raw input row in place."""
    if v.is_category():
        code = category_code(v, value)
        for c, cd in zip(v.columns, v.codes):
            raw[c] = 1.0 if cd == code else 0.0
    else:
        raw[v.columns[0]] = float(value)


def raw_value(v, raw: np.ndarray):
    if v.is_category():
        code = next((cd for c, cd in zip(v.columns, v.codes) if raw[c] > 0.5), v.implicit_off)
        return code
    return float(raw[v.columns[0]])


def describe_value(v, raw: np.ndarray) -> str:
    val = raw_value(v, raw)
    if v.is_category():
        return f"{v.level(val)['label']} ({v.base}={val})" if val is not None else "none"
    return v.fmt(val)


def hold_row(vm, hold: str = "typical") -> np.ndarray:
    """The case other inputs are held at: ``typical`` (median number, most common category),
    ``row:N`` (training row N) or ``where:<expr>`` (the typical case of a subgroup)."""
    from ..analysis.plain_language import typical_raw
    h = (hold or "typical").strip()
    if h in ("typical", "median", "reference"):
        return typical_raw(vm.m).copy()
    if h.startswith("row:"):
        return vm.m.raw_train[int(h[4:])].astype(float).copy()
    if h.startswith("where:"):
        mask = query_mask(vm, h[6:])
        if not mask.any():
            raise ValueError(f"no training rows match {h[6:]!r}")
        return typical_raw(vm.m, mask).copy()
    raise ValueError(f"unknown --hold {hold!r} (typical, row:N, where:<expr>)")


def to_z(vm, raw: np.ndarray) -> np.ndarray:
    raw = np.atleast_2d(raw)
    return (raw - vm.m.mean) / vm.m.scale


def parse_assign(s: str) -> Tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"expected var=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def sweep_values(v, spec: str) -> List[Any]:
    """``a:b:steps`` (numbers), ``a,b,c`` (values or category codes/labels) or ``all`` (every
    category; for a number the 1st-99th percentile in 11 steps)."""
    spec = spec.strip()
    if spec == "all":
        if v.is_category():
            return list(v.all_codes)
        lo, hi = np.percentile(v.raw_train, [1, 99])
        return [float(x) for x in np.linspace(lo, hi, 11)]
    if ":" in spec and not v.is_category():
        a, b, *n = spec.split(":")
        steps = int(n[0]) if n else 11
        return [float(x) for x in np.linspace(float(a), float(b), steps)]
    return [x.strip() for x in spec.split(",") if x.strip()]


# ============================================================================ evaluation
def evaluate(vm, raws: np.ndarray, parts: bool = True) -> List[Dict[str, Any]]:
    """Risk, score, and per part: exit values and contribution (score change when switched off)."""
    m = vm.m
    Z = to_z(vm, raws)
    logit = m.logit(Z)
    risk = m.risk(logit)
    rows = [{"risk": float(risk[i]), "score": float(logit[i])} for i in range(Z.shape[0])]
    if not parts:
        return rows
    vals = m.ev.forward(Z)
    for p in vm.parts:
        if p.is_root or not p.exits:
            continue
        off = m.logit(Z, ablate=p.baselines)
        for i in range(Z.shape[0]):
            exits = {x: float(vals[x][i]) for x in p.exits}
            on = any(abs(exits[x] - p.baselines.get(x, 0.0)) > ON_TOL for x in p.exits)
            rows[i].setdefault("parts", {})[p.id] = {
                "name": p.name, "exits": exits, "active": bool(on),
                "contribution": float(logit[i] - off[i]),
                "risk_without": float(m.risk(off[i:i + 1])[0])}
    return rows


def probe(vm, sets: Sequence[str] = (), hold: str = "typical", sweeps: Sequence[str] = (),
          parts: bool = True) -> Dict[str, Any]:
    """The model's prediction for constructed inputs (real units)."""
    if len(sweeps) > 2:
        raise ValueError("at most two --sweep (a 2-D grid)")
    base = hold_row(vm, hold)
    fixed = []
    for s in sets:
        k, val = parse_assign(s)
        v = variable(vm, k)
        set_raw(base, v, val)
        fixed.append(v)
    axes = []
    for s in sweeps:
        k, spec = parse_assign(s)
        v = variable(vm, k)
        axes.append((v, sweep_values(v, spec)))
    combos = list(itertools.product(*[vals for _, vals in axes])) if axes else [()]
    raws = []
    for combo in combos:
        r = base.copy()
        for (v, _), val in zip(axes, combo):
            set_raw(r, v, val)
        raws.append(r)
    raws = np.array(raws)
    ev = evaluate(vm, raws, parts=parts)
    out_rows = []
    for combo, r, e in zip(combos, raws, ev):
        row = {"inputs": {v.base: describe_value(v, r) for v, _ in axes}}
        row.update(e)
        out_rows.append(row)
    shown = {v.base for v in fixed} | {v.base for v, _ in axes}
    return {
        "genome_id": vm.genome_id, "hold": hold,
        "held_case": {v.base: describe_value(v, base) for v in vm.m.variables if v.base not in shown},
        "set": {v.base: describe_value(v, base) for v in fixed},
        "sweep": [v.base for v, _ in axes],
        "parts": {p.id: p.name for p in vm.parts},
        "rows": out_rows,
        "support": {v.base: support_note(vm, v, [raw_value(v, r) for r in raws]) for v in fixed + [a for a, _ in axes]},
    }


def support_note(vm, v, values) -> str:
    """Whether the probed values lie inside the training data (no extrapolation)."""
    if v.is_category():
        counts = v.counts()
        out = [f"{c}: n={counts.get(c, 0)}" for c in sorted({str(x) for x in values if x is not None})]
        return "training counts " + ", ".join(out)
    lo, hi = float(np.min(v.raw_train)), float(np.max(v.raw_train))
    p1, p99 = np.percentile(v.raw_train, [1, 99])
    vals = [float(x) for x in values]
    outside = [x for x in vals if x < lo or x > hi]
    note = f"training range {v.fmt(lo)} to {v.fmt(hi)} (1st-99th percentile {v.fmt(p1)} to {v.fmt(p99)})"
    if outside:
        note += f"; OUTSIDE the data: {', '.join(v.fmt(x) for x in outside)}"
    return note


# ============================================================================ ablation
def _auc(y, s) -> Optional[float]:
    from ..analysis.explanation_tools import default_auc
    a = default_auc(y, s)
    return None if a is None or (isinstance(a, float) and math.isnan(a)) else round(float(a), 4)


def ablate(vm, part_ref: str, hold: str = "typical", split: str = "train", to: str = "baseline",
           where: Optional[str] = None, titles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Prediction change with a part switched off: its exits fixed at ``to`` (``baseline``:
    0 for a gated part, the training mean otherwise; ``zero``; ``mean``)."""
    m = vm.m
    p = vm.part(part_ref, titles)
    if p.is_root:
        raise ValueError("the root is the whole model; ablate one of its parts")
    vals = m._vals_train()
    if to == "baseline":
        fix = dict(p.baselines)
    elif to == "zero":
        fix = {x: 0.0 for x in p.exits}
    elif to == "mean":
        fix = {x: float(np.mean(vals[x])) for x in p.exits}
    else:
        raise ValueError("--to baseline | zero | mean")
    raw0 = hold_row(vm, hold)
    z0 = to_z(vm, raw0)
    s_on, s_off = float(m.logit(z0)[0]), float(m.logit(z0, ablate=fix)[0])
    if split == "test":
        if m.Xt is None:
            raise ValueError("no test split")
        X, y, raw = m.Xt, m.yt, m.raw_test
    else:
        X, y, raw = m.X, m.y, m.raw_train
    mask = np.ones(X.shape[0], dtype=bool)
    if where:
        mask = query_mask(vm, where, split=split)
    X, y = X[mask], y[mask]
    on = m.logit(X)
    off = m.logit(X, ablate=fix)
    d = on - off
    r_on, r_off = m.risk(on), m.risk(off)
    affected = np.abs(d) > ON_TOL
    flips = int(np.sum((r_on >= 0.5) != (r_off >= 0.5)))
    return {
        "genome_id": vm.genome_id, "part": {"id": p.id, "name": p.name, "exits": p.exits,
                                             "switched_off_at": fix, "gated": p.gated},
        "typical": {"hold": hold, "risk_with": float(m.risk(np.array([s_on]))[0]),
                    "risk_without": float(m.risk(np.array([s_off]))[0]),
                    "score_with": s_on, "score_without": s_off, "acts": abs(s_on - s_off) > ON_TOL},
        "data": {"split": split, "where": where, "n": int(X.shape[0]),
                 "n_affected": int(affected.sum()),
                 "share_affected": float(affected.mean()) if len(d) else 0.0,
                 "mean_score_change": float(np.mean(d)) if len(d) else 0.0,
                 "mean_abs_score_change": float(np.mean(np.abs(d))) if len(d) else 0.0,
                 "mean_risk_change": float(np.mean(r_on - r_off)) if len(d) else 0.0,
                 "mean_abs_risk_change": float(np.mean(np.abs(r_on - r_off))) if len(d) else 0.0,
                 "max_abs_risk_change": float(np.max(np.abs(r_on - r_off))) if len(d) else 0.0,
                 "risk_change_p10_p90": [float(x) for x in np.percentile(r_on - r_off, [10, 90])] if len(d) else None,
                 "auc_with": _auc(y, on), "auc_without": _auc(y, off),
                 "accuracy_with": float(np.mean((r_on >= 0.5) == (y > 0.5))) if len(d) else None,
                 "accuracy_without": float(np.mean((r_off >= 0.5) == (y > 0.5))) if len(d) else None,
                 "predictions_flipped": flips,
                 "outcome_rate_affected": float(np.mean(y[affected])) if affected.any() else None,
                 "outcome_rate_unaffected": float(np.mean(y[~affected])) if (~affected).any() else None},
    }


# ============================================================================ subgroups
def frame(vm, split: str = "train"):
    """A table of the rows in real units: numbers as values, categories as codes, plus
    ``<var>_label`` (the plain label), ``outcome`` (0/1) and ``risk`` (predicted)."""
    import pandas as pd
    m = vm.m
    raw = m.raw_test if split == "test" else m.raw_train
    X = m.Xt if split == "test" else m.X
    y = m.yt if split == "test" else m.y
    cols: Dict[str, Any] = {}
    for v in m.variables:
        if v.is_category():
            codes = v.raw_test if split == "test" else v.raw_train
            cols[v.base] = np.array([str(c) if c is not None else "" for c in codes], dtype=object)
            cols[v.base + "_label"] = np.array([v.level(c)["label"] if c is not None else "" for c in codes],
                                               dtype=object)
        else:
            cols[v.base] = raw[:, v.columns[0]].astype(float)
    cols["outcome"] = np.asarray(y, dtype=float)
    cols["risk"] = m.risk(m.logit(X))
    return pd.DataFrame(cols)


def _safe_name(s: str) -> str:
    return re.sub(r"\W", "_", s)


def query_mask(vm, expr: str, split: str = "train") -> np.ndarray:
    """Rows matching a pandas ``query`` expression over :func:`frame` (e.g.
    ``"age > 60 and sex_label == 'male'"``, ``"chol >= 258"``, ``"cp == '4'"``)."""
    df = frame(vm, split)
    ren = {c: _safe_name(c) for c in df.columns}
    df = df.rename(columns=ren)
    e = expr
    for c, s in sorted(ren.items(), key=lambda kv: -len(kv[0])):
        if c != s:
            e = e.replace(c, s)
    return np.asarray(df.eval(e, engine="python"), dtype=bool)


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def subgroup(vm, where: str, split: str = "train") -> Dict[str, Any]:
    """n, outcome rate and part activity for rows matching ``where`` (and the rest)."""
    from ..analysis.plain_language import typical_raw, typical_values
    m = vm.m
    mask = query_mask(vm, where, split)
    X = m.Xt if split == "test" else m.X
    y = m.yt if split == "test" else m.y
    on = m.logit(X)
    vals = m.ev.forward(X)

    def side(sel):
        n = int(sel.sum())
        k = int(np.sum(y[sel] > 0.5))
        out = {"n": n, "share_of_rows": n / len(y) if len(y) else 0.0,
               "outcome_n": k, "outcome_rate": k / n if n else None,
               "outcome_rate_ci95": _wilson(k, n),
               "mean_predicted_risk": float(np.mean(m.risk(on[sel]))) if n else None, "parts": {}}
        for p in vm.parts:
            if p.is_root or not p.exits:
                continue
            act = np.zeros(len(y), dtype=bool)
            for x in p.exits:
                act |= np.abs(vals[x] - p.baselines.get(x, 0.0)) > ON_TOL
            contrib = on - m.logit(X, ablate=p.baselines)
            out["parts"][p.id] = {"name": p.name,
                                  "share_active": float(act[sel].mean()) if n else None,
                                  "n_active": int(act[sel].sum()),
                                  "mean_contribution": float(np.mean(contrib[sel])) if n else None,
                                  "mean_abs_contribution": float(np.mean(np.abs(contrib[sel]))) if n else None}
        return out

    res = {"genome_id": vm.genome_id, "where": where, "split": split, "n_total": int(len(y)),
           "subgroup": side(mask), "rest": side(~mask)}
    if split == "train" and mask.any():
        raw = typical_raw(m, mask)
        res["typical_case_in_subgroup"] = {"values": typical_values(m, raw),
                                           "risk": float(m.risk(m.logit(to_z(vm, raw)))[0])}
    return res


def variables_table(vm) -> List[Dict[str, Any]]:
    """Every variable with its unit, range or categories (for writing --set and --where)."""
    out = []
    for v in vm.m.variables:
        d = {"name": v.base, "label": v.label, "kind": v.kind}
        if v.is_category():
            counts = v.counts()
            d["categories"] = [{"code": c, "label": v.level(c)["label"], "n_train": counts.get(c, 0)}
                               for c in v.all_codes]
        else:
            d["unit"] = v.unit
            d["range"] = [float(np.min(v.raw_train)), float(np.max(v.raw_train))]
            d["median"] = float(np.median(v.raw_train))
        out.append(d)
    return out
