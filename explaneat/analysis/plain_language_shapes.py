"""Shapes of effects: how the model's score responds across a variable's range.

Every claim about a numeric or ordinal variable (direction, magnitude, threshold, interaction per
group) gets a ``shape``: the model's response curve on its **score scale** (the log-odds, i.e.
the output's pre-activation; the "mechanism"), classified into a small plain taxonomy, checked
for consistency across patients and for robustness to the tolerances. The risk-scale curve is
recorded separately, with whether it saturates near 0% or 100% (the "appearance").

Taxonomy (``SHAPE_CLASSES``) and how it is decided (tolerances in ``TOL``; ``x`` is the variable
over its observed range, 1st to 99th percentile, or every observed level of an ordinal):

* ``flat``: the score moves by less than ``flat_abs`` (log-odds) over the whole range;
* ``peak`` / ``dip``: the curve reverses inside the range: an interior maximum (minimum) beats
  both ends by at least ``reversal`` of the curve's range; the turning point is named;
* the rest is split into linear pieces (ReLU networks give piecewise-linear scores): adjacent
  grid intervals belong to one piece while their slopes differ by at most ``kink`` of the
  largest slope; pieces shorter than ``min_piece`` of the range are merged; a piece is flat if
  its slope is at most ``flat_slope`` of the largest slope;
* ``steady``: no point of the curve is further than ``straight`` of its range from the straight
  line between its ends (a constant slope);
* ``step``: at least ``step_share`` of the change happens within ``step_width`` of the range;
  the window's centre is named;
* ``hinge``: two pieces, one of them flat (flat, then steady; or steady, then flat); the point
  where they meet is named;
* ``levelling_off`` / ``accelerating``: every piece goes the same way and the slopes keep
  shrinking (growing) in the direction of change;
* ``multi_segment``: three or more pieces otherwise; the pieces are summarised.

Ordinals also get the increment per level and an ``ordinal_pattern``: ``equal_steps`` (every
increment within ``equal_steps`` of the mean), ``first_step`` (the first step carries at least
``first_step`` of the change) or ``uneven``.

Categorical variables get ``categorical``: each level's effect relative to the reference and
the ordering.

Consistency: the same curve for a sample of training patients (``PATIENT_SAMPLE``); the share
whose class equals the typical patient's is recorded, ``consistent`` if at least
``consistent_share``. Robustness: the typical patient's class must not change when every
tolerance is scaled by 0.75 and by 1.25. A shape is ``verified`` only when it is consistent and
robust; otherwise the safe description applies (direction only, "in an uneven way"), unless a
clear majority shape has its exceptions characterised by a category (``mixed``: "for most
patients ..., but for patients with ... it ...").
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SHAPE_CLASSES = ("flat", "steady", "hinge", "step", "levelling_off", "accelerating", "peak", "dip",
                 "multi_segment", "categorical")

TOL = {
    "flat_abs": 0.05,        # log-odds: a curve moving less than this is flat
    "flat_slope": 0.05,      # a piece is flat if |slope| <= this share of the largest slope
    "kink": 0.15,            # slopes within this share of the largest slope are one piece
    "min_piece": 0.05,       # pieces shorter than this share of the range are merged
    "step_share": 0.70,      # a step: at least this share of the change ...
    "step_width": 0.10,      # ... within this share of the range
    "reversal": 0.10,        # a peak/dip: the interior extreme beats both ends by this share
    "equal_steps": 0.25,     # ordinal increments within this share of their mean: equal steps
    "first_step": 0.60,      # ordinal: the first step carries at least this share of the change
    "straight": 0.05,        # steady: no point further than this share of the range from the chord
}
CONSISTENT_SHARE = 0.90
MIXED_MAJORITY = 0.60
SENSITIVITY = (0.75, 1.25)
GRID = 41
PATIENT_SAMPLE = 300
ORDINAL_MAX_LEVELS = 12
SATURATION = 0.03            # risk below 3% or above 97% at an end of the range: saturates

# Plain phrases the prose must use for each shape (the lint checks them; never "linear", "convex").
SHAPE_PHRASES = {
    "flat": r"barely|hardly|almost no change|no real change",
    "steady": r"steadily|the same amount (for|with) each|each extra .{0,40}adds? about the same",
    "hinge": (r"nothing until|no change until|only once|only above|only below|only up to|only from|"
              r"until about|then no further|no further|stops? (adding|changing|mattering)|from about .{0,40}"
              r"\b(on|onwards|upwards)\b|then steadily|stays below about|rises above about|once it (rises|passes)"),
    "step": r"jump|in a step|mostly (between|around|from)|sharply around",
    "levelling_off": r"levels? off|levelling off|quickly at first|then more slowly|flattens? out|flattening",
    "accelerating": r"slowly at first|then faster|speeds? up|more and more|increasingly",
    "peak": r"highest at|peaks? at|lower (on )?either side",
    "dip": r"lowest at|dips? at|higher (on )?either side",
    "multi_segment": r"in stages|in several stages|by different amounts",
    "fallback": r"in an uneven way|unevenly",
    "first_step": r"mainly the first|mostly the first|first .{0,20}(step|vessel|level|one) does most",
    "equal_steps": r"each extra|the same amount for each|steadily",
}


# ============================================================================ the classifier

def _pieces(x: np.ndarray, L: np.ndarray, tol: Dict[str, float]) -> List[Dict[str, Any]]:
    dx = np.diff(x)
    s = np.diff(L) / np.where(dx == 0, 1e-12, dx)
    smax = float(np.max(np.abs(s))) if len(s) else 0.0
    if smax == 0:
        return [{"i0": 0, "i1": len(x) - 1, "slope": 0.0, "flat": True}]
    pieces = [[0]]
    for i in range(1, len(s)):
        cur = pieces[-1]
        if abs(s[i] - float(np.mean(s[cur]))) <= tol["kink"] * smax:
            cur.append(i)
        else:
            pieces.append([i])
    span = x[-1] - x[0]
    # merge short pieces into the neighbour with the closer slope
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for k, p in enumerate(pieces):
            if (x[p[-1] + 1] - x[p[0]]) < tol["min_piece"] * span:
                mine = float(np.mean(s[p]))
                left = float(np.mean(s[pieces[k - 1]])) if k > 0 else None
                right = float(np.mean(s[pieces[k + 1]])) if k + 1 < len(pieces) else None
                j = k - 1 if right is None or (left is not None and abs(left - mine) <= abs(right - mine)) else k + 1
                lo, hi = min(j, k), max(j, k)
                pieces[lo] = sorted(pieces[lo] + pieces[hi])
                del pieces[hi]
                changed = True
                break
    out = []
    for p in pieces:
        i0, i1 = p[0], p[-1] + 1
        slope = float((L[i1] - L[i0]) / (x[i1] - x[i0])) if x[i1] != x[i0] else 0.0
        out.append({"i0": i0, "i1": i1, "slope": slope, "flat": abs(slope) <= tol["flat_slope"] * smax})
    # merge adjacent pieces that ended up with similar slopes
    merged = [out[0]]
    for p in out[1:]:
        q = merged[-1]
        if abs(p["slope"] - q["slope"]) <= tol["kink"] * smax or (p["flat"] and q["flat"]):
            i0, i1 = q["i0"], p["i1"]
            slope = float((L[i1] - L[i0]) / (x[i1] - x[i0]))
            merged[-1] = {"i0": i0, "i1": i1, "slope": slope, "flat": abs(slope) <= tol["flat_slope"] * smax}
        else:
            merged.append(p)
    return merged


def _meet(x, L, p, q) -> float:
    """x where the lines of two adjacent pieces meet (clipped to the boundary interval)."""
    b = x[p["i1"]]
    if p["slope"] == q["slope"]:
        return float(b)
    # line p through (x[i0], L[i0]); line q through (x[i1q], L[i1q])
    xp, lp = x[p["i0"]], L[p["i0"]]
    xq, lq = x[q["i1"]], L[q["i1"]]
    t = (lq - lp + p["slope"] * xp - q["slope"] * xq) / (p["slope"] - q["slope"])
    lo, hi = x[max(p["i1"] - 1, 0)], x[min(q["i0"] + 1, len(x) - 1)]
    return float(min(max(t, lo), hi))


def classify_curve(x: Sequence[float], L: Sequence[float], tol: Optional[Dict[str, float]] = None,
                   scale: float = 1.0, ordinal: bool = False) -> Dict[str, Any]:
    """The shape class of a response curve (score scale) with its named points."""
    t = {k: v * scale for k, v in (tol or TOL).items()}
    x = np.asarray(x, dtype=float)
    L = np.asarray(L, dtype=float)
    R = float(L.max() - L.min())
    total = float(L[-1] - L[0])
    out: Dict[str, Any] = {"class": None, "range_score": R, "change_score": total, "points": {}}
    if R < t["flat_abs"] or len(x) < 2:
        out["class"] = "flat"
        return out
    ends_max, ends_min = max(L[0], L[-1]), min(L[0], L[-1])
    imax, imin = int(np.argmax(L)), int(np.argmin(L))
    if 0 < imax < len(L) - 1 and L[imax] - ends_max >= t["reversal"] * R:
        out.update({"class": "peak", "points": {"turning": float(x[imax])}})
        return out
    if 0 < imin < len(L) - 1 and ends_min - L[imin] >= t["reversal"] * R:
        out.update({"class": "dip", "points": {"turning": float(x[imin])}})
        return out
    if ordinal:
        inc = np.diff(L)
        mean = float(np.mean(inc))
        out["increments"] = [float(v) for v in inc]
        if len(inc) >= 2 and abs(inc[0]) >= t["first_step"] * abs(total) and np.all(np.sign(inc) * np.sign(total) >= 0):
            out["ordinal_pattern"] = "first_step"
        elif mean != 0 and np.max(np.abs(inc - mean)) <= t["equal_steps"] * abs(mean):
            out["ordinal_pattern"] = "equal_steps"
        else:
            out["ordinal_pattern"] = "uneven"
    pieces = _pieces(x, L, t)
    out["pieces"] = [{"from": float(x[p["i0"]]), "to": float(x[p["i1"]]), "slope": p["slope"], "flat": p["flat"]}
                     for p in pieces]
    if ordinal and out.get("ordinal_pattern") == "equal_steps":
        out["class"] = "steady"
        return out
    if ordinal and out.get("ordinal_pattern") == "first_step":
        out.update({"class": "step", "points": {"from": float(x[0]), "to": float(x[1]),
                                                "at": float((x[0] + x[1]) / 2)}})
        return out
    # step: most of the change within a narrow window (checked before the pieces)
    span = x[-1] - x[0]
    if not ordinal and total != 0:
        w = t["step_width"] * span
        best, where = 0.0, None
        for i in range(len(x)):
            j = int(np.searchsorted(x, x[i] + w, side="right")) - 1
            if j <= i:
                continue
            share = abs(L[j] - L[i]) / abs(total)
            if share > best:
                best, where = share, (float(x[i]), float(x[j]))
        if best >= t["step_share"] and where is not None:
            out.update({"class": "step", "points": {"from": where[0], "to": where[1],
                                                    "at": (where[0] + where[1]) / 2}})
            return out
    chord = L[0] + (L[-1] - L[0]) * (x - x[0]) / (span if span else 1.0)
    if float(np.max(np.abs(L - chord))) <= t["straight"] * R:
        out["class"] = "steady"
        return out
    moving = [p for p in pieces if not p["flat"]]
    if len(pieces) == 2 and len(moving) == 1:
        at = _meet(x, L, pieces[0], pieces[1])
        out.update({"class": "hinge", "points": {"at": at},
                    "hinge_kind": "flat_then_moving" if pieces[0]["flat"] else "moving_then_flat"})
        return out
    signs = {np.sign(p["slope"]) for p in moving}
    if len(signs) == 1:
        if len(pieces) == 1:   # a smooth bend within one piece: compare the two halves
            mid = int(len(x) // 2)
            s1 = abs((L[mid] - L[0]) / (x[mid] - x[0]))
            s2 = abs((L[-1] - L[mid]) / (x[-1] - x[mid]))
            out["class"] = "levelling_off" if s2 < s1 else "accelerating"
            return out
        mags = [abs(p["slope"]) for p in pieces]
        d = np.diff(mags)
        if np.all(d <= 0):
            out["class"] = "levelling_off"
            return out
        if np.all(d >= 0):
            out["class"] = "accelerating"
            return out
    out["class"] = "multi_segment"
    return out


def classify_robust(x, L, ordinal=False) -> Dict[str, Any]:
    base = classify_curve(x, L, ordinal=ordinal)
    alts = [classify_curve(x, L, scale=s, ordinal=ordinal)["class"] for s in SENSITIVITY]
    base["stable"] = all(a == base["class"] for a in alts)
    base["sensitivity"] = {f"x{s}": a for s, a in zip(SENSITIVITY, alts)}
    return base


# ============================================================================ curves on the model

def grid_for(v) -> Tuple[np.ndarray, bool]:
    """(grid in raw units, ordinal?): every common observed level of an ordinal, else 41 points
    from the 1st to the 99th percentile."""
    vals, cnt = np.unique(v.raw_train, return_counts=True)
    n = len(v.raw_train)
    keep = cnt >= max(5, 0.01 * n)
    common = vals[keep]
    integer = bool(np.all(np.abs(common - np.round(common)) < 1e-9))
    coverage = float(cnt[keep].sum()) / n if n else 0.0
    # an ordinal: at most ORDINAL_MAX_LEVELS whole-number levels that hold at least 95% of patients
    if integer and 2 <= len(common) <= ORDINAL_MAX_LEVELS and coverage >= 0.95:
        return common.astype(float), True
    lo, hi = np.percentile(v.raw_train, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(v.raw_train)), float(np.max(v.raw_train))
    return np.linspace(lo, hi, GRID), False


def _curves(m, v, grid, scope, rows: np.ndarray, extra: Optional[Dict[int, float]] = None) -> np.ndarray:
    X = m.X[rows].copy()
    if extra:
        for c, z in extra.items():
            X[:, c] = z
    G = len(grid)
    Xr = np.repeat(X, G, axis=0)
    z = (np.tile(grid, X.shape[0]) - v.mean[0]) / v.scale[0]
    return m.logit(Xr, {v.columns[0]: z}, scope).reshape(X.shape[0], G)


def _sample(rows_mask: np.ndarray) -> np.ndarray:
    idx = np.where(rows_mask)[0]
    if len(idx) > PATIENT_SAMPLE:
        idx = idx[np.linspace(0, len(idx) - 1, PATIENT_SAMPLE).round().astype(int)]
    return idx


def _points_text(v, typ: Dict[str, Any], ordinal: bool) -> Dict[str, str]:
    pts = typ.get("points", {}) or {}
    out = {k: v.fmt(round(val, v.decimals)) for k, val in pts.items()}
    if ordinal and typ.get("class") == "step" and "from" in pts and "to" in pts:
        a, b = v.fmt(pts["from"]), v.fmt(pts["to"])
        out["at"] = f"between {a.split(' ')[0]} and {b}"
    return out


def numeric_shape(m, v, scope, typical_raw_row: np.ndarray, rows_mask: np.ndarray,
                  describe_vars: Sequence[str] = (), extra_raw: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    """Shape of ``v``'s effect (on ``scope``) for the typical patient, plus consistency across a
    sample of the patients in ``rows_mask``."""
    from .plain_language import sig
    grid, ordinal = grid_for(v)
    zrow = ((typical_raw_row - m.mean) / m.scale)[None, :]
    if extra_raw:
        for c, val in extra_raw.items():
            zrow[0, c] = (val - m.mean[c]) / m.scale[c]
    Lt = m.logit(np.repeat(zrow, len(grid), axis=0), {v.columns[0]: (grid - v.mean[0]) / v.scale[0]}, scope)
    typ = classify_robust(grid, Lt, ordinal)
    risk = m.risk(Lt)
    sat = None
    if float(np.max(risk)) > 1 - SATURATION:
        sat = "near 100%"
    elif float(np.min(risk)) < SATURATION:
        sat = "near 0%"
    idx = _sample(rows_mask)
    extra = None
    if extra_raw:
        extra = {c: (val - m.mean[c]) / m.scale[c] for c, val in extra_raw.items()}
    C = _curves(m, v, grid, scope, idx, extra)
    per = [classify_curve(grid, C[i], ordinal=ordinal) for i in range(len(idx))]
    classes_all = [r["class"] for r in per]
    rebased = False
    if typ["class"] == "flat":
        moving = [r for r in per if r["class"] != "flat"]
        if len(moving) >= max(3, 0.1 * len(per)):
            # the typical patient is not affected (e.g. a part that is off for them): describe the
            # shape among the patients whose curve moves, with the median named point
            mc: Dict[str, int] = {}
            for r in moving:
                mc[r["class"]] = mc.get(r["class"], 0) + 1
            major = max(mc, key=lambda k: mc[k])
            pts = [r.get("points", {}) for r in moving if r["class"] == major]
            med = {}
            for k in {k for p in pts for k in p}:
                vals = [p[k] for p in pts if k in p]
                med[k] = float(np.median(vals))
            ex = next(r for r in moving if r["class"] == major)
            typ = {**typ, "class": major, "points": med, "hinge_kind": ex.get("hinge_kind"),
                   "ordinal_pattern": ex.get("ordinal_pattern"), "pieces": ex.get("pieces", [])}
            typ["stable"] = all(classify_curve(grid, C[i], scale=sc, ordinal=ordinal)["class"] == major
                                for i, r in enumerate(per) if r is ex for sc in SENSITIVITY)
            rebased = True
    # patients whose curve does not move at all are not affected by this scope; consistency is
    # judged among the affected patients (the unaffected share is recorded)
    affected = np.array([c != "flat" for c in classes_all]) if typ["class"] != "flat" else np.ones(len(idx), bool)
    share_unaffected = float(1 - affected.mean()) if len(idx) else 0.0
    idx = idx[affected]
    classes = [c for c, a in zip(classes_all, affected) if a]
    share = float(np.mean([c == typ["class"] for c in classes])) if classes else 0.0
    counts: Dict[str, int] = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    out = {"class": typ["class"], "scale": "model score (log-odds)", "ordinal": ordinal,
           "grid": [sig(g) for g in grid], "grid_basis": ("every common observed level" if ordinal else
                                                          "41 points from the 1st to the 99th percentile"),
           "typical_curve": [sig(l) for l in Lt], "typical_risk": [sig(r) for r in risk],
           "points": {k: sig(val) for k, val in typ.get("points", {}).items()},
           "points_text": _points_text(v, typ, ordinal),
           "typical_patient_unaffected": rebased,
           "pieces": [{"from": sig(p["from"]), "to": sig(p["to"]), "slope": sig(p["slope"]), "flat": p["flat"]}
                      for p in typ.get("pieces", [])],
           "hinge_kind": typ.get("hinge_kind"), "ordinal_pattern": typ.get("ordinal_pattern"),
           "increments": [sig(x) for x in typ.get("increments", [])] or None,
           "stable": typ["stable"], "sensitivity": typ["sensitivity"],
           "patients_sampled": int(len(classes_all)), "share_unaffected": sig(share_unaffected),
           "share_same_shape": sig(share), "share_basis": "among sampled patients whose curve moves",
           "classes_across_patients": dict(sorted(counts.items())),
           "consistent": share >= CONSISTENT_SHARE, "risk_saturation": sat,
           "tolerances": dict(TOL), "consistent_share": CONSISTENT_SHARE}
    status = "verified" if (out["consistent"] and out["stable"]) else "fallback"
    out["exceptions"] = []
    if not out["consistent"] and classes:
        major = max(counts, key=lambda k: counts[k])
        if major == typ["class"] and share >= MIXED_MAJORITY and out["stable"]:
            # characterise the exceptions by a category (patients with another shape)
            other = np.array([c != typ["class"] for c in classes])
            conds = []
            for b in describe_vars:
                W = m.var_by_base[b]
                if not W.is_category():
                    continue
                codes = W.raw_train[idx]
                for code in W.all_codes:
                    ins = float(np.mean(codes[other] == code)) if other.any() else 0.0
                    oth = float(np.mean(codes[~other] == code)) if (~other).any() else 0.0
                    if ins >= 0.8 and ins - oth >= 0.3:
                        conds.append({"group": W.level(code)["group"], "share_in": sig(ins), "share_out": sig(oth)})
            alt = max((k for k in counts if k != typ["class"]), key=lambda k: counts[k])
            if conds:
                out["exceptions"] = [{"class": alt, "share": sig(1 - share),
                                      "who": [c["group"] for c in conds]}]
                status = "mixed"
    out["status"] = status
    out["verified"] = status in ("verified", "mixed")
    if status == "fallback":
        out["description"] = "direction only, in an uneven way"
    return out


def categorical_shape(m, v, scope, typical_raw_row: np.ndarray) -> Dict[str, Any]:
    from .plain_language import level_classes, sig
    read = m.read_codes(v, scope)
    if not read:
        return {"class": "categorical", "levels": []}
    classes = level_classes(v, read)
    ref = next(c for c in classes if c["is_reference"])
    zrow = ((typical_raw_row - m.mean) / m.scale)[None, :]
    base = float(m.set_logit(zrow, v, ref["rep"], scope)[0])
    levels = []
    for c in classes:
        lz = float(m.set_logit(zrow, v, c["rep"], scope)[0])
        levels.append({"label": c["label"], "effect_score": sig(lz - base),
                       "risk": sig(float(m.risk(np.array([lz]))[0])), "reference": c is ref,
                       "n_train": c["n_train"]})
    order = [l["label"] for l in sorted(levels, key=lambda l: -(l["effect_score"] or 0))]
    return {"class": "categorical", "scale": "model score (log-odds) relative to the reference level",
            "reference": ref["label"], "levels": levels, "ordering_high_to_low": order,
            "status": "verified", "verified": True}


# ============================================================================ attaching shapes to claims

def attach_shapes(m, parts, claims: List[Dict[str, Any]]) -> None:
    """Give every effect claim a ``shape`` (uses the private ``_ctx`` the builders leave)."""
    from .plain_language import typical_raw
    part_by_id = {p.id: p for p in parts}
    folds: Dict[str, List[Dict[str, Any]]] = {}
    for c in claims:
        if c["type"] == "threshold" and c.get("verified"):
            folds.setdefault(c["statement"]["variable"], []).append(
                {"claim": c.get("id"), "part": c["annotation"]["id"], "kind": c["statement"]["kind"],
                 "value": c["statement"]["threshold"]["value"], "text": c["statement"]["threshold"]["text"],
                 "p10_p90": (c.get("estimate") or {}).get("threshold", {}).get("p10_p90_across_patients"),
                 "p10_p90_text": (c.get("estimate") or {}).get("threshold", {}).get("p10_p90_text"),
                 "count_step": c["statement"]["threshold"].get("count_step")})
    cache: Dict[Tuple, Dict[str, Any]] = {}
    for c in claims:
        ctx = c.pop("_ctx", None)
        if ctx is None or c["type"] in ("null", "rule", "subgroup"):
            continue
        v = m.var_by_base[ctx["var"]]
        part = part_by_id.get(c["annotation"]["id"])
        scope = ctx["scope"] if "scope" in ctx else (None if part is None or part.is_root else part.input_nodes)
        rows = ctx.get("rows")
        rows = np.ones(m.X.shape[0], dtype=bool) if rows is None else rows
        raw = ctx.get("typical") if ctx.get("typical") is not None else typical_raw(m)
        dvars = list(part.variables) if part is not None else []
        if v.is_category():
            key = ("cat", v.base, c["annotation"]["id"], c.get("scope"), tuple(raw))
            if key not in cache:
                cache[key] = categorical_shape(m, v, scope, raw)
            c["shape"] = cache[key]
            continue
        if c["type"] == "interaction":
            groups = []
            for g in ctx["groups"]:
                sh = numeric_shape(m, v, None, raw, g["rows"], dvars, extra_raw=g["extra_raw"])
                sh["group"] = g["label"]
                groups.append(sh)
            strong = c["statement"]["stronger_group"]["label"]
            main = next((g for g in groups if g["group"] == strong), groups[0])
            c["shape"] = {**{k: main[k] for k in ("class", "status", "verified", "points", "points_text",
                                                  "ordinal_pattern", "risk_saturation", "hinge_kind", "exceptions",
                                                  "share_same_shape", "consistent", "stable")},
                          "scale": main["scale"], "basis": "the stronger group's typical patient",
                          "groups": groups}
            continue
        key = ("num", v.base, c["annotation"]["id"], c.get("scope"), tuple(raw), tuple(np.where(rows)[0][:5]), int(rows.sum()))
        if key not in cache:
            cache[key] = numeric_shape(m, v, scope, raw, rows, dvars)
        sh = dict(cache[key])
        fl = [f for f in folds.get(v.base, [])]
        if fl:
            sh["structure_folds"] = fl
            span = (sh["grid"][-1] or 0) - (sh["grid"][0] or 0)
            at = (sh.get("points") or {}).get("at")
            if at is not None and span:
                near = [f for f in fl if abs((f["value"] or 0) - at) <= 0.1 * span]
                sh["matched_fold"] = near[0] if near else None
        if c["type"] == "threshold":
            th = c["statement"]["threshold"]["value"]
            at = (sh.get("points") or {}).get("at")
            span = (sh["grid"][-1] or 0) - (sh["grid"][0] or 0)
            sh["agrees_with_threshold"] = bool(sh["class"] in ("hinge", "step") and at is not None and span
                                               and abs(at - th) <= 0.15 * span)
        reconcile_point(sh, c, v, fl)
        c["shape"] = sh


WIDE_SPREAD = 0.10   # a fold spread (10th-90th percentile) wider than this share of the range is "wide"
OUTSIDE_TOL = 0.10   # the typical hinge may sit this share of the range (4 grid steps) outside the spread


def reconcile_point(sh: Dict[str, Any], c: Dict[str, Any], v, folds: List[Dict[str, Any]]) -> None:
    """One named point per fold.

    A shape's hinge/step point and a threshold claim refer to the same fold when they concern the
    same variable within the same annotation (part). Then the named point is the threshold claim's
    median fold across affected patients; the typical patient's hinge is recorded alongside, with
    the spread of per-patient folds (10th to 90th percentile). Points of different parts (or a
    whole-model curve against a part's fold) are kept separate, each named with its own part.
    """
    from .plain_language import sig
    at = (sh.get("points") or {}).get("at")
    same = [f for f in folds if f["part"] == c["annotation"]["id"]]
    if sh.get("class") not in ("hinge", "step") or at is None:
        sh["point_reconciled"] = False
        sh["point_reason"] = f"no fold in this shape ({sh.get('class')})"
        return
    if c.get("scope") == "model" or c["type"] == "interaction":
        sh["point_reconciled"] = False
        sh["point_reason"] = ("whole-model curve: its point is kept separate from the parts' folds"
                              + (f" ({', '.join(f['part'] for f in folds)})" if folds else ""))
        return
    if not same:
        sh["point_reconciled"] = False
        sh["point_reason"] = ("no verified threshold claim for this variable in this part"
                              + (f"; folds of other parts kept separate ({', '.join(f['part'] for f in folds)})"
                                 if folds else ""))
        return
    f = same[0]
    span = (sh["grid"][-1] or 0) - (sh["grid"][0] or 0)
    lo, hi = (f.get("p10_p90") or [None, None])
    # The typical patient's hinge is read off a 41-point grid (or one level of a count), so it may
    # sit a grid step from the per-patient folds; beyond that it is a different fold for this patient.
    tol = max(OUTSIDE_TOL * span, float(f.get("count_step") or 0), 0.0)
    if lo is not None and (at < lo - tol or at > hi + tol):
        sh["point_reconciled"] = False
        sh["point_reason"] = (f"the typical patient's fold ({(sh.get('points_text') or {}).get('at')}) lies outside "
                              f"the per-patient folds of threshold claim {f['claim']} "
                              f"({' to '.join(f.get('p10_p90_text') or [])}): kept separate")
        return
    wide = bool(lo is not None and span and (hi - lo) > WIDE_SPREAD * span)
    typical_text = (sh.get("points_text") or {}).get("at")
    sh["point"] = {"value": f["value"], "text": f["text"], "source_claim": f["claim"],
                   "basis": "median fold across the part's affected patients (threshold claim)",
                   "typical_hinge": {"value": sig(at), "text": typical_text},
                   "spread_p10_p90": [lo, hi], "spread_text": f.get("p10_p90_text"), "wide": wide}
    sh["points_text"] = dict(sh.get("points_text") or {})   # the shape dict is shared via a cache
    if f.get("count_step") is None:
        sh["points_text"]["at"] = f["text"]
    sh["point_reconciled"] = True
    sh["point_reason"] = "same variable and same part as threshold claim " + str(f["claim"])


def spread_phrase(sh: Dict[str, Any], who: str = "patients") -> str:
    """'(between 0.8 and 1.9 mm for different patients)' when a reconciled fold spreads widely."""
    pt = (sh or {}).get("point") or {}
    if not pt.get("wide") or not pt.get("spread_text"):
        return ""
    a, b = pt["spread_text"]
    if a == b:
        return ""
    a, b = a.split(" (")[0], b.split(" (")[0]          # no second unit inside the brackets
    na, _, ua = a.partition(" ")
    nb, _, ub = b.partition(" ")
    if ua and ua == ub:                                  # '0.8 and 1.9 mm', not '0.8 mm and 1.9 mm'
        a = na
    return f" (between {a} and {b} for different {who})"


def old_spread_phrase(sh: Dict[str, Any], who: str = "patients") -> str:
    """The first v3.1 wording, kept so stored drafts can be migrated to :func:`spread_phrase`."""
    pt = (sh or {}).get("point") or {}
    if not pt.get("wide") or not pt.get("spread_text"):
        return ""
    a, b = pt["spread_text"]
    if a == b:
        return ""
    return f" (between {a.split(' (')[0]} and {b} for different {who})"


def shape_phrase_ok(text: str, shape: Optional[Dict[str, Any]], ctype: str) -> Tuple[bool, str]:
    """Does ``text`` carry a phrase that matches ``shape`` (or the fallback)?"""
    import re
    if not shape or shape.get("class") == "categorical":
        return True, ""
    st = shape.get("status")
    if st == "fallback":
        return bool(re.search(SHAPE_PHRASES["fallback"], text, re.I)), "the fallback phrase 'in an uneven way'"
    cls = shape["class"]
    pats = [SHAPE_PHRASES[cls]]
    if shape.get("ordinal_pattern") == "first_step":
        pats.append(SHAPE_PHRASES["first_step"])
    if shape.get("ordinal_pattern") == "equal_steps":
        pats.append(SHAPE_PHRASES["equal_steps"])
    if ctype == "threshold":
        pats += [SHAPE_PHRASES["hinge"], SHAPE_PHRASES["step"]]
    ok = any(re.search(p, text, re.I) for p in pats)
    if st == "mixed":
        ok = ok and bool(re.search(r"\bbut for\b|\bexcept for\b", text, re.I))
    want = f"a phrase for the shape '{cls}'" + (" plus 'but for …' for the exceptions" if st == "mixed" else "")
    return ok, want
