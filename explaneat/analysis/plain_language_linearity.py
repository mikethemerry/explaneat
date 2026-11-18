"""Linear factors: which inputs (and parts) act at a fixed rate, and how much of the model they carry.

Every input variable (one-hot levels of a category together) is classified from its paths to the
output, by the hidden units each path passes through:

* ``exact_linear``: every path is linear (direct edges, identity units): a constant change in the
  score per unit of the input, everywhere;
* ``support_linear``: the paths also pass through ReLU units that are linear on the training data,
  so the effect is affine on the data (it could bend away from the data). A unit is linear on the
  data when it is active on every training row (``always_on``), or when it is a *binary one-step*
  unit (``binary_step``): a ReLU that switches on the data but reads exactly one input node, whose
  data column takes exactly two distinct values on the training rows (a yes/no input or one level
  of a one-hot category). Such a unit has only two possible outputs on the data, one per input
  value, so it is exactly an adjustment to that input's coefficient plus a constant; each one is
  checked numerically (its activation equals an affine function of the column on every training
  row, max error < 1e-9) and stays ``switching`` if the check fails. It holds only for the data's
  two values. Units downstream of a binary-step unit, and units with several inputs (even several
  levels of one one-hot group), are not reclassified; the one-group multi-input units that are also
  exactly affine on the data are listed separately (``one_group_exact``) for review;
* ``mixed``: at least one path is linear (as above) and at least one goes through a unit that
  switches on and off on the data;
* ``nonlinear``: every path goes through a switching unit.

A unit that is never active on the training data (dead) carries nothing, so paths through it are
left out. Each class is checked numerically: slopes of the score against the input on every
training row equal the linear-path coefficient (exact everywhere; 99% of rows for support-linear)
or, for mixed and nonlinear inputs, vary across rows. The slope is a finite difference, except for
a column that feeds a binary-step unit: there it is the change in the score between the column's
two values divided by their difference (the per-level effect), since a small step at the data point
would read the unit's local slope (0 or its weight) rather than its effect.

Per part: exact or support-linear when every hidden unit inside it is identity, always on, binary
one-step or dead (evidence: ``binary_step_units``), otherwise nonlinear. Per model: the share of the
score's variation carried by the linear-path contribution, and the AUC of the linearised model
(every switching unit replaced by its least-squares affine fit on the training rows) against the
full model, on train and test. A binary-step unit's affine fit is exact on the data, so leaving it
out of the linearisation does not change the linearised model (checked:
``binary_step_linearisation_max_diff``).

The claims are of type ``linearity``; they are appended after every other claim so existing claim
ids do not change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NodeType
from .explanation_tools import Linearisation, default_auc, input_column

KINDS = {"linear": 0, "always_on": 1, "binary_step": 1, "switching": 2}
CLASS_OF = {frozenset({0}): "exact_linear", frozenset({1}): "support_linear",
            frozenset({0, 1}): "support_linear"}
SUPPORT_SHARE = 0.99        # support-linear: slopes match on at least this share of training rows
FD_STEP = 1e-4              # finite-difference step (z units)
BINARY_STEP_TOL = 1e-9      # binary one-step unit: max |activation - affine fit| on the training rows


def unit_kinds(m, binary_step: bool = True, details: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Each non-input node: ``linear`` (identity activation, or an output: the score is its
    pre-activation), ``always_on`` (ReLU active on every training row), ``dead`` (never active),
    ``binary_step`` (switching, but reads one two-valued input only and is checked to be exactly
    affine in it on the data; see :func:`binary_step_check`) or ``switching``. Uses the training
    rows. ``binary_step=False`` gives the classification without the binary one-step kind.
    ``details`` (a dict) receives the check of every candidate binary-step unit."""
    pre = m.ev.pre_activations(m.X)
    out = {}
    for n, node in m.nodes.items():
        if node.type == NodeType.INPUT:
            continue
        if node.type == NodeType.OUTPUT or node.activation == "identity":
            out[n] = "linear"
            continue
        z = pre[n]
        if np.all(z > 0):
            out[n] = "always_on"
        elif np.all(z <= 0):
            out[n] = "dead"
        else:
            out[n] = "switching"
            if binary_step:
                chk = binary_step_check(m, n, z)
                if chk is not None:
                    if details is not None:
                        details[n] = chk
                    if chk["passed"]:
                        out[n] = "binary_step"
    return out


def _direct_inputs(m, n) -> List[Tuple[str, float]]:
    return [(str(p), float(w)) for p, w in m.ev.preds.get(n, []) if w != 0]


def binary_step_check(m, n: str, z: np.ndarray) -> Optional[Dict[str, Any]]:
    """For a switching ReLU ``n`` with pre-activation ``z`` on the training rows: if it reads
    exactly one input node and that node's data column takes exactly two distinct values on the
    training rows, the numerical check that its activation is an affine function of the column on
    every training row (least squares; max error < BINARY_STEP_TOL). ``None`` when it is not a
    candidate. ``gain``: the unit's output change over its pre-activation change between the two
    values (its slope on the data, used in place of the ReLU's 1 on the linear paths)."""
    ins = _direct_inputs(m, n)
    if len(ins) != 1:
        return None
    src, w = ins[0]
    node = m.nodes.get(src)
    if node is None or node.type != NodeType.INPUT:
        return None
    col = input_column(src)
    x = m.X[:, col]
    vals = np.unique(x)
    if len(vals) != 2:
        return None
    a = np.maximum(z, 0.0)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    err = float(np.max(np.abs(A @ coef - a)))
    lo, hi = float(vals[0]), float(vals[1])
    a_lo, a_hi = float(np.mean(a[x == lo])), float(np.mean(a[x == hi]))
    z_lo, z_hi = float(np.mean(z[x == lo])), float(np.mean(z[x == hi]))
    gain = (a_hi - a_lo) / (z_hi - z_lo) if z_hi != z_lo else 0.0
    return {"input": src, "column": int(col), "feature": m.feature_names[col] if col < len(m.feature_names) else str(col),
            "weight": w, "values_z": [lo, hi], "activation": [a_lo, a_hi],
            "slope_z": (a_hi - a_lo) / (hi - lo), "gain": float(gain),
            "rows_active": int(np.sum(z > 0)), "n": int(len(x)),
            "max_affine_error": err, "passed": bool(err < BINARY_STEP_TOL)}


def one_group_exact(m, kinds: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Switching ReLU units that read two or more input nodes, all columns of one one-hot category,
    and are exactly affine in those columns on the training rows (max error < BINARY_STEP_TOL):
    exact by the same argument as a binary one-step unit (one output per level), but not
    reclassified. Reported for review only."""
    pre = None
    out = {}
    for n, k in kinds.items():
        if k != "switching":
            continue
        ins = _direct_inputs(m, n)
        if len(ins) < 2 or any(m.nodes.get(p) is None or m.nodes[p].type != NodeType.INPUT for p, _ in ins):
            continue
        cols = sorted({input_column(p) for p, _ in ins})
        bases = {m.base_of_col.get(c) for c in cols}
        if len(bases) != 1 or len(cols) < 2:
            continue
        v = m.var_by_base.get(next(iter(bases)))
        if v is None or not v.is_category():
            continue
        if pre is None:
            pre = m.ev.pre_activations(m.X)
        a = np.maximum(pre[n], 0.0)
        A = np.column_stack([np.ones(m.X.shape[0])] + [m.X[:, c] for c in cols])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        err = float(np.max(np.abs(A @ coef - a)))
        if err < BINARY_STEP_TOL:
            out[n] = {"variable": v.base, "columns": [m.feature_names[c] for c in cols],
                      "inputs": [p for p, _ in ins], "max_affine_error": err}
    return out


def _succ(m) -> Dict[str, List[Tuple[str, float]]]:
    reach = m.reaching_output()
    succ: Dict[str, List[Tuple[str, float]]] = {n: [] for n in m.nodes}
    for c in m.structure.connections:
        u, v = str(c.from_node), str(c.to_node)
        if c.enabled and c.weight != 0 and u in m.nodes and v in reach:
            succ[u].append((v, float(c.weight)))
    return succ


def path_analysis(m, kinds: Optional[Dict[str, str]] = None, gains: Optional[Dict[str, float]] = None):
    """(path kinds per node: the set of worst-node kinds over its paths to the output;
    linear-path derivative of the score per node). ``gains``: a binary-step unit's slope on the
    data (output change over pre-activation change between its input's two values); 1 for the
    other linear-on-the-data units."""
    kinds = kinds or unit_kinds(m)
    gains = gains or {}
    succ = _succ(m)
    order = m.ev._order({})
    K: Dict[str, set] = {}
    g: Dict[str, float] = {}
    for n in reversed(order):
        if n in m.out_ids:
            K[n], g[n] = {0}, 1.0
            continue
        ks, d = set(), 0.0
        for s, w in succ.get(n, []):
            if s not in K:
                continue
            sk = kinds.get(s)
            if sk == "dead" or not K[s]:
                continue
            nk = KINDS[sk]
            ks |= {max(nk, k) for k in K[s]}
            if nk < 2:
                d += w * gains.get(s, 1.0) * g[s]
        K[n], g[n] = ks, d
    return K, g


def _class(kset: set) -> Optional[str]:
    if not kset:
        return None
    if 2 not in kset:
        return CLASS_OF[frozenset(kset)]
    return "mixed" if kset - {2} else "nonlinear"


def _gains(bstep: Dict[str, Dict[str, Any]], kinds: Dict[str, str]) -> Dict[str, float]:
    return {n: d["gain"] for n, d in bstep.items() if kinds.get(n) == "binary_step"}


def _step_columns(bstep: Dict[str, Dict[str, Any]], kinds: Dict[str, str]) -> Dict[int, List[str]]:
    """Data column -> the binary-step units its input nodes feed."""
    out: Dict[int, List[str]] = {}
    for n, d in bstep.items():
        if kinds.get(n) == "binary_step":
            out.setdefault(d["column"], []).append(n)
    return out


def classify_variables(m, kinds=None, bstep: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Per used variable: class, the linear-path coefficient of each column (score per z unit and
    per raw unit), and the slope check (finite differences; the two-value difference for a column
    that feeds a binary-step unit). ``bstep``: the binary-step checks from :func:`unit_kinds`."""
    if kinds is None:
        bstep = {} if bstep is None else bstep
        kinds = unit_kinds(m, details=bstep)
    bstep = bstep or {}
    K, g = path_analysis(m, kinds, _gains(bstep, kinds))
    step_cols = _step_columns(bstep, kinds)
    out = {}
    base = m.base_logit
    for v in m.variables:
        cols = [c for c in v.columns if c in m.read_columns]
        if not cols:
            continue
        kset = set()
        coef = {}
        for c in cols:
            for nd in m.input_nodes_by_col.get(c, []):
                kset |= K.get(nd, set())
            coef[c] = sum(g.get(nd, 0.0) for nd in m.input_nodes_by_col.get(c, []))
        cls = _class(kset)
        if cls is None:
            continue
        # finite-difference slopes on every training row, column by column
        match, varies, max_dev, slopes, two_value = [], [], 0.0, {}, []
        for c in cols:
            if c in step_cols:
                # the per-level effect: the score change between the column's two values
                lo, hi = np.unique(m.X[:, c])
                slope = (m.logit(m.X, {c: hi}) - m.logit(m.X, {c: lo})) / (hi - lo)
                two_value.append(int(c))
            else:
                X1 = m.X.copy()
                X1[:, c] = X1[:, c] + FD_STEP
                slope = (m.logit(X1) - base) / FD_STEP
            slopes[c] = slope
            dev = np.abs(slope - coef[c])
            tol = 1e-5 * max(1.0, abs(coef[c]))
            match.append(float(np.mean(dev <= tol)))
            varies.append(float(np.ptp(slope)) > tol)
            max_dev = max(max_dev, float(np.max(dev)))
        if cls in ("mixed", "nonlinear") and not any(varies):
            # its switching paths never switch with it on the data (the units' state is set by other
            # inputs, or they only pass a constant): a fixed rate across the observed data
            cls = "support_linear"
            coef = {c: float(np.median(slopes[c])) for c in cols}
            match = [1.0 for _ in cols]
            max_dev = max(float(np.max(np.abs(slopes[c] - coef[c]))) for c in cols)
        out[v.base] = {"class": cls, "columns": cols,
                       "coef_z": {c: float(coef[c]) for c in cols},
                       "coef_raw": {c: float(coef[c]) / float(v.scale[v.columns.index(c)]) for c in cols},
                       "fd_match_share": min(match), "fd_varies": any(varies), "fd_max_dev": max_dev,
                       "two_value_columns": two_value,
                       "binary_step_units": sorted({u for c in cols for u in step_cols.get(c, [])})}
    return out


def part_class(m, p, kinds) -> str:
    """exact_linear / support_linear when every hidden unit of the part is identity, always on,
    binary one-step or dead; nonlinear otherwise."""
    inside = {str(x) for x in p.unit.get("subgraph_nodes", [])} | {str(x) for x in p.exits}
    inside -= {str(x) for x in p.unit.get("entry_nodes", [])}
    ks = {kinds.get(n) for n in inside if n in kinds}
    if "switching" in ks:
        return "nonlinear"
    return "support_linear" if ks & {"always_on", "binary_step"} else "exact_linear"


def linear_component(m, g: Dict[str, float], X) -> np.ndarray:
    """The score's linear-path contribution: sum over input nodes of their linear-path derivative
    times the data column."""
    L = np.zeros(np.asarray(X).shape[0])
    for c, nds in m.input_nodes_by_col.items():
        k = sum(g.get(nd, 0.0) for nd in nds)
        if k:
            L = L + k * np.asarray(X)[:, c]
    return L


def linearised_auc(m, kinds) -> Dict[str, Any]:
    """AUC of the model with every switching unit replaced by its least-squares affine fit on the
    training rows (of the unit's output on its inputs), against the full model. Binary-step units
    are not replaced (their fit is exact on the data); ``binary_step_linearisation_max_diff`` is the
    largest change in the linearised score (train rows) if they were replaced as well."""
    reach = m.reaching_output()
    sw = [n for n, k in kinds.items() if k == "switching" and n in reach]
    bs = [n for n, k in kinds.items() if k == "binary_step" and n in reach]
    units = {n: {"entry_nodes": [p for p, _ in m.ev.preds[n]], "exit_nodes": [n]} for n in sw}
    out = {"switching_units": len(sw)}
    if units:
        lin = Linearisation(m.ev, units, m.X)
        f_tr, l_tr = m.ev.logit(m.X), lin.evaluator.logit(m.X, lin.overrides(sw))
    else:
        lin, f_tr, l_tr = None, m.ev.logit(m.X), m.ev.logit(m.X)
    if bs:
        both = sw + bs
        lin2 = Linearisation(m.ev, {n: {"entry_nodes": [p for p, _ in m.ev.preds[n]], "exit_nodes": [n]}
                                    for n in both}, m.X)
        l2 = lin2.evaluator.logit(m.X, lin2.overrides(both))
        out["binary_step_linearisation_max_diff"] = float(np.max(np.abs(l2 - l_tr)))
    out["auc_full_train"] = default_auc(m.y, f_tr)
    out["auc_linearised_train"] = default_auc(m.y, l_tr)
    if m.Xt is not None and m.yt is not None and len(m.Xt):
        f_te = m.ev.logit(m.Xt)
        l_te = lin.evaluator.logit(m.Xt, lin.overrides(sw)) if lin else f_te
        out["auc_full_test"] = default_auc(m.yt, f_te)
        out["auc_linearised_test"] = default_auc(m.yt, l_te)
    return out


def _fmt_coef(x: float) -> str:
    ax = abs(x)
    s = f"{x:+.2g}" if ax < 0.1 else (f"{x:+.2f}" if ax < 10 else f"{x:+.1f}")
    return s.replace("+", "+").replace("-", "−")


def _singular(unit: str) -> str:
    """'beats per minute' -> 'beat per minute', 'US dollars' -> 'US dollar', 'years' -> 'year'."""
    words = unit.split()
    i = 0 if "per" in words else len(words) - 1
    w = words[i]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        words[i] = w[:-1]
    return " ".join(words)


def _per_unit(v, info) -> List[Dict[str, Any]]:
    rows = []
    for c in info["columns"]:
        k = info["coef_raw"][c]
        if v.is_category():
            code = v.codes[v.columns.index(c)]
            lab = v.level(code)["label"]
            rows.append({"column": int(c), "level": str(code), "label": lab, "score_per_unit": k,
                         "text": f"{_fmt_coef(k)} for {lab}"})
        else:
            two = _two_values(v)
            if two is not None:
                lo, hi = two
                rows.append({"column": int(c), "label": v.short or v.label, "score_per_unit": k,
                             "values": [lo, hi], "score_change": k * (hi - lo),
                             "text": f"{_fmt_coef(k * (hi - lo))} for {v.fmt(hi)} against {v.fmt(lo)}"})
                continue
            rows.append({"column": int(c), "label": v.short or v.label, "score_per_unit": k,
                         "text": f"{_fmt_coef(k)} per {_singular(v.unit or 'unit')}"})
    return rows


def _two_values(v) -> Optional[Tuple[float, float]]:
    """(lower, higher) when a numeric variable takes exactly two values on the training rows."""
    vals = np.unique(np.asarray(v.raw_train, float))
    return (float(vals[0]), float(vals[1])) if len(vals) == 2 else None


def _typical_change(m, v, info, typ_raw) -> Dict[str, Any]:
    from .plain_language import _z
    base = float(m.risk(m.logit(_z(m, typ_raw)))[0])
    if v.is_category():
        cols = info["columns"]
        best = max(cols, key=lambda c: abs(info["coef_z"][c] * (1.0 / float(v.scale[v.columns.index(c)]))))
        code = v.codes[v.columns.index(best)]
        X = _z(m, typ_raw).copy()
        for c, z in v.encode(code).items():
            X[0, c] = z
        to = float(m.risk(m.logit(X))[0])
        n = int(np.sum(np.asarray(v.raw_train).astype(str) == str(code)))
        return {"from": base, "to": to, "change": f"to {v.level(code)['label']}", "n": n}
    x = np.asarray(v.raw_train, float)
    two = _two_values(v)       # a two-valued number: its two values, not the 10th/90th percentiles
    lo, hi = two if two is not None else np.percentile(x, [10, 90])
    out = []
    for val in (lo, hi):
        X = _z(m, typ_raw).copy()
        X[0, v.columns[0]] = (val - v.mean[0]) / v.scale[0]
        out.append(float(m.risk(m.logit(X))[0]))
    return {"from": out[0], "to": out[1], "change": f"from {v.fmt(lo)} to {v.fmt(hi)}", "n": int(len(x))}


def linearity_claims(m, parts, binary_step: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(claims, graph record): one claim per used variable, one per part, one for the model.
    ``binary_step=False``: without the binary one-step kind (the classification before it)."""
    from .plain_language import _claim, _finish, _var_desc, sig, typical_raw
    root = next((p for p in parts if p.is_root), None)
    if root is None:
        return [], {}
    bstep: Dict[str, Dict[str, Any]] = {}
    kinds = unit_kinds(m, binary_step=binary_step, details=bstep)
    K, g = path_analysis(m, kinds, _gains(bstep, kinds))
    vinfo = classify_variables(m, kinds, bstep)
    reach = m.reaching_output()
    n_step = sum(1 for n, k in kinds.items() if k == "binary_step" and n in reach)
    typ = typical_raw(m)
    claims: List[Dict[str, Any]] = []
    for v in m.variables:
        info = vinfo.get(v.base)
        if info is None:
            continue
        cls = info["class"]
        ch = _typical_change(m, v, info, typ) if cls in ("exact_linear", "support_linear") else None
        c = _claim(root, "linearity", f"var:{v.base}", variables=[_var_desc(v)],
                   statement={"kind": "variable", "variable": v.base, "class": cls,
                              "per_unit": _per_unit(v, info) if cls != "nonlinear" else [],
                              "per_unit_scope": ("every path" if cls in ("exact_linear", "support_linear")
                                                 else "its paths that do not switch" if cls == "mixed" else None),
                              "direction": (("raises" if sum(info["coef_z"].values()) > 0 else "lowers")
                                            if cls != "nonlinear" and not v.is_category() else None)},
                   n=int(m.X.shape[0]))
        if ch:
            c["risk"] = {"reference": "typical patient", "from": sig(ch["from"]), "to": sig(ch["to"]),
                         "delta": sig(ch["to"] - ch["from"]), "change": ch["change"], "n": ch["n"]}
        c["estimate"] = {"method": "linear-path coefficient of the score (log-odds) per unit, from the "
                                   "weights along the paths that do not switch; checked by finite "
                                   "differences on every training row", "value": sig(sum(info["coef_z"].values()))}
        c["evidence"] = {"fd_match_share": sig(info["fd_match_share"]), "fd_varies": info["fd_varies"],
                         "fd_max_deviation": sig(info["fd_max_dev"])}
        if info["binary_step_units"]:
            c["evidence"]["binary_step_units"] = len(info["binary_step_units"])
            c["evidence"]["two_value_columns"] = [m.feature_names[x] for x in info["two_value_columns"]]
        if cls == "exact_linear":
            chk = {"slope_constant": (info["fd_match_share"] >= 1.0,
                                      f"slope equals the coefficient on {info['fd_match_share']:.3f} of rows (needs all)")}
        elif cls == "support_linear":
            chk = {"slope_constant_on_data": (info["fd_match_share"] >= SUPPORT_SHARE,
                                              f"slope equals the coefficient on {info['fd_match_share']:.3f} of rows "
                                              f"(needs {SUPPORT_SHARE})")}
        else:
            chk = {"slope_varies": (info["fd_varies"], "the slope does not vary across rows")}
        claims.append(_finish(c, chk))
    # parts
    pcls = {}
    for p in parts:
        if p.is_root or p.duplicate_of:
            continue
        cls = part_class(m, p, kinds)
        pcls[p.id] = cls
        c = _claim(p, "linearity", "part", statement={"kind": "part", "class": cls}, n=int(m.X.shape[0]))
        inside = {str(x) for x in p.unit.get("subgraph_nodes", [])} | {str(x) for x in p.exits}
        sw = sorted(n for n in inside if kinds.get(n) == "switching")
        on = sorted(n for n in inside if kinds.get(n) == "always_on")
        st = sorted(n for n in inside if kinds.get(n) == "binary_step")
        c["evidence"] = {"switching_units": len(sw), "always_on_units": len(on)}
        if st:
            c["evidence"]["binary_step_units"] = len(st)
        c["estimate"] = {"method": "activity of the part's hidden units on the training rows", "value": len(sw)}
        claims.append(_finish(c, {"activity_checked": (True, "")}))
    # model
    L = linear_component(m, g, m.X)
    R = m.base_logit - L
    vl, vr = float(np.var(L)), float(np.var(R))
    share = vl / (vl + vr) if (vl + vr) > 0 else 1.0
    auc = linearised_auc(m, kinds)
    lin_diff = auc.pop("binary_step_linearisation_max_diff", None)
    counts = {k: sum(1 for i in vinfo.values() if i["class"] == k)
              for k in ("exact_linear", "support_linear", "mixed", "nonlinear")}
    nl_parts = sum(1 for k in pcls.values() if k == "nonlinear")
    c = _claim(root, "linearity", "model",
               statement={"kind": "model", "n_inputs": len(vinfo), **{f"n_{k}": v for k, v in counts.items()},
                          "variance_share_linear": sig(share), "n_parts": len(pcls), "n_nonlinear_parts": nl_parts,
                          **({"binary_step_units": n_step} if n_step else {}),
                          **{k: (sig(v) if isinstance(v, float) else v) for k, v in auc.items()}},
               n=int(m.X.shape[0]))
    c["estimate"] = {"method": "variance of the linear-path contribution over (its variance + the variance of the "
                               "rest of the score), training rows; linearised model: every switching unit "
                               "replaced by its least-squares affine fit on the training rows", "value": sig(share)}
    c["evidence"] = {"always_on_min_margin": sig(float(min([np.min(m.ev.pre_activations(m.X)[n])
                                                             for n, k in kinds.items() if k == "always_on"] or [0.0])))}
    og = one_group_exact(m, kinds)
    chk = {"score_split": (bool(np.allclose(L + R, m.base_logit)), "split does not add up"),
           "auc_computed": (auc.get("auc_full_train") == auc.get("auc_full_train"), "AUC not computable")}
    if bstep:
        failed = sorted(n for n, d in bstep.items() if not d["passed"])
        c["evidence"]["binary_step_units"] = n_step
        c["evidence"]["binary_step_max_affine_error"] = float(max(d["max_affine_error"] for d in bstep.values()))
        if failed:
            c["evidence"]["binary_step_check_failed"] = failed
        if lin_diff is not None:
            c["evidence"]["binary_step_linearisation_max_diff"] = lin_diff
            chk["binary_step_linearisation_unchanged"] = (lin_diff < 1e-6, "replacing the binary one-step units by "
                                                          "their affine fits changes the linearised score")
    if og:
        c["evidence"]["one_group_exact_not_reclassified"] = sorted(og)
    claims.append(_finish(c, chk))
    graph = {"columns": {m.feature_names[c]: vinfo[v.base]["class"] for v in m.variables if v.base in vinfo
                         for c in vinfo[v.base]["columns"]},
             "inputs": {nd: vinfo[v.base]["class"] for v in m.variables if v.base in vinfo
                        for c in vinfo[v.base]["columns"] for nd in m.input_nodes_by_col.get(c, [])},
             "variables": {b: i["class"] for b, i in vinfo.items()},
             "parts": pcls, "units": {n: k for n, k in kinds.items() if n not in m.out_ids}}
    if bstep:
        graph["binary_step"] = {n: {k: d[k] for k in ("input", "feature", "gain", "slope_z", "activation",
                                                       "max_affine_error", "passed")} for n, d in sorted(bstep.items())}
    if og:
        graph["one_group_exact"] = og
    return claims, graph
