"""Linear factors: which inputs (and parts) act at a fixed rate, and how much of the model they carry.

Every input variable (one-hot levels of a category together) is classified from its paths to the
output, by the hidden units each path passes through:

* ``exact_linear``: every path is linear (direct edges, identity units): a constant change in the
  score per unit of the input, everywhere;
* ``support_linear``: the paths also pass through ReLU units that are active on every training
  row, so the effect is affine on the data (it could bend away from the data);
* ``mixed``: at least one path is linear (as above) and at least one goes through a unit that
  switches on and off on the data;
* ``nonlinear``: every path goes through a switching unit.

A unit that is never active on the training data (dead) carries nothing, so paths through it are
left out. Each class is checked numerically: finite-difference slopes of the score against the
input on every training row equal the linear-path coefficient (exact everywhere; 99% of rows for
support-linear) or, for mixed and nonlinear inputs, vary across rows.

Per part: exact or support-linear when every hidden unit inside it is (identity / always on / dead),
otherwise nonlinear. Per model: the share of the score's variation carried by the linear-path
contribution, and the AUC of the linearised model (every switching unit replaced by its least-
squares affine fit on the training rows) against the full model, on train and test.

The claims are of type ``linearity``; they are appended after every other claim so existing claim
ids do not change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NodeType
from .explanation_tools import Linearisation, default_auc, input_column

KINDS = {"linear": 0, "always_on": 1, "switching": 2}
CLASS_OF = {frozenset({0}): "exact_linear", frozenset({1}): "support_linear",
            frozenset({0, 1}): "support_linear"}
SUPPORT_SHARE = 0.99        # support-linear: slopes match on at least this share of training rows
FD_STEP = 1e-4              # finite-difference step (z units)


def unit_kinds(m) -> Dict[str, str]:
    """Each non-input node: ``linear`` (identity activation, or an output: the score is its
    pre-activation), ``always_on`` (ReLU active on every training row), ``dead`` (never active)
    or ``switching``. Uses the training rows."""
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
    return out


def _succ(m) -> Dict[str, List[Tuple[str, float]]]:
    reach = m.reaching_output()
    succ: Dict[str, List[Tuple[str, float]]] = {n: [] for n in m.nodes}
    for c in m.structure.connections:
        u, v = str(c.from_node), str(c.to_node)
        if c.enabled and c.weight != 0 and u in m.nodes and v in reach:
            succ[u].append((v, float(c.weight)))
    return succ


def path_analysis(m, kinds: Optional[Dict[str, str]] = None):
    """(path kinds per node: the set of worst-node kinds over its paths to the output;
    linear-path derivative of the score per node)."""
    kinds = kinds or unit_kinds(m)
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
                d += w * g[s]
        K[n], g[n] = ks, d
    return K, g


def _class(kset: set) -> Optional[str]:
    if not kset:
        return None
    if 2 not in kset:
        return CLASS_OF[frozenset(kset)]
    return "mixed" if kset - {2} else "nonlinear"


def classify_variables(m, kinds=None) -> Dict[str, Dict[str, Any]]:
    """Per used variable: class, the linear-path coefficient of each column (score per z unit and
    per raw unit), and the finite-difference check."""
    kinds = kinds or unit_kinds(m)
    K, g = path_analysis(m, kinds)
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
        match, varies, max_dev, slopes = [], [], 0.0, {}
        for c in cols:
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
                       "fd_match_share": min(match), "fd_varies": any(varies), "fd_max_dev": max_dev}
    return out


def part_class(m, p, kinds) -> str:
    """exact_linear / support_linear when every hidden unit of the part is identity, always on or
    dead; nonlinear otherwise."""
    inside = {str(x) for x in p.unit.get("subgraph_nodes", [])} | {str(x) for x in p.exits}
    inside -= {str(x) for x in p.unit.get("entry_nodes", [])}
    ks = {kinds.get(n) for n in inside if n in kinds}
    if "switching" in ks:
        return "nonlinear"
    return "support_linear" if "always_on" in ks else "exact_linear"


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
    training rows (of the unit's output on its inputs), against the full model."""
    sw = [n for n, k in kinds.items() if k == "switching" and n in m.reaching_output()]
    units = {n: {"entry_nodes": [p for p, _ in m.ev.preds[n]], "exit_nodes": [n]} for n in sw}
    out = {"switching_units": len(sw)}
    if units:
        lin = Linearisation(m.ev, units, m.X)
        f_tr, l_tr = m.ev.logit(m.X), lin.evaluator.logit(m.X, lin.overrides(sw))
    else:
        lin, f_tr, l_tr = None, m.ev.logit(m.X), m.ev.logit(m.X)
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
            rows.append({"column": int(c), "label": v.short or v.label, "score_per_unit": k,
                         "text": f"{_fmt_coef(k)} per {_singular(v.unit or 'unit')}"})
    return rows


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
    lo, hi = np.percentile(x, [10, 90])
    out = []
    for val in (lo, hi):
        X = _z(m, typ_raw).copy()
        X[0, v.columns[0]] = (val - v.mean[0]) / v.scale[0]
        out.append(float(m.risk(m.logit(X))[0]))
    return {"from": out[0], "to": out[1], "change": f"from {v.fmt(lo)} to {v.fmt(hi)}", "n": int(len(x))}


def linearity_claims(m, parts) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(claims, graph record): one claim per used variable, one per part, one for the model."""
    from .plain_language import _claim, _finish, _var_desc, sig, typical_raw
    root = next((p for p in parts if p.is_root), None)
    if root is None:
        return [], {}
    kinds = unit_kinds(m)
    K, g = path_analysis(m, kinds)
    vinfo = classify_variables(m, kinds)
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
        c["evidence"] = {"switching_units": len(sw), "always_on_units": len(on)}
        c["estimate"] = {"method": "activity of the part's hidden units on the training rows", "value": len(sw)}
        claims.append(_finish(c, {"activity_checked": (True, "")}))
    # model
    L = linear_component(m, g, m.X)
    R = m.base_logit - L
    vl, vr = float(np.var(L)), float(np.var(R))
    share = vl / (vl + vr) if (vl + vr) > 0 else 1.0
    auc = linearised_auc(m, kinds)
    counts = {k: sum(1 for i in vinfo.values() if i["class"] == k)
              for k in ("exact_linear", "support_linear", "mixed", "nonlinear")}
    nl_parts = sum(1 for k in pcls.values() if k == "nonlinear")
    c = _claim(root, "linearity", "model",
               statement={"kind": "model", "n_inputs": len(vinfo), **{f"n_{k}": v for k, v in counts.items()},
                          "variance_share_linear": sig(share), "n_parts": len(pcls), "n_nonlinear_parts": nl_parts,
                          **{k: (sig(v) if isinstance(v, float) else v) for k, v in auc.items()}},
               n=int(m.X.shape[0]))
    c["estimate"] = {"method": "variance of the linear-path contribution over (its variance + the variance of the "
                               "rest of the score), training rows; linearised model: every switching unit "
                               "replaced by its least-squares affine fit on the training rows", "value": sig(share)}
    c["evidence"] = {"always_on_min_margin": sig(float(min([np.min(m.ev.pre_activations(m.X)[n])
                                                             for n, k in kinds.items() if k == "always_on"] or [0.0])))}
    claims.append(_finish(c, {"score_split": (bool(np.allclose(L + R, m.base_logit)), "split does not add up"),
                              "auc_computed": (auc.get("auc_full_train") == auc.get("auc_full_train"),
                                               "AUC not computable")}))
    graph = {"columns": {m.feature_names[c]: vinfo[v.base]["class"] for v in m.variables if v.base in vinfo
                         for c in vinfo[v.base]["columns"]},
             "inputs": {nd: vinfo[v.base]["class"] for v in m.variables if v.base in vinfo
                        for c in vinfo[v.base]["columns"] for nd in m.input_nodes_by_col.get(c, [])},
             "variables": {b: i["class"] for b, i in vinfo.items()},
             "parts": pcls, "units": {n: k for n, k in kinds.items() if n not in m.out_ids}}
    return claims, graph
