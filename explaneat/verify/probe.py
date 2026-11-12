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


def fmt_full(v, value) -> str:
    """A number in real units at full precision (no rounding to the codebook's decimals)."""
    s = f"{float(value):.10g}"
    return s + (f" {v.unit}" if v.unit else "")


def describe_value(v, raw: np.ndarray) -> str:
    val = raw_value(v, raw)
    if v.is_category():
        return f"{v.level(val)['label']} ({v.base}={val})" if val is not None else "none"
    return fmt_full(v, val)


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


def parse_assign(s: str, vm=None) -> Tuple[str, str]:
    """``var=value``. With ``vm``, a variable whose own name contains ``=`` (a one-hot column
    such as ``horTh=yes`` from scikit-survival's encoder) is found by trying the longest prefix
    that names a variable first: ``horTh=yes=1`` sets ``horTh=yes`` to 1."""
    if "=" not in s:
        raise ValueError(f"expected var=value, got {s!r}")
    if vm is not None:
        for i in [j for j, ch in enumerate(s) if ch == "="][::-1]:
            k = s[:i].strip()
            try:
                variable(vm, k)
            except KeyError:
                continue
            return k, s[i + 1:].strip()
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


def _leading_number(text: str) -> Optional[float]:
    m = re.search(r"[-−]?\d[\d,]*(?:\.\d+)?", str(text))
    return float(m.group(0).replace(",", "").replace("−", "-")) if m else None


# ============================================================================ a claim's own case
def claim_case(vm, claim: Dict[str, Any]) -> Dict[str, Any]:
    """The reference case a claim was computed on, as probe settings: ``hold`` raw row,
    ``sets`` (the reference values the claim lists), and ``sweeps`` (the claim's change, and
    for an interaction the modifier's settings)."""
    st = claim.get("statement") or {}
    risk = claim.get("risk") or {}
    raw = hold_row(vm, "typical")
    sets: List[str] = []
    notes: List[str] = []
    for key, text in (risk.get("reference_values") or {}).items():
        try:
            v = variable(vm, key)
        except KeyError:
            notes.append(f"reference value for {key!r} not matched to a variable")
            continue
        try:
            if v.is_category():
                val = category_code(v, text)
            else:
                val = _leading_number(text)
                if val is None:
                    raise ValueError(text)
            set_raw(raw, v, val)
            sets.append(f"{v.base}={val}")
        except ValueError:
            notes.append(f"reference value {key}={text!r} not parsed")
    sweeps: List[str] = []
    var = st.get("variable") or ((claim.get("variables") or [{}])[0].get("base"))
    ch = st.get("change") or {}
    if var and ch.get("from") is not None and ch.get("to") is not None:
        v = variable(vm, var)
        f, t = ch["from"], ch["to"]
        if v.is_category():
            a, b = f.get("code"), t.get("code")
        else:
            a, b = f.get("value"), t.get("value")
        if a is not None and b is not None:
            sweeps.append(f"{v.base}={a},{b}")
    mod = st.get("modifier")
    settings = (claim.get("evidence") or {}).get("counterfactual_modifier_settings")
    if mod and settings:
        try:
            mv = variable(vm, mod)
            sweeps.append(f"{mv.base}=" + ",".join(str(s) for s in settings))
        except KeyError:
            notes.append(f"modifier {mod!r} not matched to a variable")
    return {"raw": raw, "sets": sets, "sweeps": sweeps, "notes": notes,
            "reference": risk.get("reference"),
            "stated_risk": {k: risk.get(k) for k in ("from", "to", "groups") if risk.get(k) is not None}}


# ============================================================================ evaluation
def _node_ids(vm, spec: Optional[Sequence[str]]) -> List[str]:
    """Node ids for ``--nodes``: ids, display names, a part id/name (its internal nodes and
    exits), or ``all`` (every hidden node)."""
    from ..core.genome_network import NodeType
    if not spec:
        return []
    st = vm.structure
    nodes = {n.id: n for n in st.nodes}
    dm = st.get_display_map()
    by_name = {v: k for k, v in dm.items()}
    out: List[str] = []
    for s in spec:
        for tok in str(s).split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok == "all":
                out += [n.id for n in st.nodes if n.type not in (NodeType.INPUT,)]
                continue
            if tok in nodes:
                out.append(tok)
                continue
            if tok in by_name:
                out.append(by_name[tok])
                continue
            try:
                p = vm.part(tok)
                inner = [str(n) for n in p.unit.get("subgraph_nodes", [])
                         if str(n) in nodes and nodes[str(n)].type != NodeType.INPUT]
                out += inner + [x for x in p.exits if x not in inner]
                continue
            except KeyError:
                pass
            raise KeyError(f"no node or part {tok!r}")
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def evaluate(vm, raws: np.ndarray, parts: bool = True, nodes: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Risk, score; per part its exit values and contribution (score change when switched off;
    ``acts`` = contribution not zero); per requested node its value and pre-activation."""
    m = vm.m
    Z = to_z(vm, raws)
    logit = m.logit(Z)
    risk = m.risk(logit)
    rows = [{"risk": float(risk[i]), "score": float(logit[i])} for i in range(Z.shape[0])]
    if parts:
        vals = m.ev.forward(Z)
        for p in vm.parts:
            if p.is_root or not p.exits:
                continue
            off = m.logit(Z, ablate=p.baselines)
            for i in range(Z.shape[0]):
                exits = {x: float(vals[x][i]) for x in p.exits}
                contrib = float(logit[i] - off[i])
                rows[i].setdefault("parts", {})[p.id] = {
                    "name": p.name, "exits": exits, "acts": abs(contrib) > ON_TOL,
                    "exit_nonzero": any(abs(v) > ON_TOL for v in exits.values()),
                    "contribution": contrib, "risk_without": float(m.risk(off[i:i + 1])[0])}
    if nodes:
        vals = m.ev.forward(Z)
        pre = m.ev.pre_activations(Z)
        for i in range(Z.shape[0]):
            rows[i]["nodes"] = {n: {"value": float(vals[n][i]),
                                    "pre_activation": float(pre[n][i]) if n in pre else float(vals[n][i])}
                                for n in nodes}
    return rows


def probe(vm, sets: Sequence[str] = (), hold: str = "typical", sweeps: Sequence[str] = (),
          parts: bool = True, nodes: Sequence[str] = (), claim: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The model's prediction for constructed inputs (real units). With ``claim``, start from
    the claim's own reference case and sweep its change (extra ``sets``/``sweeps`` apply on top)."""
    case = None
    if claim is not None:
        case = claim_case(vm, claim)
        base = case["raw"]
        sweeps = list(sweeps) or case["sweeps"]
        hold = f"the reference case of claim {claim.get('id')} ({case['reference'] or 'typical'})"
    else:
        base = hold_row(vm, hold)
    if len(sweeps) > 2:
        raise ValueError("at most two --sweep (a 2-D grid)")
    fixed = []
    for s in sets:
        k, val = parse_assign(s, vm)
        v = variable(vm, k)
        set_raw(base, v, val)
        fixed.append(v)
    axes = []
    for s in sweeps:
        k, spec = parse_assign(s, vm)
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
    node_ids = _node_ids(vm, nodes)
    ev = evaluate(vm, raws, parts=parts, nodes=node_ids)
    out_rows = []
    for combo, r, e in zip(combos, raws, ev):
        row = {"inputs": {v.base: describe_value(v, r) for v, _ in axes}}
        row.update(e)
        out_rows.append(row)
    shown = {v.base for v in fixed} | {v.base for v, _ in axes}
    dm = vm.structure.get_display_map()
    res = {
        "genome_id": vm.genome_id, "hold": hold,
        "held_case": {v.base: describe_value(v, base) for v in vm.m.variables if v.base not in shown},
        "set": {v.base: describe_value(v, base) for v in fixed},
        "sweep": [v.base for v, _ in axes],
        "parts": {p.id: p.name for p in vm.parts},
        "nodes": {n: dm.get(n, n) for n in node_ids},
        "rows": out_rows,
        "support": {v.base: support_note(vm, v, [raw_value(v, r) for r in raws]) for v in fixed + [a for a, _ in axes]},
    }
    if case is not None:
        res["claim"] = {"id": claim.get("id"), "key": claim.get("key"), "stated_risk": case["stated_risk"],
                        "reference_sets": case["sets"], "notes": case["notes"]}
    return res


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
    note = (f"training range {fmt_full(v, lo)} to {fmt_full(v, hi)} (1st-99th percentile "
            f"{fmt_full(v, p1)} to {fmt_full(v, p99)})")
    if outside:
        note += f"; OUTSIDE the data: {', '.join(fmt_full(v, x) for x in outside)}"
    return note


# ============================================================================ ablation
TO_HELP = ("baseline (default): a gated ReLU exit is set to 0 (the unit is off) and an always-on exit to "
           "its training mean (its variation removed, its average kept); zero: every exit to 0 "
           "(for an always-on unit this also removes its average); mean: every exit to its training mean. "
           "Edges: zero sets the weight to 0; baseline and mean replace the edge's contribution by its "
           "training mean (moved into the target's bias).")


def _auc(y, s) -> Optional[float]:
    from ..analysis.explanation_tools import default_auc
    a = default_auc(y, s)
    return None if a is None or (isinstance(a, float) and math.isnan(a)) else round(float(a), 4)


def _edge_ref(s: str) -> Tuple[str, str]:
    m = re.fullmatch(r"\s*(.+?)\s*(?:->|→|:)\s*(.+?)\s*", s)
    if not m:
        raise ValueError(f"--edge wants FROM->TO, got {s!r}")
    return m.group(1), m.group(2)


class Ablation:
    """Switch off parts, nodes and edges together, on a copy of the model state."""

    def __init__(self, vm, parts: Sequence[str] = (), nodes: Sequence[str] = (), edges: Sequence[str] = (),
                 direct: bool = False, to: str = "baseline", titles: Optional[Dict[str, str]] = None):
        import copy as _copy
        from ..analysis.explanation_tools import StateEvaluator
        from ..core.genome_network import NodeType
        if to not in ("baseline", "zero", "mean"):
            raise ValueError("--to baseline | zero | mean")
        self.vm, self.to = vm, to
        m = vm.m
        vals = m._vals_train()
        st = _copy.deepcopy(vm.structure)
        nodes_by_id = {n.id: n for n in st.nodes}
        dm = st.get_display_map()
        by_name = {v: k for k, v in dm.items()}
        self.fix: Dict[str, float] = {}
        self.what: List[Dict[str, Any]] = []
        for ref in parts:
            p = vm.part(ref, titles)
            if p.is_root:
                raise ValueError("the root is the whole model; use --direct for its direct edges, or ablate its parts")
            if to == "baseline":
                f = dict(p.baselines)
            elif to == "zero":
                f = {x: 0.0 for x in p.exits}
            else:
                f = {x: float(np.mean(vals[x])) for x in p.exits}
            self.fix.update(f)
            self.what.append({"part": p.id, "name": p.name, "exits_fixed_at": f, "gated": p.gated})
        for ref in nodes:
            nid = ref if ref in nodes_by_id else by_name.get(ref)
            if nid is None:
                raise KeyError(f"no node {ref!r}")
            v = vals[nid]
            if to == "zero":
                k = 0.0
            elif to == "mean":
                k = float(np.mean(v))
            else:
                n = nodes_by_id[nid]
                k = 0.0 if (n.activation != "identity" and float(np.min(v)) <= 1e-12) else float(np.mean(v))
            self.fix[nid] = k
            self.what.append({"node": nid, "name": dm.get(nid, nid), "fixed_at": k,
                              "note": "only this node is fixed; the units it feeds still compute from their other inputs"})
        targets: List[Tuple[str, str]] = [_edge_ref(e) for e in edges]
        targets = [(a if a in nodes_by_id else by_name.get(a, a), b if b in nodes_by_id else by_name.get(b, b))
                   for a, b in targets]
        if direct:
            # the root's direct input edges: input -> output, also through identity relays
            # (add_identity_node on the output puts identity_<k> between the inputs and the output)
            def relay(n) -> bool:
                return (n.type not in (NodeType.INPUT, NodeType.OUTPUT) and str(n.activation or "").lower() == "identity"
                        and float(n.bias or 0.0) == 0.0)
            frontier = [str(o) for o in st.output_node_ids]
            seen = set(frontier)
            while frontier:
                b = frontier.pop()
                for c in st.connections:
                    if not (c.enabled and str(c.to_node) == b):
                        continue
                    a = nodes_by_id[str(c.from_node)]
                    if a.type == NodeType.INPUT:
                        targets.append((str(c.from_node), b))
                    elif relay(a) and str(c.from_node) not in seen:
                        seen.add(str(c.from_node))
                        frontier.append(str(c.from_node))
        done = []
        for a, b in targets:
            conns = [c for c in st.connections if c.enabled and str(c.from_node) == a and str(c.to_node) == b]
            if not conns:
                raise KeyError(f"no enabled edge {a}->{b}")
            for c in conns:
                shift = 0.0 if to == "zero" else float(c.weight * np.mean(vals[a]))
                nodes_by_id[b].bias = float(nodes_by_id[b].bias or 0.0) + shift
                done.append({"edge": f"{a}->{b}", "weight": float(c.weight), "bias_shift": shift})
                c.weight = 0.0
        if direct:
            self.what.append({"direct_edges_of_root": len(done)})
        self.what += [{"edge": d["edge"], "weight": d["weight"], "replaced_by_mean_contribution": d["bias_shift"]}
                      for d in done]
        if not self.what:
            raise ValueError("nothing to switch off: give --part, --node, --edge or --direct")
        self.ev = StateEvaluator(st)

    def logit(self, Z) -> np.ndarray:
        n = Z.shape[0]
        ov = {x: ((), (lambda vals, k=float(k), n=n: np.full(n, k))) for x, k in self.fix.items()}
        return self.ev.logit(Z, ov or None)


def ablate(vm, part_ref=None, hold: str = "typical", split: str = "train", to: str = "baseline",
           where: Optional[str] = None, titles: Optional[Dict[str, str]] = None,
           parts: Sequence[str] = (), nodes: Sequence[str] = (), edges: Sequence[str] = (),
           direct: bool = False) -> Dict[str, Any]:
    """Prediction change with parts, single nodes, edges or the root's direct edges switched
    off (see :data:`TO_HELP` for ``to``), for the typical case and over the data."""
    m = vm.m
    parts = list(parts) + ([part_ref] if isinstance(part_ref, str) else list(part_ref or []))
    ab = Ablation(vm, parts, nodes, edges, direct, to, titles)
    z0 = to_z(vm, hold_row(vm, hold))
    s_on, s_off = float(m.logit(z0)[0]), float(ab.logit(z0)[0])
    if split == "test":
        if m.Xt is None:
            raise ValueError("no test split")
        X, y = m.Xt, m.yt
    else:
        X, y = m.X, m.y
    if where:
        mask = query_mask(vm, where, split=split)
        X, y = X[mask], y[mask]
    on = m.logit(X)
    off = ab.logit(X)
    d = on - off
    r_on, r_off = m.risk(on), m.risk(off)
    affected = np.abs(d) > ON_TOL
    flips = int(np.sum((r_on >= 0.5) != (r_off >= 0.5)))
    first = ab.what[0]
    return {
        "genome_id": vm.genome_id, "to": to, "to_meaning": TO_HELP, "switched_off": ab.what,
        "part": ({"id": first.get("part"), "name": first.get("name"), "exits": list(first.get("exits_fixed_at", {})),
                  "switched_off_at": first.get("exits_fixed_at"), "gated": first.get("gated")}
                 if len(ab.what) == 1 and "part" in first else None),
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
    """A table of the rows in real units: numbers as values, categories as codes (strings),
    plus ``<var>_label`` (the plain label), ``outcome`` (0/1) and ``risk`` (predicted)."""
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
            for code in v.all_codes:     # 0/1 indicators: a5_1 == 1, or counts such as a1_1 + a2_1 == 2
                cols[f"{v.base}_{code}"] = (codes == code).astype(int)
        else:
            cols[v.base] = raw[:, v.columns[0]].astype(float)
    cols["outcome"] = np.asarray(y, dtype=float)
    cols["risk"] = m.risk(m.logit(X))
    # node values: node_290 > 0 (on), node_204_a == 0 (off); hidden and identity nodes only
    vals = m._vals_train() if split != "test" else m.ev.forward(X)
    inputs = {str(n.id) for n in m.structure.nodes if n.type.name == "INPUT"}
    outs = set(m.out_ids)
    for nid, v in vals.items():
        if str(nid) in inputs or str(nid) in outs:
            continue
        key = "node_" + re.sub(r"[^0-9A-Za-z_]", "_", str(nid))
        if key not in cols:
            cols[key] = np.asarray(v, dtype=float)
    return pd.DataFrame(cols)


QUERY_HELP = ("Write a pandas query over real-unit columns. Category columns hold their codes as text, "
              "so quote them: a5 == '1', thal == '7' (or use the label: sex_label == 'female'). Put names "
              "with spaces or hyphens in backticks: `capital-gain` > 5000, `total day minutes` > 200, "
              "`voice mail plan_label` == 'no voice-mail plan'. Each category level also has a 0/1 column "
              "<var>_<code> (a5_1 == 1; counts: a1_1 + a2_1 + a3_1 == 2). Also: outcome (0/1), risk (predicted), "
              "and each hidden node's value node_<id> (node_290 > 0 and node_204_a == 0).")


def query_mask(vm, expr: str, split: str = "train") -> np.ndarray:
    """Rows matching a pandas ``query`` expression over :func:`frame` (e.g.
    ``"age > 60 and sex_label == 'male'"``, ``"chol >= 258"``, ``"cp == '4'"``,
    ``"`capital-gain` > 5000"``)."""
    df = frame(vm, split)
    cats = [v.base for v in vm.m.variables if v.is_category()]
    for c in cats:   # a category compared with a bare number is always False: say so
        rx = r"(?:`%s`|(?<![\w`])%s(?![\w`]))\s*(?:==|!=|<|>|<=|>=)\s*-?\d" % (re.escape(c), re.escape(c))
        if re.search(rx, expr):
            raise ValueError(f"{c} is a category: its codes are text, so write {c} == '1' (quoted), "
                             f"not {c} == 1. " + QUERY_HELP)
    try:
        res = df.eval(expr, engine="python")
    except Exception as e:
        raise ValueError(f"cannot evaluate --where {expr!r} ({type(e).__name__}: {e}). " + QUERY_HELP) from None
    return np.asarray(res, dtype=bool)


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def subgroup(vm, where: str, split: str = "train") -> Dict[str, Any]:
    """n, outcome rate and, per part, where it acts (contribution not zero) for rows matching
    ``where`` and for the rest. ``exit_nonzero`` is also given: for a gated ReLU part it is the
    same as acting; an always-on (linear) part has non-zero exits everywhere but may act on
    only some rows."""
    from ..analysis.plain_language import typical_raw, typical_values
    m = vm.m
    mask = query_mask(vm, where, split)
    X = m.Xt if split == "test" else m.X
    y = m.yt if split == "test" else m.y
    on = m.logit(X)
    vals = m.ev.forward(X)
    contribs = {}
    for p in vm.parts:
        if p.is_root or not p.exits:
            continue
        contribs[p.id] = on - m.logit(X, ablate=p.baselines)

    def side(sel):
        n = int(sel.sum())
        k = int(np.sum(y[sel] > 0.5))
        out = {"n": n, "share_of_rows": n / len(y) if len(y) else 0.0,
               "outcome_n": k, "outcome_rate": k / n if n else None,
               "outcome_rate_ci95": _wilson(k, n),
               "mean_predicted_risk": float(np.mean(m.risk(on[sel]))) if n else None, "parts": {}}
        for p in vm.parts:
            if p.id not in contribs:
                continue
            contrib = contribs[p.id]
            acts = np.abs(contrib) > ON_TOL
            nz = np.zeros(len(y), dtype=bool)
            for x in p.exits:
                nz |= np.abs(vals[x]) > ON_TOL
            out["parts"][p.id] = {"name": p.name, "gated": p.gated,
                                  "n_acts_on": int(acts[sel].sum()),
                                  "share_acts_on": float(acts[sel].mean()) if n else None,
                                  "share_exit_nonzero": float(nz[sel].mean()) if n else None,
                                  "mean_contribution": float(np.mean(contrib[sel])) if n else None,
                                  "mean_abs_contribution": float(np.mean(np.abs(contrib[sel]))) if n else None}
        return out

    res = {"genome_id": vm.genome_id, "where": where, "split": split, "n_total": int(len(y)),
           "subgroup": side(mask), "rest": side(~mask)}
    if not mask.any():
        res["hint"] = "no rows match. " + QUERY_HELP
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
