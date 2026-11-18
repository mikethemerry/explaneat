"""Feature dossier: what one input does to the model, patient by patient (read-only).

For one continuous input of an explained genome this builds, on the **model state** (the
phenotype with every operation replayed) and the **training split**:

* **ICE curves** for every training patient over the input's observed training range, in real
  units (``raw = z * scale + mean`` with the split's StandardScaler), the curve with every other
  column at its training median (the evidence panel's line plot) and the partial dependence.
* **Bend attribution.** A NEAT network here is ReLU hidden units into a sigmoid output, so each
  patient's log-odds curve is piecewise linear in the input and bends exactly where a hidden
  unit downstream of the input crosses zero. Every crossing is located (grid, then bisection),
  the change of slope it makes is measured, and every numerical kink of the curve is checked
  to be one of them. For each unit the input reaches, the threshold is given in closed form
  as a function of the other inputs (real units), for each on/off pattern of the units
  upstream of it, and checked against the located crossings.
* **Shape families.** Patients are grouped by their symbolic signature (which units switch,
  on or off, in which order, within the observed range, counting only bends that change the
  curve by at least ``min_bend`` log-odds across the range); rare signatures are merged into
  the nearest common one. A k-means clustering of the curves is reported as a cross-check.
  Each family gets its characteristic (median) curve, the median bend positions, the
  conditions that define it (a shallow decision tree on the other inputs in real units and
  decoded labels, with its training and cross-validated accuracy) and its data support.
* **Data check.** Observed outcome rate in each family across bands of the input (training;
  the test split as a labelled check, families assigned by the same signature rule), against
  the model's mean predicted risk.

Nothing is written to the database. The curves are the model's, not the world's: a shape the
data barely covers is flagged, never interpreted.

    uv run python -m explaneat dossier <genome_id> <input> [--annotation NAME] [--out DIR]
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NodeType
from .composition import annotation_dicts_from_operations
from .explanation_tools import as_unit, input_column

DOSSIER_SCHEMA_VERSION = "1.0.0"


@dataclass
class DossierConfig:
    grid_points: int = 401          # ICE grid over the observed training range
    bisect_iters: int = 48          # crossing refinement (exact to ~range / 2**48)
    min_bend: float = 0.05          # a bend counts if |change of slope| * range >= this (log-odds)
    max_families: int = 7
    min_family: int = 5             # signatures with fewer patients are merged into the nearest
    tree_depth: int = 3
    tree_min_leaf: int = 5
    support_window: float = 0.05    # bend region widened by this share of the range each side
    thin_support: int = 5           # fewer family patients than this in a bend region: flagged
    n_bands: int = 5                # outcome bands (training quantiles of the input)
    flat_logodds: float = 0.5       # a segment reads as flat if |slope| * range < this (log-odds)
    min_segment: float = 0.03       # segments shorter than this share of the range are not words
    seed: int = 0


# ============================================================================ helpers

def _sig(x, n: int = 4):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x) or x == 0:
        return x
    return float(f"{x:.{n}g}")


def _wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [_sig(max(0.0, c - h), 3), _sig(min(1.0, c + h), 3)]


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def resolve_variable(m, name: str):
    """The model's variable for ``name``: a base name (``trestbps``), a column name, a
    codebook label, or the panel's ``"trestbps · resting blood pressure"``."""
    cand = [s.strip() for s in str(name).replace("·", "|").split("|") if s.strip()]
    for c in cand:
        if c in m.var_by_base:
            return m.var_by_base[c]
        for v in m.variables:
            if c.lower() in (v.label.lower(), v.base.lower()):
                return v
    raise ValueError(f"input {name!r} not found; inputs are {[v.base for v in m.variables]}")


# ============================================================================ the analyser

class FeatureAnalyser:
    """Forward passes with one input varied, pre-activations of every unit, and closed-form
    unit thresholds. ``m`` is a :class:`~explaneat.analysis.plain_language.PlainModel`."""

    def __init__(self, m, var, scope: Optional[Sequence[str]] = None, output_index: int = 0):
        if var.is_category():
            raise ValueError(f"{var.base} is a category; the dossier is for numeric inputs")
        self.m = m
        self.var = var
        self.col = var.columns[0]
        self.mean = float(var.mean[0])
        self.scale = float(var.scale[0])
        self.ev = m.ev
        self.nodes = m.nodes
        self.out = m.out_ids[output_index]
        copies = m.input_nodes_by_col.get(self.col, [])
        self.varied = [n for n in copies if scope is None or n in set(scope)]
        if not self.varied:
            raise ValueError(f"no copy of {var.base} in the given scope")
        self.held_copies = [n for n in copies if n not in self.varied]
        self.preds = self.ev.preds
        succ: Dict[str, List[str]] = {}
        for to, ps in self.preds.items():
            for p, w in ps:
                if w != 0:
                    succ.setdefault(p, []).append(to)
        reach = m.reaching_output()
        down, stack = set(), list(self.varied)
        while stack:
            x = stack.pop()
            for y in succ.get(x, []):
                if y not in down:
                    down.add(y)
                    stack.append(y)
        self.downstream = down & reach
        self.units = sorted(
            (n for n in self.downstream
             if self.nodes[n].type == NodeType.HIDDEN and self.nodes[n].activation != "identity"),
            key=self._topo_key())
        self.relu_all = [n for n, nd in self.nodes.items()
                         if nd.type == NodeType.HIDDEN and nd.activation != "identity"]

    def _topo_key(self):
        order = self.ev._order({})
        pos = {n: i for i, n in enumerate(order)}
        return lambda n: pos[n]

    # ------------------------------------------------------------------ forward
    def z_of(self, raw):
        return (np.asarray(raw, dtype=float) - self.mean) / self.scale

    def pre(self, X: np.ndarray, x_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Pre-activation of every non-input node (and the output's log-odds) for rows ``X``
        (z-scored) with the varied copies set to ``x_raw`` (real units, one per row)."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        z = np.broadcast_to(self.z_of(x_raw), (n,)).astype(float)
        overrides = {node: ((), (lambda vals, a=z: a)) for node in self.varied}
        vals = self.ev.forward(X, overrides)
        out = {}
        for nid, node in self.nodes.items():
            if node.type == NodeType.INPUT:
                continue
            s = np.full(n, float(node.bias or 0.0))
            for p, w in self.preds[nid]:
                s = s + w * vals[p]
            out[nid] = s
        return out

    def logit(self, X, x_raw) -> np.ndarray:
        return self.pre(X, x_raw)[self.out]

    def curves(self, X: np.ndarray, grid: np.ndarray, chunk: int = 200_000):
        """(N, G) log-odds and {unit: (N, G)} pre-activations on the grid."""
        N, G = X.shape[0], len(grid)
        L = np.empty((N, G))
        P = {u: np.empty((N, G)) for u in self.units}
        per = max(1, chunk // G)
        for s in range(0, N, per):
            Xc = X[s:s + per]
            k = Xc.shape[0]
            pr = self.pre(np.repeat(Xc, G, axis=0), np.tile(grid, k))
            L[s:s + k] = pr[self.out].reshape(k, G)
            for u in self.units:
                P[u][s:s + k] = pr[u].reshape(k, G)
        return L, P

    # ------------------------------------------------------------------ crossings
    def crossings(self, X: np.ndarray, grid: np.ndarray, P: Dict[str, np.ndarray], iters: int):
        """Every zero crossing of a unit's pre-activation along each row's curve, refined by
        bisection. Returns a list of dicts (row, unit, x, direction)."""
        cand = []
        for u in self.units:
            on = P[u] > 0
            rows, ks = np.nonzero(on[:, 1:] != on[:, :-1])
            for r, k in zip(rows, ks):
                cand.append((int(r), u, float(grid[k]), float(grid[k + 1]), bool(on[r, k + 1])))
        if not cand:
            return []
        lo = np.array([c[2] for c in cand])
        hi = np.array([c[3] for c in cand])
        rows = np.array([c[0] for c in cand])
        Xr = X[rows]
        up = np.array([c[4] for c in cand])
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            pr = self.pre(Xr, mid)
            v = np.array([pr[c[1]][i] for i, c in enumerate(cand)])
            right = (v > 0) == up          # mid is already on the far side
            hi = np.where(right, mid, hi)
            lo = np.where(right, lo, mid)
        xs = 0.5 * (lo + hi)
        return [{"row": c[0], "unit": c[1], "x": float(x), "on": c[4]} for c, x in zip(cand, xs)]

    def slope_change(self, X: np.ndarray, events: List[Dict[str, Any]], delta: float):
        if not events:
            return
        rows = np.array([e["row"] for e in events])
        x = np.array([e["x"] for e in events])
        Xr = X[rows]
        l0 = self.logit(Xr, x)
        lm = self.logit(Xr, x - delta)
        lp = self.logit(Xr, x + delta)
        for i, e in enumerate(events):
            e["slope_left"] = float((l0[i] - lm[i]) / delta)
            e["slope_right"] = float((lp[i] - l0[i]) / delta)
            e["dslope"] = e["slope_right"] - e["slope_left"]

    def states_at(self, X: np.ndarray, events: List[Dict[str, Any]], delta: float):
        """On/off state of every ReLU unit just left of each event (the pattern under which
        the crossing unit's threshold formula holds)."""
        if not events:
            return
        rows = np.array([e["row"] for e in events])
        x = np.array([e["x"] for e in events])
        pr = self.pre(X[rows], x)
        for i, e in enumerate(events):
            e["pattern"] = {u: bool(pr[u][i] > 0) for u in self.ancestors(e["unit"])}

    # ------------------------------------------------------------------ closed forms
    def ancestors(self, u: str) -> List[str]:
        """ReLU units upstream of ``u`` (any path), in topological order."""
        seen, stack = set(), [u]
        while stack:
            x = stack.pop()
            for p, w in self.preds.get(x, []):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        key = self._topo_key()
        return sorted((n for n in seen if n in self.nodes and self.nodes[n].type == NodeType.HIDDEN
                       and self.nodes[n].activation != "identity"), key=key)

    def affine(self, u: str, pattern: Dict[str, bool]) -> Tuple[float, Dict[Any, float]]:
        """Pre-activation of ``u`` as const + sum(coef * symbol) under ``pattern``; symbols are
        ``"x"`` (the varied input, real units) and data-column indices (real units) of the
        other inputs (a held copy of the input itself is its own column)."""
        memo: Dict[str, Tuple[float, Dict[Any, float]]] = {}
        m = self.m

        def val(n: str):
            if n in memo:
                return memo[n]
            node = self.nodes[n]
            if node.type == NodeType.INPUT:
                c = input_column(n)
                mu, sd = float(m.mean[c]), float(m.scale[c])
                sym = "x" if n in self.varied else c
                r = (-mu / sd, {sym: 1.0 / sd})
            else:
                c0, co = pre_of(n)
                if node.type == NodeType.HIDDEN and node.activation != "identity" and not pattern.get(n, True):
                    r = (0.0, {})
                else:
                    r = (c0, co)
            memo[n] = r
            return r

        def pre_of(n: str):
            c0 = float(self.nodes[n].bias or 0.0)
            co: Dict[Any, float] = {}
            for p, w in self.preds[n]:
                pc, pco = val(p)
                c0 += w * pc
                for k, v in pco.items():
                    co[k] = co.get(k, 0.0) + w * v
            return c0, co

        return pre_of(u)

    def threshold_formula(self, u: str, pattern: Dict[str, bool]) -> Optional[Dict[str, Any]]:
        c0, co = self.affine(u, pattern)
        cx = co.get("x", 0.0)
        if abs(cx) < 1e-15:
            return None
        terms = {k: -v / cx for k, v in co.items() if k != "x" and abs(v) > 1e-15}
        return {"const": -c0 / cx, "terms": terms, "slope_in_x": cx,
                "direction": "on above" if cx > 0 else "on below"}

    def eval_formula(self, f: Dict[str, Any], raw_row: np.ndarray) -> float:
        return float(f["const"] + sum(v * raw_row[k] for k, v in f["terms"].items()))

    def condition_formula(self, a: str, pattern: Dict[str, bool]) -> Tuple[float, Dict[Any, float]]:
        return self.affine(a, pattern)


# ============================================================================ text helpers

def _col_term(m, c: int) -> Dict[str, Any]:
    """How a data column reads in a formula: a number (with unit) or a 0/1 category flag."""
    name = m.feature_names[c]
    base = m.base_of_col.get(c, name)
    v = m.var_by_base.get(base)
    if v is not None and v.is_category():
        code = v.codes[v.columns.index(c)]
        return {"column": name, "kind": "flag", "text": f"[{v.label} = {v.level(code)['label']}]",
                "short": f"[{name}]"}
    unit = (v.unit if v is not None else None) or ""
    return {"column": name, "kind": "number", "text": v.label if v is not None else name,
            "short": name, "unit": unit, "decimals": v.decimals if v is not None else 2}


def formula_text(m, f: Dict[str, Any], x_label: str, x_unit: str = "", short: bool = False,
                 held_label: Optional[str] = None) -> str:
    parts = [f"{_sig(f['const'], 4)}"]
    for c, v in sorted(f["terms"].items(), key=lambda kv: -abs(kv[1])):
        t = _col_term(m, c)
        name = t["short"] if short else t["text"]
        sign = "−" if v < 0 else "+"
        if t["kind"] == "flag":
            parts.append(f"{sign} {_sig(abs(v), 3)}·{name}")
        else:
            parts.append(f"{sign} {_sig(abs(v), 3)}·{name}")
    return f"{x_label}* = " + " ".join(parts) + (f"  ({x_unit})" if x_unit else "")


def inequality_text(m, c0: float, co: Dict[Any, float], x_label: str = "x",
                    negate: bool = False) -> str:
    """``c0 + sum co·sym > 0`` in readable form (``<= 0`` with ``negate``); a single-variable
    condition is solved for that variable."""
    co = {k: v for k, v in co.items() if abs(v) > 1e-15}
    if negate:
        c0, co = -c0, {k: -v for k, v in co.items()}
    if not co:
        return "always" if c0 > 0 else "never"
    if len(co) == 1:
        (k, v), = co.items()
        t = -c0 / v
        if k == "x":
            name, unit, dec = x_label, "", 1
        else:
            tt = _col_term(m, k)
            if tt["kind"] == "flag":
                # c0 + v*flag > 0
                on, off = c0 + v > 0, c0 > 0
                if on and off:
                    return "always"
                if not on and not off:
                    return "never"
                return f"{tt['text']} is {'1' if on else '0'}"
            name, unit, dec = tt["text"], tt.get("unit", ""), tt.get("decimals", 2)
        op = ">" if v > 0 else "<"
        return f"{name} {op} {t:.{max(dec, 1)}f}{(' ' + unit) if unit else ''}"
    terms = []
    for k, v in sorted(co.items(), key=lambda kv: -abs(kv[1])):
        name = x_label if k == "x" else _col_term(m, k)["text"]
        terms.append(f"{'−' if v < 0 else '+'} {_sig(abs(v), 3)}·{name}")
    return f"{_sig(c0, 4)} " + " ".join(terms) + " > 0"


# ============================================================================ parts (annotations)

def unit_parts(m) -> Dict[str, List[Dict[str, Any]]]:
    """For every node, the annotations that hold it as an own (non-entry) node, smallest first."""
    anns = annotation_dicts_from_operations(m.ops)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for a in anns:
        try:
            unit = as_unit(a, anns, m.structure)
        except Exception:   # noqa: BLE001 - a malformed annotation never stops the dossier
            unit = a
        entries = {str(e) for e in unit.get("entry_nodes", [])}
        own = [str(n) for n in unit.get("subgraph_nodes", []) if str(n) not in entries]
        hyp = (a.get("hypothesis") or "").strip()
        first = hyp.split(". ")[0].strip() + ("." if hyp and not hyp.split(". ")[0].endswith(".") else "")
        for n in own:
            out.setdefault(n, []).append({"id": a["id"], "name": a["name"], "size": len(own),
                                          "kind": "composition" if a.get("child_annotation_ids") else "leaf",
                                          "hypothesis_first_sentence": first[:400] if hyp else None})
    for n in out:
        out[n].sort(key=lambda d: d["size"])
    return out


# ============================================================================ the dossier

def _token(e: Dict[str, Any]) -> str:
    return f"{e['unit']}{'+' if e['on'] else '-'}"


WORD = {"up": "rises", "flat": "flat", "down": "falls"}


def _shape(es: List[Dict[str, Any]], lo: float, hi: float, s0: float, cfg: DossierConfig,
           flat_slope: float):
    """Segments between the visible bends of one curve, grouped into **pieces** (consecutive
    segments that read the same: rises / flat / falls; segments shorter than ``min_segment``
    of the range join their neighbour) and the qualitative shape words (one per piece)."""
    R = hi - lo
    xs = [lo] + [e["x"] for e in es] + [hi]
    slopes = [s0] + [e["slope_right"] for e in es]
    segs = []
    for i, s in enumerate(slopes):
        a, b = xs[i], xs[i + 1]
        w = "flat" if abs(s) * R < flat_slope else ("up" if s > 0 else "down")
        segs.append({"from": a, "to": b, "slope": s, "word": w,
                     "ends_with": _token(es[i]) if i < len(es) else None})
    long_ = [(sg["to"] - sg["from"]) >= cfg.min_segment * R for sg in segs]
    if not any(long_):
        long_ = [True] * len(segs)
    first_word = next(sg["word"] for sg, ok in zip(segs, long_) if ok)
    pieces: List[Dict[str, Any]] = []
    for sg, ok in zip(segs, long_):
        w = sg["word"] if ok else (pieces[-1]["word"] if pieces else first_word)
        dl = sg["slope"] * (sg["to"] - sg["from"])
        if pieces and pieces[-1]["word"] == w:
            p = pieces[-1]
            if p["ends_with"]:
                p["bends_inside"].append(p["ends_with"])
            p["to"], p["dlogit"], p["ends_with"] = sg["to"], p["dlogit"] + dl, sg["ends_with"]
        else:
            pieces.append({"word": w, "from": sg["from"], "to": sg["to"], "dlogit": dl,
                           "ends_with": sg["ends_with"], "bends_inside": []})
    return segs, tuple(p["word"] for p in pieces), pieces


def _family_pieces(rows, rows_info, words, L, Prob, grid, typ, raw):
    """Per piece of the family's shape: median start/end, log-odds change, the bend that ends
    it and the bends inside it (over members whose own shape is exactly the family's), and
    the typical member's predicted risk at the ends."""
    exact = [r for r in rows if rows_info[r]["words"] == tuple(words)]
    out = []
    for k, w in enumerate(words):
        ps = [rows_info[r]["pieces"][k] for r in exact]
        if not ps:
            continue
        ends: Dict[str, int] = {}
        inside: Dict[str, int] = {}
        for p in ps:
            if p["ends_with"]:
                ends[p["ends_with"]] = ends.get(p["ends_with"], 0) + 1
            for t in p["bends_inside"]:
                inside[t] = inside.get(t, 0) + 1
        tp = rows_info[typ]["pieces"][k] if rows_info[typ]["words"] == tuple(words) else None
        risk = None
        if tp is not None:
            ia = int(np.argmin(np.abs(grid - tp["from"])))
            ib = int(np.argmin(np.abs(grid - tp["to"])))
            risk = {"from_x": _sig(tp["from"], 4), "to_x": _sig(tp["to"], 4),
                    "risk_from": _sig(float(Prob[typ, ia]), 3), "risk_to": _sig(float(Prob[typ, ib]), 3)}
        occ = int(sum(1 for r in exact if rows_info[r]["pieces"][k]["from"] <= raw[r] <= rows_info[r]["pieces"][k]["to"]))
        out.append({"piece": k + 1, "word": w, "reads": WORD[w], "n_exact": len(exact),
                    "members_whose_own_value_is_on_this_piece": occ,
                    "from_median": _sig(np.median([p["from"] for p in ps]), 4),
                    "to_median": _sig(np.median([p["to"] for p in ps]), 4),
                    "from_p10_p90": [_sig(np.percentile([p["from"] for p in ps], 10), 4),
                                     _sig(np.percentile([p["from"] for p in ps], 90), 4)],
                    "to_p10_p90": [_sig(np.percentile([p["to"] for p in ps], 10), 4),
                                   _sig(np.percentile([p["to"] for p in ps], 90), 4)],
                    "logodds_change_median": _sig(np.median([p["dlogit"] for p in ps]), 3),
                    "ends_with": dict(sorted(ends.items(), key=lambda kv: -kv[1])),
                    "bends_inside": dict(sorted(inside.items(), key=lambda kv: -kv[1])),
                    "typical_member": risk})
    return out


def build_dossier(m, input_name: str, annotation: Optional[str] = None,
                  cfg: Optional[DossierConfig] = None, meta: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """The feature dossier (a JSON-ready dict) plus private arrays under ``"_arrays"`` for the
    figures (removed by :func:`write_dossier`)."""
    cfg = cfg or DossierConfig()
    v = resolve_variable(m, input_name)
    parts_of = unit_parts(m)
    scope = None
    scope_part = None
    if annotation:
        anns = annotation_dicts_from_operations(m.ops)
        a = next((x for x in anns if annotation in (x["name"], x["id"])), None)
        if a is None:
            raise ValueError(f"annotation {annotation!r} not found")
        unit = as_unit(a, anns, m.structure)
        scope = [str(e) for e in unit["entry_nodes"]]
        scope_part = a["name"]
    fa = FeatureAnalyser(m, v, scope)
    raw = np.asarray(v.raw_train, dtype=float)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    R = hi - lo
    grid = np.linspace(lo, hi, cfg.grid_points)
    X = m.X
    N = X.shape[0]
    y = m.y
    unit_label = v.unit or ""
    xl = v.label

    # ---------------------------------------------------------------- curves
    L, P = fa.curves(X, grid)
    Prob = m.risk(L)
    med_z = np.median(X, axis=0)
    Lmed, Pmed_units = fa.curves(med_z[None, :], grid)
    pd_prob = Prob.mean(axis=0)
    pd_logit = L.mean(axis=0)
    own_logit = fa.logit(X, raw) if scope is None else m.base_logit
    sn_check = _structure_network_check(m, fa, med_z, grid, Lmed[0]) if scope is None else None

    # ---------------------------------------------------------------- bends
    delta = 1e-4 * R
    events = fa.crossings(X, grid, P, cfg.bisect_iters)
    fa.slope_change(X, events, delta)
    fa.states_at(X, events, delta)
    for e in events:
        e["visible"] = bool(abs(e["dslope"]) * R >= cfg.min_bend)
    by_row: Dict[int, List[Dict[str, Any]]] = {r: [] for r in range(N)}
    for e in events:
        by_row[e["row"]].append(e)
    for r in by_row:
        by_row[r].sort(key=lambda e: e["x"])
    # every numerical kink is one of the located crossings
    d2 = np.abs(np.diff(L, 2, axis=1))
    tol = 1e-7 * max(1.0, float(np.max(np.abs(L))))
    unattributed = 0
    h = grid[1] - grid[0]
    for r, k in zip(*np.nonzero(d2 > tol)):
        a_, b_ = grid[k] - 1e-9, grid[k + 2] + 1e-9
        if not any(a_ <= e["x"] <= b_ for e in by_row[int(r)]):
            unattributed += 1

    # ---------------------------------------------------------------- unit thresholds
    units_out = []
    formula_err = 0.0
    for u in fa.units:
        evs = [e for e in events if e["unit"] == u]
        pats: Dict[Tuple, Dict[str, Any]] = {}
        for e in evs:
            key = tuple(sorted(e["pattern"].items()))
            d = pats.setdefault(key, {"pattern": dict(e["pattern"]), "events": []})
            d["events"].append(e)
        pat_out = []
        for key, d in sorted(pats.items(), key=lambda kv: -len(kv[1]["events"])):
            f = fa.threshold_formula(u, d["pattern"])
            errs = [abs(fa.eval_formula(f, m.raw_train[e["row"]]) - e["x"]) for e in d["events"]] if f else []
            if errs:
                formula_err = max(formula_err, max(errs))
            conds = []
            for a_, on in d["pattern"].items():
                c0, co = fa.condition_formula(a_, d["pattern"])
                conds.append({"unit": a_, "on": on, "on_when": inequality_text(m, c0, co, xl),
                              "holds": inequality_text(m, c0, co, xl, negate=not on)})
            pat_out.append({
                "upstream_states": {a_: ("on" if on else "off") for a_, on in d["pattern"].items()},
                "upstream_conditions": conds,
                "n_crossings": len(d["events"]),
                "formula": None if f is None else {
                    "const": _sig(f["const"], 6),
                    "terms": {m.feature_names[k]: _sig(c, 6) for k, c in f["terms"].items()},
                    "text": formula_text(m, f, xl, unit_label),
                    "text_columns": formula_text(m, f, v.base, unit_label, short=True),
                    "unit_on": "above the threshold" if f["slope_in_x"] > 0 else "below the threshold"},
                "max_abs_err_vs_crossings": _sig(max(errs), 3) if errs else None,
            })
        xs = np.array([e["x"] for e in evs])
        ds = np.array([e["dslope"] for e in evs])
        direct = [(p, w) for p, w in fa.preds[u] if p in fa.varied]
        via = [p for p, w in fa.preds[u] if p in fa.downstream or p in fa.units]
        pre_all = P[u]
        units_out.append({
            "unit": u,
            "parts": parts_of.get(u, []),
            "part": (parts_of.get(u) or [{}])[0].get("name"),
            "reached_directly_from": [{"copy": p, "weight": _sig(w, 4)} for p, w in direct],
            "reached_through": via,
            "n_patients_switching_in_range": int(len({e["row"] for e in evs})),
            "n_on_throughout": int(np.sum(np.all(pre_all > 0, axis=1))),
            "n_off_throughout": int(np.sum(np.all(pre_all <= 0, axis=1))),
            "switch": ("on" if all(e["on"] for e in evs) else "off" if evs and not any(e["on"] for e in evs)
                       else "both" if evs else None),
            "threshold_quantiles": None if not len(xs) else {
                q: _sig(np.percentile(xs, p), 4) for q, p in (("min", 0), ("p10", 10), ("median", 50),
                                                                ("p90", 90), ("max", 100))},
            "dslope_median": _sig(np.median(ds), 4) if len(ds) else None,
            "bend_logodds_over_range_median": _sig(np.median(ds) * R, 4) if len(ds) else None,
            "patterns": pat_out,
        })

    # ---------------------------------------------------------------- signatures and shapes
    flat_slope = cfg.flat_logodds
    rows_info = []
    for r in range(N):
        es = [e for e in by_row[r] if e["visible"]]
        s0 = float((L[r, 1] - L[r, 0]) / h)
        segs, words, pieces = _shape(es, lo, hi, s0, cfg, flat_slope)
        rows_info.append({"tokens": tuple(_token(e) for e in es), "words": words, "segs": segs,
                          "pieces": pieces, "events": es})
    fam_of, families = _families([ri["words"] for ri in rows_info], cfg)
    labels = np.array(fam_of)

    # k-means cross-check
    kmeans = _kmeans_check(L, labels, len(families), cfg.seed)

    # ---------------------------------------------------------------- features for conditions
    feats, fnames, fkinds = _condition_features(m, v, include_self=scope is not None)
    tree_all = _tree(feats, fnames, fkinds, labels, cfg, [f["key"] for f in families])

    # ---------------------------------------------------------------- per family
    edges = _bands(raw, cfg.n_bands, v.decimals)
    test = _test_assign(m, fa, v, grid, cfg, flat_slope, families) if m.Xt is not None else None
    pred_own = m.risk(m.base_logit)
    fam_out = []
    for fi, f in enumerate(families):
        rows = np.nonzero(labels == fi)[0]
        tok_counts: Dict[Tuple, int] = {}
        for r in rows:
            tok_counts[rows_info[r]["tokens"]] = tok_counts.get(rows_info[r]["tokens"], 0) + 1
        variants = [{"bends_in_order": list(k), "n": c} for k, c in
                    sorted(tok_counts.items(), key=lambda kv: -kv[1])]
        # bends: every token in the family's patients
        bends = {}
        for r in rows:
            for e in rows_info[r]["events"]:
                bends.setdefault(_token(e), []).append(e)
        bend_out = []
        for tkn, es in sorted(bends.items(), key=lambda kv: np.median([e["x"] for e in kv[1]])):
            xs = np.array([e["x"] for e in es])
            q10, q90 = np.percentile(xs, 10), np.percentile(xs, 90)
            w = cfg.support_window * R
            a_, b_ = q10 - w, q90 + w
            own = raw[[e["row"] for e in es]]
            in_region_fam = int(np.sum((raw[rows] >= a_) & (raw[rows] <= b_)))
            in_region_all = int(np.sum((raw >= a_) & (raw <= b_)))
            u = es[0]["unit"]
            beyond = int(np.sum(own > xs))
            bend_out.append({
                "bend": tkn, "unit": u, "part": (parts_of.get(u) or [{}])[0].get("name"),
                "unit_switches": "on" if es[0]["on"] else "off",
                "n_patients_with_bend": len(es), "share_of_family": _sig(len(es) / len(rows), 3),
                "x_median": _sig(np.median(xs), 4), "x_p10": _sig(q10, 4), "x_p90": _sig(q90, 4),
                "dslope_median": _sig(np.median([e["dslope"] for e in es]), 4),
                "bend_logodds_over_range": _sig(np.median([e["dslope"] for e in es]) * R, 4),
                "region": [_sig(a_, 4), _sig(b_, 4)],
                "family_patients_with_value_in_region": in_region_fam,
                "all_training_patients_in_region": in_region_all,
                "family_patients_past_their_own_bend": beyond,
                "thin": bool(in_region_fam < cfg.thin_support),
            })
        # where the family's patients sit on their own curves
        seg_occ: Dict[int, int] = {}
        for r in rows:
            k = int(sum(1 for e in rows_info[r]["events"] if e["x"] < raw[r]))
            seg_occ[k] = seg_occ.get(k, 0) + 1
        med_L = np.median(L[rows], axis=0)
        med_P = np.median(Prob[rows], axis=0)
        exact = np.array([r for r in rows if rows_info[r]["words"] == tuple(f["words"])] or list(rows))
        typ = int(exact[np.argmin(np.sum((L[exact] - med_L) ** 2, axis=1))])
        pieces = _family_pieces(rows, rows_info, f["words"], L, Prob, grid, typ, raw)
        segments = _family_segments(bend_out, med_L, med_P, grid, lo, hi, flat_slope, R)
        cond = _family_rule(feats, fnames, fkinds, labels == fi, cfg)
        outcome = _outcome_bands(raw[rows], y[rows], pred_own[rows], edges)
        bend_check = _bend_check(raw[rows], y[rows], pred_own[rows], bend_out)
        fam_out.append({
            "family": f["key"], "shape": list(f["words"]), "shape_text": " → ".join(WORD[w] for w in f["words"]),
            "n": int(len(rows)), "share": _sig(len(rows) / N, 3), "merged_from": f["merged_from"],
            "variants": variants, "pieces": pieces, "bends": bend_out, "segments_of_median_curve": segments,
            "patients_by_own_segment": {str(k): c for k, c in sorted(seg_occ.items())},
            "typical_patient_row": typ,
            "typical_patient_values": _row_values(m, m.raw_train[typ], v),
            "characteristic_curve": {"x": [_sig(g, 5) for g in grid[::10]],
                                     "logit_median": [_sig(z, 4) for z in med_L[::10]],
                                     "risk_median": [_sig(z, 4) for z in med_P[::10]]},
            "conditions": cond,
            "outcome_by_band_train": outcome,
            "bend_data_check_train": bend_check,
            "outcome_by_band_test": None if test is None else _outcome_bands(
                test["raw"][test["labels"] == fi], m.yt[test["labels"] == fi],
                test["pred"][test["labels"] == fi], edges),
            "n_test": None if test is None else int(np.sum(test["labels"] == fi)),
        })

    # ---------------------------------------------------------------- median-others curve
    med_events = fa.crossings(med_z[None, :], grid, {u: Pmed_units[u] for u in fa.units}, cfg.bisect_iters)
    fa.slope_change(med_z[None, :], med_events, delta)
    med_raw = med_z * m.scale + m.mean

    out = {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "kind": "feature_dossier",
        "meta": meta or {},
        "input": {"base": v.base, "label": v.label, "unit": v.unit, "column": m.feature_names[fa.col],
                  "copies_varied": fa.varied, "copies_held": fa.held_copies,
                  "scope_annotation": scope_part,
                  "observed_range_train": [lo, hi], "grid_points": cfg.grid_points,
                  "train_quantiles": {q: _sig(np.percentile(raw, p), 4) for q, p in
                                      (("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))},
                  "decoding": "raw = z * scale + mean (split StandardScaler)",
                  "scale": fa.scale, "mean": fa.mean},
        "config": asdict(cfg),
        "data": {"split": "train", "n_train": int(N), "n_test": None if m.Xt is None else int(m.Xt.shape[0]),
                 "outcome_rate_train": _sig(float(np.mean(y)), 3), "bands": edges},
        "checks": {
            "kinks_unattributed": int(unattributed),
            "n_crossings": len(events),
            "threshold_formula_max_abs_err": _sig(formula_err, 3),
            "structure_network_median_curve_max_abs_err": sn_check,
            "note": ("every kink of every training curve (second difference above tolerance on the "
                     "grid) lies within one grid step of a located unit crossing; threshold formulas "
                     "are checked against the bisection crossings"),
        },
        "units": units_out,
        "median_others_curve": {
            "definition": "every other column at its training median (z), as the evidence panel's line plot",
            "values_held": _row_values(m, med_raw, v),
            "x": [_sig(g, 5) for g in grid[::5]], "logit": [_sig(z, 4) for z in Lmed[0][::5]],
            "risk": [_sig(z, 4) for z in m.risk(Lmed[0])[::5]],
            "bends": [{"bend": _token(e), "unit": e["unit"], "part": (parts_of.get(e["unit"]) or [{}])[0].get("name"),
                       "x": _sig(e["x"], 4), "dslope": _sig(e["dslope"], 4),
                       "risk_at": _sig(float(m.risk(fa.logit(med_z[None, :], np.array([e["x"]])))[0]), 3)}
                      for e in sorted(med_events, key=lambda e: e["x"])],
        },
        "partial_dependence": {"x": [_sig(g, 5) for g in grid[::5]],
                               "risk_mean": [_sig(z, 4) for z in pd_prob[::5]],
                               "logit_mean": [_sig(z, 4) for z in pd_logit[::5]]},
        "families": fam_out,
        "family_tree": tree_all,
        "kmeans_check": kmeans,
        "rare_shapes_merged": [f for f in families if f["merged_from"]],
    }
    out["_arrays"] = {"grid": grid, "L": L, "Prob": Prob, "labels": labels, "raw": raw, "y": y,
                      "events": events, "Lmed": Lmed[0], "pd": pd_prob, "med_events": med_events,
                      "units": fa.units, "P": P, "test": test}
    return out


# ============================================================================ pieces

def _structure_network_check(m, fa, med_z, grid, Lmed) -> Optional[float]:
    """Median-others curve through StructureNetwork (the evidence panel's engine)."""
    try:
        import torch
        from ..core.structure_network import StructureNetwork
        sn = StructureNetwork(m.structure)
        sn.override_output_activation(sn.model_output_activation)
        Xg = np.tile(med_z, (len(grid), 1))
        Xg[:, fa.col] = fa.z_of(grid)
        with torch.no_grad():
            out = sn.forward(torch.tensor(Xg, dtype=torch.float64)).numpy()
        out = out[:, 0] if out.ndim == 2 else out
        return _sig(float(np.max(np.abs(out - m.risk(Lmed)))), 3)
    except Exception as exc:   # noqa: BLE001
        return f"not run: {type(exc).__name__}: {exc}"


def _families(words: List[Tuple[str, ...]], cfg: DossierConfig):
    counts: Dict[Tuple[str, ...], int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda k: (-counts[k], k))
    keep = [k for k in order if counts[k] >= cfg.min_family][:cfg.max_families] or order[:1]
    target = {}
    for k in order:
        if k in keep:
            target[k] = k
        else:
            target[k] = min(keep, key=lambda q: (_levenshtein(k, q), -counts[q]))
    fams = [{"key": f"F{i + 1}", "words": k, "merged_from": []} for i, k in enumerate(keep)]
    idx = {k: i for i, k in enumerate(keep)}
    for k in order:
        if target[k] != k:
            fams[idx[target[k]]]["merged_from"].append({"shape": list(k), "n": counts[k]})
    return [idx[target[w]] for w in words], fams


def _kmeans_check(L, labels, k, seed):
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        Z = np.diff(L, axis=1)    # slopes: shape without the patient's offset
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        return {"method": "k-means on the slope profile of each log-odds curve (offset removed), "
                          "k = number of families", "k": int(k),
                "adjusted_rand_vs_families": _sig(adjusted_rand_score(labels, km.labels_), 3)}
    except Exception as exc:   # noqa: BLE001
        return {"error": str(exc)}


def _condition_features(m, v, include_self: bool = False):
    feats, names, kinds = [], [], []
    for w in m.variables:
        if w.base == v.base and not include_self:
            continue
        if w.is_category():
            for c, code in zip(w.columns, w.codes):
                feats.append((m.raw_train[:, c] > 0.5).astype(float))
                names.append(f"{w.label} = {w.level(code)['label']}")
                kinds.append({"kind": "flag", "column": m.feature_names[c]})
        else:
            feats.append(m.raw_train[:, w.columns[0]].astype(float))
            names.append(w.label)
            kinds.append({"kind": "number", "column": m.feature_names[w.columns[0]],
                          "unit": w.unit or "", "decimals": w.decimals})
    return np.column_stack(feats) if feats else np.zeros((m.X.shape[0], 0)), names, kinds


def _cond_text(name, kind, op, t):
    if kind["kind"] == "flag":
        return name if op == ">" else name.replace(" = ", " ≠ ", 1)
    d = kind.get("decimals", 2)
    t = math.floor(t) + 0.5 if d == 0 and abs(t - round(t)) > 1e-9 else t
    unit = (" " + kind["unit"]) if kind.get("unit") else ""
    return f"{name} {op} {t:.{max(d, 0) if d else 1}f}{unit}" if d else f"{name} {op} {t:g}{unit}"


def _paths(tree, names, kinds):
    t = tree.tree_
    out = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            out.append((node, conds))
            return
        f, thr = t.feature[node], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<", thr)])
        walk(t.children_right[node], conds + [(f, ">", thr)])

    walk(0, [])
    res = []
    for node, conds in out:
        # simplify: keep the tightest bound per feature and side
        best: Dict[Tuple[int, str], float] = {}
        for f, op, thr in conds:
            k = (f, op)
            if k not in best or (op == "<" and thr < best[k]) or (op == ">" and thr > best[k]):
                best[k] = thr
        texts = []
        for (f, op), thr in sorted(best.items()):
            texts.append(_cond_text(names[f], kinds[f], "<" if op == "<" else ">", thr))
        res.append((node, texts))
    return res


def _tree(feats, names, kinds, labels, cfg, keys):
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:   # noqa: BLE001
        return {"error": str(exc)}
    if feats.shape[1] == 0 or len(set(labels)) < 2:
        return {"note": "one family; no tree"}
    clf = DecisionTreeClassifier(max_depth=cfg.tree_depth, min_samples_leaf=cfg.tree_min_leaf,
                                 random_state=cfg.seed).fit(feats, labels)
    acc = float(clf.score(feats, labels))
    ncls = np.bincount(labels)
    cv = None
    nfold = int(min(5, ncls[ncls > 0].min())) if (ncls > 0).any() else 0
    if nfold >= 2:
        cv = float(np.mean(cross_val_score(
            DecisionTreeClassifier(max_depth=cfg.tree_depth, min_samples_leaf=cfg.tree_min_leaf,
                                   random_state=cfg.seed), feats, labels,
            cv=StratifiedKFold(nfold, shuffle=True, random_state=cfg.seed))))
    leaves = []
    t = clf.tree_
    for node, conds in _paths(clf, names, kinds):
        vals = t.value[node][0]
        cnt = vals / vals.sum() * t.n_node_samples[node] if vals.sum() > 0 else vals
        k = int(np.argmax(vals))
        leaves.append({"if": conds, "family": keys[clf.classes_[k]], "n": int(t.n_node_samples[node]),
                       "purity": _sig(float(vals[k] / vals.sum()), 3),
                       "counts": {keys[clf.classes_[i]]: int(round(c)) for i, c in enumerate(cnt) if round(c)}})
    return {"method": f"decision tree, depth <= {cfg.tree_depth}, leaves >= {cfg.tree_min_leaf} "
                      "patients, on the other inputs in real units (training patients)",
            "accuracy_train": _sig(acc, 3), "accuracy_cv": None if cv is None else _sig(cv, 3),
            "cv_folds": nfold, "leaves": leaves}


def _family_rule(feats, names, kinds, member, cfg):
    """One-vs-rest shallow tree: the leaves that hold the family, with precision and recall."""
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception as exc:   # noqa: BLE001
        return {"error": str(exc)}
    member = np.asarray(member, dtype=bool)
    if feats.shape[1] == 0 or member.all() or not member.any():
        return {"rules": [], "note": "no contrast"}
    clf = DecisionTreeClassifier(max_depth=cfg.tree_depth, min_samples_leaf=cfg.tree_min_leaf,
                                 random_state=cfg.seed, class_weight="balanced").fit(feats, member)
    leaf_of = clf.apply(feats)
    rules = []
    for node, conds in _paths(clf, names, kinds):
        inleaf = leaf_of == node
        k = int(np.sum(inleaf & member))
        n = int(np.sum(inleaf))
        if n == 0 or k / n < 0.5:
            continue
        rules.append({"if": conds or ["(all patients)"], "n": n, "members": k,
                      "precision": _sig(k / n, 3), "recall": _sig(k / member.sum(), 3)})
    rules.sort(key=lambda r: -r["members"])
    pred = clf.predict(feats).astype(bool)
    return {"method": "one-vs-rest decision tree (balanced), the leaves where the family is the majority",
            "rules": rules,
            "balanced_accuracy_train": _sig(0.5 * (np.mean(pred[member]) + np.mean(~pred[~member])), 3)}


def _bands(raw, n_bands, decimals):
    qs = np.unique(np.round(np.percentile(raw, np.linspace(0, 100, n_bands + 1)[1:-1]), decimals))
    edges = [float(np.min(raw))] + [float(q) for q in qs] + [float(np.max(raw))]
    bands = []
    for i in range(len(edges) - 1):
        bands.append({"from": edges[i], "to": edges[i + 1],
                      "label": (f"≤ {edges[i + 1]:g}" if i == 0 else
                                f"> {edges[i]:g}" if i == len(edges) - 2 else
                                f"{edges[i]:g}–{edges[i + 1]:g}")})
    return bands


def _band_index(x, bands):
    idx = np.zeros(len(x), dtype=int)
    for i, b in enumerate(bands[1:], 1):
        idx[x > b["from"]] = i
    return idx


def _outcome_bands(x, y, pred, bands):
    x = np.asarray(x, dtype=float)
    idx = _band_index(x, bands)
    out = []
    for i, b in enumerate(bands):
        sel = idx == i
        n, k = int(sel.sum()), int(np.sum(np.asarray(y)[sel] > 0.5))
        out.append({"band": b["label"], "n": n, "events": k,
                    "rate": _sig(k / n, 3) if n else None, "rate_ci95": _wilson(k, n),
                    "model_mean_risk": _sig(float(np.mean(pred[sel])), 3) if n else None})
    return out


def _bend_check(x, y, pred, bends):
    out = []
    for b in bends:
        t = b["x_median"]
        lo_, hi_ = x <= t, x > t
        res = {"bend": b["bend"], "at": t}
        for name, sel in (("below", lo_), ("above", hi_)):
            n, k = int(sel.sum()), int(np.sum(y[sel] > 0.5))
            res[name] = {"n": n, "events": k, "rate": _sig(k / n, 3) if n else None,
                         "rate_ci95": _wilson(k, n),
                         "model_mean_risk": _sig(float(np.mean(pred[sel])), 3) if n else None}
        out.append(res)
    return out


def _family_segments(bends, med_L, med_P, grid, lo, hi, flat_slope, R):
    """The characteristic curve cut at the family's median bend positions."""
    cuts = [lo] + sorted({b["x_median"] for b in bends if b["share_of_family"] >= 0.5}) + [hi]
    by_x = {}
    for b in bends:
        if b["share_of_family"] >= 0.5:
            by_x.setdefault(b["x_median"], []).append(b)
    segs = []
    for a, c in zip(cuts[:-1], cuts[1:]):
        ia, ic = int(np.argmin(np.abs(grid - a))), int(np.argmin(np.abs(grid - c)))
        if ic <= ia:
            continue
        dl = float(med_L[ic] - med_L[ia])
        s = dl / max(c - a, 1e-12)
        w = "flat" if abs(s) * R < flat_slope else ("up" if s > 0 else "down")
        segs.append({"from": _sig(a, 4), "to": _sig(c, 4), "word": w,
                     "logodds_change": _sig(dl, 3), "slope_per_unit": _sig(s, 3),
                     "risk_from": _sig(float(med_P[ia]), 3), "risk_to": _sig(float(med_P[ic]), 3),
                     "ends_with": [b["bend"] for b in by_x.get(c, [])]})
    return segs


def _row_values(m, raw_row, v) -> Dict[str, str]:
    out = {}
    for w in m.variables:
        if w.base == v.base:
            continue
        if w.is_category():
            code = next((cd for c, cd in zip(w.columns, w.codes) if raw_row[c] > 0.5), w.implicit_off)
            out[w.label] = w.level(code)["label"] if code is not None else "none"
        else:
            out[w.label] = w.fmt(raw_row[w.columns[0]])
    return out


def _test_assign(m, fa, v, grid, cfg, flat_slope, families):
    """Test patients: curves, shapes and family by the same rule (labelled check only)."""
    Xt = m.Xt
    Lt, Pt = fa.curves(Xt, grid)
    lo, hi = float(grid[0]), float(grid[-1])
    R = hi - lo
    ev = fa.crossings(Xt, grid, Pt, cfg.bisect_iters)
    fa.slope_change(Xt, ev, 1e-4 * R)
    by: Dict[int, List[Dict[str, Any]]] = {r: [] for r in range(Xt.shape[0])}
    for e in ev:
        if abs(e["dslope"]) * R >= cfg.min_bend:
            by[e["row"]].append(e)
    keys = [f["words"] for f in families]
    labels = []
    h = grid[1] - grid[0]
    for r in range(Xt.shape[0]):
        es = sorted(by[r], key=lambda e: e["x"])
        _, words, _p = _shape(es, lo, hi, float((Lt[r, 1] - Lt[r, 0]) / h), cfg, flat_slope)
        if words in keys:
            labels.append(keys.index(words))
        else:
            labels.append(min(range(len(keys)), key=lambda i: _levenshtein(words, keys[i])))
    raw_t = np.asarray(v.raw_test, dtype=float)
    return {"labels": np.array(labels), "raw": raw_t, "pred": m.risk(m.base_logit_test), "L": Lt}
