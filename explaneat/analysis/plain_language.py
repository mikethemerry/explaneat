"""Plain-language explanation layer: structured claims, checked against the model and data.

Every plain-language statement about an explained genome rests on one or more **claims**.
A claim is derived from an annotation (a "part" of the explained network), computed on the
model state and the split's data, and verified automatically. Only verified claims may be
turned into sentences; the others are listed as "not supported by the model".

Claim types (``CLAIM_TYPES``):

* ``direction``: the model treats a higher value of X (or category L rather than the
  reference category) as raising or lowering the risk. Checked: the sign of X's effect on the
  model's log-odds for every patient, its mean with a bootstrap 95% interval, the share of
  patients with the opposite sign, and whether the average curve is monotone.
* ``threshold``: through a part, X only matters once it rises above (or while it stays below)
  about T. Checked: the fold of each patient's curve (where the part's effect starts or
  stops), the median fold and its spread, and the size of the effect past the fold.
* ``interaction``: X's effect differs with Y. Checked: X's effect within each Y group (or high
  vs low Y), the difference with a stratified bootstrap interval, and the sign of the
  counterfactual difference-in-differences through the part itself.
* ``magnitude``: the odds ratio of a change in X for a stated reference patient (median of
  each number, most common category), with a bootstrap interval over the reference patient.
* ``subgroup``: the share of patients a gated part acts on, who they are, and its effect on
  them.
* ``null``: the model ignores Z (exact Null rule: no path to the output over enabled,
  non-zero-weight edges, confirmed numerically), or a part barely changes the predictions
  (ablation).

Effects are on the model's log-odds (the output's pre-activation); odds ratios are
``exp`` of it, "on the model's odds scale". They describe what the model does, not the world.

Everything is deterministic: bootstraps use ``numpy.random.default_rng(SEED)`` afresh for every
claim, so a claim does not depend on which claims were computed before it.

Units: data columns are z-scored by the split's ``StandardScaler`` (``scaler_params`` mean and
scale, fitted on the training rows). Values are converted back with ``raw = z * scale + mean``
and the round trip against the stored raw data is checked and recorded. One-hot columns
``base:code`` are decoded with a codebook (:mod:`.plain_language_codebooks`); codes the codebook
does not document are reported as gaps, never guessed.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.genome_network import NetworkStructure, NodeType
from .composition import annotation_dicts_from_operations
from .explanation_tools import StateEvaluator, as_unit, default_auc, input_column
from .plain_language_codebooks import (base_of, codebook_gaps, find_codebook, generic_codebook,
                                       level_info)

LAYER_SCHEMA_VERSION = "1.0.0"
CLAIM_TYPES = ("direction", "threshold", "interaction", "magnitude", "subgroup", "null")

SEED = 42
BOOTSTRAP_B = 1000
SMALL_N = 20                 # groups smaller than this are flagged small_n
NEGLIGIBLE = 0.05            # |log-odds| below this: negligible
SLIGHT = 0.25                # |log-odds| below this: slight
STRONG = 1.0                 # |log-odds| at least this: strong
MIN_INTERACTION = 0.1        # |difference in log-odds| an interaction must reach
MIN_THRESHOLD_EFFECT = 0.1   # |log-odds| the effect past a fold must reach
MAX_CONTRADICTING = 0.10     # share of patients allowed the opposite sign (direction)
MAX_THRESHOLD_SPREAD = 0.20  # IQR of the folds, as a share of the observed range
MIN_THRESHOLD_PATIENTS = 5
MAX_INTERACTIONS_PER_PART = 4
TOP_MAGNITUDES = 5
INERT_CANDIDATE = 0.25       # parts whose |contribution| never exceeds this are null candidates
INERT_MAX_DP = 0.02          # ... verified if predictions move by at most this
INERT_MAX_DAUC = 0.002
GRID_POINTS = 101
FLAT_TOL = 1e-9
SIG = 6



# ============================================================================ small helpers

def sig(x, n: int = SIG):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return float(f"{x:.{n}g}")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _rng():
    return np.random.default_rng(SEED)


def boot_mean(e: np.ndarray, B: int = BOOTSTRAP_B) -> Tuple[float, List[float]]:
    e = np.asarray(e, dtype=float)
    n = len(e)
    if n == 0:
        return float("nan"), [float("nan"), float("nan")]
    idx = _rng().integers(0, n, size=(B, n))
    m = e[idx].mean(axis=1)
    return float(e.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_stat(e: np.ndarray, stat, B: int = BOOTSTRAP_B) -> Tuple[float, List[float]]:
    e = np.asarray(e, dtype=float)
    n = len(e)
    idx = _rng().integers(0, n, size=(B, n))
    vals = np.array([stat(e[i]) for i in idx])
    return float(stat(e)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_diff(e1: np.ndarray, e0: np.ndarray, B: int = BOOTSTRAP_B) -> Tuple[float, List[float]]:
    """Mean(e1) - mean(e0) with a stratified bootstrap interval (each group resampled)."""
    e1 = np.asarray(e1, dtype=float)
    e0 = np.asarray(e0, dtype=float)
    if len(e1) == 0 or len(e0) == 0:
        return float("nan"), [float("nan"), float("nan")]
    rng = _rng()
    m1 = e1[rng.integers(0, len(e1), size=(B, len(e1)))].mean(axis=1)
    m0 = e0[rng.integers(0, len(e0), size=(B, len(e0)))].mean(axis=1)
    d = m1 - m0
    return float(e1.mean() - e0.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def excludes_zero(ci) -> bool:
    lo, hi = ci
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def strength(x: float) -> str:
    a = abs(x)
    if a < NEGLIGIBLE:
        return "negligible"
    if a < SLIGHT:
        return "slight"
    if a < STRONG:
        return "moderate"
    return "strong"


def _sig_ci(ci):
    return [sig(ci[0]), sig(ci[1])]


def _or(x):
    return sig(math.exp(x)) if x is not None and math.isfinite(x) else None


def estimate_block(value, ci, method: str, unit: str = "log-odds (model)", B: int = BOOTSTRAP_B,
                   resample: str = "patients", odds: bool = True) -> Dict[str, Any]:
    out = {"value": sig(value), "ci95": _sig_ci(ci), "unit": unit, "method": method,
           "bootstrap": {"B": B, "seed": SEED, "resample": resample, "interval": "percentile"}}
    if odds:
        out["odds_ratio"] = _or(value)
        out["odds_ratio_ci95"] = [_or(ci[0]), _or(ci[1])]
    return out


# ============================================================================ variables

@dataclass
class Variable:
    base: str
    kind: str                      # continuous | count | category
    columns: List[int]
    codes: List[Optional[str]]     # category: code of each column; else [None]
    cb: Dict[str, Any]
    mean: np.ndarray               # per column
    scale: np.ndarray
    raw_train: np.ndarray          # continuous: values; category: code per row (object)
    raw_test: np.ndarray
    reference: Any = None
    decoded: bool = True

    @property
    def label(self) -> str:
        return self.cb.get("label") or self.base

    @property
    def short(self) -> str:
        return self.cb.get("short") or self.label

    @property
    def unit(self) -> Optional[str]:
        return self.cb.get("unit")

    @property
    def decimals(self) -> int:
        d = self.cb.get("decimals")
        return 2 if d is None else int(d)

    def is_category(self) -> bool:
        return self.kind == "category"

    def encode(self, value) -> Dict[int, float]:
        """Data-column z-values that represent ``value`` (a raw number, or a category code)."""
        if self.is_category():
            return {c: float(((1.0 if code == str(value) else 0.0) - m) / s)
                    for c, code, m, s in zip(self.columns, self.codes, self.mean, self.scale)}
        c = self.columns[0]
        return {c: float((float(value) - self.mean[0]) / self.scale[0])}

    def decode(self, z: float) -> float:
        return float(z) * float(self.scale[0]) + float(self.mean[0])

    def level(self, code) -> Dict[str, Any]:
        return level_info(self.cb, str(code))

    def fmt(self, value) -> str:
        """A value in real-world units, for printing."""
        if self.is_category():
            return self.level(value)["label"]
        special = (self.cb.get("special_values") or {}).get(repr(float(value)))
        if special:
            return special
        d = self.decimals
        s = f"{float(value):.{d}f}"
        if self.unit:
            s += f" {self.unit}"
        alt = self.cb.get("alt_unit")
        if alt:
            s += f" ({float(value) * alt['factor']:.{alt['decimals']}f} {alt['unit']})"
        return s

    def fmt_number(self, value) -> str:
        return f"{float(value):.{self.decimals}f}"


def _category_codes(raw_cols: np.ndarray, codes: List[str]) -> np.ndarray:
    """Per row, the code whose one-hot column is on (None if none or several are on)."""
    out = np.empty(raw_cols.shape[0], dtype=object)
    for i, row in enumerate(raw_cols):
        on = [codes[j] for j in range(len(codes)) if row[j] > 0.5]
        out[i] = on[0] if len(on) == 1 else None
    return out


def build_variables(feature_names: Sequence[str], raw_train: np.ndarray, raw_test: np.ndarray,
                    mean: np.ndarray, scale: np.ndarray, codebook: Dict[str, Any]) -> List[Variable]:
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, f in enumerate(feature_names):
        b = base_of(f)
        if b not in groups:
            groups[b] = []
            order.append(b)
        groups[b].append(i)
    out = []
    for b in order:
        cols = groups[b]
        cb = (codebook.get("variables") or {}).get(b) or {"label": b}
        is_cat = any(":" in feature_names[c] for c in cols)
        if is_cat:
            codes = [feature_names[c].split(":", 1)[1] for c in cols]
            v = Variable(b, "category", cols, codes, cb, mean[cols], scale[cols],
                         _category_codes(raw_train[:, cols], codes),
                         _category_codes(raw_test[:, cols], codes) if raw_test is not None else None)
            counts = {c: int(np.sum(v.raw_train == c)) for c in codes}
            # reference: the most common category (ties: codebook order)
            v.reference = max(codes, key=lambda c: (counts[c], -codes.index(c)))
            v.decoded = all(level_info(cb, c)["decoded"] for c in codes)
        else:
            kind = cb.get("kind") or "continuous"
            v = Variable(b, kind, cols, [None], cb, mean[cols], scale[cols],
                         raw_train[:, cols[0]].astype(float),
                         raw_test[:, cols[0]].astype(float) if raw_test is not None else None)
            v.reference = float(np.median(v.raw_train))
            v.decoded = bool(cb.get("unit"))
        out.append(v)
    return out


def _lo_hi(v: Variable) -> Tuple[float, float]:
    """10th and 90th percentile of the training values, rounded to the variable's decimals."""
    lo = round(float(np.percentile(v.raw_train, 10)), v.decimals)
    hi = round(float(np.percentile(v.raw_train, 90)), v.decimals)
    if hi <= lo:
        lo, hi = float(np.min(v.raw_train)), float(np.max(v.raw_train))
    return lo, hi


# ============================================================================ the model view

@dataclass
class Part:
    id: str
    name: str
    ann: Dict[str, Any]
    unit: Dict[str, Any]
    kind: str
    is_root: bool
    parent: Optional[str]
    children: List[str]
    input_nodes: List[str]
    upstream_nodes: List[str]
    variables: List[str]
    exits: List[str]
    baselines: Dict[str, float] = field(default_factory=dict)
    gated: bool = False
    nonlinear: bool = False
    contribution: Optional[np.ndarray] = None
    contribution_test: Optional[np.ndarray] = None
    duplicate_of: Optional[str] = None
    order: int = 0


class PlainModel:
    """Forward passes on the explained model state with real-world manipulations."""

    def __init__(self, structure: NetworkStructure, operations: Sequence[Dict[str, Any]],
                 X_train, y_train, X_test, y_test, feature_names: Sequence[str],
                 mean, scale, codebook: Dict[str, Any], raw_train=None, raw_test=None):
        self.structure = structure
        self.ops = [dict(o) for o in operations]
        self.X = np.asarray(X_train, dtype=float)
        self.y = np.asarray(y_train, dtype=float).ravel()
        self.Xt = None if X_test is None else np.asarray(X_test, dtype=float)
        self.yt = None if y_test is None else np.asarray(y_test, dtype=float).ravel()
        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.codebook = codebook
        dec_tr = self.X * self.scale + self.mean
        dec_te = None if self.Xt is None else self.Xt * self.scale + self.mean
        self.raw_train = dec_tr if raw_train is None else np.asarray(raw_train, dtype=float)
        self.raw_test = dec_te if raw_test is None else np.asarray(raw_test, dtype=float)
        self.round_trip_err = None
        if raw_train is not None:
            z_again = (self.raw_train - self.mean) / self.scale
            self.round_trip_err = float(max(np.max(np.abs(dec_tr - self.raw_train)),
                                            np.max(np.abs(z_again - self.X))))
        self.ev = StateEvaluator(structure)
        self.nodes = {n.id: n for n in structure.nodes}
        self.out_ids = [str(o) for o in structure.output_node_ids]
        self.variables = build_variables(self.feature_names, self.raw_train, self.raw_test,
                                         self.mean, self.scale, codebook)
        self.var_by_base = {v.base: v for v in self.variables}
        self.col_var = {c: v.base for v in self.variables for c in v.columns}
        self.input_nodes_by_col: Dict[int, List[str]] = {}
        for n in structure.nodes:
            if n.type == NodeType.INPUT:
                self.input_nodes_by_col.setdefault(input_column(n.id), []).append(n.id)
        self.base_logit = self.ev.logit(self.X)
        self.base_logit_test = None if self.Xt is None else self.ev.logit(self.Xt)

    # ------------------------------------------------------------------ forward passes
    def logit(self, X, assign: Optional[Dict[int, Any]] = None, scope: Optional[Sequence[str]] = None,
              ablate: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Log-odds with data columns set by ``assign`` ({column: value or per-row array}).

        ``scope=None`` sets the column everywhere (the whole model sees the new value); a list
        of input node ids sets it only on those copies (the change reaches the output only
        through the part those copies feed). ``ablate`` fixes nodes at constants.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        overrides = {}
        if assign:
            if scope is None:
                X = X.copy()
                for c, val in assign.items():
                    X[:, c] = val
            else:
                scope = set(scope)
                for c, val in assign.items():
                    for node in self.input_nodes_by_col.get(c, []):
                        if node in scope:
                            arr = np.broadcast_to(np.asarray(val, dtype=float), (n,)).copy()
                            overrides[node] = ((), (lambda vals, a=arr: a))
        for node, const in (ablate or {}).items():
            overrides[node] = ((), (lambda vals, k=float(const), m=n: np.full(m, k)))
        return self.ev.logit(X, overrides or None)

    def set_logit(self, X, var: Variable, value, scope=None, extra=None) -> np.ndarray:
        assign = var.encode(value)
        if extra:
            assign = {**assign, **extra}
        return self.logit(X, assign, scope)

    def effect(self, X, var: Variable, v_from, v_to, scope=None, extra=None) -> np.ndarray:
        """Per-row change in log-odds when ``var`` goes from ``v_from`` to ``v_to``."""
        return (self.set_logit(X, var, v_to, scope, extra) -
                self.set_logit(X, var, v_from, scope, extra))

    # ------------------------------------------------------------------ structure
    def reaching_output(self) -> set:
        """Nodes with a path to an output over enabled, non-zero-weight edges."""
        preds: Dict[str, List[str]] = {}
        for c in self.structure.connections:
            if c.enabled and c.weight != 0:
                preds.setdefault(str(c.to_node), []).append(str(c.from_node))
        seen, stack = set(), list(self.out_ids)
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(preds.get(x, []))
        return seen

    def vars_of_nodes(self, nodes: Sequence[str]) -> List[str]:
        out = []
        for v in self.variables:
            if any(nd in nodes for c in v.columns for nd in self.input_nodes_by_col.get(c, [])):
                out.append(v.base)
        return out

    def scope_columns(self, var: Variable, scope) -> List[int]:
        if scope is None:
            return list(var.columns)
        return [c for c in var.columns if any(nd in scope for nd in self.input_nodes_by_col.get(c, []))]


# ============================================================================ parts

def build_parts(m: PlainModel) -> List[Part]:
    anns = annotation_dicts_from_operations(m.ops)
    by_ref = {}
    for a in anns:
        by_ref[a["name"]] = a
        by_ref[a["id"]] = a
    parent_of: Dict[str, str] = {}
    for a in anns:
        for c in a.get("child_annotation_ids") or []:
            ch = by_ref.get(c)
            if ch is not None:
                parent_of[ch["name"]] = a["name"]
    vals = m.ev.forward(m.X)
    vals_t = None if m.Xt is None else m.ev.forward(m.Xt)
    parts: List[Part] = []
    for a in anns:
        unit = as_unit(a, anns, m.structure)
        entries = [str(e) for e in unit["entry_nodes"]]
        inputs = [e for e in entries if e in m.nodes and m.nodes[e].type == NodeType.INPUT]
        upstream = [e for e in entries if e not in inputs]
        exits = [str(x) for x in unit["exit_nodes"]]
        is_root = any(x in m.out_ids for x in exits)
        kids = [by_ref[c]["name"] for c in (a.get("child_annotation_ids") or []) if c in by_ref]
        p = Part(id=a["id"], name=a["name"], ann=a, unit=unit,
                 kind="composition" if kids else "leaf", is_root=is_root,
                 parent=parent_of.get(a["name"]), children=kids, input_nodes=inputs,
                 upstream_nodes=upstream, variables=m.vars_of_nodes(inputs), exits=exits)
        if not is_root:
            gated = False
            for x in exits:
                node = m.nodes.get(x)
                v = vals[x]
                if node is not None and node.activation != "identity" and np.min(v) <= 1e-12:
                    p.baselines[x] = 0.0
                    gated = True
                else:
                    p.baselines[x] = float(np.mean(v))
            p.gated = gated and all(b == 0.0 for b in p.baselines.values())
            p.contribution = m.base_logit - m.logit(m.X, ablate=p.baselines)
            if m.Xt is not None:
                p.contribution_test = m.base_logit_test - m.logit(m.Xt, ablate=p.baselines)
            p.nonlinear = _is_nonlinear(m, p, vals)
        parts.append(p)
    # duplicates: a leaf whose exits equal an earlier leaf's exits on every training row
    leaves = [p for p in parts if p.kind == "leaf" and not p.is_root]
    for i, p in enumerate(leaves):
        for q in leaves[:i]:
            if q.duplicate_of or len(q.exits) != len(p.exits) or set(q.variables) != set(p.variables):
                continue
            if all(np.max(np.abs(vals[a] - vals[b])) < 1e-12 for a, b in zip(p.exits, q.exits)):
                if vals_t is None or all(np.max(np.abs(vals_t[a] - vals_t[b])) < 1e-12
                                         for a, b in zip(p.exits, q.exits)):
                    p.duplicate_of = q.id
                    break
    # order: root first, then by the spread of the part's contribution, then by sequence
    seq = {a["name"]: i for i, a in enumerate(anns)}
    parts.sort(key=lambda p: (not p.is_root, p.duplicate_of is not None,
                              -float(np.std(p.contribution)) if p.contribution is not None else 0.0,
                              seq[p.name]))
    for i, p in enumerate(parts):
        p.order = i
    return parts


def _is_nonlinear(m: PlainModel, p: Part, vals) -> bool:
    """True when the part's exits are not an affine function of its entries on the data."""
    entries = p.input_nodes + p.upstream_nodes
    if not entries:
        return False
    n = m.X.shape[0]
    A = np.column_stack([np.ones(n)] + [vals[e] for e in entries])
    for x in p.exits:
        y = vals[x]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if np.max(np.abs(y - A @ coef)) > 1e-7 * max(1.0, float(np.max(np.abs(y)))):
            return True
    return False


# ============================================================================ group descriptions

def describe_group(m: PlainModel, rows: np.ndarray, var_names: Sequence[str]) -> Dict[str, Any]:
    """Who is in a group of training patients: category shares and number ranges."""
    rows = np.asarray(rows, dtype=bool)
    out: Dict[str, Any] = {"n": int(rows.sum()), "n_other": int((~rows).sum()), "conditions": [],
                           "categories": [], "numbers": []}
    for b in var_names:
        v = m.var_by_base[b]
        if v.is_category():
            for code in v.codes:
                ins = float(np.mean(v.raw_train[rows] == code)) if rows.any() else 0.0
                oth = float(np.mean(v.raw_train[~rows] == code)) if (~rows).any() else 0.0
                rec = {"variable": b, "code": code, "label": v.level(code)["label"],
                       "group": v.level(code)["group"], "share_in": sig(ins),
                       "share_out": sig(oth), "decoded": v.level(code)["decoded"]}
                out["categories"].append(rec)
                if ins >= 0.95 and ins - oth >= 0.2:
                    out["conditions"].append(rec)
        else:
            x = v.raw_train
            out["numbers"].append({
                "variable": b, "label": v.label, "unit": v.unit,
                "median_in": sig(np.median(x[rows])) if rows.any() else None,
                "median_out": sig(np.median(x[~rows])) if (~rows).any() else None,
                "min_in": sig(np.min(x[rows])) if rows.any() else None,
                "max_in": sig(np.max(x[rows])) if rows.any() else None})
    return out


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [sig(max(0.0, c - h)), sig(min(1.0, c + h))]


# ============================================================================ claim builders

def _claim(part: Part, ctype: str, key: str, **kw) -> Dict[str, Any]:
    c = {"key": f"{part.id}:{ctype}:{key}", "type": ctype,
         "annotation": {"id": part.id, "name": part.name},
         "scope": "model" if part.is_root else "part", "split": "train", "test_check": None,
         "small_n": False, "small_n_groups": [], "verified": False, "reasons": [],
         "checks": {}}
    c.update(kw)
    return c


def _finish(c: Dict[str, Any], checks: Dict[str, Tuple[bool, str]]):
    c["checks"] = {k: bool(v[0]) for k, v in checks.items()}
    c["reasons"] = [msg for ok, msg in checks.values() if not ok]
    c["verified"] = all(ok for ok, _ in checks.values())
    if c["verified"]:
        c["reasons"] = ["all checks passed"]
    return c


def _settings(m: PlainModel, v: Variable, scope) -> List[Tuple[Any, Any]]:
    """(from, to) settings that describe a change in ``v`` the scope can see."""
    if not v.is_category():
        return [_lo_hi(v)]
    cols = m.scope_columns(v, scope)
    read = [v.codes[v.columns.index(c)] for c in cols]
    counts = {c: int(np.sum(v.raw_train == c)) for c in v.codes}
    if not read:
        return []
    if len(read) == 1 and read[0] == v.reference:
        others = [c for c in v.codes if c != v.reference]
        if not others:
            return []
        alt = max(others, key=lambda c: (counts[c], -v.codes.index(c)))
        return [(v.reference, alt)]   # always phrased from the reference category
    return [(v.reference, code) for code in read if code != v.reference]


def _setting_desc(v: Variable, a, b) -> Dict[str, Any]:
    if v.is_category():
        return {"from": {"code": a, **_lvl(v, a)}, "to": {"code": b, **_lvl(v, b)}}
    return {"from": {"value": a, "text": v.fmt(a)}, "to": {"value": b, "text": v.fmt(b)},
            "basis": "10th and 90th percentile of the training patients"}


def _lvl(v: Variable, code) -> Dict[str, Any]:
    li = v.level(code)
    return {"label": li["label"], "group": li["group"], "cond": li["cond"], "decoded": li["decoded"],
            "n_train": int(np.sum(v.raw_train == code))}


def _var_desc(v: Variable) -> Dict[str, Any]:
    return {"base": v.base, "label": v.label, "short": v.short, "unit": v.unit, "kind": v.kind,
            "decoded": v.decoded}


def direction_claims(m: PlainModel, part: Part, scope) -> List[Dict[str, Any]]:
    out = []
    for b in part.variables:
        v = m.var_by_base[b]
        for (a, z) in _settings(m, v, scope):
            e = m.effect(m.X, v, a, z, scope)
            if np.max(np.abs(e)) < 1e-12:
                continue
            mean, ci = boot_mean(e)
            sgn = np.sign(mean)
            contra = float(np.mean(np.sign(e) == -sgn) if sgn != 0 else 1.0)
            zero = float(np.mean(np.abs(e) < 1e-12))
            affected = np.abs(e) >= 1e-12
            key = b if not v.is_category() else f"{b}:{z if z != v.reference else a}"
            c = _claim(part, "direction", key, variables=[_var_desc(v)],
                       statement={"variable": b, "change": _setting_desc(v, a, z),
                                  "direction": "raises" if mean > 0 else "lowers",
                                  "strength": strength(mean)})
            c["estimate"] = estimate_block(mean, ci, "mean over training patients of the change "
                                           "in the model's log-odds when the variable is changed "
                                           + ("everywhere in the model" if scope is None else
                                              "only where this part reads it"))
            c["evidence"] = {"share_opposite_sign": sig(contra), "share_unaffected": sig(zero),
                             "mean_among_affected": sig(float(np.mean(e[affected]))) if affected.any() else None,
                             "n_affected": int(affected.sum())}
            # monotone average curve (continuous only)
            mono = None
            if not v.is_category():
                grid = np.unique(np.round(np.percentile(v.raw_train, np.arange(5, 96, 5)), v.decimals))
                pd = np.array([np.mean(m.set_logit(m.X, v, g, scope)) for g in grid])
                d = np.diff(pd)
                mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
                c["evidence"]["average_curve"] = {"values": [sig(g) for g in grid],
                                                  "mean_log_odds": [sig(x) for x in pd]}
            c["statement"]["monotone"] = mono
            n = len(e)
            c["n"] = n
            groups = []
            if v.is_category():
                for code in (a, z):
                    k = int(np.sum(v.raw_train == code))
                    groups.append({"label": v.level(code)["label"], "n": k})
                    if k < SMALL_N:
                        c["small_n_groups"].append(v.level(code)["label"])
            c["groups"] = groups
            c["small_n"] = bool(c["small_n_groups"])
            c["test_check"] = _test_direction(m, v, a, z, scope, mean)
            checks = {
                "ci_excludes_zero": (excludes_zero(ci), "the 95% interval of the mean effect includes zero"),
                "consistent_sign": (contra <= MAX_CONTRADICTING,
                                    f"{contra:.0%} of patients have the opposite sign (allowed: "
                                    f"{MAX_CONTRADICTING:.0%})"),
            }
            if v.is_category():
                checks["level_observed"] = (all(g["n"] > 0 for g in groups),
                                            "a category in the comparison has no training patients")
            out.append(_finish(c, checks))
    return out


def _test_direction(m: PlainModel, v, a, z, scope, train_mean) -> Optional[Dict[str, Any]]:
    if m.Xt is None:
        return None
    e = m.effect(m.Xt, v, a, z, scope)
    mean, ci = boot_mean(e)
    if np.max(np.abs(e)) < 1e-12:
        return {"split": "test", "n": len(e), "value": 0.0, "ci95": [0.0, 0.0],
                "consistent": None, "note": "no test patient is affected"}
    cons = bool(np.sign(mean) == np.sign(train_mean) or
                (abs(mean) < NEGLIGIBLE and abs(train_mean) < NEGLIGIBLE))
    return {"split": "test", "n": len(e), "value": sig(mean), "ci95": _sig_ci(ci), "consistent": cons}


def _reference_rows(m: PlainModel, rows_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Raw reference patient(s): median of every number, most common category."""
    raw = m.raw_train if rows_idx is None else m.raw_train[rows_idx]
    ref = np.zeros(m.raw_train.shape[1])
    for v in m.variables:
        if v.is_category():
            codes = v.raw_train if rows_idx is None else v.raw_train[rows_idx]
            counts = {c: int(np.sum(codes == c)) for c in v.codes}
            best = max(v.codes, key=lambda c: (counts[c], -v.codes.index(c)))
            for c, code in zip(v.columns, v.codes):
                ref[c] = 1.0 if code == best else 0.0
        else:
            ref[v.columns[0]] = float(np.median(raw[:, v.columns[0]]))
    return ref


def reference_patient(m: PlainModel) -> Dict[str, Any]:
    raw = _reference_rows(m)
    z = (raw - m.mean) / m.scale
    p = float(_sigmoid(m.logit(z[None, :]))[0])
    vals = {}
    for v in m.variables:
        if v.is_category():
            vals[v.base] = v.level(v.reference)["label"]
        else:
            vals[v.base] = v.fmt(raw[v.columns[0]])
    return {"definition": "the median training patient for every number and the most common "
                          "training category for every category (not a real patient)",
            "values": vals, "predicted_risk": sig(p), "raw": [sig(x) for x in raw]}


def magnitude_claims(m: PlainModel, part: Part, directions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def decoded(d):
        ch = d["statement"]["change"]
        return all((ch.get(k) or {}).get("decoded", True) for k in ("from", "to"))
    # magnitudes for the biggest verified effects with adequate data and decoded categories
    cands = [d for d in directions if d["verified"] and not d["small_n"] and decoded(d)]
    cands.sort(key=lambda d: (-abs(d["estimate"]["value"]), d["key"]))
    seen, chosen = set(), []
    for d in cands:
        b = d["statement"]["variable"]
        if b in seen:
            continue
        seen.add(b)
        chosen.append(d)
        if len(chosen) >= TOP_MAGNITUDES:
            break
    if not chosen:
        return []
    n = m.X.shape[0]
    rng = _rng()
    boot_idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
    ref_raw = _reference_rows(m)
    boot_raw = np.array([_reference_rows(m, idx) for idx in boot_idx])
    out = []
    for d in chosen:
        v = m.var_by_base[d["statement"]["variable"]]
        ch = d["statement"]["change"]
        a = ch["from"].get("code", ch["from"].get("value"))
        z_ = ch["to"].get("code", ch["to"].get("value"))
        Z = (np.vstack([ref_raw[None, :], boot_raw]) - m.mean) / m.scale
        e = m.effect(Z, v, a, z_, None)
        val = float(e[0])
        ci = [float(np.percentile(e[1:], 2.5)), float(np.percentile(e[1:], 97.5))]
        c = _claim(part, "magnitude", d["statement"]["variable"]
                   if not v.is_category() else d["key"].split(":", 2)[2],
                   variables=[_var_desc(v)],
                   statement={"variable": v.base, "change": ch,
                              "direction": "raises" if val > 0 else "lowers",
                              "strength": strength(val), "reference_patient": "typical",
                              "related_direction": d["key"]})
        c["estimate"] = estimate_block(val, ci, "change in the model's log-odds for the reference "
                                       "patient; interval from recomputing the reference patient "
                                       "on bootstrap resamples of the training patients",
                                       resample="patients (reference patient recomputed)")
        c["n"] = n
        c["groups"] = d.get("groups", [])
        c["small_n_groups"] = list(d.get("small_n_groups", []))
        c["small_n"] = d.get("small_n", False)
        c["test_check"] = None
        checks = {"ci_excludes_zero": (excludes_zero(ci), "the 95% interval includes no change"),
                  "agrees_with_direction": (np.sign(val) == np.sign(d["estimate"]["value"]),
                                            "the sign at the reference patient differs from the "
                                            "average direction")}
        out.append(_finish(c, checks))
    return out


def _ice(m: PlainModel, v: Variable, grid: np.ndarray, scope) -> np.ndarray:
    """(n_patients, len(grid)) log-odds with ``v`` set to each grid value on the scope."""
    n, G = m.X.shape[0], len(grid)
    Xr = np.repeat(m.X, G, axis=0)
    vals = np.tile(grid, n)
    col = v.columns[0]
    z = (vals - v.mean[0]) / v.scale[0]
    return m.logit(Xr, {col: z}, scope).reshape(n, G)


def _folds(L: np.ndarray, grid: np.ndarray):
    """Per patient: the fold of a piecewise-linear curve that is flat on one side.

    Returns (onset, offset, depth_on, depth_off): onset where a curve that is flat from the
    lowest value starts to move; offset where a moving curve goes flat up to the highest value;
    depth = how far the curve is from the straight-line continuation of its moving part at the
    flat end (in log-odds), i.e. how much the fold matters.
    """
    h = grid[1] - grid[0]
    D = np.diff(L, axis=1)
    flat = np.abs(D) <= FLAT_TOL
    n = L.shape[0]
    onset, offset = np.full(n, np.nan), np.full(n, np.nan)
    d_on, d_off = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        f = flat[i]
        if f.all() or (~f).all():
            continue
        if f[0]:
            k = int(np.argmax(~f))
            s = D[i, k + 1] / h if k + 1 < D.shape[1] and not f[k + 1] else D[i, k] / h
            t = grid[k + 1] - (L[i, k + 1] - L[i, k]) / s if s != 0 else grid[k] + h / 2
            onset[i] = min(max(t, grid[k]), grid[k + 1])
            d_on[i] = abs(s) * (onset[i] - grid[0])
        elif f[-1]:
            k = int(len(f) - 1 - np.argmax(~f[::-1]))
            s = D[i, k - 1] / h if k - 1 >= 0 and not f[k - 1] else D[i, k] / h
            t = grid[k] + (L[i, k + 1] - L[i, k]) / s if s != 0 else grid[k] + h / 2
            offset[i] = min(max(t, grid[k]), grid[k + 1])
            d_off[i] = abs(s) * (grid[-1] - offset[i])
    return onset, offset, d_on, d_off


def _threshold_groups(m: PlainModel, part: Part, b: str, rows: np.ndarray, ts_all: np.ndarray,
                      span: float) -> List[Tuple[Optional[Dict[str, Any]], np.ndarray]]:
    """Candidate patient groups for one threshold: everyone with a fold, then each category
    level of the part's other variables (largest group whose folds agree first)."""
    out: List[Tuple[Optional[Dict[str, Any]], np.ndarray]] = [(None, rows)]
    for w in part.variables:
        if w == b:
            continue
        W = m.var_by_base[w]
        if not W.is_category():
            continue
        for code in W.codes:
            g = rows & (W.raw_train == code)
            if g.sum() >= MIN_THRESHOLD_PATIENTS:
                out.append(({"variable": w, "code": code, **_lvl(W, code)}, g))
    return out


def threshold_claims(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    out = []
    for b in part.variables:
        v = m.var_by_base[b]
        if v.is_category():
            continue
        lo, hi = float(np.min(v.raw_train)), float(np.max(v.raw_train))
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, GRID_POINTS)
        L = _ice(m, v, grid, part.input_nodes)
        onset, offset, d_on, d_off = _folds(L, grid)
        n_on, n_off = int(np.sum(~np.isnan(onset))), int(np.sum(~np.isnan(offset)))
        if max(n_on, n_off) < MIN_THRESHOLD_PATIENTS:
            continue
        kind = "above" if n_on >= n_off else "below"
        t_all, depth = (onset, d_on) if kind == "above" else (offset, d_off)
        with_fold = ~np.isnan(t_all)
        # the group: everyone with a fold if their folds agree, else the largest category
        # group (of the part's other variables) whose folds agree
        best = None
        for cond, g in _threshold_groups(m, part, b, with_fold, t_all, hi - lo):
            q1, q3 = np.percentile(t_all[g], [25, 75])
            spread = float((q3 - q1) / (hi - lo))
            cand = (spread <= MAX_THRESHOLD_SPREAD, int(g.sum()), cond, g, spread)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
            if cond is None and cand[0]:
                break
        agrees, _, cond, rows, spread = best
        ts = t_all[rows]
        med, ci = boot_stat(ts, np.median)
        q1, q3 = np.percentile(ts, [25, 75])
        hinge = float(np.median(depth[rows]))
        p5, p95 = np.percentile(v.raw_train, [5, 95])
        if kind == "above":
            eff = (m.set_logit(m.X[rows], v, p95, part.input_nodes) -
                   m.set_logit(m.X[rows], v, lo, part.input_nodes))
        else:
            eff = (m.set_logit(m.X[rows], v, hi, part.input_nodes) -
                   m.set_logit(m.X[rows], v, p5, part.input_nodes))
        emean, eci = boot_mean(eff)
        actual = v.raw_train[rows]
        past = int(np.sum(actual > ts)) if kind == "above" else int(np.sum(actual < ts))
        who = describe_group(m, rows, [x for x in part.variables if x != b])
        conds = [x["group"] for x in who["conditions"]]
        if cond is not None and cond["group"] not in conds:
            conds = [cond["group"]] + conds
        th = {"value": sig(med), "text": v.fmt(round(med, v.decimals))}
        if v.kind == "count":   # a fold between two whole numbers: name the step
            step = int(math.floor(med)) + 1
            th.update({"count_step": step, "text": v.fmt(step)})
        c = _claim(part, "threshold", b, variables=[_var_desc(v)],
                   statement={"variable": b, "kind": kind,
                              "threshold": th,
                              "direction": "raises" if emean > 0 else "lowers",
                              "strength": strength(emean), "group": cond,
                              "conditions": conds})
        c["estimate"] = {
            "threshold": {"value": sig(med), "ci95": _sig_ci(ci), "unit": v.unit or "raw units",
                          "iqr": [sig(q1), sig(q3)], "spread_share_of_range": sig(spread),
                          "method": "median over patients of the fold of each patient's curve "
                                    "(log-odds against the variable, changed only where this part "
                                    "reads it); bootstrap of the median",
                          "bootstrap": {"B": BOOTSTRAP_B, "seed": SEED, "resample": "patients in the group"}},
            "fold_depth": {"value": sig(hinge), "unit": "log-odds",
                           "method": "median distance at the flat end between the curve and the "
                                     "straight-line continuation of its moving part"},
            **estimate_block(emean, eci, "mean change in log-odds from "
                             + (f"the lowest value to the 95th percentile ({v.fmt(p95)})" if kind == "above"
                                else f"the 5th percentile ({v.fmt(p5)}) to the highest value")
                             + " for the patients in the group")}
        c["n"] = int(rows.sum())
        c["groups"] = [{"label": ("patients whose curve has a fold" if cond is None else
                                  f"{cond['group']} whose curve has a fold"), "n": int(rows.sum())},
                       {"label": "of them, patients already past their fold", "n": past}]
        c["evidence"] = {"who": who, "n_onset": n_on, "n_offset": n_off,
                         "n_with_fold": int(with_fold.sum()),
                         "observed_range": [sig(lo), sig(hi)], "grid_points": GRID_POINTS}
        if past < SMALL_N:
            c["small_n_groups"].append("patients past the fold")
        c["small_n"] = bool(c["small_n_groups"])
        checks = {
            "enough_patients": (rows.sum() >= MIN_THRESHOLD_PATIENTS,
                                f"fewer than {MIN_THRESHOLD_PATIENTS} patients have a fold"),
            "single_threshold": (spread <= MAX_THRESHOLD_SPREAD,
                                 f"the folds are spread over {spread:.0%} of the observed range "
                                 f"(allowed {MAX_THRESHOLD_SPREAD:.0%}), also within each category "
                                 f"group: no single threshold"),
            "fold_matters": (hinge >= MIN_THRESHOLD_EFFECT,
                             f"the fold changes the curve by only {hinge:.3f} log-odds compared with "
                             f"a straight line (needed {MIN_THRESHOLD_EFFECT})"),
            "effect_past_threshold": (excludes_zero(eci) and abs(emean) >= MIN_THRESHOLD_EFFECT,
                                      "the effect past the fold is too small or its interval includes zero"),
            "patients_past_threshold": (past >= 1, "no training patient is past the fold"),
        }
        out.append(_finish(c, checks))
    return out


def _groups_for(m: PlainModel, w: Variable, w0, w1, X_raw_codes=None):
    """Boolean rows of the two observed groups for a modifier."""
    if w.is_category():
        g1 = w.raw_train == w1
        g0 = (w.raw_train == w0) if (w0 is not None) else ~g1
        return g0, g1
    med = float(np.median(w.raw_train))
    return w.raw_train <= med, w.raw_train > med


def interaction_claims(m: PlainModel, part: Part, total_effect: Dict[str, float]) -> List[Dict[str, Any]]:
    vs = list(part.variables)
    if len(vs) < 2:
        return []
    cands = []
    scope = part.input_nodes
    for va, vb in itertools.combinations(vs, 2):
        A, Bv = m.var_by_base[va], m.var_by_base[vb]
        best = None
        for sa in _settings(m, A, scope):
            for sb in (_settings(m, Bv, scope) if Bv.is_category() else
                       [tuple(np.round(np.percentile(Bv.raw_train, [25, 75]), Bv.decimals))]):
                e1 = m.effect(m.X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[1]))
                e0 = m.effect(m.X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[0]))
                did = e1 - e0
                if np.max(np.abs(did)) < 1e-9:
                    continue
                score = abs(float(np.mean(did)))
                if best is None or score > best[0]:
                    best = (score, sa, sb, float(np.mean(did)))
        if best is not None:
            cands.append((va, vb, best))
    cands.sort(key=lambda t: (-t[2][0], t[0], t[1]))
    out = []
    for va, vb, (score, sa, sb, did_mean) in cands[:MAX_INTERACTIONS_PER_PART]:
        # main variable: the one with the larger total effect in the model; prefer a
        # category as the modifier when exactly one of them is a category
        A, Bv = m.var_by_base[va], m.var_by_base[vb]
        swap = False
        if A.is_category() != Bv.is_category():
            swap = A.is_category()
        elif abs(total_effect.get(vb, 0.0)) > abs(total_effect.get(va, 0.0)):
            swap = True
        if swap:
            A, Bv, sa, sb = Bv, A, sb, sa
            if not Bv.is_category():
                sb = tuple(np.round(np.percentile(Bv.raw_train, [25, 75]), Bv.decimals))
            if not A.is_category():
                sa = _lo_hi(A)
            did = (m.effect(m.X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[1])) -
                   m.effect(m.X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[0])))
            did_mean = float(np.mean(did))
        g0, g1 = _groups_for(m, Bv, sb[0], sb[1])
        e_total = m.effect(m.X, A, sa[0], sa[1], None)
        e_part = m.effect(m.X, A, sa[0], sa[1], scope)
        e1, e0 = e_total[g1], e_total[g0]
        m1, ci1 = boot_mean(e1)
        m0, ci0 = boot_mean(e0)
        d, dci = boot_diff(e1, e0)
        if Bv.is_category():
            lab1, lab0 = Bv.level(sb[1]), Bv.level(sb[0]) if sb[0] is not None else None
            gl1 = {"label": lab1["label"], "cond": lab1["cond"], "group": lab1["group"]}
            gl0 = ({"label": lab0["label"], "cond": lab0["cond"], "group": lab0["group"]} if lab0 else
                   {"label": "other", "cond": "otherwise", "group": "other patients"})
        else:
            med = float(np.median(Bv.raw_train))
            gl1 = {"label": f"{Bv.short} above {Bv.fmt(med)}", "cond": f"{Bv.short} is above {Bv.fmt(med)}",
                   "group": f"patients with {Bv.short} above {Bv.fmt(med)}"}
            gl0 = {"label": f"{Bv.short} at or below {Bv.fmt(med)}",
                   "cond": f"{Bv.short} is at or below {Bv.fmt(med)}",
                   "group": f"patients with {Bv.short} at or below {Bv.fmt(med)}"}
        groups = [{**gl1, "n": int(g1.sum()), "estimate": sig(m1), "ci95": _sig_ci(ci1),
                   "odds_ratio": _or(m1), "part_effect": sig(float(np.mean(e_part[g1]))) if g1.any() else None},
                  {**gl0, "n": int(g0.sum()), "estimate": sig(m0), "ci95": _sig_ci(ci0),
                   "odds_ratio": _or(m0), "part_effect": sig(float(np.mean(e_part[g0]))) if g0.any() else None}]
        strong, weak = (0, 1) if abs(m1) >= abs(m0) else (1, 0)
        ms, mw = (m1, m0) if strong == 0 else (m0, m1)
        ciw = ci0 if strong == 0 else ci1
        if np.sign(ms) != np.sign(mw) and excludes_zero(ci0) and excludes_zero(ci1):
            pattern = "reversed"
        elif not excludes_zero(ciw) or abs(mw) < NEGLIGIBLE:
            pattern = "only_if"
        else:
            pattern = "amplified"
        key = f"{A.base}x{Bv.base}"
        c = _claim(part, "interaction", key, variables=[_var_desc(A), _var_desc(Bv)],
                   statement={"variable": A.base, "modifier": Bv.base,
                              "change": _setting_desc(A, sa[0], sa[1]),
                              "direction": "raises" if ms > 0 else "lowers",
                              "pattern": pattern, "stronger_group": groups[strong],
                              "weaker_group": groups[weak], "strength": strength(ms)})
        c["estimate"] = estimate_block(d, dci, "difference between the two groups (first minus "
                                       "second) of the mean change in the model's log-odds when the "
                                       "main variable changes everywhere; stratified bootstrap",
                                       resample="patients within each group")
        c["groups"] = groups
        c["n"] = int(g1.sum() + g0.sum())
        c["evidence"] = {"counterfactual_did_through_part": sig(did_mean),
                         "counterfactual_modifier_settings": [str(sb[0]), str(sb[1])],
                         "method_note": "groups are the patients' actual values; the "
                                        "difference-in-differences sets the modifier for every "
                                        "patient, only where this part reads it"}
        for g in groups:
            if g["n"] < SMALL_N:
                c["small_n_groups"].append(g["label"])
        c["small_n"] = bool(c["small_n_groups"])
        c["test_check"] = _test_interaction(m, A, Bv, sa, sb, d)
        checks = {
            "ci_excludes_zero": (excludes_zero(dci), "the 95% interval of the group difference includes zero"),
            "large_enough": (abs(d) >= MIN_INTERACTION,
                             f"the group difference is below {MIN_INTERACTION} log-odds"),
            "mechanism_agrees": (np.sign(d) == np.sign(did_mean),
                                 "the part's own counterfactual difference has the opposite sign"),
            "both_groups_observed": (g1.sum() > 0 and g0.sum() > 0, "a group has no training patients"),
        }
        out.append(_finish(c, checks))
    return out


def _test_interaction(m, A, Bv, sa, sb, d_train):
    if m.Xt is None or Bv.raw_test is None:
        return None
    if Bv.is_category():
        g1 = Bv.raw_test == sb[1]
        g0 = (Bv.raw_test == sb[0]) if sb[0] is not None else ~g1
    else:
        med = float(np.median(Bv.raw_train))
        g1, g0 = Bv.raw_test > med, Bv.raw_test <= med
    e = m.effect(m.Xt, A, sa[0], sa[1], None)
    if g1.sum() == 0 or g0.sum() == 0:
        return {"split": "test", "n": int(g1.sum() + g0.sum()), "value": None, "consistent": None,
                "note": "a group has no test patients", "groups_n": [int(g1.sum()), int(g0.sum())]}
    d, dci = boot_diff(e[g1], e[g0])
    return {"split": "test", "n": int(g1.sum() + g0.sum()), "groups_n": [int(g1.sum()), int(g0.sum())],
            "value": sig(d), "ci95": _sig_ci(dci), "consistent": bool(np.sign(d) == np.sign(d_train))}


def subgroup_claims(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    if not part.gated or part.contribution is None:
        return []
    c_ = part.contribution
    active = np.abs(c_) > 1e-12
    k, n = int(active.sum()), len(c_)
    if k == n:
        return []
    share, sci = boot_mean(active.astype(float))
    who = describe_group(m, active, part.variables)
    eff, eci = boot_mean(c_[active]) if k else (float("nan"), [float("nan")] * 2)
    pos = int(np.sum(m.y[active] == 1)) if k else 0
    c = _claim(part, "subgroup", "active",
               variables=[_var_desc(m.var_by_base[b]) for b in part.variables],
               statement={"n_active": k, "n": n, "share": sig(k / n),
                          "direction": "raises" if eff > 0 else "lowers",
                          "strength": strength(eff) if k else None,
                          "conditions": [x["group"] for x in who["conditions"]],
                          "outcome_in_group": {"positives": pos, "n": k}})
    c["estimate"] = {"share": {"value": sig(k / n), "ci95": _sig_ci(sci), "unit": "share of patients",
                               "bootstrap": {"B": BOOTSTRAP_B, "seed": SEED, "resample": "patients"}},
                     **estimate_block(eff, eci, "mean contribution of the part to the log-odds among "
                                      "the patients it acts on (the part switched off: exits at 0)")}
    c["n"] = n
    c["groups"] = [{"label": "patients the part acts on", "n": k},
                   {"label": "patients it does not act on", "n": n - k}]
    c["evidence"] = {"who": who, "outcome_rate_in_group": sig(pos / k) if k else None,
                     "outcome_rate_ci95": _wilson(pos, k), "outcome_rate_overall": sig(float(np.mean(m.y)))}
    if k < SMALL_N:
        c["small_n_groups"].append("patients the part acts on")
    c["small_n"] = bool(c["small_n_groups"])
    if part.contribution_test is not None:
        kt = int(np.sum(np.abs(part.contribution_test) > 1e-12))
        c["test_check"] = {"split": "test", "n": len(part.contribution_test), "n_active": kt,
                           "share": sig(kt / len(part.contribution_test)),
                           "consistent": None if kt == 0 else bool(
                               np.sign(np.mean(part.contribution_test[np.abs(part.contribution_test) > 1e-12]))
                               == np.sign(eff))}
    checks = {"acts_on_someone": (k >= 1, "the part acts on no training patient"),
              "effect_ci_excludes_zero": (k >= 1 and excludes_zero(eci),
                                          "the interval of its effect includes zero"),
              "not_negligible": (k >= 1 and abs(eff) >= NEGLIGIBLE, "its effect is negligible")}
    return [_finish(c, checks)]


def null_claims_model(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    reach = m.reaching_output()
    out = []
    for v in m.variables:
        nodes = [nd for c in v.columns for nd in m.input_nodes_by_col.get(c, [])]
        if any(nd in reach for nd in nodes):
            continue
        # numeric confirmation: every column moved to its min and max
        dmax = 0.0
        for c in v.columns:
            for val in (np.min(m.X[:, c]), np.max(m.X[:, c])):
                dmax = max(dmax, float(np.max(np.abs(m.logit(m.X, {c: val}) - m.base_logit))))
        c = _claim(part, "null", v.base, variables=[_var_desc(v)],
                   statement={"variable": v.base, "method": "exact"})
        c["estimate"] = {"value": sig(dmax), "unit": "max |change in log-odds|",
                         "method": "exact Null rule (no path to the output over enabled, non-zero-weight "
                                   "edges), confirmed by moving every column to its minimum and maximum"}
        c["n"] = m.X.shape[0]
        c["groups"] = []
        checks = {"no_path": (True, ""), "no_numeric_effect": (dmax <= 1e-12, "moving it changes the output")}
        out.append(_finish(c, checks))
    return out


def null_claim_part(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    c_ = part.contribution
    if c_ is None or float(np.max(np.abs(c_))) >= INERT_CANDIDATE:
        return []
    p_full = _sigmoid(m.base_logit)
    p_abl = _sigmoid(m.base_logit - c_)
    dp = float(np.max(np.abs(p_full - p_abl)))
    dauc = float(default_auc(m.y, p_full) - default_auc(m.y, p_abl))
    c = _claim(part, "null", "inert", variables=[_var_desc(m.var_by_base[b]) for b in part.variables],
               statement={"method": "ablation", "max_change_in_risk": sig(dp)})
    c["estimate"] = {"value": sig(dp), "unit": "max |change in predicted risk|",
                     "max_abs_log_odds": sig(float(np.max(np.abs(c_)))), "auc_change": sig(dauc),
                     "method": "the part switched off (exits at 0, or at their mean when always on)"}
    c["n"] = len(c_)
    c["groups"] = []
    checks = {"small_change": (dp <= INERT_MAX_DP, f"predictions move by up to {dp:.3f}"),
              "auc_unchanged": (abs(dauc) <= INERT_MAX_DAUC, f"training AUC changes by {dauc:.4f}")}
    return [_finish(c, checks)]


# ============================================================================ the layer

TYPE_ORDER = {t: i for i, t in enumerate(("direction", "magnitude", "interaction", "threshold",
                                          "subgroup", "null"))}


def _claim_sort_key(c):
    est = c.get("estimate") or {}
    v = est.get("value")
    return (TYPE_ORDER[c["type"]], -abs(v) if isinstance(v, (int, float)) and v is not None else 0.0, c["key"])


def _dedupe_interactions(parts: List[Part], by_part: Dict[str, List[Dict[str, Any]]]) -> None:
    """An interaction's estimate is measured on the whole model, so a composition and the
    child that realises it (or two copies of one calculation) report the same claim. Keep it
    once, in the most specific part (fewest variables, then the more important part), and
    record the other parts in ``evidence.also_in``."""
    rank = {p.id: (len(p.variables), p.order) for p in parts}
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for p in sorted(parts, key=lambda p: rank[p.id]):
        keep = []
        for c in by_part.get(p.id, []):
            if c["type"] != "interaction":
                keep.append(c)
                continue
            sig_key = (c["key"].split(":", 2)[2], c["estimate"]["value"],
                       tuple(g["n"] for g in c["groups"]))
            if sig_key in seen:
                seen[sig_key]["evidence"].setdefault("also_in", []).append(p.id)
                continue
            seen[sig_key] = c
            keep.append(c)
        by_part[p.id] = keep


def extract_claims(m: PlainModel) -> Dict[str, Any]:
    """Every claim for every part (deterministic order and ids)."""
    parts = build_parts(m)
    root = next((p for p in parts if p.is_root), None)
    total_effect: Dict[str, float] = {}
    by_part: Dict[str, List[Dict[str, Any]]] = {}
    if root is not None:
        root.variables = [v.base for v in m.variables]   # the whole model reads every variable
        dirs = direction_claims(m, root, None)
        for d in dirs:
            b = d["statement"]["variable"]
            if abs(d["estimate"]["value"]) > abs(total_effect.get(b, 0.0)):
                total_effect[b] = d["estimate"]["value"]
        by_part[root.id] = dirs + magnitude_claims(m, root, dirs) + null_claims_model(m, root)
    for p in parts:
        if p.is_root:
            continue
        if p.duplicate_of:
            by_part[p.id] = []
            continue
        cl = direction_claims(m, p, p.input_nodes)
        if p.nonlinear:
            cl += interaction_claims(m, p, total_effect) + threshold_claims(m, p)
        cl += subgroup_claims(m, p) + null_claim_part(m, p)
        by_part[p.id] = cl
    _dedupe_interactions(parts, by_part)
    claims = []
    for p in parts:
        claims.extend(sorted(by_part.get(p.id, []), key=_claim_sort_key))
    width = max(2, len(str(len(claims))))
    for i, c in enumerate(claims):
        c["id"] = f"C{i + 1:0{width}d}"
    return {"parts": parts, "claims": claims}


def part_record(m: PlainModel, p: Part) -> Dict[str, Any]:
    return {"id": p.id, "name": p.name, "kind": p.kind, "role": "root" if p.is_root else
            ("duplicate" if p.duplicate_of else "part"), "duplicate_of": p.duplicate_of,
            "parent": p.parent, "children": p.children, "order": p.order,
            "variables": [m.var_by_base[b].label for b in p.variables],
            "variable_keys": list(p.variables), "upstream_nodes": len(p.upstream_nodes),
            "nonlinear": p.nonlinear, "gated": p.gated,
            "contribution_sd": sig(float(np.std(p.contribution))) if p.contribution is not None else None,
            "original_hypothesis": p.ann.get("hypothesis")}


def units_record(m: PlainModel) -> Dict[str, Any]:
    vs = []
    for v in m.variables:
        rec = {"base": v.base, "label": v.label, "kind": v.kind, "unit": v.unit,
               "columns": [m.feature_names[c] for c in v.columns], "decoded": v.decoded}
        if v.is_category():
            rec["levels"] = [{"code": code, "column": m.feature_names[c], **_lvl(v, code)}
                             for c, code in zip(v.columns, v.codes)]
            rec["reference"] = {"code": v.reference, "label": v.level(v.reference)["label"]}
        else:
            x = v.raw_train
            rec.update({"min": sig(x.min()), "p10": sig(np.percentile(x, 10)), "median": sig(np.median(x)),
                        "p90": sig(np.percentile(x, 90)), "max": sig(x.max()),
                        "mean": sig(v.mean[0]), "scale": sig(v.scale[0])})
        vs.append(rec)
    return {"method": "raw = z * scale + mean, with the split's StandardScaler parameters "
                      "(scaler_params mean and scale, fitted on the training rows)",
            "round_trip_max_abs_err": sig(m.round_trip_err) if m.round_trip_err is not None else None,
            "round_trip_note": ("max |decoded - stored raw| and |re-encoded - model input| over the "
                                "training rows" if m.round_trip_err is not None else
                                "stored raw values not available: not checked"),
            "variables": vs}


def claims_hash(claims: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(_clean(claims), sort_keys=True).encode()).hexdigest()


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _clean(float(o))
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def build_layer(m: PlainModel, meta: Dict[str, Any]) -> Dict[str, Any]:
    """The plain-language layer document (claims, template text; no draft)."""
    from .plain_language_templates import render_template
    res = extract_claims(m)
    parts = res["parts"]
    claims = _clean(res["claims"])
    layer = {
        "schema_version": LAYER_SCHEMA_VERSION,
        "label": meta.get("label"),
        "genome_id": meta.get("genome_id"),
        "explanation_id": meta.get("explanation_id"),
        "dataset": meta.get("dataset"),
        "split": {**(meta.get("split") or {}), "primary": "train", "check": "test",
                  "train_n": int(m.X.shape[0]), "test_n": int(m.Xt.shape[0]) if m.Xt is not None else 0},
        "outcome": m.codebook.get("outcome"),
        "codebook": {"id": m.codebook.get("id"), "source": m.codebook.get("source"),
                     "gaps": codebook_gaps(m.codebook, m.feature_names)},
        "units": units_record(m),
        "reference_patient": reference_patient(m),
        "settings": {"seed": SEED, "bootstrap_B": BOOTSTRAP_B, "small_n": SMALL_N,
                     "negligible": NEGLIGIBLE, "slight": SLIGHT, "strong": STRONG,
                     "min_interaction": MIN_INTERACTION, "min_threshold_effect": MIN_THRESHOLD_EFFECT,
                     "max_opposite_sign_share": MAX_CONTRADICTING,
                     "max_threshold_spread": MAX_THRESHOLD_SPREAD,
                     "max_interactions_per_part": MAX_INTERACTIONS_PER_PART,
                     "top_magnitudes": TOP_MAGNITUDES, "effect_scale": "the model's log-odds "
                     "(output pre-activation); odds ratios are exp of it"},
        "parts": [_clean(part_record(m, p)) for p in parts],
        "claims": claims,
        "claims_hash": claims_hash(claims),
        "counts": {"claims": len(claims), "verified": sum(1 for c in claims if c["verified"]),
                   "not_supported": sum(1 for c in claims if not c["verified"]),
                   "by_type": {t: [sum(1 for c in claims if c["type"] == t and c["verified"]),
                                   sum(1 for c in claims if c["type"] == t)] for t in CLAIM_TYPES}},
        "drafted": None,
        "verification": {"verified": None, "verified_by": None, "verified_at": None, "notes": ""},
    }
    layer["template"] = render_template(layer)
    return _clean(layer)


# ============================================================================ loading from the DB

def load_model(genome_id: str, db=None) -> Tuple[PlainModel, Dict[str, Any]]:
    """Build the :class:`PlainModel` for a genome from the database (read-only)."""
    import uuid
    from sqlalchemy import text
    from mcp_server.helpers import build_engine, load_split_data
    from mcp_server.server import get_db
    from ..db.models import Dataset, DatasetSplit, Experiment, Explanation, Genome, Population
    from ..export.collect import resolve_target
    db = db or get_db()
    with db.session_scope() as s:
        if s.bind.dialect.name == "postgresql":
            s.execute(text("SET TRANSACTION READ ONLY"))
        genome_id, explanation_id = resolve_target(s, genome=genome_id)
        g = s.get(Genome, uuid.UUID(genome_id))
        pop = s.get(Population, g.population_id)
        exp = s.get(Experiment, pop.experiment_id)
        ds = s.get(Dataset, exp.dataset_id)
        split = s.get(DatasetSplit, exp.split_id)
        ex = s.get(Explanation, uuid.UUID(explanation_id))
        ops = copy.deepcopy(list(ex.operations or []))
        engine = build_engine(s, genome_id)
        X_tr, y_tr, names, *_ = load_split_data(s, str(split.id), "train", 1.0, 10**9)
        X_te, y_te, *_ = load_split_data(s, str(split.id), "test", 1.0, 10**9)
        X_full, _ = ds.get_data()
        raw_tr = np.asarray(X_full[split.train_indices], dtype=float)
        raw_te = np.asarray(X_full[split.test_indices], dtype=float)
        if split.scaler_type != "StandardScaler":
            raise ValueError(f"scaler {split.scaler_type!r} is not supported (StandardScaler only)")
        mean = np.asarray(split.scaler_params["mean"], dtype=float)
        scale = np.asarray(split.scaler_params["scale"], dtype=float)
        meta = {"genome_id": genome_id, "explanation_id": explanation_id,
                "dataset": {"id": str(ds.id), "name": ds.name},
                "split": {"id": str(split.id), "scaler_type": split.scaler_type},
                "experiment": {"id": str(exp.id), "name": exp.name}}
        names = list(names or ds.feature_names)
        dsname = ds.name
    cb = find_codebook(dsname, names) or generic_codebook(names)
    m = PlainModel(engine.current_state, ops, X_tr, y_tr, X_te, y_te, names, mean, scale, cb,
                   raw_train=raw_tr, raw_test=raw_te)
    return m, meta
