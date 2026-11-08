"""Plain-language explanation layer: structured claims, checked against the model and data.

Every plain-language statement about an explained genome rests on one or more **claims**.
A claim is derived from an annotation (a "part" of the explained network), computed on the
model state and the split's data, and checked automatically. Only verified claims may be
turned into sentences; the others are listed as "not supported by the model".

Two separate questions are answered for every claim (layer schema 2):

* **Is it true of the model?** (``verified``). Checked on the training patients: the effect is
  non-zero and goes the same way for (nearly) every patient ("consistently across patients");
  thresholds have one fold; interactions have a large enough group difference whose sign the
  part's own counterfactual confirms. The spread of per-patient effects is reported as a
  **range across patients** (10th to 90th percentile), not as a confidence interval.
* **How strongly may it be said?** (``confidence``), from (a) the size of the change in
  **predicted risk for a typical patient** (probability scale), (b) the **test-split check**
  (bootstrap over test patients, seed 42) and (c) n. The words are defined in
  ``CONFIDENCE_RULES`` and used by the template renderer, the writing protocol and the lint.

Claim types (``CLAIM_TYPES``): ``direction``, ``threshold``, ``interaction``, ``magnitude``,
``subgroup``, ``null``, and ``rule`` (a count rule from the codebook, e.g. MONK-2: "how many
attributes take their first value").

Magnitudes are led by the change in predicted risk for a typical patient (median of every
number, most common category; for a part that does not act on that patient, the typical
patient it acts on). Log-odds and odds ratios stay in the evidence. The risk scale is pluggable
(:class:`BinaryRisk`; :class:`SurvivalRisk` gives the predicted risk at a stated horizon from a
Breslow baseline, for ``explaneat.survival`` models).

Units: data columns are z-scored by the split's ``StandardScaler`` (``scaler_params``);
``raw = z * scale + mean``, round trip recorded. One-hot columns ``base:code`` (and documented
binary columns) are decoded with a codebook (:mod:`.plain_language_codebooks`); codes it does not
document are reported as gaps, never guessed.

Everything is deterministic: bootstraps use ``numpy.random.default_rng(SEED)`` afresh for every
claim.
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
from .plain_language_codebooks import (base_of, codebook_checks, codebook_gaps, find_codebook,
                                       generic_codebook, level_info)

LAYER_SCHEMA_VERSION = "2.0.0"
CLAIM_TYPES = ("direction", "threshold", "interaction", "magnitude", "subgroup", "null", "rule")

SEED = 42
BOOTSTRAP_B = 1000
SMALL_N = 20                 # groups smaller than this are flagged small_n
NEGLIGIBLE = 0.05            # |log-odds| below this: no effect worth a claim
MIN_INTERACTION = 0.1        # |difference in log-odds| an interaction must reach
MIN_THRESHOLD_EFFECT = 0.1   # |log-odds| the effect past a fold must reach
MAX_CONTRADICTING = 0.10     # share of patients allowed the opposite sign
MAX_THRESHOLD_SPREAD = 0.20  # IQR of the folds, as a share of the observed range
MIN_THRESHOLD_PATIENTS = 5
MAX_INTERACTIONS_PER_PART = 4
TOP_MAGNITUDES = 5
INERT_CANDIDATE = 0.25
INERT_MAX_DP = 0.02
INERT_MAX_DAUC = 0.002
RULE_PURITY = 0.95
GRID_POINTS = 101
FLAT_TOL = 1e-9
SCREEN_ROWS = 4000           # interaction candidates are ranked on at most this many rows
CHUNK_ROWS = 200_000         # forward passes are chunked to this many rows
SIG = 6

# Confidence words (probability scale at the typical patient; test split; n).
PROB_SLIGHT = 0.05           # |change in predicted risk| below 5 points: "slightly"
PROB_LARGE = 0.20            # at least 20 points: a large change
CONFIDENCE_RULES = {
    "clearly": ("a meaningful change in predicted risk for a typical patient (at least "
                f"{PROB_SLIGHT:.0%} points), the test split agrees (same sign, and its bootstrap "
                "interval over test patients excludes no benefit), and every group has at least "
                f"{SMALL_N} training patients"),
    "slightly": f"the change in predicted risk for a typical patient is below {PROB_SLIGHT:.0%} points",
    "the model has little data on this": f"a compared group has fewer than {SMALL_N} training patients",
    "on the training data only": ("the test split disagrees (opposite sign, or no benefit on "
                                  "test patients) or is unclear (its interval includes no benefit), "
                                  "or no test check is available"),
    "consistently across patients": ("the claim holds for at least "
                                     f"{1 - MAX_CONTRADICTING:.0%} of training patients (range "
                                     "across patients); this is not a significance statement"),
}


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
    mm = e[idx].mean(axis=1)
    return float(e.mean()), [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]


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
    """Size class of a log-odds change (kept for the evidence; words use the risk scale)."""
    a = abs(x)
    if a < NEGLIGIBLE:
        return "negligible"
    if a < 0.25:
        return "slight"
    if a < 1.0:
        return "moderate"
    return "strong"


def _sig_list(xs):
    return [sig(x) for x in xs]


def _or(x):
    return sig(math.exp(x)) if x is not None and math.isfinite(x) else None


def spread(e: np.ndarray) -> Dict[str, Any]:
    """Mean and range across patients (10th-90th percentile) of per-patient effects."""
    e = np.asarray(e, dtype=float)
    if len(e) == 0:
        return {"mean": None, "range_across_patients": [None, None], "share_opposite_sign": None}
    mean = float(e.mean())
    s = np.sign(mean)
    return {"mean": sig(mean),
            "range_across_patients": _sig_list(np.percentile(e, [10, 90])),
            "range_basis": "10th to 90th percentile of per-patient effects (training patients)",
            "share_opposite_sign": sig(float(np.mean(np.sign(e) == -s))) if s != 0 else 1.0,
            "share_unaffected": sig(float(np.mean(np.abs(e) < 1e-12)))}


def estimate_block(value: float, e: Optional[np.ndarray], method: str,
                   unit: str = "log-odds (model)") -> Dict[str, Any]:
    out = {"value": sig(value), "unit": unit, "method": method, "odds_ratio": _or(value)}
    if e is not None:
        sp = spread(e)
        out["range_across_patients"] = sp["range_across_patients"]
        out["range_basis"] = sp.get("range_basis")
        out["odds_ratio_range"] = [_or(x) if x is not None else None for x in sp["range_across_patients"]]
    return out


# ============================================================================ risk scales

class BinaryRisk:
    """Predicted probability of the positive class from the model's log-odds."""
    kind = "binary"
    horizon = None

    def risk(self, eta):
        return _sigmoid(eta)

    def loss_p(self, p, y):
        """Per-row log-loss of predicted probabilities (the test check's yardstick)."""
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        y = np.asarray(y, dtype=float)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def describe(self) -> Dict[str, Any]:
        return {"kind": "binary", "label": "predicted risk", "horizon": None,
                "formula": "risk = 1 / (1 + exp(-log-odds))"}


class SurvivalRisk:
    """Predicted risk by a horizon for a Cox model: ``1 - S0(t)^exp(g)`` (Breslow baseline).

    Use with ``explaneat.survival`` models (identity output = log-risk ``g``). The test check
    needs a survival loss, which is not implemented yet: survival claims get the test status
    "not available" (so never "clearly").
    """
    kind = "survival"

    def __init__(self, baseline_survival_at_horizon: float, horizon: float, horizon_label: str):
        self.s0 = float(baseline_survival_at_horizon)
        self.horizon = float(horizon)
        self.horizon_label = horizon_label

    @classmethod
    def from_breslow(cls, baseline, horizon: float, horizon_label: str) -> "SurvivalRisk":
        """From a fitted ``explaneat.survival.breslow.BreslowBaseline``."""
        s0 = float(np.asarray(baseline.baseline_survival(np.array([horizon]))).ravel()[0])
        return cls(s0, horizon, horizon_label)

    def risk(self, eta):
        return 1.0 - np.power(self.s0, np.exp(np.asarray(eta, dtype=float)))

    loss_p = None

    def describe(self) -> Dict[str, Any]:
        return {"kind": "survival", "label": f"predicted {self.horizon_label} risk",
                "horizon": self.horizon, "baseline_survival_at_horizon": sig(self.s0),
                "formula": "risk = 1 - S0(horizon) ** exp(log-risk)"}


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
    implicit_off: Optional[str] = None   # binary column: the code when the column is 0

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

    @property
    def all_codes(self) -> List[str]:
        return list(self.codes) + ([self.implicit_off] if self.implicit_off is not None else [])

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
        s = f"{float(value):,.{d}f}" if abs(float(value)) >= 10000 else f"{float(value):.{d}f}"
        if self.unit:
            s += f" {self.unit}"
        alt = self.cb.get("alt_unit")
        if alt:
            s += f" ({float(value) * alt['factor']:.{alt['decimals']}f} {alt['unit']})"
        return s

    def counts(self) -> Dict[str, int]:
        return {c: int(np.sum(self.raw_train == c)) for c in self.all_codes}


def _category_codes(raw_cols: np.ndarray, codes: List[str], implicit_off: Optional[str] = None) -> np.ndarray:
    """Per row, the code whose one-hot column is on (None if none or several are on)."""
    out = np.empty(raw_cols.shape[0], dtype=object)
    for i, row in enumerate(raw_cols):
        on = [codes[j] for j in range(len(codes)) if row[j] > 0.5]
        out[i] = on[0] if len(on) == 1 else (implicit_off if not on else None)
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
        is_binary = (not is_cat and len(cols) == 1 and cb.get("kind") == "binary")
        if is_cat or is_binary:
            if is_cat:
                codes = [feature_names[c].split(":", 1)[1] for c in cols]
                off = None
            else:
                codes, off = ["1"], "0"
            v = Variable(b, "category", cols, codes, cb, mean[cols], scale[cols],
                         _category_codes(raw_train[:, cols], codes, off),
                         _category_codes(raw_test[:, cols], codes, off) if raw_test is not None else None,
                         implicit_off=off)
            counts = v.counts()
            allc = v.all_codes
            v.reference = max(allc, key=lambda c: (counts[c], -allc.index(c)))
            v.decoded = all(level_info(cb, c)["decoded"] for c in allc)
        else:
            kind = cb.get("kind") if cb.get("kind") in ("continuous", "count") else "continuous"
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
    if hi <= lo:   # mostly one value (e.g. capital gains of 0): to the median of the larger values
        above = v.raw_train[v.raw_train > lo]
        hi = round(float(np.median(above)), v.decimals) if len(above) else float(np.max(v.raw_train))
    return lo, hi


# ============================================================================ level classes

def level_classes(v: Variable, read_codes: Sequence[str]) -> List[Dict[str, Any]]:
    """Categories a scope can tell apart: every level it reads, plus one class for the rest.

    A scope that reads only some one-hot columns cannot distinguish the levels it does not
    read, so they form a single class ("any other ...") instead of many identical claims.
    """
    counts = v.counts()
    read = [c for c in v.all_codes if c in set(read_codes)]
    others = [c for c in v.all_codes if c not in set(read)]
    classes = [[c] for c in read] + ([others] if others else [])
    out = []
    for codes in classes:
        rep = max(codes, key=lambda c: (counts[c], -v.all_codes.index(c)))
        if len(codes) == 1:
            li = v.level(codes[0])
            d = {"label": li["label"], "group": li["group"], "cond": li["cond"], "decoded": li["decoded"]}
        else:
            names = ", ".join(v.level(c)["label"] for c in read) or "the listed values"
            d = {"label": f"{v.label} other than {names}",
                 "group": f"{v.cb.get('patients', 'patients')} with {v.label} other than {names}",
                 "cond": f"{v.label} is something other than {names}",
                 "decoded": all(v.level(c)["decoded"] for c in read)}
        out.append({**d, "codes": list(codes), "rep": rep, "key": "+".join(codes) if len(codes) <= 3
                    else f"other{len(codes)}", "n_train": int(sum(counts[c] for c in codes)),
                    "is_reference": v.reference in codes})
    return out


def class_rows(v: Variable, cls: Dict[str, Any], test: bool = False) -> np.ndarray:
    raw = v.raw_test if test else v.raw_train
    return np.isin(raw.astype(object), np.array(cls["codes"], dtype=object))


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
                 mean, scale, codebook: Dict[str, Any], raw_train=None, raw_test=None,
                 risk_scale=None):
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
        self.risk_scale = risk_scale or BinaryRisk()
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
        self.input_nodes_by_col: Dict[int, List[str]] = {}
        for n in structure.nodes:
            if n.type == NodeType.INPUT:
                self.input_nodes_by_col.setdefault(input_column(n.id), []).append(n.id)
        self.base_logit = self.logit(self.X)
        self.base_logit_test = None if self.Xt is None else self.logit(self.Xt)
        n = self.X.shape[0]
        step = max(1, int(math.ceil(n / SCREEN_ROWS)))
        self.screen_idx = np.arange(0, n, step)
        reach = self.reaching_output()
        self.read_columns = {c for c, nodes in self.input_nodes_by_col.items()
                             if any(nd in reach for nd in nodes)}

    # ------------------------------------------------------------------ forward passes
    def logit(self, X, assign: Optional[Dict[int, Any]] = None, scope: Optional[Sequence[str]] = None,
              ablate: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Log-odds (log-risk for survival) with data columns set by ``assign``.

        ``assign`` maps a column to a value or a per-row array. ``scope=None`` sets the column
        everywhere; a list of input node ids sets it only on those copies (the change reaches
        the output only through the part those copies feed). ``ablate`` fixes nodes at
        constants. Large inputs are processed in chunks.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if n > CHUNK_ROWS:
            outs = []
            for s in range(0, n, CHUNK_ROWS):
                sl = slice(s, min(n, s + CHUNK_ROWS))
                a = None if assign is None else {
                    c: (np.asarray(v)[sl] if np.ndim(v) else v) for c, v in assign.items()}
                outs.append(self.logit(X[sl], a, scope, ablate))
            return np.concatenate(outs)
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

    def risk(self, eta):
        return self.risk_scale.risk(eta)

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
            return [c for c in var.columns if c in self.read_columns]
        return [c for c in var.columns if any(nd in scope for nd in self.input_nodes_by_col.get(c, []))]

    def read_codes(self, var: Variable, scope) -> List[str]:
        cols = self.scope_columns(var, scope)
        return [var.codes[var.columns.index(c)] for c in cols]


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


# ============================================================================ groups and the typical patient

def describe_group(m: PlainModel, rows: np.ndarray, var_names: Sequence[str]) -> Dict[str, Any]:
    """Who is in a group of training patients: category shares and number ranges."""
    rows = np.asarray(rows, dtype=bool)
    out: Dict[str, Any] = {"n": int(rows.sum()), "n_other": int((~rows).sum()), "conditions": [],
                           "categories": [], "numbers": []}
    for b in var_names:
        v = m.var_by_base[b]
        if v.is_category():
            for code in v.all_codes:
                ins = float(np.mean(v.raw_train[rows] == code)) if rows.any() else 0.0
                oth = float(np.mean(v.raw_train[~rows] == code)) if (~rows).any() else 0.0
                if ins == 0 and oth < 0.05:
                    continue
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


def typical_raw(m: PlainModel, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """The typical patient (raw values): median of every number and the most common category,
    over all training patients or the given rows."""
    mask = np.ones(m.X.shape[0], dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    ref = np.zeros(m.raw_train.shape[1])
    for v in m.variables:
        if v.is_category():
            codes = v.raw_train[mask]
            allc = v.all_codes
            counts = {c: int(np.sum(codes == c)) for c in allc}
            best = max(allc, key=lambda c: (counts[c], -allc.index(c)))
            for c, code in zip(v.columns, v.codes):
                ref[c] = 1.0 if code == best else 0.0
        else:
            ref[v.columns[0]] = float(np.median(m.raw_train[mask, v.columns[0]]))
    return ref


def typical_values(m: PlainModel, raw: np.ndarray) -> Dict[str, str]:
    out = {}
    for v in m.variables:
        if v.is_category():
            code = next((cd for c, cd in zip(v.columns, v.codes) if raw[c] > 0.5), v.implicit_off)
            out[v.base] = v.level(code)["label"] if code is not None else "none"
        else:
            out[v.base] = v.fmt(raw[v.columns[0]])
    return out


def _z(m: PlainModel, raw: np.ndarray) -> np.ndarray:
    return ((raw - m.mean) / m.scale)[None, :]


def reference_patient(m: PlainModel) -> Dict[str, Any]:
    raw = typical_raw(m)
    p = float(m.risk(m.logit(_z(m, raw)))[0])
    return {"definition": "the typical patient: the median training patient for every number and "
                          "the most common training category for every category (not a real patient)",
            "values": typical_values(m, raw), "predicted_risk": sig(p),
            "risk_scale": m.risk_scale.describe(), "raw": _sig_list(raw)}


def risk_pair(m: PlainModel, raw: np.ndarray, var: Variable, a, b, scope=None,
              extra_raw: Optional[Dict[int, float]] = None) -> Tuple[float, float]:
    """Predicted risk of the patient ``raw`` with ``var`` at ``a`` and at ``b`` (on ``scope``)."""
    z = _z(m, raw)
    extra = None
    if extra_raw:
        extra = {c: float((val - m.mean[c]) / m.scale[c]) for c, val in extra_raw.items()}
        for c, val in extra.items():
            z[0, c] = val
    pa = float(m.risk(m.set_logit(z, var, a, scope))[0])
    pb = float(m.risk(m.set_logit(z, var, b, scope))[0])
    return pa, pb


def risk_block(pa: float, pb: float, reference: str, values: Optional[Dict[str, str]] = None,
               **kw) -> Dict[str, Any]:
    out = {"reference": reference, "from": sig(pa), "to": sig(pb), "delta": sig(pb - pa)}
    if values is not None:
        out["reference_values"] = values
    out.update(kw)
    return out


# ============================================================================ test checks and confidence

def support(v: Variable) -> List[Tuple[Any, float]]:
    """The variable's training distribution as (value, weight) pairs: every category (or every
    distinct value when there are at most 25), else 20 equally weighted quantiles."""
    if v.is_category():
        counts = v.counts()
        tot = sum(counts.values()) or 1
        return [(c, counts[c] / tot) for c in v.all_codes if counts[c] > 0]
    vals, cnt = np.unique(v.raw_train, return_counts=True)
    if len(vals) <= 25:
        return [(float(x), float(k) / cnt.sum()) for x, k in zip(vals, cnt)]
    qs = np.percentile(v.raw_train, np.arange(2.5, 100, 5))
    return [(float(q), 1.0 / len(qs)) for q in qs]


def _removed_risk(m: PlainModel, X, var: Optional[Variable], scope, part: Optional[Part]) -> np.ndarray:
    """Predicted risk with the mechanism's information removed, keeping its average effect:
    the prediction averaged over the variable's training distribution (on the scope), or over
    the training distribution of the part's outputs."""
    if var is not None:
        acc = np.zeros(X.shape[0])
        for val, w in support(var):
            acc += w * m.risk(m.set_logit(X, var, val, scope))
        return acc
    vals = m.ev.forward(m.X[m.screen_idx])
    rows = np.linspace(0, len(m.screen_idx) - 1, 20).round().astype(int)
    acc = np.zeros(X.shape[0])
    for r in rows:
        acc += m.risk(m.logit(X, ablate={x: float(vals[x][r]) for x in part.exits})) / len(rows)
    return acc


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC by the rank-sum formula (ties get average ranks)."""
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def test_benefit(m: PlainModel, var: Optional[Variable] = None, scope=None,
                 part: Optional[Part] = None) -> Dict[str, Any]:
    """Does the model's use of this mechanism help it rank test patients?

    benefit = test AUC of the full model minus test AUC with the mechanism's information removed
    (positive: it helps). Removing averages the prediction over the variable's training
    distribution on the claim's scope (or over the part's training outputs), so only the
    information is lost. The interval is a paired bootstrap over test patients (seed 42;
    ``BOOTSTRAP_B`` resamples, 200 when there are more than 2,000 test patients). Discrimination
    is used rather than log-loss, which would also penalise an over-confident model for its
    confidence. The training value is reported for comparison.
    """
    if getattr(m.risk_scale, "loss_p", None) is None or m.Xt is None or m.yt is None or len(m.yt) == 0:
        return {"split": "test", "status": "not available",
                "note": "no test split, or no test measure for this risk scale (survival)"}
    yt = m.yt
    full = m.risk(m.base_logit_test)
    rem = _removed_risk(m, m.Xt, var, scope, part)
    b = fast_auc(yt, full) - fast_auc(yt, rem)
    B = BOOTSTRAP_B if len(yt) <= 2000 else 200
    idx = _rng().integers(0, len(yt), size=(B, len(yt)))
    diffs = np.array([fast_auc(yt[i], full[i]) - fast_auc(yt[i], rem[i]) for i in idx])
    diffs = diffs[np.isfinite(diffs)]
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if len(diffs) else [float("nan")] * 2
    Xs, ys = m.X[m.screen_idx], m.y[m.screen_idx]
    tr = fast_auc(ys, m.risk(m.base_logit[m.screen_idx])) - fast_auc(ys, _removed_risk(m, Xs, var, scope, part))
    return {"split": "test", "n": int(len(yt)), "benefit_mean": sig(b), "benefit_ci95": _sig_list(ci),
            "bootstrap_B": int(B), "train_benefit_mean": sig(tr),
            "measure": "test AUC of the full model minus test AUC with the mechanism's information removed "
                       "(prediction averaged over its training distribution); positive = it helps; paired "
                       "bootstrap over test patients, seed 42"}


def _status(benefit: Dict[str, Any], same_sign: Optional[bool] = True,
            diff_ci: Optional[List[float]] = None) -> str:
    if benefit.get("status") == "not available" and diff_ci is None:
        return "not available"
    if same_sign is False:
        return "disagrees"
    if diff_ci is not None:
        if diff_ci[0] is None:
            return "unclear"
        return "agrees" if (diff_ci[0] > 0 or diff_ci[1] < 0) else "unclear"
    b, ci = benefit.get("benefit_mean"), benefit.get("benefit_ci95") or [None, None]
    if b is None:
        return "not available"
    if b < 0:
        return "disagrees"
    return "agrees" if (b > 0 and ci[0] is not None and ci[0] > 0) else "unclear"


def confidence(c: Dict[str, Any]) -> Dict[str, Any]:
    """The confidence words a statement of this claim may use (``CONFIDENCE_RULES``)."""
    r = c.get("risk") or {}
    delta = abs(r.get("delta") or 0.0)
    size = "slight" if delta < PROB_SLIGHT else ("large" if delta >= PROB_LARGE else "meaningful")
    test = (c.get("test_check") or {}).get("status", "not available")
    n_ok = not c.get("small_n")
    quals = []
    if size == "slight":
        quals.append("slightly")
    if not n_ok:
        quals.append("the model has little data on this")
    if test != "agrees":
        quals.append("on the training data only")
    word = "clearly" if (size != "slight" and test == "agrees" and n_ok) else None
    return {"word": word, "size": size, "delta_risk": sig(delta), "test": test, "n_ok": n_ok,
            "qualifiers": quals}


# ============================================================================ claim builders

def _claim(part: Part, ctype: str, key: str, **kw) -> Dict[str, Any]:
    c = {"key": f"{part.id}:{ctype}:{key}", "type": ctype,
         "annotation": {"id": part.id, "name": part.name},
         "scope": "model" if part.is_root else "part", "split": "train", "test_check": None,
         "small_n": False, "small_n_groups": [], "verified": False, "reasons": [],
         "checks": {}, "risk": None}
    c.update(kw)
    return c


def _finish(c: Dict[str, Any], checks: Dict[str, Tuple[bool, str]]):
    c["checks"] = {k: bool(v[0]) for k, v in checks.items()}
    c["reasons"] = [msg for ok, msg in checks.values() if not ok]
    c["verified"] = all(ok for ok, _ in checks.values())
    if c["verified"]:
        c["reasons"] = ["all checks passed"]
    if c["type"] != "null":
        c["confidence"] = confidence(c)
    return c


def _var_desc(v: Variable) -> Dict[str, Any]:
    return {"base": v.base, "label": v.label, "short": v.short, "unit": v.unit, "kind": v.kind,
            "decoded": v.decoded}


def _settings(m: PlainModel, v: Variable, scope) -> List[Tuple[Any, Any, Optional[Dict], Optional[Dict]]]:
    """(from, to, from_class, to_class): changes in ``v`` the scope can see."""
    if not v.is_category():
        lo, hi = _lo_hi(v)
        return [(lo, hi, None, None)]
    read = m.read_codes(v, scope)
    if not read:
        return []
    classes = level_classes(v, read)
    ref = next(c for c in classes if c["is_reference"])
    return [(ref["rep"], c["rep"], ref, c) for c in classes if c is not ref]


def _change_desc(v: Variable, a, b, ca=None, cb=None) -> Dict[str, Any]:
    if v.is_category():
        def d(code, cls):
            cls = cls or {}
            return {"code": code, "codes": cls.get("codes", [code]), "label": cls.get("label"),
                    "group": cls.get("group"), "cond": cls.get("cond"),
                    "decoded": cls.get("decoded", True), "n_train": cls.get("n_train")}
        return {"from": d(a, ca), "to": d(b, cb)}
    return {"from": {"value": a, "text": v.fmt(a)}, "to": {"value": b, "text": v.fmt(b)},
            "basis": "10th and 90th percentile of the training patients"}


def _part_typical(m: PlainModel, part: Part, var: Variable, a, b, scope):
    """Typical patient for a statement about a part: the overall typical patient, or (if the
    part does not move that patient's risk) the typical patient the part acts on."""
    raw = typical_raw(m)
    pa, pb = risk_pair(m, raw, var, a, b, scope)
    if abs(pb - pa) > 1e-9 or part.is_root or part.contribution is None:
        return pa, pb, "typical patient", None
    active = np.abs(part.contribution) > 1e-12
    if active.any() and not active.all():
        raw2 = typical_raw(m, active)
        pa2, pb2 = risk_pair(m, raw2, var, a, b, scope)
        return pa2, pb2, "typical patient this part acts on", typical_values(m, raw2)
    return pa, pb, "typical patient", None


def direction_claims(m: PlainModel, part: Part, scope) -> List[Dict[str, Any]]:
    out = []
    for bname in part.variables:
        v = m.var_by_base[bname]
        for (a, z, ca, cz) in _settings(m, v, scope):
            e = m.effect(m.X, v, a, z, scope)
            if np.max(np.abs(e)) < 1e-12:
                continue
            mean = float(e.mean())
            sp = spread(e)
            key = bname if not v.is_category() else f"{bname}:{cz['key']}"
            pa, pb, ref_name, ref_vals = _part_typical(m, part, v, a, z, scope)
            c = _claim(part, "direction", key, variables=[_var_desc(v)],
                       statement={"variable": bname, "change": _change_desc(v, a, z, ca, cz),
                                  "direction": "raises" if mean > 0 else "lowers",
                                  "strength": strength(mean)})
            c["estimate"] = estimate_block(mean, e, "mean over training patients of the change in "
                                           "the model's log-odds when the variable is changed "
                                           + ("everywhere in the model" if scope is None else
                                              "only where this part reads it"))
            c["risk"] = risk_block(pa, pb, ref_name, ref_vals)
            c["evidence"] = {"share_opposite_sign": sp["share_opposite_sign"],
                             "share_unaffected": sp["share_unaffected"]}
            mono = None
            if not v.is_category():
                grid = np.unique(np.round(np.percentile(v.raw_train, np.arange(5, 96, 5)), v.decimals))
                Xs = m.X[m.screen_idx]
                pd = np.array([np.mean(m.set_logit(Xs, v, g, scope)) for g in grid])
                d = np.diff(pd)
                mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
                c["evidence"]["average_curve"] = {"values": _sig_list(grid), "mean_log_odds": _sig_list(pd),
                                                  "rows": int(len(Xs))}
            c["statement"]["monotone"] = mono
            c["n"] = int(len(e))
            groups = []
            if v.is_category():
                for cls in (ca, cz):
                    groups.append({"label": cls["label"], "n": cls["n_train"]})
                    if cls["n_train"] < SMALL_N:
                        c["small_n_groups"].append(cls["label"])
            c["groups"] = groups
            c["small_n"] = bool(c["small_n_groups"])
            et = m.effect(m.Xt, v, a, z, scope) if m.Xt is not None else None
            same = None if et is None or np.max(np.abs(et)) < 1e-12 else bool(np.sign(et.mean()) == np.sign(mean))
            ben = test_benefit(m, v, scope)
            c["test_check"] = {**ben, "model_effect_mean": sig(float(et.mean())) if et is not None else None,
                               "same_sign": same, "status": _status(ben, same)}
            checks = {
                "non_zero": (abs(mean) >= 1e-9, "the mean effect is zero"),
                "consistent_across_patients": (sp["share_opposite_sign"] <= MAX_CONTRADICTING,
                                               f"{sp['share_opposite_sign']:.0%} of patients have the "
                                               f"opposite sign (allowed: {MAX_CONTRADICTING:.0%})"),
            }
            if v.is_category():
                checks["level_observed"] = (all(g["n"] > 0 for g in groups),
                                            "a category in the comparison has no training patients")
            out.append(_finish(c, checks))
    return out


def magnitude_claims(m: PlainModel, part: Part, directions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The biggest verified effects as changes for the typical patient (headline numbers)."""
    def decoded(d):
        ch = d["statement"]["change"]
        return all((ch.get(k) or {}).get("decoded", True) for k in ("from", "to"))
    cands = [d for d in directions if d["verified"] and not d["small_n"] and decoded(d)]
    cands.sort(key=lambda d: (-abs(d["risk"]["delta"] or 0), d["key"]))
    seen, out = set(), []
    raw = typical_raw(m)
    for d in cands:
        b = d["statement"]["variable"]
        if b in seen:
            continue
        seen.add(b)
        v = m.var_by_base[b]
        ch = d["statement"]["change"]
        a = ch["from"].get("code", ch["from"].get("value"))
        z_ = ch["to"].get("code", ch["to"].get("value"))
        e0 = float(m.effect(_z(m, raw), v, a, z_, None)[0])
        pa, pb = risk_pair(m, raw, v, a, z_, None)
        c = _claim(part, "magnitude", d["key"].split(":", 2)[2], variables=[_var_desc(v)],
                   statement={"variable": b, "change": ch, "direction": "raises" if e0 > 0 else "lowers",
                              "strength": strength(e0), "related_direction": d["key"]})
        c["estimate"] = estimate_block(e0, None, "change in the model's log-odds for the typical patient")
        c["risk"] = risk_block(pa, pb, "typical patient")
        c["n"] = d["n"]
        c["groups"] = d.get("groups", [])
        c["small_n_groups"] = list(d.get("small_n_groups", []))
        c["small_n"] = d.get("small_n", False)
        c["test_check"] = dict(d["test_check"])
        checks = {"non_zero": (abs(e0) >= 1e-9, "no change for the typical patient"),
                  "agrees_with_direction": (np.sign(e0) == np.sign(d["estimate"]["value"]),
                                            "the sign for the typical patient differs from the "
                                            "average direction")}
        out.append(_finish(c, checks))
        if len(out) >= TOP_MAGNITUDES:
            break
    return out


def _ice(m: PlainModel, v: Variable, grid: np.ndarray, scope, rows: np.ndarray) -> np.ndarray:
    """(len(rows), len(grid)) log-odds with ``v`` set to each grid value on the scope."""
    X = m.X[rows]
    G = len(grid)
    out = np.empty((X.shape[0], G))
    per = max(1, CHUNK_ROWS // G)
    col = v.columns[0]
    for s in range(0, X.shape[0], per):
        Xc = X[s:s + per]
        Xr = np.repeat(Xc, G, axis=0)
        z = (np.tile(grid, Xc.shape[0]) - v.mean[0]) / v.scale[0]
        out[s:s + per] = m.logit(Xr, {col: z}, scope).reshape(Xc.shape[0], G)
    return out


def _folds(L: np.ndarray, grid: np.ndarray):
    """Per patient: the fold of a piecewise-linear curve that is flat on one side.

    Returns (onset, offset, depth_on, depth_off). depth = how far the curve is from the
    straight-line continuation of its moving part at the flat end (log-odds).
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


def threshold_claims(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    out = []
    for bname in part.variables:
        v = m.var_by_base[bname]
        if v.is_category():
            continue
        lo, hi = float(np.min(v.raw_train)), float(np.max(v.raw_train))
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, GRID_POINTS)
        rows_all = np.arange(m.X.shape[0])
        L = _ice(m, v, grid, part.input_nodes, rows_all)
        onset, offset, d_on, d_off = _folds(L, grid)
        n_on, n_off = int(np.sum(~np.isnan(onset))), int(np.sum(~np.isnan(offset)))
        if max(n_on, n_off) < MIN_THRESHOLD_PATIENTS:
            continue
        kind = "above" if n_on >= n_off else "below"
        t_all, depth = (onset, d_on) if kind == "above" else (offset, d_off)
        with_fold = ~np.isnan(t_all)
        best = None
        cands: List[Tuple[Optional[Dict[str, Any]], np.ndarray]] = [(None, with_fold)]
        for w in part.variables:
            W = m.var_by_base[w]
            if w == bname or not W.is_category():
                continue
            for code in W.all_codes:
                g = with_fold & (W.raw_train == code)
                if g.sum() >= MIN_THRESHOLD_PATIENTS:
                    li = W.level(code)
                    cands.append(({"variable": w, "code": code, "label": li["label"], "group": li["group"],
                                   "cond": li["cond"], "decoded": li["decoded"]}, g))
        for cond, g in cands:
            q1, q3 = np.percentile(t_all[g], [25, 75])
            sp_ = float((q3 - q1) / (hi - lo))
            cand = (sp_ <= MAX_THRESHOLD_SPREAD, int(g.sum()), cond, g, sp_)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
            if cond is None and cand[0]:
                break
        _, _, cond, rows, spread_share = best
        ts = t_all[rows]
        med = float(np.median(ts))
        q1, q3 = np.percentile(ts, [25, 75])
        hinge = float(np.median(depth[rows]))
        p5, p95 = np.percentile(v.raw_train, [5, 95])
        va, vb = (lo, p95) if kind == "above" else (p5, hi)
        eff = (m.set_logit(m.X[rows], v, vb, part.input_nodes) - m.set_logit(m.X[rows], v, va, part.input_nodes))
        emean = float(eff.mean())
        sp = spread(eff)
        actual = v.raw_train[rows]
        past = int(np.sum(actual > ts)) if kind == "above" else int(np.sum(actual < ts))
        who = describe_group(m, rows, [x for x in part.variables if x != bname])
        conds = [x["group"] for x in who["conditions"]]
        if cond is not None and cond["group"] not in conds:
            conds = [cond["group"]] + conds
        th = {"value": sig(med), "text": v.fmt(round(med, v.decimals))}
        if v.kind == "count":
            step = int(math.floor(med)) + 1
            th.update({"count_step": step, "text": v.fmt(step)})
        raw_t = typical_raw(m, rows)
        pa, pb = risk_pair(m, raw_t, v, va, vb, part.input_nodes)
        c = _claim(part, "threshold", bname, variables=[_var_desc(v)],
                   statement={"variable": bname, "kind": kind, "threshold": th,
                              "direction": "raises" if emean > 0 else "lowers",
                              "strength": strength(emean), "group": cond, "conditions": conds,
                              "change": {"from": {"value": sig(va), "text": v.fmt(round(va, v.decimals))},
                                         "to": {"value": sig(vb), "text": v.fmt(round(vb, v.decimals))}}})
        c["estimate"] = {
            "threshold": {"value": sig(med), "unit": v.unit or "raw units",
                          "iqr_across_patients": _sig_list([q1, q3]), "spread_share_of_range": sig(spread_share),
                          "method": "median over patients of the fold of each patient's curve "
                                    "(log-odds against the variable, changed only where this part reads it)"},
            "fold_depth": {"value": sig(hinge), "unit": "log-odds",
                           "method": "median distance at the flat end between the curve and the "
                                     "straight-line continuation of its moving part"},
            **estimate_block(emean, eff, f"mean change in log-odds from {v.fmt(round(va, v.decimals))} to "
                             f"{v.fmt(round(vb, v.decimals))} for the patients in the group")}
        c["risk"] = risk_block(pa, pb, "typical patient with a fold" + (f" ({cond['group']})" if cond else ""),
                               typical_values(m, raw_t))
        c["n"] = int(rows.sum())
        c["groups"] = [{"label": ("patients whose curve has a fold" if cond is None else
                                  f"{cond['group']} whose curve has a fold"), "n": int(rows.sum())},
                       {"label": "of them, patients already past their fold", "n": past}]
        c["evidence"] = {"who": who, "n_onset": n_on, "n_offset": n_off, "n_with_fold": int(with_fold.sum()),
                         "observed_range": _sig_list([lo, hi]), "grid_points": GRID_POINTS}
        if past < SMALL_N:
            c["small_n_groups"].append("patients past the fold")
        c["small_n"] = bool(c["small_n_groups"])
        ben = test_benefit(m, v, part.input_nodes)
        c["test_check"] = {**ben, "status": _status(ben)}
        checks = {
            "enough_patients": (rows.sum() >= MIN_THRESHOLD_PATIENTS,
                                f"fewer than {MIN_THRESHOLD_PATIENTS} patients have a fold"),
            "single_threshold": (spread_share <= MAX_THRESHOLD_SPREAD,
                                 f"the folds are spread over {spread_share:.0%} of the observed range "
                                 f"(allowed {MAX_THRESHOLD_SPREAD:.0%}), also within each category "
                                 f"group: no single threshold"),
            "fold_matters": (hinge >= MIN_THRESHOLD_EFFECT,
                             f"the fold changes the curve by only {hinge:.3f} log-odds compared with "
                             f"a straight line (needed {MIN_THRESHOLD_EFFECT})"),
            "effect_past_threshold": (abs(emean) >= MIN_THRESHOLD_EFFECT and sp["share_opposite_sign"] <= MAX_CONTRADICTING,
                                      "the effect past the fold is too small or not consistent across patients"),
            "patients_past_threshold": (past >= 1, "no training patient is past the fold"),
        }
        out.append(_finish(c, checks))
    return out


def _modifier_settings(m: PlainModel, W: Variable, scope):
    """(w0, w1, class0, class1) for a modifier: category classes, or the 25th/75th percentile."""
    if W.is_category():
        return [(s[0], s[1], s[2], s[3]) for s in _settings(m, W, scope)]
    q = tuple(np.round(np.percentile(W.raw_train, [25, 75]), W.decimals))
    return [(q[0], q[1], None, None)]


def _modifier_groups(W: Variable, s, test: bool = False):
    """Observed groups (rows0, rows1) and their descriptions for a modifier setting."""
    w0, w1, c0, c1 = s
    raw = W.raw_test if test else W.raw_train
    if W.is_category():
        r1, r0 = class_rows(W, c1, test), class_rows(W, c0, test)
        d1 = {"label": c1["label"], "cond": c1["cond"], "group": c1["group"]}
        d0 = {"label": c0["label"], "cond": c0["cond"], "group": c0["group"]}
        return r0, r1, d0, d1, w0, w1
    med = float(np.median(W.raw_train))
    lo_vals, hi_vals = W.raw_train[W.raw_train <= med], W.raw_train[W.raw_train > med]
    d1 = {"label": f"{W.short} above {W.fmt(med)}", "cond": f"{W.short} is above {W.fmt(med)}",
          "group": f"patients with {W.short} above {W.fmt(med)}"}
    d0 = {"label": f"{W.short} at or below {W.fmt(med)}", "cond": f"{W.short} is at or below {W.fmt(med)}",
          "group": f"patients with {W.short} at or below {W.fmt(med)}"}
    w0r = float(np.median(lo_vals)) if len(lo_vals) else med
    w1r = float(np.median(hi_vals)) if len(hi_vals) else med
    return raw <= med, raw > med, d0, d1, w0r, w1r


def _roles(A: Variable, B: Variable, total_effect: Dict[str, float]) -> Tuple[Variable, Variable]:
    """(main, modifier): a category is the modifier when exactly one is a category; otherwise
    the variable with the larger whole-model effect is the main one."""
    if A.is_category() != B.is_category():
        return (B, A) if A.is_category() else (A, B)
    if abs(total_effect.get(B.base, 0.0)) > abs(total_effect.get(A.base, 0.0)):
        return B, A
    return A, B


def _did(m: PlainModel, X, A: Variable, sa, Bv: Variable, sb, scope) -> np.ndarray:
    return (m.effect(X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[1])) -
            m.effect(X, A, sa[0], sa[1], scope, extra=Bv.encode(sb[0])))


def interaction_claims(m: PlainModel, part: Part, total_effect: Dict[str, float]) -> List[Dict[str, Any]]:
    vs = list(part.variables)
    if len(vs) < 2:
        return []
    scope = part.input_nodes
    Xs = m.X[m.screen_idx]
    cands = []
    for va, vb in itertools.combinations(vs, 2):
        A, Bv = _roles(m.var_by_base[va], m.var_by_base[vb], total_effect)
        best = None
        for sa in _settings(m, A, scope):
            for sb in _modifier_settings(m, Bv, scope):
                did = _did(m, Xs, A, sa, Bv, sb, scope)
                if np.max(np.abs(did)) < 1e-9:
                    continue
                score = abs(float(np.mean(did)))
                if best is None or score > best[0]:
                    best = (score, sa, sb)
        if best is not None:
            cands.append((A.base, Bv.base, best))
    cands.sort(key=lambda t: (-t[2][0], t[0], t[1]))
    out = []
    raw = typical_raw(m)
    for ab, bb, (score, sa, sb) in cands[:MAX_INTERACTIONS_PER_PART]:
        A, Bv = m.var_by_base[ab], m.var_by_base[bb]
        did = _did(m, m.X, A, sa, Bv, sb, scope)
        did_mean = float(did.mean())
        nz = np.abs(did) > 1e-9
        did_consistent = float(np.mean(np.sign(did[nz]) == np.sign(did_mean))) if nz.any() else 0.0
        g0, g1, d0, d1, w0v, w1v = _modifier_groups(Bv, sb)
        e_total = m.effect(m.X, A, sa[0], sa[1], None)
        e1, e0 = e_total[g1], e_total[g0]
        m1 = float(e1.mean()) if len(e1) else float("nan")
        m0 = float(e0.mean()) if len(e0) else float("nan")
        d = m1 - m0
        # the typical patient of each group: the typical patient with the modifier set to the group
        risks = []
        for wv, cls in ((w1v, sb[3]), (w0v, sb[2])):
            if Bv.is_category():
                extra_raw = {c: (1.0 if code == cls["rep"] else 0.0) for c, code in zip(Bv.columns, Bv.codes)}
            else:
                extra_raw = {Bv.columns[0]: wv}
            risks.append(risk_pair(m, raw, A, sa[0], sa[1], None, extra_raw))
        groups = []
        for dd, rows, e, (pa, pb), mm in ((d1, g1, e1, risks[0], m1), (d0, g0, e0, risks[1], m0)):
            groups.append({**dd, "n": int(rows.sum()), "estimate": sig(mm), "odds_ratio": _or(mm),
                           "range_across_patients": spread(e)["range_across_patients"],
                           "risk_from": sig(pa), "risk_to": sig(pb),
                           "part_effect": sig(float(np.mean(m.effect(m.X[rows], A, sa[0], sa[1], scope))))
                           if rows.any() else None})
        strong, weak = (0, 1) if abs(m1) >= abs(m0) else (1, 0)
        ms, mw = groups[strong]["estimate"] or 0.0, groups[weak]["estimate"] or 0.0
        ew = e1 if weak == 0 else e0
        weak_consistent = bool(len(ew)) and spread(ew)["share_opposite_sign"] <= MAX_CONTRADICTING
        if np.sign(ms) != np.sign(mw) and abs(mw) >= NEGLIGIBLE and weak_consistent:
            pattern = "reversed"
        elif abs(mw) < NEGLIGIBLE:
            pattern = "only_if"
        else:
            pattern = "amplified"
        dps = [abs((g["risk_to"] or 0) - (g["risk_from"] or 0)) for g in groups]
        c = _claim(part, "interaction", f"{A.base}x{Bv.base}", variables=[_var_desc(A), _var_desc(Bv)],
                   statement={"variable": A.base, "modifier": Bv.base,
                              "change": _change_desc(A, sa[0], sa[1], sa[2], sa[3]),
                              "direction": "raises" if ms > 0 else "lowers",
                              "pattern": pattern, "stronger_group": groups[strong],
                              "weaker_group": groups[weak], "strength": strength(ms)})
        c["estimate"] = estimate_block(d, None, "difference between the two groups (first minus second) "
                                       "of the mean change in the model's log-odds when the main "
                                       "variable changes everywhere")
        c["risk"] = {"reference": "typical patient of each group (the typical patient with the "
                                  "modifier set to the group's value)",
                     "groups": [{"label": g["label"], "from": g["risk_from"], "to": g["risk_to"]} for g in groups],
                     "from": None, "to": None,
                     "delta": sig(dps[strong] - dps[weak]),
                     "delta_basis": "change in predicted risk in the stronger group minus in the weaker group"}
        c["groups"] = groups
        c["n"] = int(g1.sum() + g0.sum())
        c["evidence"] = {"counterfactual_did_through_part": sig(did_mean),
                         "did_share_same_sign": sig(did_consistent),
                         "counterfactual_modifier_settings": [str(sb[0]), str(sb[1])],
                         "method_note": "groups are the patients' actual values; the "
                                        "difference-in-differences sets the modifier for every "
                                        "patient, only where this part reads it"}
        for g in groups:
            if g["n"] < SMALL_N:
                c["small_n_groups"].append(g["label"])
        c["small_n"] = bool(c["small_n_groups"])
        c["test_check"] = _test_interaction(m, A, Bv, sa, sb, d, part)
        checks = {
            "large_enough": (abs(d) >= MIN_INTERACTION, f"the group difference is below {MIN_INTERACTION} log-odds"),
            "mechanism_agrees": (np.sign(d) == np.sign(did_mean),
                                 "the part's own counterfactual difference has the opposite sign"),
            "consistent_across_patients": (did_consistent >= 1 - MAX_CONTRADICTING,
                                           f"the part's counterfactual difference has the same sign for only "
                                           f"{did_consistent:.0%} of affected patients"),
            "both_groups_observed": (g1.sum() > 0 and g0.sum() > 0, "a group has no training patients"),
        }
        out.append(_finish(c, checks))
    return out


def _test_interaction(m, A, Bv, sa, sb, d_train, part):
    if m.Xt is None or Bv.raw_test is None:
        return {"split": "test", "status": "not available"}
    g0, g1, *_ = _modifier_groups(Bv, sb, test=True)
    e = m.effect(m.Xt, A, sa[0], sa[1], None)
    ben = test_benefit(m, Bv, part.input_nodes)
    if g1.sum() == 0 or g0.sum() == 0:
        return {**ben, "split": "test", "groups_n": [int(g1.sum()), int(g0.sum())], "difference": None,
                "same_sign": None, "status": "unclear", "note": "a group has no test patients"}
    dt, ci = boot_diff(e[g1], e[g0])
    same = bool(np.sign(dt) == np.sign(d_train))
    return {**ben, "split": "test", "groups_n": [int(g1.sum()), int(g0.sum())], "difference": sig(dt),
            "difference_ci95": _sig_list(ci), "same_sign": same,
            "status": _status(ben, same, ci if same else None),
            "measure_difference": "group difference of the main variable's effect on test patients, "
                                  "stratified bootstrap over test patients (seed 42)"}


def subgroup_claims(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    if not part.gated or part.contribution is None:
        return []
    c_ = part.contribution
    active = np.abs(c_) > 1e-12
    k, n = int(active.sum()), len(c_)
    if k == n or k == 0:
        return []
    who = describe_group(m, active, part.variables)
    eff = c_[active]
    emean = float(eff.mean())
    sp = spread(eff)
    pos = int(np.sum(m.y[active] == 1))
    raw = typical_raw(m, active)
    z = _z(m, raw)
    p_on = float(m.risk(m.logit(z))[0])
    p_off = float(m.risk(m.logit(z, ablate=part.baselines))[0])
    c = _claim(part, "subgroup", "active", variables=[_var_desc(m.var_by_base[b]) for b in part.variables],
               statement={"n_active": k, "n": n, "share": sig(k / n),
                          "direction": "raises" if emean > 0 else "lowers", "strength": strength(emean),
                          "conditions": [x["group"] for x in who["conditions"]],
                          "outcome_in_group": {"positives": pos, "n": k}})
    c["estimate"] = {"share": {"value": sig(k / n), "unit": "share of training patients"},
                     **estimate_block(emean, eff, "mean contribution of the part to the log-odds among "
                                      "the patients it acts on (the part switched off: exits at 0)")}
    c["risk"] = risk_block(p_off, p_on, "typical patient this part acts on (risk without the part, then with it)",
                           typical_values(m, raw))
    c["n"] = n
    c["groups"] = [{"label": "patients the part acts on", "n": k},
                   {"label": "patients it does not act on", "n": n - k}]
    c["evidence"] = {"who": who, "outcome_rate_in_group": sig(pos / k),
                     "outcome_rate_ci95": _wilson(pos, k), "outcome_rate_overall": sig(float(np.mean(m.y)))}
    if k < SMALL_N:
        c["small_n_groups"].append("patients the part acts on")
    c["small_n"] = bool(c["small_n_groups"])
    ben = test_benefit(m, None, None, part=part)
    tc = {**ben, "status": _status(ben)}
    if part.contribution_test is not None:
        act_t = np.abs(part.contribution_test) > 1e-12
        tc.update({"n_active": int(act_t.sum()), "share_active": sig(float(act_t.mean()))})
        if act_t.sum() == 0:
            tc["status"] = "unclear"
        elif np.sign(np.mean(part.contribution_test[act_t])) != np.sign(emean):
            tc["status"] = "disagrees"
    c["test_check"] = tc
    checks = {"acts_on_someone": (k >= 1, "the part acts on no training patient"),
              "not_negligible": (abs(emean) >= NEGLIGIBLE, "its effect is negligible"),
              "consistent_across_patients": (sp["share_opposite_sign"] <= MAX_CONTRADICTING,
                                             "its effect goes both ways across the patients it acts on")}
    return [_finish(c, checks)]


def null_claims_model(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    reach = m.reaching_output()
    out = []
    Xs = m.X[m.screen_idx]
    base = m.base_logit[m.screen_idx]
    for v in m.variables:
        nodes = [nd for c in v.columns for nd in m.input_nodes_by_col.get(c, [])]
        if any(nd in reach for nd in nodes):
            continue
        dmax = 0.0
        for c in v.columns:
            for val in (np.min(m.X[:, c]), np.max(m.X[:, c])):
                dmax = max(dmax, float(np.max(np.abs(m.logit(Xs, {c: val}) - base))))
        c = _claim(part, "null", v.base, variables=[_var_desc(v)], statement={"variable": v.base, "method": "exact"})
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
    p_full = m.risk(m.base_logit)
    p_abl = m.risk(m.base_logit - c_)
    dp = float(np.max(np.abs(p_full - p_abl)))
    dauc = float(default_auc(m.y, p_full) - default_auc(m.y, p_abl)) if m.risk_scale.kind == "binary" else 0.0
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


def rule_claims(m: PlainModel, part: Part) -> List[Dict[str, Any]]:
    """Count rules from the codebook (e.g. MONK-2: how many attributes take their first value).

    For each count k, the share of patients the model predicts positive; the claim is that the
    model's prediction is a function of the count: every k (with patients) has at least
    ``RULE_PURITY`` of one prediction. The set of counts predicted positive is reported.
    """
    out = []
    for rule in (m.codebook.get("count_rules") or []):
        terms = [(m.var_by_base.get(b), str(code)) for b, code in rule["terms"]]
        if any(v is None for v, _ in terms):
            continue

        def counts(test=False):
            k = np.zeros(len(m.yt) if test else len(m.y), dtype=int)
            for v, code in terms:
                k += ((v.raw_test if test else v.raw_train) == code).astype(int)
            return k
        k = counts()
        pred = (m.risk(m.base_logit) >= 0.5).astype(int)
        table, positive, pure = [], [], True
        for kv in sorted(set(k.tolist())):
            rows = k == kv
            share = float(pred[rows].mean())
            purity = max(share, 1 - share)
            pure &= purity >= RULE_PURITY
            if share >= 0.5:
                positive.append(int(kv))
            table.append({"count": int(kv), "n": int(rows.sum()), "share_predicted_positive": sig(share),
                          "share_actually_positive": sig(float(m.y[rows].mean()))})
        c = _claim(part, "rule", rule["id"], variables=[_var_desc(v) for v, _ in terms],
                   statement={"rule": rule["name"], "description": rule.get("description"),
                              "counts_predicted_positive": positive, "positive_label": rule.get("positive_label"),
                              "count_phrase": rule.get("count_phrase")})
        c["estimate"] = {"value": sig(min(max(t["share_predicted_positive"], 1 - t["share_predicted_positive"])
                                          for t in table)), "unit": "lowest purity over counts",
                         "table": table, "method": "share of training patients predicted positive "
                                                   "(predicted risk at least 0.5) at each count"}
        c["n"] = len(k)
        c["groups"] = [{"label": f"count {t['count']}", "n": t["n"]} for t in table]
        c["small_n_groups"] = [f"count {t['count']}" for t in table if t["n"] < SMALL_N]
        c["small_n"] = False   # every count is reported; purity is checked per count
        pos_rows = np.isin(k, positive)
        raw_pos, raw_neg = typical_raw(m, pos_rows) if pos_rows.any() else None, typical_raw(m, ~pos_rows) if (~pos_rows).any() else None
        pp = float(m.risk(m.logit(_z(m, raw_pos)))[0]) if raw_pos is not None else None
        pn = float(m.risk(m.logit(_z(m, raw_neg)))[0]) if raw_neg is not None else None
        c["risk"] = risk_block(pn if pn is not None else 0.0, pp if pp is not None else 0.0,
                               "typical case with a count predicted negative, then positive")
        status = "not available"
        if m.Xt is not None:
            kt = counts(True)
            pt = (m.risk(m.base_logit_test) >= 0.5).astype(int)
            agree = float(np.mean(pt == np.isin(kt, positive).astype(int)))
            status = "agrees" if agree >= RULE_PURITY else "disagrees"
            c["test_check"] = {"split": "test", "n": int(len(kt)), "share_following_rule": sig(agree), "status": status}
        else:
            c["test_check"] = {"split": "test", "status": status}
        checks = {"pure": (pure, f"at some count fewer than {RULE_PURITY:.0%} of cases get the same prediction"),
                  "has_positive": (bool(positive) and len(positive) < len(table),
                                   "the model predicts the same class at every count")}
        out.append(_finish(c, checks))
    return out


# ============================================================================ the layer

TYPE_ORDER = {t: i for i, t in enumerate(("rule", "direction", "magnitude", "interaction", "threshold",
                                          "subgroup", "null"))}


def _claim_sort_key(c):
    r = (c.get("risk") or {}).get("delta")
    est = (c.get("estimate") or {}).get("value")
    v = r if isinstance(r, (int, float)) else (est if isinstance(est, (int, float)) else 0.0)
    return (TYPE_ORDER[c["type"]], -abs(v or 0.0), c["key"])


def _dedupe_interactions(parts: List[Part], by_part: Dict[str, List[Dict[str, Any]]]) -> None:
    """An interaction's estimate is measured on the whole model, so a composition and the child
    that realises it (or two copies of one calculation) report the same claim. Keep it once, in
    the most specific part, and record the others in ``evidence.also_in``."""
    rank = {p.id: (len(p.variables), p.order) for p in parts}
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for p in sorted(parts, key=lambda p: rank[p.id]):
        keep = []
        for c in by_part.get(p.id, []):
            if c["type"] != "interaction":
                keep.append(c)
                continue
            sig_key = (c["key"].split(":", 2)[2], c["estimate"]["value"], tuple(g["n"] for g in c["groups"]))
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
        root.variables = [v.base for v in m.variables]
        dirs = direction_claims(m, root, None)
        for d in dirs:
            b = d["statement"]["variable"]
            if abs(d["estimate"]["value"]) > abs(total_effect.get(b, 0.0)):
                total_effect[b] = d["estimate"]["value"]
        by_part[root.id] = (rule_claims(m, root) + dirs + magnitude_claims(m, root, dirs)
                            + null_claims_model(m, root))
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
            counts = v.counts()
            rec["levels"] = [{"code": code, "label": v.level(code)["label"], "decoded": v.level(code)["decoded"],
                              "n_train": counts[code]} for code in v.all_codes]
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
        "population": {"plural": m.codebook.get("patient", "patients"),
                       "singular": m.codebook.get("patient_one", "patient")},
        "risk_scale": m.risk_scale.describe(),
        "codebook": {"id": m.codebook.get("id"), "source": m.codebook.get("source"),
                     "gaps": codebook_gaps(m.codebook, m.feature_names),
                     "checks": codebook_checks(m.codebook, m.feature_names, m.raw_train, m.y)},
        "units": units_record(m),
        "reference_patient": reference_patient(m),
        "settings": {"seed": SEED, "bootstrap_B": BOOTSTRAP_B, "small_n": SMALL_N,
                     "prob_slight": PROB_SLIGHT, "prob_large": PROB_LARGE,
                     "confidence_rules": CONFIDENCE_RULES,
                     "min_interaction": MIN_INTERACTION, "min_threshold_effect": MIN_THRESHOLD_EFFECT,
                     "max_opposite_sign_share": MAX_CONTRADICTING,
                     "max_threshold_spread": MAX_THRESHOLD_SPREAD,
                     "max_interactions_per_part": MAX_INTERACTIONS_PER_PART,
                     "top_magnitudes": TOP_MAGNITUDES, "screen_rows": SCREEN_ROWS,
                     "effect_scale": "log-odds (output pre-activation) for verification; predicted "
                                     "risk for the typical patient for wording"},
        "parts": [_clean(part_record(m, p)) for p in parts],
        "claims": claims,
        "claims_hash": claims_hash(claims),
        "counts": {"claims": len(claims), "verified": sum(1 for c in claims if c["verified"]),
                   "not_supported": sum(1 for c in claims if not c["verified"]),
                   "clearly": sum(1 for c in claims if c["verified"] and (c.get("confidence") or {}).get("word") == "clearly"),
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
    failed = [c for c in codebook_checks(cb, names, raw_tr, y_tr) if not c["passed"]]
    if failed:   # never decode on a mapping the data contradicts
        note = ("codebook " + str(cb.get("id")) + " not used: its checks failed ("
                + "; ".join(c["check"] for c in failed) + ")")
        cb = generic_codebook(names)
        cb["gaps"] = cb["gaps"] + [note]
    m = PlainModel(engine.current_state, ops, X_tr, y_tr, X_te, y_te, names, mean, scale, cb,
                   raw_train=raw_tr, raw_test=raw_te)
    return m, meta
