"""Linked inputs: inputs whose values the data ties together (layer v5g, 2026-11-11).

A counterfactual that changes one input and holds the rest can build a case no real row has:
an education category changed while the years-of-education number stays at 10, a "wife" who
stays male, more day minutes at the same day charge, voice-mail messages without a plan. The
number reproduces exactly, so neither slots nor ``recheck`` catch it; the drafter then states
it about real people (authoring-v4 evaluation, K1).

This module finds such links in the training data and moves the linked inputs with the changed
one:

* **Detection** (:class:`Links`). Every variable is binned (a category per level; a
  number with at most 25 values per value; otherwise its point masses, each at least 5% of
  rows, plus deciles of the rest). ``A`` is linked to ``V`` when some bin pair is a structural
  zero: a level of ``A`` with at least 20 rows, a ``V`` bin it would contain at least 20 times
  if they were independent, and in fact at most 0.2% of that level's rows and at most 5% of the
  expected count; and these zeros rule out at least 2% of ``V``'s distribution on average over
  the rows (a constraint, not a rare association such as country of birth and race). Examples:
  education 9th grade
  with education-num 10; relationship wife with sex male; plan "no" with messages above 0; a
  day-minutes decile with a day-charge decile). A per-dataset override in the codebook
  (``linked_inputs: {"add": [[A, V]], "remove": [[A, V]]}``) adds or removes links.
* **Moving** (:meth:`Links.align_raw`). When ``A`` is set to a value, every row whose ``V`` falls
  in a bin that (almost) never occurs with that value of ``A`` (the same test, from 3 expected
  rows) gets ``V`` set to its conditional
  typical value (the median, or the most common level, among the training rows with that value
  of ``A``). Moves chain (relationship, then marital status) and each input moves at most once.
  Inputs that the counterfactual itself sets (``fixed``) are never moved; if one of them is off
  the data the case is **off the data** and the claim says so (``off_data``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAX_DISCRETE = 25          # a number with at most this many values is binned per value
POINT_MASS = 0.05          # a value with at least this share of rows gets its own bin
DECILES = 10
MIN_LEVEL_ROWS = 20        # a level of A needs this many rows to define a structural zero
MIN_EXPECTED = 20          # ... and the V bin this many expected rows under independence
MAX_ZERO_SHARE = 0.002     # observed share of the level's rows that still counts as "never"
MAX_RATIO = 0.05           # ... and at most this share of the rows expected under independence
MIN_RULED_OUT = 0.02       # a link: A's value rules out at least this share of V's distribution
MIN_EXPECTED_MOVE = 3      # moving: a pair counts as off the data from this many expected rows

SETTINGS = {"max_discrete": MAX_DISCRETE, "point_mass": POINT_MASS, "deciles": DECILES,
            "min_level_rows": MIN_LEVEL_ROWS, "min_expected": MIN_EXPECTED,
            "max_zero_share": MAX_ZERO_SHARE, "max_ratio": MAX_RATIO, "min_ruled_out": MIN_RULED_OUT,
            "min_expected_move": MIN_EXPECTED_MOVE}


@dataclass
class Binner:
    """Bins of one variable over the training rows."""
    kind: str                         # category | discrete | continuous
    codes: List[Any] = field(default_factory=list)       # category codes / discrete values
    masses: List[float] = field(default_factory=list)    # continuous: point-mass values
    edges: Optional[np.ndarray] = None                   # continuous: inner decile edges

    @property
    def n_bins(self) -> int:
        if self.kind in ("category", "discrete"):
            return len(self.codes)
        return len(self.masses) + (0 if self.edges is None else len(self.edges) + 1)

    def bin(self, values) -> np.ndarray:
        vals = np.asarray(values, dtype=object if self.kind == "category" else float)
        if self.kind != "category":
            vals = np.round(vals, 6)   # raw values come back from z-scores with rounding error
        if self.kind == "category":
            idx = {c: i for i, c in enumerate(self.codes)}
            return np.array([idx.get(x, -1) for x in vals], dtype=int)
        if self.kind == "discrete":
            u = np.asarray(self.codes, dtype=float)
            j = np.searchsorted(u, vals)
            j = np.clip(j, 0, len(u) - 1)
            jl = np.clip(j - 1, 0, len(u) - 1)
            return np.where(np.abs(u[jl] - vals) < np.abs(u[j] - vals), jl, j).astype(int)
        out = np.full(vals.shape, -1, dtype=int)
        mass = np.zeros(vals.shape, dtype=bool)
        for i, x in enumerate(self.masses):
            hit = vals == round(x, 6)
            out[hit] = i
            mass |= hit
        if self.edges is not None:
            out[~mass] = len(self.masses) + np.searchsorted(self.edges, vals[~mass], side="right")
        return out


def make_binner(v) -> Binner:
    if v.is_category():
        return Binner("category", codes=list(v.all_codes))
    raw = np.asarray(v.raw_train, dtype=float)
    u, cnt = np.unique(raw, return_counts=True)
    if len(u) <= MAX_DISCRETE:
        return Binner("discrete", codes=[float(x) for x in u])
    masses = [float(x) for x, k in zip(u, cnt) if k / len(raw) >= POINT_MASS]
    rest = raw[~np.isin(raw, masses)]
    edges = None
    if len(rest):
        e = np.unique(np.quantile(rest, np.linspace(0, 1, DECILES + 1))[1:-1])
        edges = np.round(e, 6)
    return Binner("continuous", masses=masses, edges=edges)


def _cat_index(v, raw_rows: np.ndarray) -> np.ndarray:
    """Per-row index into ``v.all_codes`` (-1 if several one-hot columns are on, or none and
    the variable has no implicit level)."""
    on = np.asarray(raw_rows[:, v.columns]) > 0.5
    k = on.sum(axis=1)
    idx = np.argmax(on, axis=1)
    out = np.where(k == 1, idx, -1)
    if v.implicit_off is not None:
        out = np.where(k == 0, len(v.codes), out)
    return out.astype(int)


def _raw_values(v, raw_rows: np.ndarray) -> np.ndarray:
    """Per-row raw value of ``v`` from raw (decoded) rows: a category code or a number."""
    if v.is_category():
        codes = np.array(list(v.all_codes) + [None], dtype=object)
        return codes[_cat_index(v, raw_rows)]
    return raw_rows[:, v.columns[0]].astype(float)


def _bins_of(binner, v, raw_rows: np.ndarray) -> np.ndarray:
    if v.is_category():
        return _cat_index(v, raw_rows)
    return binner.bin(raw_rows[:, v.columns[0]])


@dataclass
class Link:
    a: str
    v: str
    allowed: np.ndarray        # [a_bin, v_bin] bool: the pair occurs in the data
    typical: List[Any]         # per a_bin: V's conditional typical value (None: no rows)
    n_a: np.ndarray            # rows per a_bin
    example: Dict[str, Any]    # the strongest structural zero, for the record
    source: str = "data"


class Links:
    """The linked inputs of a model's training data, and the moves they imply."""

    def __init__(self, m, overrides: Optional[Dict[str, Any]] = None):
        self.m = m
        self.by_a: Dict[str, List[Link]] = {}
        self.binners = {v.base: make_binner(v) for v in m.variables}
        self.bins = {v.base: _bins_of(self.binners[v.base], v, m.raw_train) for v in m.variables}
        self._detect(overrides or {})

    # ------------------------------------------------------------------ detection
    def _table(self, A, V) -> Tuple[np.ndarray, np.ndarray]:
        ba, bv = self.bins[A.base], self.bins[V.base]
        na, nv = self.binners[A.base].n_bins, self.binners[V.base].n_bins
        ok = (ba >= 0) & (bv >= 0)
        N = np.zeros((na, nv), dtype=float)
        np.add.at(N, (ba[ok], bv[ok]), 1.0)
        return N, ok

    def _link(self, A, V, N, force: bool = False) -> Optional[Link]:
        n_a = N.sum(axis=1)
        tot = N.sum()
        if tot == 0:
            return None
        p_v = N.sum(axis=0) / tot
        exp = n_a[:, None] * p_v[None, :]
        never = (N <= MAX_ZERO_SHARE * n_a[:, None]) & (N <= MAX_RATIO * exp)
        zero = never & (n_a[:, None] >= MIN_LEVEL_ROWS) & (exp >= MIN_EXPECTED)
        # how much of V's distribution the value of A rules out, averaged over the rows
        ruled_out = float(np.sum((n_a / tot)[:, None] * p_v[None, :] * zero))
        if ruled_out < MIN_RULED_OUT and not force:
            return None
        allowed = ~(never & (exp >= MIN_EXPECTED_MOVE))
        ba, bv = self.bins[A.base], self.bins[V.base]
        vals = V.raw_train
        typical: List[Any] = []
        for i in range(N.shape[0]):
            rows = ba == i
            if not rows.any():
                typical.append(None)
            elif V.is_category():
                codes = vals[rows]
                allc = V.all_codes
                cnt = {c: int(np.sum(codes == c)) for c in allc}
                typical.append(max(allc, key=lambda c: (cnt[c], -allc.index(c))))
            else:
                typical.append(float(np.median(np.asarray(vals[rows], dtype=float))))
        ex = {}
        if zero.any():
            i, j = max(zip(*np.where(zero)), key=lambda ij: exp[ij])
            ex = {"a_bin": self.bin_label(A, int(i)), "v_bin": self.bin_label(V, int(j)),
                  "rows_of_a": int(n_a[i]), "expected": round(float(exp[i, j]), 1), "observed": int(N[i, j]),
                  "ruled_out_share": round(ruled_out, 4)}
        return Link(A.base, V.base, allowed, typical, n_a, ex, "override" if force and ruled_out < MIN_RULED_OUT else "data")

    def _detect(self, overrides: Dict[str, Any]) -> None:
        m = self.m
        remove = {tuple(p) for p in overrides.get("remove", [])}
        add = {tuple(p) for p in overrides.get("add", [])}
        for A in m.variables:
            for V in m.variables:
                if A is V or (A.base, V.base) in remove:
                    continue
                N, _ = self._table(A, V)
                lk = self._link(A, V, N, force=(A.base, V.base) in add)
                if lk is not None:
                    self.by_a.setdefault(A.base, []).append(lk)

    def bin_label(self, v, i: int) -> str:
        b = self.binners[v.base]
        if b.kind == "category":
            return v.level(b.codes[i])["label"]
        if b.kind == "discrete":
            return v.fmt(b.codes[i])
        if i < len(b.masses):
            return v.fmt(b.masses[i])
        k = i - len(b.masses)
        e = list(b.edges) if b.edges is not None else []
        lo = None if k == 0 else e[k - 1]
        hi = None if k >= len(e) else e[k]
        if lo is None:
            return f"below {v.fmt(hi)}"
        if hi is None:
            return f"{v.fmt(lo)} or more"
        return f"{v.fmt(lo)} to {v.fmt(hi)}"

    def record(self) -> Dict[str, Any]:
        """For the layer: the detected links (which inputs move with which)."""
        out = []
        for a, lks in sorted(self.by_a.items()):
            for lk in lks:
                out.append({"changed": a, "moves": lk.v, "source": lk.source, "example": lk.example})
        return {"links": out, "settings": SETTINGS,
                "rule": "when a counterfactual sets an input, every linked input whose value (almost) never "
                        "occurs with the new value in the training data is set to its typical value among "
                        "the training rows with that value (median or most common level); moves chain, each "
                        "input moves at most once, inputs the counterfactual sets itself never move (if "
                        "they are off the data, the claim is marked off_data)"}

    def linked(self, base: str) -> List[str]:
        return [lk.v for lk in self.by_a.get(base, [])]

    # ------------------------------------------------------------------ moving
    def align_raw(self, raw_rows: np.ndarray, changed: Sequence[str], fixed: Iterable[str] = ()
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Raw rows with the inputs linked to ``changed`` moved onto the data.

        Returns the new rows and ``info``: ``moved`` {V: rows moved}, ``off`` [(A, V, rows)]
        for fixed inputs that are off the data, ``none`` [(A, rows)] when a value of ``A`` has
        no training rows to take a typical value from."""
        m = self.m
        R = np.array(raw_rows, dtype=float, copy=True)
        fixed = set(fixed) | set(changed)
        info: Dict[str, Any] = {"moved": {}, "off": [], "none": []}
        queue = [b for b in changed if b in self.by_a]
        touched = {b: np.ones(R.shape[0], dtype=bool) for b in changed}   # rows whose value was set
        done = set()
        while queue:
            a = queue.pop(0)
            if a in done:
                continue
            done.add(a)
            A = m.var_by_base[a]
            ab = _bins_of(self.binners[a], A, R)
            for lk in self.by_a.get(a, []):
                V = m.var_by_base[lk.v]
                vb = _bins_of(self.binners[lk.v], V, R)
                okb = (ab >= 0) & (vb >= 0)
                off = np.zeros(R.shape[0], dtype=bool)
                off[okb] = ~lk.allowed[ab[okb], vb[okb]]
                off |= (ab >= 0) & (vb < 0) & V.is_category()
                off &= touched[a]
                if not off.any():
                    continue
                if lk.v in fixed or lk.v in info["moved"]:
                    if lk.v in fixed:
                        info["off"].append((a, lk.v, int(off.sum())))
                    continue
                n_none = 0
                moved = np.zeros(R.shape[0], dtype=bool)
                for i in np.unique(ab[off]):
                    t = lk.typical[i]
                    rows = off & (ab == i)
                    if t is None:
                        n_none += int(rows.sum())
                        continue
                    moved |= rows
                    if V.is_category():
                        for c, code in zip(V.columns, V.codes):
                            R[rows, c] = 1.0 if code == t else 0.0
                    else:
                        R[rows, V.columns[0]] = float(t)
                if n_none:
                    info["none"].append((a, n_none))
                info["moved"][lk.v] = int(moved.sum())
                touched[lk.v] = moved
                queue.append(lk.v)
        return R, info

    def describe_row(self, before: np.ndarray, after: np.ndarray) -> Dict[str, Dict[str, str]]:
        """{V: {"from": text, "to": text}} for the inputs that differ between two raw rows."""
        m = self.m
        out = {}
        for v in m.variables:
            if np.allclose(before[v.columns], after[v.columns]):
                continue
            a = _raw_values(v, before[None, :])[0]
            b = _raw_values(v, after[None, :])[0]
            ft = (lambda x: v.level(x)["label"] if x is not None else "none") if v.is_category() else v.fmt
            out[v.base] = {"label": v.label, "from": ft(a), "to": ft(b)}
        return out
