"""Concept libraries: named derived features of a dataset's raw inputs, for the concept probe.

A *concept* is a function of the raw (real-unit) input rows that a domain reader would
recognise: "number of threats x has", "the black king can capture the rook", "age above the
median". The probe (:mod:`.concept_probe`) scores how well a concept predicts a leaf's exit
activation, which is the test of the third level of explanation (understood): a leaf maps to a
concept, not only to a closed form.

Libraries are keyed by dataset name:

* ``connect4_win_ordinal`` (Connect-4, 42 ordinal cells): threat counts per side, open lines
  of 1-3 per side, centre-column occupancy, row placement, threat row parity and immediacy,
  piece counts; from ``experiments/c4_capacity/c4_features.py`` and ``C4_LINES`` in
  ``experiments/complex_real/data.py`` (re-implemented here so the library ships with the package).
* ``connect4_lines``, ``connect4_lines_cells``, ``connect4_perline`` (Connect-4 with derived line
  features as inputs, ``experiments/c4_line_features``): every non-cell input is itself a domain
  concept (an open-line, threat, centre, parity or per-line count), plus the differences
  x minus o of each paired count; when the 42 cells are among the inputs the full Connect-4
  library above is computed from them and the cells are offered as inputs.
* ``krk_draw_ordinal`` (KRK, 6 ordinal inputs): rook adjacency and capture, check, stalemate,
  king distances (the exact rule features of ``krk_rules``).
* every other dataset (heart, adult, churn, monk2, the ladder's synthetic data): the generic
  tabular library: each input as it is, each continuous input above its training median,
  each one-hot level; with ``extra_pairs`` the pairwise products and conjunctions of a few
  named inputs (the leaf's own entry inputs), which is the "simple combinations" vocabulary.

An **ad-hoc concept** is a pandas expression over the input names (backticks for names with
spaces or hyphens) and the library's concept names, e.g. ``"x_thr - o_thr"`` or
``"age > 55 and sex == 1"`` (:func:`evaluate_expression`). The scope of this vocabulary is
stated in ``docs/annotation_coverage_model.md``: per-input and feature-based; a relational
domain needs its library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# ============================================================================ concept
@dataclass
class Concept:
    name: str
    description: str
    kind: str                       # "binary" | "count" | "continuous"
    source: str                     # "domain" | "input" | "threshold" | "level" | "pair" | "expression"
    values: np.ndarray = field(repr=False)

    @property
    def is_binary(self) -> bool:
        return self.kind == "binary"


def _kind_of(v: np.ndarray) -> str:
    u = np.unique(v[np.isfinite(v)])
    if len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}:
        return "binary"
    if np.all(u == np.round(u)) and len(u) <= 50:
        return "count"
    return "continuous"


def _concept(name: str, desc: str, values, source: str, kind: Optional[str] = None) -> Concept:
    v = np.asarray(values, dtype=float).ravel()
    return Concept(name=name, description=desc, kind=kind or _kind_of(v), source=source, values=v)


# ============================================================================ Connect-4
C4_FEATURES = [f"c_{c}{r}" for c in "abcdefg" for r in range(1, 7)]

C4_LINES: List[List[tuple]] = []
for _c in range(7):
    for _r in range(6):
        for _dc, _dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            _cells = [(_c + i * _dc, _r + i * _dr) for i in range(4)]
            if all(0 <= a < 7 and 0 <= b < 6 for a, b in _cells):
                C4_LINES.append(_cells)
_C4_LINE_IDX = np.array([[c * 6 + r for c, r in line] for line in C4_LINES])   # [69, 4] flat cell index


def c4_concepts(X_raw: np.ndarray) -> List[Concept]:
    """Connect-4 domain concepts from raw boards (+1 = x, -1 = o, 0 = blank; x to move).
    Rows are 0-based from the bottom, so an even row index is an odd row number (1, 3, 5)."""
    X = np.asarray(X_raw, dtype=np.float64)
    N = len(X)
    cells = X[:, _C4_LINE_IDX]                       # [N, 69, 4]
    nx = (cells == 1).sum(-1)
    no = (cells == -1).sum(-1)
    out: List[Concept] = []
    for k in (1, 2, 3):
        out.append(_concept(f"x_open_{k}", f"lines of 4 with {k} x piece(s) and no o piece",
                            ((nx == k) & (no == 0)).sum(1), "domain"))
        out.append(_concept(f"o_open_{k}", f"lines of 4 with {k} o piece(s) and no x piece",
                            ((no == k) & (nx == 0)).sum(1), "domain"))
    filled = X != 0
    below_filled = np.ones_like(filled)
    for c in range(7):
        for r in range(1, 6):
            below_filled[:, c * 6 + r] = filled[:, c * 6 + r - 1]
    row = np.tile(np.arange(6), 7)
    for p, cnt, other in (("x", nx, no), ("o", no, nx)):
        thr_cell = np.zeros((N, 42), bool)
        sel = (cnt == 3) & (other == 0)
        for li in range(len(C4_LINES)):
            rows_ = np.where(sel[:, li])[0]
            if len(rows_) == 0:
                continue
            empt = ~filled[np.ix_(rows_, _C4_LINE_IDX[li])]
            for j in range(4):
                thr_cell[rows_[empt[:, j]], _C4_LINE_IDX[li, j]] = True
        who = "x (to move)" if p == "x" else "o"
        out.append(_concept(f"{p}_thr", f"distinct empty cells that complete a line of 3 for {who} (threats)",
                            thr_cell.sum(1), "domain"))
        out.append(_concept(f"{p}_thr_odd", f"threats of {who} on odd rows (1, 3, 5)",
                            thr_cell[:, row % 2 == 0].sum(1), "domain"))
        out.append(_concept(f"{p}_thr_even", f"threats of {who} on even rows (2, 4, 6)",
                            thr_cell[:, row % 2 == 1].sum(1), "domain"))
        out.append(_concept(f"{p}_thr_now", f"threats of {who} playable now (cell below filled)",
                            (thr_cell & below_filled).sum(1), "domain"))
        out.append(_concept(f"{p}_has_thr", f"{who} has at least one threat", thr_cell.sum(1) > 0, "domain"))
        out.append(_concept(f"{p}_thr_now_any", f"{who} has a threat playable now",
                            (thr_cell & below_filled).sum(1) > 0, "domain"))
    B = X.reshape(-1, 7, 6)
    out.append(_concept("centre_x", "x pieces in the centre column d", (B[:, 3, :] == 1).sum(1), "domain"))
    out.append(_concept("centre_o", "o pieces in the centre column d", (B[:, 3, :] == -1).sum(1), "domain"))
    out.append(_concept("centre_diff", "x minus o pieces in the centre column", (B[:, 3, :] == 1).sum(1)
                        - (B[:, 3, :] == -1).sum(1), "domain"))
    out.append(_concept("x_thr_minus_o_thr", "x threats minus o threats", _thr_diff(out), "domain"))
    for r in range(6):
        out.append(_concept(f"x_row{r + 1}", f"x pieces on row {r + 1}", (B[:, :, r] == 1).sum(1), "domain"))
        out.append(_concept(f"o_row{r + 1}", f"o pieces on row {r + 1}", (B[:, :, r] == -1).sum(1), "domain"))
    out.append(_concept("x_mean_row", "mean row (1-6) of x pieces", _mean_row(B, 1), "domain"))
    out.append(_concept("o_mean_row", "mean row (1-6) of o pieces", _mean_row(B, -1), "domain"))
    heights = filled.reshape(-1, 7, 6).sum(2)        # [N, 7]
    out.append(_concept("n_pieces", "pieces on the board", heights.sum(1), "domain"))
    out.append(_concept("n_odd_empty_playable", "playable cells on odd rows (parity: first player's rows)",
                        _playable_parity(heights, 0), "domain"))
    out.append(_concept("n_even_empty_playable", "playable cells on even rows (parity: second player's rows)",
                        _playable_parity(heights, 1), "domain"))
    for c, letter in enumerate("abcdefg"):
        out.append(_concept(f"col_{letter}_x", f"x pieces in column {letter}", (B[:, c, :] == 1).sum(1), "domain"))
        out.append(_concept(f"col_{letter}_o", f"o pieces in column {letter}", (B[:, c, :] == -1).sum(1), "domain"))
    return out


def _thr_diff(concepts: List[Concept]) -> np.ndarray:
    d = {c.name: c.values for c in concepts}
    return d["x_thr"] - d["o_thr"]


def _mean_row(B: np.ndarray, who: int) -> np.ndarray:
    m = B == who                                     # [N, 7, 6]
    rows = np.arange(1, 7)[None, None, :]
    n = m.sum((1, 2))
    return np.where(n > 0, (m * rows).sum((1, 2)) / np.maximum(n, 1), 0.0)


def _playable_parity(heights: np.ndarray, parity: int) -> np.ndarray:
    # the next playable cell of each column is at row index = height; odd row number <=> even index
    playable = heights < 6
    return ((heights % 2 == parity) & playable).sum(1)


def c4_line_input_concepts(X: np.ndarray, names: Sequence[str], include_inputs: bool = True) -> List[Concept]:
    """Connect-4 with derived line features as inputs: each non-cell input is a domain concept
    (its description from the standard library when it is one of its features, else from its
    name), plus x minus o for each paired count; the full cell library when the 42 cells are
    among the inputs (and the cells as inputs)."""
    names = [str(n) for n in names]
    cols = {n: i for i, n in enumerate(names)}
    cell_cols = [cols.get(n) for n in C4_FEATURES]
    have_cells = all(c is not None for c in cell_cols)
    std = c4_concepts(X[:, cell_cols]) if have_cells else c4_concepts(np.zeros((1, 42)))  # descriptions
    std_desc = {c.name: c.description for c in std}
    out: List[Concept] = []
    for n, i in cols.items():
        if n in C4_FEATURES:
            continue
        if n in std_desc:
            desc = std_desc[n]
        elif n.startswith(("x_line_", "o_line_")):
            who, line = n[0], n[len("x_line_"):]
            desc = f"{who} pieces on the line {line} when no {'o' if who == 'x' else 'x'} piece is on it (0 if blocked or empty)"
        else:
            desc = f"line feature {n}"
        out.append(_concept(n, desc, X[:, i], "domain"))
    for n in list(cols):
        if n.startswith("x_") and ("o_" + n[2:]) in cols and n not in C4_FEATURES:
            m = "o_" + n[2:]
            out.append(_concept(f"{n}_minus_{m}", f"{n} minus {m}", X[:, cols[n]] - X[:, cols[m]], "domain"))
    if have_cells:
        seen = {c.name for c in out}
        out += [c for c in std if c.name not in seen]
        if include_inputs:
            out += [_concept(n, f"cell {n} (+1 x, -1 o, 0 blank)", X[:, cols[n]], "input", "count") for n in C4_FEATURES]
    return out


# ============================================================================ KRK
KRK_FEATURES = ["wk_file", "wk_rank", "wr_file", "wr_rank", "bk_file", "bk_rank"]


def krk_concepts(X_raw: np.ndarray) -> List[Concept]:
    """KRK (black to move) rule concepts: the only draws are an immediate rook capture by the
    black king or stalemate. Files and ranks are 1-8."""
    X = np.asarray(X_raw, dtype=int)
    wkf, wkr, wrf, wrr, bkf, bkr = (X[:, i] for i in range(6))

    def cheb(f1, r1, f2, r2):
        return np.maximum(np.abs(f1 - f2), np.abs(r1 - r2))

    bk_wr = cheb(bkf, bkr, wrf, wrr)
    wk_wr = cheb(wkf, wkr, wrf, wrr)
    bk_wk = cheb(bkf, bkr, wkf, wkr)
    capture = (bk_wr == 1) & (wk_wr > 1)

    def attacked(f, r):
        by_k = cheb(f, r, wkf, wkr) <= 1
        same_f = (f == wrf) & ~((wkf == f) & (np.minimum(r, wrr) < wkr) & (wkr < np.maximum(r, wrr)))
        same_r = (r == wrr) & ~((wkr == r) & (np.minimum(f, wrf) < wkf) & (wkf < np.maximum(f, wrf)))
        on_rook = (f == wrf) & (r == wrr)
        return by_k | ((same_f | same_r) & ~on_rook)

    in_check = attacked(bkf, bkr)
    has_move = np.zeros(len(X), bool)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == dr == 0:
                continue
            f, r = bkf + df, bkr + dr
            inside = (f >= 1) & (f <= 8) & (r >= 1) & (r <= 8)
            on_rook = (f == wrf) & (r == wrr)
            safe = ~attacked(f, r) | (on_rook & (cheb(f, r, wkf, wkr) > 1))
            has_move |= inside & safe
    stalemate = ~in_check & ~has_move
    same_file = bkf == wrf
    same_rank = bkr == wrr
    return [
        _concept("rook_adjacent", "the white rook is adjacent to the black king", bk_wr == 1, "domain"),
        _concept("rook_protected", "the white king is adjacent to the rook", wk_wr <= 1, "domain"),
        _concept("capture", "the black king can capture the rook (adjacent, unprotected)", capture, "domain"),
        _concept("in_check", "the black king is in check", in_check, "domain"),
        _concept("stalemate", "stalemate: not in check and no legal move", stalemate, "domain"),
        _concept("rule_draw", "draw by the rule: capture or stalemate", capture | stalemate, "domain"),
        _concept("bk_rook_dist", "king-rook distance (Chebyshev)", bk_wr, "domain"),
        _concept("wk_rook_dist", "white king-rook distance", wk_wr, "domain"),
        _concept("kings_dist", "distance between the kings", bk_wk, "domain"),
        _concept("kings_opposition", "kings two apart on a file or rank (opposition)",
                 ((bkf == wkf) & (np.abs(bkr - wkr) == 2)) | ((bkr == wkr) & (np.abs(bkf - wkf) == 2)), "domain"),
        _concept("bk_on_rook_file", "black king on the rook's file", same_file, "domain"),
        _concept("bk_on_rook_rank", "black king on the rook's rank", same_rank, "domain"),
        _concept("bk_on_edge", "black king on an edge", (bkf == 1) | (bkf == 8) | (bkr == 1) | (bkr == 8), "domain"),
        _concept("bk_in_corner", "black king in a corner", ((bkf == 1) | (bkf == 8)) & ((bkr == 1) | (bkr == 8)), "domain"),
        _concept("bk_edge_dist", "black king's distance to the nearest edge",
                 np.minimum(np.minimum(bkf - 1, 8 - bkf), np.minimum(bkr - 1, 8 - bkr)), "domain"),
    ]


# ============================================================================ generic tabular
def _sanitise(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", str(name))


def tabular_concepts(X_raw: np.ndarray, names: Sequence[str], extra_pairs: Iterable[str] = ()) -> List[Concept]:
    """Each input as it is; each continuous input above its training median; each 0/1 input as a
    level; the pairwise products and conjunctions of ``extra_pairs`` (input names)."""
    X = np.asarray(X_raw, dtype=float)
    out: List[Concept] = []
    cols = {str(n): i for i, n in enumerate(names)}
    for n, i in cols.items():
        v = X[:, i]
        kind = _kind_of(v)
        out.append(_concept(n, f"input {n}", v, "level" if kind == "binary" else "input", kind))
        if kind != "binary":
            med = float(np.median(v))
            if np.any(v > med) and np.any(v <= med):
                out.append(_concept(f"{n} > {med:g}", f"input {n} above its training median ({med:g})",
                                    v > med, "threshold", "binary"))
    # a pair name is an input, or a variable whose one-hot levels are "<name>:<code>" columns
    pairs: List[str] = []
    for p in extra_pairs:
        p = str(p)
        matches = [p] if p in cols else [n for n in cols if n.startswith(p + ":") or n.startswith(p + "_")]
        pairs += [m for m in matches if m not in pairs]
    pairs = pairs[:10]
    for a_i, a in enumerate(pairs):
        for b in pairs[a_i + 1:]:
            va, vb = X[:, cols[a]], X[:, cols[b]]
            if _kind_of(va) == "binary" and _kind_of(vb) == "binary":
                out.append(_concept(f"{a} and {b}", f"both {a} and {b}", (va > 0.5) & (vb > 0.5), "pair", "binary"))
            else:
                out.append(_concept(f"{a} * {b}", f"product of {a} and {b}", va * vb, "pair"))
    return out


# ============================================================================ libraries
def library_for(dataset_name: str) -> Optional[str]:
    """The library a dataset uses: ``c4``, ``c4_lines``, ``krk`` or ``tabular``."""
    n = (dataset_name or "").lower()
    if n.startswith("connect4_lines") or n.startswith("connect4_perline"):
        return "c4_lines"
    if n.startswith("connect4") or n.startswith("connect_4"):
        return "c4"
    if n.startswith("krk"):
        return "krk"
    return "tabular"


def library_concepts(dataset_name: str, X_raw: np.ndarray, names: Sequence[str],
                     extra_pairs: Iterable[str] = (), include_inputs: bool = True) -> List[Concept]:
    """Every concept of the dataset's library on ``X_raw`` (rows in real units, columns
    ``names``). For the domain libraries the raw inputs are also offered (``include_inputs``)
    so a per-cell reading can be told from a domain one (``source``)."""
    lib = library_for(dataset_name)
    X = np.asarray(X_raw, dtype=float)
    if lib == "c4":
        if X.shape[1] != 42:
            raise ValueError(f"Connect-4 library needs 42 inputs, got {X.shape[1]}")
        out = c4_concepts(X)
        if include_inputs:
            out += [_concept(str(n), f"cell {n} (+1 x, -1 o, 0 blank)", X[:, i], "input", "count")
                    for i, n in enumerate(names)]
        return out
    if lib == "c4_lines":
        return c4_line_input_concepts(X, names, include_inputs)
    if lib == "krk":
        if X.shape[1] != 6:
            raise ValueError(f"KRK library needs 6 inputs, got {X.shape[1]}")
        out = krk_concepts(X)
        if include_inputs:
            out += [_concept(str(n), f"input {n}", X[:, i], "input", "count") for i, n in enumerate(names)]
        return out
    return tabular_concepts(X, names, extra_pairs)


def concept_frame(dataset_name: str, X_raw: np.ndarray, names: Sequence[str], concepts: Optional[List[Concept]] = None):
    """A DataFrame with the raw inputs (by name) and the library's concepts (by name), for
    :func:`evaluate_expression`."""
    import pandas as pd
    X = np.asarray(X_raw, dtype=float)
    cols: Dict[str, Any] = {str(n): X[:, i] for i, n in enumerate(names)}
    for c in (concepts if concepts is not None else library_concepts(dataset_name, X, names, include_inputs=False)):
        cols.setdefault(c.name, c.values)
    return pd.DataFrame(cols)


_BACKTICK = re.compile(r"`([^`]+)`")
_IDENT = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?![\w(])")
#: words of the pandas expression language that are not column names
EXPR_WORDS = {"and", "or", "not", "in", "True", "False", "None", "abs", "sqrt", "log", "exp", "sin", "cos",
              "tan", "floor", "ceil", "where", "max", "min"}


def expression_variables(expr: str, columns: Optional[Iterable[str]] = None) -> List[str]:
    """The distinct column names an ad-hoc expression refers to: backtick-quoted names, and bare
    identifiers (those among ``columns`` when a column list is given, else every identifier that
    is not an operator word). Sorted."""
    names = set(_BACKTICK.findall(expr))
    rest = _BACKTICK.sub(" ", expr)
    cols = set(str(c) for c in columns) if columns is not None else None
    for m in _IDENT.finditer(rest):
        w = m.group(0)
        if w in EXPR_WORDS:
            continue
        if cols is None or w in cols:
            names.add(w)
    return sorted(names)


def evaluate_expression(expr: str, frame) -> Concept:
    """An ad-hoc concept: a pandas expression over the frame's columns (backticks for names with
    spaces or hyphens; the library's concept names are columns too)."""
    try:
        v = frame.eval(expr, engine="python")
    except Exception as e:
        raise ValueError(f"cannot evaluate concept expression {expr!r} ({type(e).__name__}: {e}); columns: "
                         + ", ".join(list(frame.columns)[:40]) + (", ..." if len(frame.columns) > 40 else "")) from None
    v = np.asarray(v, dtype=float).ravel()
    if v.shape[0] != len(frame):
        raise ValueError(f"concept expression {expr!r} must give one value per row")
    return _concept(expr, f"expression {expr}", v, "expression")
