"""Connect-4 case networks (cells: c4cap-main-h10-e50-s1; lines: c4lf-lines-heavy-s0): the data of the thesis
figure c4-case-contrast (read-only DB).

    PGOPTIONS='-c default_transaction_read_only=on' uv run python thesis/ch7/exp1/extract_c4_case.py
    # -> exp1/data/c4_case.json   (run twice: byte-identical)

The study's own engine: experiments/c4_case/probes.py (CaseModel: the genome's phenotype with the stored explanation
replayed, StructureNetwork forward passes, asserted equal to the explanation library's evaluator; the typical
position is each plain-language layer's reference_patient). Nothing is drawn here: c4_case_figures.py formats the
file.

(a) cells network: the weight of each cell's direct edge to the output (on the standardised cell, as trained), by
board square; per row, the mean |weight|, the mean share of training positions with a piece on a cell of the row
(row_mean_cell_share, the occupancy profile) and the share with a piece anywhere in the row.
(b) lines network: the predicted chance that x wins for the typical position as one line feature changes, and the
training positions at each value.
units (figures c4-shared-board, c4-units-all): the cells network's ten hidden units read from the phenotype (asserted
to reproduce the explained model's logit on every training row): per unit, points per piece w_ki / sigma_i (the
change in its pre-activation from an empty cell to an x; an o is the negative), its empty-board level, output
weight, share of training positions where it fires, SD of its contribution v_k ReLU(pre_k), the training-AUC change
with that contribution held at its mean, and its TOP_N largest-|weight| inputs; the same SD and AUC change for the
direct edges; train and test AUC.
squares (figure c4-square-variance): per square, over the training rows, occupancy, the share of the variance of
the outcome and of the network's logit between the square's three states (x, empty, o), x's win rate with x / with
o on it (at least MIN_ROWS rows) and the direction of each; the whole network's value of an x (value_x) and of an
o (value_o) on the square: the mean change in the logit over the training rows where it is empty, by a full forward
pass (gravity and turn order ignored by design).

Design choices (literals): the sweeps of (b): x's open twos and o's open twos 0..SWEEP_MAX (the record's queries
q_652fe92047 and q_3e23718b44 use 0..6); o's threats 0, 1, 2 (the training range) with the threat-parity inputs set
consistently, every threat on an even row (the common case: odd-row o threats are rare; both counts are stored), and,
for comparison, the record's own sweep (q_daf2178e5e: o_thr alone, parity inputs left at 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments" / "c4_case"))
from probes import COLS, ROWS, CaseModel, cell_index  # noqa: E402

OUT = HERE / "data" / "c4_case.json"
SWEEP_MAX = 6          # design choice: the record's open-twos sweeps run 0..6
THREATS = (0, 1, 2)    # the training range of o's threats
TOP_N = 8              # design choice: the inputs per unit drawn in c4-shared-board (largest |weight|)
MIN_ROWS = 30          # design choice: a square's x / o win rate is reported only with at least this many rows


def _r(v, n=6):
    return [round(float(x), n) for x in np.atleast_1d(v)]


def cells_part(M):
    w = M.direct_edges()
    raw = M.raw
    board, occ_cell = [], []
    for r in range(1, ROWS + 1):
        board.append([float(w[M.col[cell_index(c, r)]]) for c in COLS])
        occ_cell.append([float(np.mean(raw[:, M.col[cell_index(c, r)]] != 0)) for c in COLS])
    B = np.array(board)
    assert not np.isnan(B).any(), "every cell has a direct edge"
    row_any = []
    for r in range(1, ROWS + 1):
        cols = [M.col[cell_index(c, r)] for c in COLS]
        row_any.append(float(np.mean((raw[:, cols] != 0).any(axis=1))))
    return dict(rows=list(range(1, ROWS + 1)), cols=list(COLS),
                direct_weight=[_r(b) for b in board],
                row_mean_abs_weight=_r(np.mean(np.abs(B), axis=1)),
                col_mean_abs_weight=_r(np.mean(np.abs(B), axis=0)),
                row_share_with_piece=_r(row_any),
                row_mean_cell_share=_r(np.mean(np.array(occ_cell), axis=1)),
                cell_share_with_piece=[_r(o) for o in occ_cell],
                typical_chance=round(float(M.chance(M.ref)[0]), 6))


def sweep(M, name, values, **extra):
    X = np.array([M.position(**{name: float(v)}, **{k: f(v) for k, f in extra.items()}) for v in values])
    return _r(M.chance(X))


def counts(M, name, values):
    v = M.raw[:, M.col[name]]
    return [int(np.sum(v == x)) for x in values]


def lines_part(M):
    g = list(range(0, SWEEP_MAX + 1))
    out = dict(typical=M.layer["reference_patient"]["values"], typical_chance=round(float(M.chance(M.ref)[0]), 6))
    out["x_open_2"] = dict(values=g, chance=sweep(M, "x_open_2", g), train_n=counts(M, "x_open_2", g),
                           train_n_above=int(np.sum(M.raw[:, M.col["x_open_2"]] > SWEEP_MAX)))
    out["o_open_2"] = dict(values=g, chance=sweep(M, "o_open_2", g), train_n=counts(M, "o_open_2", g),
                           train_n_above=int(np.sum(M.raw[:, M.col["o_open_2"]] > SWEEP_MAX)))
    t = list(THREATS)
    out["o_thr"] = dict(values=t,
                        chance_even=sweep(M, "o_thr", t, o_thr_even=lambda v: float(v), o_thr_odd=lambda v: 0.0),
                        chance_record=sweep(M, "o_thr", t),
                        train_n=counts(M, "o_thr", t),
                        train_n_even=counts(M, "o_thr_even", t), train_n_odd=counts(M, "o_thr_odd", t))
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _top(w, n):
    """Indices of the n largest |w|, ties broken by index."""
    return sorted(range(len(w)), key=lambda i: (-abs(float(w[i])), i))[:n]


def units_part(M):
    """The cells network's hidden units in raw piece units (figures c4-shared-board, c4-units-all). Asserted: the
    phenotype's units and direct edges, recombined, give the explained model's logit on every training row (so the
    explanation routes them unchanged), the empty-board level, and the points-per-piece reading of every unit."""
    U = M.hidden_units()
    W, b, v, b_out = U["W"], U["b"], U["v"], U["b_out"]
    d = M.direct_edges()
    assert not np.isnan(d).any()
    raw, y = M.raw, np.asarray(M.base.y_train).ravel()
    Z = M.z(raw)
    pre = Z @ W.T + b
    contrib = np.maximum(pre, 0.0) * v
    direct = Z @ d
    logit = b_out + direct + contrib.sum(1)
    lt = M.logit(raw)
    assert np.allclose(logit, lt, atol=1e-10, rtol=0), np.max(np.abs(logit - lt))
    points = W / M.scale                                # change in pre_k from an empty cell to an x piece
    start = b - (W * M.mean / M.scale).sum(1)           # pre_k on an empty board
    empty = M.z(np.zeros(len(M.names)))[0] @ W.T + b
    assert np.allclose(start, empty, atol=1e-12, rtol=0), np.max(np.abs(start - empty))
    e_raw = np.max(np.abs(start + raw @ points.T - pre))
    assert e_raw < 1e-10, e_raw
    base = _auc(y, lt)
    units = []
    for k, uid in enumerate(U["ids"]):
        c = contrib[:, k]
        units.append(dict(id=uid, points=_r(points[k]), w=_r(W[k]), start=round(float(start[k]), 6),
                          out_weight=round(float(v[k]), 6), fires_share=round(float(np.mean(pre[:, k] > 0)), 6),
                          contrib_sd=round(float(np.std(c)), 6),
                          dauc_mean=round(_auc(y, lt - c + c.mean()) - base, 6), top8=_top(W[k], TOP_N)))
    order = [u["id"] for u in sorted(units, key=lambda u: (-u["contrib_sd"], int(u["id"])))]
    test_auc = _auc(np.asarray(M.base.y_test).ravel(), M.logit(M.base.raw_test))
    return dict(names=list(M.names), units=units, order=order, top_n=TOP_N,
                direct=dict(contrib_sd=round(float(np.std(direct)), 6),
                            dauc_mean=round(_auc(y, lt - direct + direct.mean()) - base, 6)),
                train_auc=round(base, 6), test_auc=round(test_auc, 6), n_train=int(len(y)),
                occupancy=_r(np.mean(raw != 0, axis=0)))


def _between(v, col):
    """Share of Var(v) between the groups of a cell's three states (x, empty, o)."""
    return float(sum(np.mean(col == s) * (v[col == s].mean() - v.mean()) ** 2
                     for s in (-1, 0, 1) if np.any(col == s)) / v.var())


def squares_part(M):
    """Per square, over the training rows: the share of the outcome's and of the network's logit's variance
    between the square's three states, and the direction of each; and the whole network's value of a piece on the
    square: over the rows where it is empty, the mean change in the logit when an x (value_x) or an o (value_o) is
    put there, by the full forward pass (gravity and turn order ignored by design: a probe of the network, not a
    legal move) (figure c4-square-variance)."""
    raw, y = M.raw, np.asarray(M.base.y_train, dtype=float).ravel()
    lt = M.logit(raw)
    out = dict(occupancy=[], outcome_var_share=[], model_var_share=[], x_rate=[], o_rate=[], outcome_dir=[],
               model_dir=[], value_x=[], value_o=[], n_empty=[])
    for i in range(raw.shape[1]):
        col = raw[:, i]
        rate = {s: (float(y[col == s].mean()) if np.sum(col == s) >= MIN_ROWS else None) for s in (1, -1)}
        out["occupancy"].append(round(float(np.mean(col != 0)), 6))
        out["outcome_var_share"].append(round(_between(y, col), 6))
        out["model_var_share"].append(round(_between(lt, col), 6))
        out["x_rate"].append(None if rate[1] is None else round(rate[1], 6))
        out["o_rate"].append(None if rate[-1] is None else round(rate[-1], 6))
        out["outcome_dir"].append(0 if None in rate.values() else int(np.sign(rate[1] - rate[-1])))
        both = np.any(col == 1) and np.any(col == -1)
        out["model_dir"].append(int(np.sign(lt[col == 1].mean() - lt[col == -1].mean())) if both else 0)
        empty = col == 0
        for key, piece in (("value_x", 1.0), ("value_o", -1.0)):
            X = raw[empty].copy()
            X[:, i] = piece
            M.chance(X)   # the forward pass, asserted equal to the library evaluator (logit() would also round-trip
            #               the sigmoid, which saturates on some extrapolated positions)
            out[key].append(round(float(np.mean(M.m.logit(M.z(X)) - lt[empty])), 6))
        out["n_empty"].append(int(np.sum(empty)))
    out["names"] = list(M.names)
    out["min_rows"] = MIN_ROWS
    return out


def main():
    C = CaseModel("cells")
    L = CaseModel("lines")
    d = dict(source=dict(cells=C.meta, lines=L.meta, generator="thesis/ch7/exp1/extract_c4_case.py"),
             cells=cells_part(C), lines=lines_part(L), units=units_part(C), squares=squares_part(C))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
