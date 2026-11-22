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


def main():
    C = CaseModel("cells")
    L = CaseModel("lines")
    d = dict(source=dict(cells=C.meta, lines=L.meta, generator="thesis/ch7/exp1/extract_c4_case.py"),
             cells=cells_part(C), lines=lines_part(L))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
