"""Connect-4 domain features from the raw board (``c4_board``, ``C4_LINES`` in
experiments/complex_real/data.py), used for the threat-feature baseline and for reading the
learnt non-linearity.

Per position (x to move; rows 0-based from the bottom, so an even row index is an odd row
number, 1/3/5):
    x_open_k, o_open_k   lines with k = 1, 2, 3 pieces of that player and none of the other
    x_thr, o_thr         distinct empty cells that complete a line of 3 (a "threat")
    x_thr_odd/even, o_thr_odd/even   threats by row parity (row number 1/3/5 = odd)
    x_thr_now, o_thr_now threats whose cell is playable now (bottom row or the cell below filled)
    centre_x, centre_o   pieces in column d
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "complex_real"))
from data import C4_LINES, c4_board  # noqa: E402

LINE_IDX = np.array([[c * 6 + r for c, r in line] for line in C4_LINES])  # [69, 4] flat cell index


def features(X_raw):
    X = np.asarray(X_raw, dtype=np.float64)
    B = c4_board(X)
    N = len(X)
    cells = X[:, LINE_IDX]                      # [N, 69, 4]
    nx = (cells == 1).sum(-1)
    no = (cells == -1).sum(-1)
    out = {}
    for k in (1, 2, 3):
        out[f"x_open_{k}"] = ((nx == k) & (no == 0)).sum(1)
        out[f"o_open_{k}"] = ((no == k) & (nx == 0)).sum(1)
    filled = X != 0
    below_filled = np.ones_like(filled)
    for c in range(7):
        for r in range(1, 6):
            below_filled[:, c * 6 + r] = filled[:, c * 6 + r - 1]
    row = np.tile(np.arange(6), 7)
    for p, cnt in (("x", nx), ("o", no)):
        other = no if p == "x" else nx
        thr_cell = np.zeros((N, 42), bool)
        sel = (cnt == 3) & (other == 0)             # [N, 69]
        for li in range(len(C4_LINES)):
            rows_ = np.where(sel[:, li])[0]
            if len(rows_) == 0:
                continue
            empt = ~filled[np.ix_(rows_, LINE_IDX[li])]          # [n, 4]
            for j in range(4):
                thr_cell[rows_[empt[:, j]], LINE_IDX[li, j]] = True
        out[f"{p}_thr"] = thr_cell.sum(1)
        out[f"{p}_thr_odd"] = thr_cell[:, row % 2 == 0].sum(1)
        out[f"{p}_thr_even"] = thr_cell[:, row % 2 == 1].sum(1)
        out[f"{p}_thr_now"] = (thr_cell & below_filled).sum(1)
    out["centre_x"] = (B[:, 3, :] == 1).sum(1)
    out["centre_o"] = (B[:, 3, :] == -1).sum(1)
    names = list(out)
    return np.stack([out[n] for n in names], 1).astype(np.float64), names
