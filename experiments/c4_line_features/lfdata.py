"""Connect-4 line-features base test: derived line features as network inputs
(exploratory, not pre-registered).

Same rows, split and target as experiments/complex_real (67,557 eight-ply positions, x to move;
y = 1 iff x wins with perfect play; stratified 80/20 split, seed 42). What changes is the
**base**: the inputs the network sees.

    lines        the 16 line features of CODEBOOK (from c4_features.py on exp/c4-capacity and
                 the Connect-4 concept library in explaneat/analysis/concepts.py): open lines of
                 1-3 per side, threats per side with their row parity and immediacy, centre
                 column pieces, playable-cell parity. The playable-now threats (x_thr_now,
                 o_thr_now) are constant zero on this dataset (UCI excludes positions with a
                 forced move) and are left out
    lines_cells  the 16 line features followed by the 42 raw cells c_a1 .. c_g6
    perline      the 138 per-line features (for each of the 69 lines of 4 and each side, the
                 side's piece count on the line when no opposing piece is on it, else 0)
                 followed by the 16 line features (added after the GBM signal check,
                 signal_check.py: this vocabulary carries more of the signal than the cells)

Inputs are z-scored with a StandardScaler fit on the training rows (as before); the database
keeps the raw values and puts the scaler on the split. The codebook (``CODEBOOK``) names every
input so that leaves can be read in game terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "complex_real"))

from data import C4_FEATURES, load_task, split_indices  # noqa: E402  (complex_real/data.py; this module is lfdata to avoid the clash)
from explaneat.analysis.concepts import c4_concepts  # noqa: E402

#: the line features, in input order, with the codebook text (x = first player, to move)
CODEBOOK = {
    "x_open_1": "lines of 4 with exactly 1 x piece and no o piece (x's open singles)",
    "o_open_1": "lines of 4 with exactly 1 o piece and no x piece (o's open singles)",
    "x_open_2": "lines of 4 with exactly 2 x pieces and no o piece (x's open twos)",
    "o_open_2": "lines of 4 with exactly 2 o pieces and no x piece (o's open twos)",
    "x_open_3": "lines of 4 with exactly 3 x pieces and no o piece (x's open threes)",
    "o_open_3": "lines of 4 with exactly 3 o pieces and no x piece (o's open threes)",
    "x_thr": "distinct empty cells that complete an open three of x (x's threats)",
    "x_thr_odd": "x's threats on odd rows (1, 3, 5; the first player's parity rows)",
    "x_thr_even": "x's threats on even rows (2, 4, 6)",
    "o_thr": "distinct empty cells that complete an open three of o (o's threats)",
    "o_thr_odd": "o's threats on odd rows (1, 3, 5)",
    "o_thr_even": "o's threats on even rows (2, 4, 6; the second player's parity rows)",
    "centre_x": "x pieces in the centre column d",
    "centre_o": "o pieces in the centre column d",
    "n_odd_empty_playable": "columns whose next playable cell is on an odd row",
    "n_even_empty_playable": "columns whose next playable cell is on an even row",
}
LINE_FEATURES = list(CODEBOOK)
CELL_CODEBOOK = {n: f"cell {n} (column {n[2]}, row {n[3]}; +1 x, -1 o, 0 blank)" for n in C4_FEATURES}


from explaneat.analysis.concepts import C4_LINES, _C4_LINE_IDX  # noqa: E402

_DIR = {(1, 0): "horizontal", (0, 1): "vertical", (1, 1): "diagonal up", (1, -1): "diagonal down"}


def _line_name(i):
    l = C4_LINES[i]
    return f"{'abcdefg'[l[0][0]]}{l[0][1] + 1}-{'abcdefg'[l[3][0]]}{l[3][1] + 1}"


def _line_dir(i):
    l = C4_LINES[i]
    return _DIR[(l[1][0] - l[0][0], l[1][1] - l[0][1])]


PERLINE_CODEBOOK = {}
for _p, _who in (("x", "x"), ("o", "o")):
    for _i in range(len(C4_LINES)):
        PERLINE_CODEBOOK[f"{_p}_line_{_line_name(_i)}"] = (
            f"{_who} pieces on the {_line_dir(_i)} line {_line_name(_i)} when no "
            f"{'o' if _p == 'x' else 'x'} piece is on it (0 if blocked or empty)")
PERLINE_FEATURES = list(PERLINE_CODEBOOK)

INPUT_SETS = {"lines": LINE_FEATURES, "lines_cells": LINE_FEATURES + C4_FEATURES,
              "perline": PERLINE_FEATURES + LINE_FEATURES}
DATASET_NAMES = {"lines": "connect4_lines", "lines_cells": "connect4_lines_cells", "perline": "connect4_perline"}


def line_features(X_cells: np.ndarray) -> np.ndarray:
    """[N, 16] line features from raw boards (+1 x, -1 o, 0 blank), via the concept library so
    the inputs and the probe's concepts are the same functions."""
    by_name = {c.name: c.values for c in c4_concepts(np.asarray(X_cells, dtype=np.float64))}
    return np.stack([by_name[n] for n in LINE_FEATURES], 1).astype(np.float64)


def perline_features(X_cells: np.ndarray) -> np.ndarray:
    """[N, 138]: per line and side, the side's piece count if the line is open for it, else 0."""
    cells = np.asarray(X_cells, dtype=np.float64)[:, _C4_LINE_IDX]      # [N, 69, 4]
    nx = (cells == 1).sum(-1)
    no = (cells == -1).sum(-1)
    return np.concatenate([np.where(no == 0, nx, 0), np.where(nx == 0, no, 0)], 1).astype(np.float64)


def codebook(inputs: str) -> dict:
    cb = dict(PERLINE_CODEBOOK) if inputs == "perline" else {}
    cb.update(CODEBOOK)
    if inputs == "lines_cells":
        cb.update(CELL_CODEBOOK)
    return cb


def prepare(inputs: str):
    """Raw and z-scored train/test arrays for an input set, in the complex_real ``prepare`` layout
    (plus ``X_cells``, the raw boards)."""
    if inputs not in INPUT_SETS:
        raise ValueError(inputs)
    X_cells, y, _, meta = load_task("c4")
    F = line_features(X_cells)
    X = {"lines": F, "lines_cells": lambda: np.concatenate([F, X_cells], 1),
         "perline": lambda: np.concatenate([perline_features(X_cells), F], 1)}[inputs]
    X = X() if callable(X) else X
    names = list(INPUT_SETS[inputs])
    tr, te = split_indices(y)
    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0)
    scale[scale == 0.0] = 1.0
    z = lambda A: (A - mean) / scale  # noqa: E731
    meta = dict(meta, task="c4", inputs=inputs, n_inputs=len(names),
                encoding={"lines": "16 line features (counts) from the 42 ordinal cells",
                          "lines_cells": "16 line features (counts) from the 42 ordinal cells; then the 42 ordinal cells (x = +1, o = -1, blank = 0)",
                          "perline": "138 per-line open counts (69 lines x 2 sides) from the 42 ordinal cells; then the 16 line features"}[inputs],
                n_rows=int(len(y)), n_train=int(len(tr)), n_test=int(len(te)),
                positive_share=float(y.mean()), positive_share_train=float(y[tr].mean()),
                split_seed=42, scaler="StandardScaler fit on the training rows",
                study="experiments/c4_line_features")
    return dict(X_raw=X, X_cells=X_cells, y_all=y, names=names, train_rows=tr, test_rows=te,
                X_train=z(X[tr]), y_train=y[tr], X_test=z(X[te]), y_test=y[te],
                mean=mean, scale=scale, meta=meta)


if __name__ == "__main__":
    d = prepare("lines")
    print(d["X_raw"].shape, d["meta"])
    for n, v in zip(d["names"], d["X_raw"].T):
        print(f"{n:24s} min {v.min():.0f} max {v.max():.0f} mean {v.mean():.3f}  {CODEBOOK[n]}")
