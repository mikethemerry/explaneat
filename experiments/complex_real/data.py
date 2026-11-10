"""Data for the complex-real exploratory runs (exploratory, not pre-registered).

Source: PMLB (``pmlb.fetch_data``), cached outside the repo under
``~/.explaneat/data/pmlb``. PMLB label-encodes string columns alphabetically; the decoding
below was checked against the UCI class counts (krkopt) and the board rules (connect_4).

``krk``  UCI "Chess (King-Rook vs. King)", PMLB ``krkopt``: 28,056 positions, white king +
         white rook vs black king, black to move. 6 inputs, ordinal integers:
         ``wk_file, wk_rank, wr_file, wr_rank, bk_file, bk_rank`` with file a-h -> 1-8
         (PMLB stores files 0-7, so +1) and rank 1-8 (stored as is). The white king is
         confined to the a1-d1-d4 triangle (symmetry reduction in the source).
         Target (binarised): y = 1 for a **draw**, y = 0 for a win for white (depth-to-mate
         0-16). PMLB target code 0 = "draw" (alphabetical: draw, eight, eleven, ..., zero;
         counts match UCI: draw 2,796, zero 27, one 78, ..., sixteen 390).
``c4``   UCI "Connect-4", PMLB ``connect_4``: 67,557 legal 8-ply positions (4 x, 4 o)
         where neither player has won and the next move is not forced; x (the first
         player) is to move. 42 inputs ``c_a1 .. c_g6`` (column a-g, row 1 = bottom to
         6 = top), ordinal: x = +1, o = -1, blank = 0 (PMLB codes b = 0, o = 1, x = 2).
         Target (binarised): y = 1 if x **wins** with perfect play, 0 for a loss or draw
         (PMLB code 2 = "win"; draw = 0, loss = 1).

Split: stratified 80:20 on the binary label, ``random_state`` 42 (sklearn
``train_test_split``). The networks see the inputs z-scored with a StandardScaler fit on
the training rows (as in the 10-seed study); the database keeps the raw ordinal values and
the scaler on the split.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("EXPLANEAT_DATA", Path.home() / ".explaneat" / "data")) / "pmlb"
SPLIT_SEED = 42
SUBSET_SEED = 12345

KRK_FEATURES = ["wk_file", "wk_rank", "wr_file", "wr_rank", "bk_file", "bk_rank"]
C4_FEATURES = [f"c_{c}{r}" for c in "abcdefg" for r in range(1, 7)]


def _fetch(name):
    from pmlb import fetch_data
    CACHE.mkdir(parents=True, exist_ok=True)
    return fetch_data(name, local_cache_dir=str(CACHE))


def load_task(task: str):
    """(X raw float64 [N, d], y int {0,1} [N], feature names, meta)."""
    if task == "krk":
        df = _fetch("krkopt")
        X = df.drop(columns="target").to_numpy().astype(np.float64)
        X[:, [0, 2, 4]] += 1.0           # files a-h: 0-7 -> 1-8
        y = (df["target"].to_numpy() == 0).astype(np.int64)   # 0 = draw
        meta = dict(task="krk", source="PMLB krkopt (UCI Chess KRK)",
                    target="1 = draw, 0 = win for white (any depth-to-mate 0-16)",
                    encoding="ordinal integers: file a-h -> 1-8, rank 1-8",
                    class_names=["win", "draw"])
        return X, y, list(KRK_FEATURES), meta
    if task == "c4":
        df = _fetch("connect_4")
        raw = df.drop(columns="target").to_numpy()
        X = np.select([raw == 2, raw == 1], [1.0, -1.0], 0.0).astype(np.float64)
        y = (df["target"].to_numpy() == 2).astype(np.int64)   # 2 = win
        meta = dict(task="c4", source="PMLB connect_4 (UCI Connect-4)",
                    target="1 = first player (x, to move) wins with perfect play; 0 = loss or draw",
                    encoding="ordinal per cell: x = +1, o = -1, blank = 0; c_<column a-g><row 1-6>, row 1 = bottom",
                    class_names=["not win", "win"])
        return X, y, list(C4_FEATURES), meta
    raise ValueError(task)


def split_indices(y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SPLIT_SEED)
    return np.sort(tr), np.sort(te)


def prepare(task: str, n_train=None):
    """Raw and z-scored train/test arrays. ``n_train`` subsamples the *training* rows
    (stratified, seed SUBSET_SEED); the test rows are always the full 20%."""
    X, y, names, meta = load_task(task)
    tr, te = split_indices(y)
    full_train = len(tr)
    if n_train is not None and n_train < len(tr):
        from sklearn.model_selection import train_test_split
        tr, _ = train_test_split(tr, train_size=n_train, stratify=y[tr], random_state=SUBSET_SEED)
        tr = np.sort(tr)
    mean = X[tr].mean(axis=0)
    scale = X[tr].std(axis=0)
    scale[scale == 0.0] = 1.0
    z = lambda A: (A - mean) / scale  # noqa: E731
    meta = dict(meta, n_rows=int(len(y)), n_train_full=int(full_train), n_train=int(len(tr)),
                n_test=int(len(te)), positive_share=float(y.mean()),
                positive_share_train=float(y[tr].mean()), split_seed=SPLIT_SEED,
                subset_seed=SUBSET_SEED if n_train else None,
                scaler="StandardScaler fit on the training rows")
    return dict(X_raw=X, y_all=y, names=names, train_rows=tr, test_rows=te,
                X_train=z(X[tr]), y_train=y[tr], X_test=z(X[te]), y_test=y[te],
                mean=mean, scale=scale, meta=meta)


# ----------------------------------------------------------------------------- ground truth
C4_LINES = []
for _c in range(7):
    for _r in range(6):
        for _dc, _dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            _cells = [(_c + i * _dc, _r + i * _dr) for i in range(4)]
            if all(0 <= a < 7 and 0 <= b < 6 for a, b in _cells):
                C4_LINES.append(_cells)


def c4_board(X_raw):
    """[N, 7 columns, 6 rows] with +1 = x, -1 = o, 0 = blank (row 0 = bottom)."""
    return np.asarray(X_raw).reshape(-1, 7, 6)


def krk_rules(X_raw):
    """Exact KRK draw rule features (black to move): the only draws in KRK are an immediate
    rook capture by the black king (rook adjacent to the black king, not adjacent to the
    white king) or stalemate. Returns dict of boolean arrays."""
    X = np.asarray(X_raw, dtype=int)
    wkf, wkr, wrf, wrr, bkf, bkr = (X[:, i] for i in range(6))
    cheb = lambda f1, r1, f2, r2: np.maximum(np.abs(f1 - f2), np.abs(r1 - r2))  # noqa: E731
    bk_wr = cheb(bkf, bkr, wrf, wrr)
    wk_wr = cheb(wkf, wkr, wrf, wrr)
    capture = (bk_wr == 1) & (wk_wr > 1)
    # stalemate: black king not in check and has no legal move
    def attacked(f, r):
        by_k = cheb(f, r, wkf, wkr) <= 1
        # rook attacks along file/rank, blocked by the white king (the black king moves away)
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
    return dict(capture=capture, stalemate=stalemate, in_check=in_check,
                bk_rook_dist=bk_wr, wk_rook_dist=wk_wr,
                rule_draw=capture | stalemate)


def feature_names(task):
    return list(KRK_FEATURES if task == "krk" else C4_FEATURES)
