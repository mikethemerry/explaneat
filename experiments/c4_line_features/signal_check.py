"""Where does the Connect-4 signal live? GBM (sklearn HistGradientBoosting, defaults, seed 0) on
nameable line vocabularies of increasing resolution, same split; a diagnostic of how much of the
GBM-on-cells signal each vocabulary carries (not a model selection).

    aggregate16    the 16 line features of lfdata.CODEBOOK (the ``lines`` base)
    direction24    open lines of k = 1..3 per side split by direction (horizontal, vertical, the two diagonals)
    rowband        open lines of k = 1..3 per side split by the lowest row of the line (1-6)
    perline138     per line (69) and side: the side's piece count if the line is open for it, else 0
    cells42        the raw cells, refit here for parity with complex_real (0.944)

    uv run python experiments/c4_line_features/signal_check.py   # -> signal_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lfdata import C4_FEATURES, prepare  # noqa: E402
from explaneat.analysis.concepts import C4_LINES, _C4_LINE_IDX  # noqa: E402

d = prepare("lines")
Xc, y, tr, te = d["X_cells"], d["y_all"], d["train_rows"], d["test_rows"]
cells = Xc[:, _C4_LINE_IDX]            # [N, 69, 4]
nx = (cells == 1).sum(-1)
no = (cells == -1).sum(-1)
x_open = np.where(no == 0, nx, 0)      # x's count on lines open for x (0 if blocked or empty)
o_open = np.where(nx == 0, no, 0)
direction = np.array([(l[1][0] - l[0][0], l[1][1] - l[0][1]) for l in C4_LINES])
dir_names = {(1, 0): "h", (0, 1): "v", (1, 1): "d_up", (1, -1): "d_down"}
dir_id = np.array([list(dir_names).index(tuple(v)) for v in direction])
low_row = np.array([min(r for _, r in l) for l in C4_LINES])

vocab = {}
vocab["aggregate16"] = (d["X_raw"], d["names"])
F, names = [], []
for p, arr in (("x", x_open), ("o", o_open)):
    for k in (1, 2, 3):
        for di, dn in enumerate(dir_names.values()):
            F.append(((arr == k) & (dir_id == di)).sum(1)); names.append(f"{p}_open_{k}_{dn}")
vocab["direction24"] = (np.stack(F, 1).astype(float), names)
F, names = [], []
for p, arr in (("x", x_open), ("o", o_open)):
    for k in (1, 2, 3):
        for r in range(6):
            F.append(((arr == k) & (low_row == r)).sum(1)); names.append(f"{p}_open_{k}_row{r + 1}")
vocab["rowband36"] = (np.stack(F, 1).astype(float), names)
vocab["aggregate16+direction24+rowband36"] = (np.concatenate([vocab["aggregate16"][0], vocab["direction24"][0], vocab["rowband36"][0]], 1),
                                            vocab["aggregate16"][1] + vocab["direction24"][1] + vocab["rowband36"][1])
vocab["perline138"] = (np.concatenate([x_open, o_open], 1).astype(float),
                       [f"x_line{i}" for i in range(69)] + [f"o_line{i}" for i in range(69)])
vocab["perline138+aggregate16"] = (np.concatenate([vocab["perline138"][0], vocab["aggregate16"][0]], 1),
                                   vocab["perline138"][1] + vocab["aggregate16"][1])
vocab["cells42"] = (Xc, list(C4_FEATURES))

out = {}
for name, (X, names) in vocab.items():
    mean, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1
    Z = (X - mean) / sd
    gbm = HistGradientBoostingClassifier(random_state=0).fit(Z[tr], y[tr])
    lr = LogisticRegression(max_iter=5000).fit(Z[tr], y[tr])
    out[name] = dict(n_inputs=int(X.shape[1]),
                     gbm=dict(train=float(roc_auc_score(y[tr], gbm.predict_proba(Z[tr])[:, 1])),
                              test=float(roc_auc_score(y[te], gbm.predict_proba(Z[te])[:, 1]))),
                     lr=dict(train=float(roc_auc_score(y[tr], lr.decision_function(Z[tr]))),
                             test=float(roc_auc_score(y[te], lr.decision_function(Z[te])))))
    print(f"{name:36s} d={X.shape[1]:3d}  LR test {out[name]['lr']['test']:.4f}  GBM test {out[name]['gbm']['test']:.4f}", flush=True)
(HERE / "signal_check.json").write_text(json.dumps(out, indent=2) + "\n")
