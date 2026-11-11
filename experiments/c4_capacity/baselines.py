"""Baselines for the Connect-4 capacity study (same split and z-scored inputs as complex_real).

    lr_cells             logistic regression on the 42 ordinal cells (= complex_real LR)
    lr_cells_threats     + the line / threat / parity counts of c4_features.py (z-scored)
    lr_onehot_threats    x and o indicators per cell (84) + the same counts
    mlp_h{10,20}         dense 42-H-1 ReLU MLP, sigmoid output, full-batch Adadelta lr 1.5 (the
                         PropNEAT optimiser), 2000 epochs, seed 0: what plain backprop reaches
                         with H dense hidden units
GBM numbers are copied from experiments/complex_real/baselines.json (not refit).

    uv run python experiments/c4_capacity/baselines.py [--no-mlp]     # -> baselines.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "complex_real"))
sys.path.insert(0, str(HERE))
from data import prepare  # noqa: E402
from c4_features import features  # noqa: E402

d = prepare("c4")
tr, te = d["train_rows"], d["test_rows"]
Xr = d["X_raw"]
F, fnames = features(Xr)
ytr, yte = d["y_train"], d["y_test"]


def lr(A_tr, A_te):
    s = StandardScaler().fit(A_tr)
    m = LogisticRegression(max_iter=5000, C=1.0).fit(s.transform(A_tr), ytr)
    return dict(train=float(roc_auc_score(ytr, m.decision_function(s.transform(A_tr)))),
                test=float(roc_auc_score(yte, m.decision_function(s.transform(A_te)))))


onehot = np.concatenate([(Xr == 1), (Xr == -1)], 1).astype(float)
out = dict(n_train=int(len(tr)), n_test=int(len(te)), feature_names=fnames,
           lr_cells=lr(d["X_train"], d["X_test"]),
           lr_cells_threats=lr(np.c_[d["X_train"], F[tr]], np.c_[d["X_test"], F[te]]),
           lr_onehot_threats=lr(np.c_[onehot[tr], F[tr]], np.c_[onehot[te], F[te]]),
           gradient_boosting=json.loads((HERE.parent / "complex_real/baselines.json").read_text())["c4"]["gradient_boosting"])

OUT = HERE / "baselines.json"
if OUT.exists():  # keep earlier entries (e.g. the MLP ones) when rerun with --no-mlp
    out = {**json.loads(OUT.read_text()), **out}
torch.set_num_threads(1)
X = torch.tensor(d["X_train"])
Xt = torch.tensor(d["X_test"])
y = torch.tensor(ytr, dtype=torch.float64)[:, None]
for H in ((10, 20) if "--no-mlp" not in sys.argv else ()):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(42, H), torch.nn.ReLU(), torch.nn.Linear(H, 1)).double()
    opt = torch.optim.Adadelta(net.parameters(), lr=1.5)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(2000):
        opt.zero_grad()
        bce(net(X), y).backward()
        opt.step()
    with torch.no_grad():
        out[f"mlp_h{H}"] = dict(train=float(roc_auc_score(ytr, net(X).numpy().ravel())),
                                test=float(roc_auc_score(yte, net(Xt).numpy().ravel())))
OUT.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "feature_names"}, indent=2))
