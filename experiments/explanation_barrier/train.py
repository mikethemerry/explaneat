"""Train plain dense ReLU MLPs (sigmoid output) for the explanation complexity ladder.

    uv run python experiments/explanation_barrier/train.py <task> [arch ...]
    arch: "4" (one hidden layer of 4), "8x8" (two layers), ...; default: the task's grid

Plain PyTorch backprop, float64: Adam (lr 1e-2, decayed), minibatch 512, BCE, weight decay 1e-5,
early stopping on a 10% validation slice of the *training* rows (seed 0; patience 25 epochs,
at most 400). The test split is used only for the reported test AUC. One torch thread.
Each model is saved to models/<task>-<arch>.pt with its metrics in models/<task>-<arch>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

MODELS = HERE / "models"
GRID = ["2", "4", "8", "16", "32", "8x8", "16x16"]
C4_GRID = ["4", "8", "16", "32", "32x16"]


class MLP(torch.nn.Module):
    def __init__(self, d, widths):
        super().__init__()
        dims = [d] + list(widths)
        self.hidden = torch.nn.ModuleList(torch.nn.Linear(a, b, dtype=torch.float64) for a, b in zip(dims, dims[1:]))
        self.out = torch.nn.Linear(dims[-1], 1, dtype=torch.float64)

    def logit(self, x):
        for h in self.hidden:
            x = torch.relu(h(x))
        return self.out(x).squeeze(-1)

    def forward(self, x):
        return torch.sigmoid(self.logit(x))

    def hidden_acts(self, x):
        acts = []
        for h in self.hidden:
            x = torch.relu(h(x))
            acts.append(x)
        return acts


def parse(arch):
    return [int(w) for w in arch.split("x")]


def train(task, arch, seed=0, max_epochs=400, patience=25, bs=512):
    d = data.prepare(task)
    Xtr, ytr = d["X_train"], d["y_train"].astype(np.float64)
    r = np.random.default_rng(seed)
    perm = r.permutation(len(ytr))
    nval = len(ytr) // 10
    va, fit = perm[:nval], perm[nval:]
    torch.manual_seed(seed)
    net = MLP(Xtr.shape[1], parse(arch))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    lossf = torch.nn.BCEWithLogitsLoss()
    X = torch.as_tensor(Xtr)
    Y = torch.as_tensor(ytr)
    best, best_state, wait = np.inf, None, 0
    t0 = time.time()
    for ep in range(max_epochs):
        net.train()
        idx = torch.as_tensor(r.permutation(fit))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = lossf(net.logit(X[b]), Y[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net.logit(X[va]), Y[va]))
        sched.step(vl)
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        lg_tr = net.logit(X).numpy()
        lg_te = net.logit(torch.as_tensor(d["X_test"])).numpy()
    m = dict(task=task, arch=arch, seed=seed, epochs=ep + 1, val_bce=best, wall_s=time.time() - t0,
             n_params=int(sum(p.numel() for p in net.parameters())),
             train_auc=float(roc_auc_score(ytr, lg_tr)), test_auc=float(roc_auc_score(d["y_test"], lg_te)))
    if d["f"] is not None:
        f = d["f"]
        ftr, fte = f[d["train_rows"]], f[d["test_rows"]]
        m.update(bayes_train_auc=float(roc_auc_score(ytr, ftr)), bayes_test_auc=float(roc_auc_score(d["y_test"], fte)),
                 logit_corr_f_train=float(np.corrcoef(lg_tr, ftr)[0, 1]),
                 logit_corr_f_test=float(np.corrcoef(lg_te, fte)[0, 1]),
                 logit_rmse_f_test=float(np.sqrt(np.mean((lg_te - fte) ** 2))))
    MODELS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), MODELS / f"{task}-{arch}.pt")
    (MODELS / f"{task}-{arch}.json").write_text(json.dumps(m, indent=2) + "\n")
    return m


if __name__ == "__main__":
    torch.set_num_threads(1)
    task = sys.argv[1]
    archs = sys.argv[2:] or (C4_GRID if task == "c4" else GRID)
    for a in archs:
        m = train(task, a)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
