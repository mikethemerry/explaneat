"""Fixed small dense CNN baseline (to keep the comparison honest).

conv3x3(1->c) ReLU -> avgpool2 -> conv3x3(c->c) ReLU -> GAP -> linear.
``c`` is chosen so the parameter count is close to a target (the evolved
genome's), or set explicitly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, c: int, n_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.c2(x))
        return self.fc(x.mean(dim=(2, 3)))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def width_for_params(target: int, in_ch: int, n_classes: int) -> int:
    best, bc = None, 1
    for c in range(1, 64):
        p = n_params(SmallCNN(in_ch, c, n_classes))
        if best is None or abs(p - target) < best:
            best, bc = abs(p - target), c
    return bc


def train_baseline(data, c: int, n_classes: int, epochs: int, seed: int, device, lr=0.01, batch=128):
    """Adam, minibatch; keeps the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_ch = data["X_train"].shape[1]
    m = SmallCNN(in_ch, c, n_classes).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    T = {s: torch.tensor(data[f"X_{s}"], device=device) for s in ("train", "val", "test")}
    Y = {s: torch.tensor(data[f"y_{s}"], device=device) for s in ("train", "val", "test")}

    def acc(s):
        m.eval()
        with torch.no_grad():
            lo = m(T[s])
            p = (lo[:, 0] > 0).long() if n_classes == 1 else lo.argmax(1)
        m.train()
        return float((p == Y[s]).float().mean())

    best = (-1, None, None)
    n = len(T["train"])
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(n), device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            lo = m(T["train"][idx])
            loss = (F.binary_cross_entropy_with_logits(lo[:, 0], Y["train"][idx].float())
                    if n_classes == 1 else F.cross_entropy(lo, Y["train"][idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        va = acc("val")
        if va > best[0]:
            best = (va, acc("test"), ep)
    return {"width": c, "n_params": n_params(m), "val_acc": best[0], "test_acc": best[1],
            "best_epoch": best[2], "epochs": epochs}
