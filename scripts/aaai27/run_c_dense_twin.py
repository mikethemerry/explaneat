"""Run C: dense twin (dense-width, dense-auc, mean-fanin, weight-spread stats).

Trains an MLP on Adult's 54 phenotype-connected features (same stored split,
same stored z-normalisation), 2 hidden ReLU layers of equal width. Width is
chosen so held-out test AUC matches the evolved network's 0.905 +/- 0.005
(search starts at 16). Standard training: Adam, modest weight decay (1e-4),
early stopping on train-loss plateau. NO deliberate sparsification (no L1,
no pruning).

Per first-layer unit: effective fan-in = number of inputs with |w| >= 10% of
that unit's max |w|. Reports mean/median/min effective fan-in and the mean %
of inputs per unit above the threshold. Evolved-network comparator: hidden
units receive 2-6 inputs.

Run: uv run python scripts/aaai27/run_c_dense_twin.py
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from aaai27.common import ADULT, SEED, load_model

from explaneat.db import db
from explaneat.core.explaneat import ExplaNEAT

TARGET_AUC = 0.905
TOL = 0.005
MAX_EPOCHS = 2000
PATIENCE = 50          # epochs without train-loss improvement
MIN_DELTA = 1e-5
WEIGHT_DECAY = 1e-4
LR = 1e-3


def connected_feature_indices(session):
    """The 54 Adult features on an active input->output path, as column indices."""
    genome, config, X, y, split, _, _ = load_model(session, ADULT)
    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    input_ids = set(pheno.input_node_ids)
    connected_keys = {c.from_node for c in pheno.connections if c.enabled} & input_ids
    # NEAT input key -k corresponds to feature column k-1
    cols = sorted(-int(k) - 1 for k in connected_keys)
    return cols, X, y, split


def train_mlp(Xtr, ytr, width, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 1),
    ).double()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossfn = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64).view(-1, 1)

    best_loss, stale = np.inf, 0
    for epoch in range(MAX_EPOCHS):
        opt.zero_grad()
        loss = lossfn(model(xt), yt)
        loss.backward()
        opt.step()
        cur = float(loss)
        if cur < best_loss - MIN_DELTA:
            best_loss, stale = cur, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return model, epoch + 1


def test_auc(model, Xte, yte):
    with torch.no_grad():
        scores = model(torch.tensor(Xte, dtype=torch.float64)).numpy().ravel()
    return roc_auc_score(yte, scores)


def fanin_stats(model, threshold_frac=0.10):
    W = model[0].weight.detach().numpy()  # (width, n_inputs)
    absW = np.abs(W)
    per_unit_max = absW.max(axis=1, keepdims=True)
    above = absW >= threshold_frac * per_unit_max
    fanin = above.sum(axis=1)
    pct = 100.0 * fanin / W.shape[1]
    return fanin, pct, W


def main():
    with db.session_scope() as s:
        cols, X, y, split = connected_feature_indices(s)
        print(f"connected features: {len(cols)} columns")
        Xtr, ytr = X[split.train_indices][:, cols], y[split.train_indices]
        Xte, yte = X[split.test_indices][:, cols], y[split.test_indices]

    width, tried = 16, {}
    while True:
        model, epochs = train_mlp(Xtr, ytr, width)
        auc = test_auc(model, Xte, yte)
        tried[width] = auc
        print(f"width={width:3d}: test AUC={auc:.4f} (epochs={epochs})")
        if abs(auc - TARGET_AUC) <= TOL:
            break
        # adjust: too low -> wider; too high -> narrower
        width = width * 2 if auc < TARGET_AUC - TOL else max(2, width // 2)
        if width in tried:
            print(f"width search cycled at {sorted(tried.items())}; taking closest")
            width = min(tried, key=lambda w: abs(tried[w] - TARGET_AUC))
            model, epochs = train_mlp(Xtr, ytr, width)
            auc = test_auc(model, Xte, yte)
            break

    fanin, pct, W = fanin_stats(model)
    print(f"\n=== dense twin: 54-{width}-{width}-1 MLP, test AUC {auc:.4f} "
          f"(evolved: 0.9055) ===")
    print(f"first-layer effective fan-in (|w| >= 10% of unit max, {W.shape[1]} inputs):")
    print(f"  mean={fanin.mean():.1f}  median={np.median(fanin):.0f}  "
          f"min={fanin.min()}  max={fanin.max()}")
    print(f"  % of inputs above threshold per unit: mean={pct.mean():.1f}%  "
          f"min={pct.min():.1f}%  max={pct.max():.1f}%")
    print(f"  per-unit fan-in: {sorted(fanin.tolist())}")
    print(f"weight spread: |w| mean={np.abs(W).mean():.4f}  "
          f"median={np.median(np.abs(W)):.4f}  max={np.abs(W).max():.4f}")
    print("evolved comparator: hidden units receive 2-6 inputs each")

    if fanin.mean() <= 10:
        print("\n*** ATTENTION: mean effective fan-in is LOW - units look sparse/"
              "nameable. Flag for Section 5.4 before using any of these numbers. ***")


if __name__ == "__main__":
    main()
