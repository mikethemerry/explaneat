"""Run C2: densified twin of the Adult paper genome (replaces the 54-16-16-1 MLP).

Instead of an arbitrary 2-layer MLP, we densify the evolved network *itself*:
take the evolved phenotype's PropNEAT layer mapping (node depths) and enable
every feed-forward connection the mapping allows — including all skip planes
(a node at depth d receives from every node at depth < d). Same 14 hidden
units at the same depths; only the wiring is made maximal. Fresh weights,
standard backprop (Adadelta, the paper/PropNEAT optimiser family), no
evolution, no sparsification. Train to matched held-out performance
(test AUC 0.905 +/- 0.005), then measure effective fan-in per hidden unit.

PropNEAT layer mapping for genome 4bc8fa07 (verified):
  depths 0..6, layer sizes [54, 8, 1, 1, 1, 3, 1]
  (54 connected inputs; 14 hidden at depths 1-5; output at depth 6)
NOTE: the mapping's longest input->output path is 6 hops (ExplaNEAT.depth()=7
layers), NOT "max active path length 4" as the paper text states. The 14
hidden-unit count reconciles; the depth wording does not. Flagged, not fatal.

Run: uv run python scripts/aaai27/run_c2_densified_twin.py
"""
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from aaai27.common import ADULT, load_model

from explaneat.db import db
from explaneat.core.explaneat import ExplaNEAT
from explaneat.core.neuralneat import NeuralNeat

SEED = 42
TARGET_AUC = 0.905
TOL = 0.005
FANIN_FRAC = 0.10
EVOLVED_MAX_FANIN = 6         # STOP if any unit's effective fan-in <= this
# Optimiser: Adam lr 1e-3, L2 weight decay 1e-4, full-batch — the exact recipe
# of the previous 54-16-16-1 twin (run_c) that this run replaces, for an
# apples-to-apples comparison. NOTE: the paper's PropNEAT *evolution* trains
# with Adadelta(lr=1.5); both full-batch and mini-batch Adadelta stalled at
# ~0.87 test AUC on this bottlenecked skip architecture, so we use the twin's
# Adam recipe as before. L2 weight decay is standard training, not
# sparsification (no L1, no pruning).
MAX_EPOCHS = 2000
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 50                 # epochs without train-loss improvement (run_c recipe)


def extract_structure(session):
    """Return (col_indices, layer_sizes_by_depth, node_depth_of_hidden/output).

    col_indices: feature-matrix columns for the 54 connected inputs (depth 0).
    layer_sizes: dict depth -> count (inputs at 0, hidden at 1..5, output at 6).
    """
    genome, config, X, y, split, expl, ds = load_model(session, ADULT)
    # GATE re-check
    enabled = sum(1 for c in genome.connections.values() if c.enabled)
    pheno = ExplaNEAT(genome, config).get_phenotype_network()
    in_ids = set(int(i) for i in pheno.input_node_ids)
    connected = sorted({int(c.from_node) for c in pheno.connections if c.enabled} & in_ids)
    assert len(genome.nodes) == 26 and len(genome.connections) == 171 \
        and len(connected) == 54, "GATE: genome stats do not reconcile"

    net = NeuralNeat(genome, config)
    node_depth = {nid: d for d, L in net.node_mapping.layers.items() for nid in L["nodes"]}

    from collections import Counter
    sizes = Counter()
    sizes[0] = len(connected)                 # 54 connected inputs at depth 0
    for nid, d in node_depth.items():
        if nid > 0:
            sizes[d] += 1                     # hidden
    sizes[node_depth[0]] += 1                 # output node id 0
    # column indices: input key -k -> feature column k-1
    col_idx = [(-k) - 1 for k in connected]
    return col_idx, dict(sizes), split, ds


class DensifiedTwin(nn.Module):
    """Dense-by-depth network: node at depth d receives from ALL nodes at
    depth < d (every skip plane). layer_sizes: dict depth -> count, depth 0 =
    inputs. Output (last depth) is a single logit."""

    def __init__(self, layer_sizes):
        super().__init__()
        self.depths = sorted(layer_sizes)
        self.sizes = [layer_sizes[d] for d in self.depths]
        self.blocks = nn.ModuleDict()
        for pos, d in enumerate(self.depths[1:], start=1):
            fan_in = sum(self.sizes[:pos])            # all prior layers (skips)
            self.blocks[str(d)] = nn.Linear(fan_in, self.sizes[pos]).double()

    def forward(self, x):
        acts = [x]                                    # depth-0 activations = inputs
        for pos, d in enumerate(self.depths[1:], start=1):
            prior = torch.cat(acts, dim=1)
            z = self.blocks[str(d)](prior)
            is_output = (pos == len(self.depths) - 1)
            acts.append(z if is_output else torch.relu(z))
        return acts[-1].squeeze(1)                     # logit

    def dense_edge_count(self):
        return sum(b.weight.numel() for b in self.blocks.values())

    def hidden_fanin(self, frac=FANIN_FRAC):
        """List of (depth, unit_index, n_incoming, effective_fanin) for every
        hidden unit (excludes the output layer)."""
        out = []
        for pos, d in enumerate(self.depths[1:], start=1):
            if pos == len(self.depths) - 1:
                continue                               # skip output layer
            W = self.blocks[str(d)].weight.detach().abs().numpy()  # (n_units, fan_in)
            for u in range(W.shape[0]):
                row = W[u]
                thr = frac * row.max() if row.max() > 0 else 0.0
                eff = int((row >= thr).sum()) if thr > 0 else 0
                out.append((d, u, row.shape[0], eff))
        return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with db.session_scope() as s:
        col_idx, sizes, split, ds = extract_structure(s)
        # load_model already returns z-normalised X (the split's StandardScaler);
        # do NOT re-scale.
        genome, config, X, y, _, _, _ = load_model(s, ADULT)
        tr, te = split.train_indices, split.test_indices

    print(f"layer sizes by depth: {sizes}")
    Xtr = X[np.ix_(tr, col_idx)]
    Xte = X[np.ix_(te, col_idx)]
    ytr, yte = y[tr], y[te]

    model = DensifiedTwin(sizes)
    n_edges = model.dense_edge_count()
    print(f"densified edge count: {n_edges} (evolved enabled: 92)")

    xt = torch.tensor(Xtr, dtype=torch.float64)
    yt = torch.tensor(ytr, dtype=torch.float64)
    xv = torch.tensor(Xte, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossfn = nn.BCEWithLogitsLoss()

    # Full-batch Adam to train-loss plateau (run_c recipe). Among all epochs
    # whose held-out AUC is in the matched band, keep the checkpoint closest to
    # TARGET_AUC — a properly-trained model at matched performance, not the
    # first band-crossing on the way up.
    import copy
    best_loss, stale = np.inf, 0
    best_gap, matched = np.inf, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            auc_te = roc_auc_score(yte, model(xv).numpy())
        gap = abs(auc_te - TARGET_AUC)
        if gap <= TOL and gap < best_gap:
            with torch.no_grad():
                auc_tr = roc_auc_score(ytr, model(xt).numpy())
            best_gap = gap
            matched = {"epoch": epoch, "train": auc_tr, "test": auc_te,
                       "state": copy.deepcopy(model.state_dict())}
        cur = float(loss)
        if cur < best_loss - 1e-6:
            best_loss, stale = cur, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"  train-loss plateau at epoch {epoch}")
            break

    if matched is None:
        with torch.no_grad():
            final_te = roc_auc_score(yte, model(xv).numpy())
        print(f"STOP: matched AUC unreachable within +/-{TOL} of {TARGET_AUC} "
              f"(best test AUC never entered band; final {final_te:.4f}). Halting.")
        matched_epoch = None
    else:
        model.load_state_dict(matched["state"])   # restore matched checkpoint
        matched_epoch = matched["epoch"]
        print(f"  matched checkpoint: epoch {matched_epoch}, "
              f"train AUC {matched['train']:.4f}, test AUC {matched['test']:.4f}")

    with torch.no_grad():
        final_tr = roc_auc_score(ytr, model(xt).numpy())
        final_te = roc_auc_score(yte, model(xv).numpy())
    in_band = abs(final_te - TARGET_AUC) <= TOL
    print(f"final: train AUC {final_tr:.4f}, test AUC {final_te:.4f}, "
          f"matched={in_band} (band [{TARGET_AUC-TOL}, {TARGET_AUC+TOL}])")
    if not in_band:
        print(f"STOP: matched AUC unreachable within +/-{TOL} of {TARGET_AUC} "
              f"(final test AUC {final_te:.4f}). Reporting and halting.")

    # ---- fan-in analysis ----
    fanin = model.hidden_fanin()
    first_layer = [f for f in fanin if f[0] == 1]           # depth-1 units
    fl_eff = np.array([f[3] for f in first_layer])
    fl_n = first_layer[0][2] if first_layer else 0
    print()
    print(f"=== first-layer (depth-1) effective fan-in, {fl_n} inputs, "
          f"threshold |w|>={FANIN_FRAC:.0%} of unit max ===")
    print(f"  units: {len(first_layer)}  per-unit: {sorted(fl_eff.tolist())}")
    print(f"  mean={fl_eff.mean():.1f} median={np.median(fl_eff):.0f} min={fl_eff.min()} "
          f"max={fl_eff.max()}  (% of inputs: mean={100*fl_eff.mean()/fl_n:.1f}%)")

    print()
    print("=== effective fan-in for EVERY hidden unit ===")
    any_le6 = False
    for d, u, n_in, eff in fanin:
        flag = "  <-- <= 6 !!" if eff <= EVOLVED_MAX_FANIN else ""
        if eff <= EVOLVED_MAX_FANIN:
            any_le6 = True
        print(f"  depth {d} unit {u}: incoming={n_in}, effective fan-in={eff}{flag}")

    all_eff = [f[3] for f in fanin]
    print()
    print(f"min effective fan-in across all {len(fanin)} hidden units: {min(all_eff)}")
    if any_le6:
        print(f"STOP CONDITION HIT: a hidden unit has effective fan-in <= "
              f"{EVOLVED_MAX_FANIN} (the evolved network's max). This would "
              f"change Section 5.5 — reporting before writing conclusions.")
    else:
        print(f"no hidden unit has effective fan-in <= {EVOLVED_MAX_FANIN}; "
              f"densified units are broadly mixed (Section 5.5 contrast holds).")

    return {"n_edges": n_edges, "matched_epoch": matched_epoch,
            "final_tr": final_tr, "final_te": final_te, "in_band": in_band,
            "fanin": fanin, "any_le6": any_le6, "fl_n": fl_n,
            "fl_stats": (float(fl_eff.mean()), float(np.median(fl_eff)),
                         int(fl_eff.min()), int(fl_eff.max()))}


if __name__ == "__main__":
    main()
