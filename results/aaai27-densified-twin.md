# AAAI-27 — densified twin of the Adult paper model (Section 5.5)

Generated 2026-07-14 on branch `analysis/aaai27-paper-results` (off `main`).
Script: `scripts/aaai27/run_c2_densified_twin.py`. Seed 42, stored split only,
paper genome not modified.

## Paper sentence (filled)

> we densified the evolved network itself (same 14 hidden units, same depth;
> **884** connections against the evolved 92), retrained it to matched
> performance (test AUC **0.905**), and **every hidden unit became broadly
> mixed — effective fan-in 38–60 inputs (first-layer mean 45 of 54), not one
> unit at or below the evolved network's maximum of 6.**

## STOP conditions — all checked, none triggered

- **Genome reconciles:** 4bc8fa07 in "No pressure" — 26 nodes / 171 connection
  genes / 92 enabled / 54 connected inputs / 14 hidden. ✓
- **Matched AUC reached:** test AUC 0.9050 ∈ [0.900, 0.910]. ✓
- **No hidden unit ≤ 6 effective fan-in:** min across all 14 units = 38. ✓

## Structure densified (the evolved phenotype's PropNEAT layer mapping)

Node depths from `NeuralNeat.node_mapping` (longest-path layering):

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| size | 54 (inputs) | 8 | 1 | 1 | 1 | 3 | 1 (output) |
| role | input | hidden | hidden | hidden | hidden | hidden | output |

- 14 hidden units at depths 1–5; output at depth 6; 54 connected inputs at depth 0.
- **Densified wiring:** every feed-forward edge the mapping allows — a node at
  depth *d* receives from **every** node at depth < *d*, including all skip
  planes (e.g. the output reads directly from all 54 inputs and every hidden
  unit). Edge count = Σ_{i<j} |L_i|·|L_j| = **884** (vs evolved enabled 92).
- Fresh weights; the paper genome and its 54-input set are otherwise unchanged.

### ⚠ Depth wording discrepancy (flag for the paper, not a blocker)

The task/paper describe the mapping as "max active path length 4". The actual
PropNEAT mapping has **7 layers (depths 0–6)**; `ExplaNEAT.depth()` returns 7,
and the longest input→output path is **6 hops** (the capital-gain cascade
`input → 3446 → 4739 → 2129 → 655 → readout → output`). The 14-hidden-unit
count reconciles exactly; the "4" does not — it appears to count hidden units
on the capital-gain cascade (a depth-3 cascade + hub = 4) rather than the full
input→output hop count. **Recommend reconciling the Section 5.2/5.5 wording**
(the densification here uses the true 7-layer/depth-6 mapping).

## Training (matched performance)

- Optimiser: **Adam, lr 1e-3, L2 weight decay 1e-4, full-batch** — the exact
  recipe of the previous 54-16-16-1 twin (`run_c_dense_twin.py`) this replaces,
  for an apples-to-apples comparison. L2 decay is standard training, not
  sparsification (no L1, no pruning). Fresh init, seed 42.
- Preprocessing: the split's stored StandardScaler (identical to the paper
  pipeline).
- Selection: trained to train-loss plateau (patience 50); among all epochs
  whose held-out AUC was in the ±0.005 band, kept the checkpoint closest to
  0.905 (a converged matched model, not a first-crossing artifact).
- **Matched checkpoint: epoch 497 — train AUC 0.9099, test AUC 0.9050.**

Baselines on this split (correct single scaling), for context:
LR(54 connected) 0.904, LR(107 all) 0.905, evolved sparse net 0.9055.
The task is nearly linear over these features, so the densified net matches
easily; the point of the exercise is the *wiring*, not the accuracy gap.

## Effective fan-in (|w| ≥ 10 % of the unit's max |w|)

### First-layer (depth-1) units — the same statistic as the previous twin

- 8 units, each over the 54 inputs.
- per-unit: [38, 40, 41, 46, 47, 48, 48, 50]
- **mean 44.8, median 46, min 38, max 50** — **82.9 %** of inputs per unit on
  average.

(For comparison, the previous 54-16-16-1 twin reported first-layer mean 31.)

### Every hidden unit

| depth | unit | incoming | effective fan-in |
|---|---|---|---|
| 1 | 0 | 54 | 41 |
| 1 | 1 | 54 | 47 |
| 1 | 2 | 54 | 48 |
| 1 | 3 | 54 | 46 |
| 1 | 4 | 54 | 38 |
| 1 | 5 | 54 | 48 |
| 1 | 6 | 54 | 50 |
| 1 | 7 | 54 | 40 |
| 2 | 0 | 62 | 49 |
| 3 | 0 | 63 | 56 |
| 4 | 0 | 64 | 60 |
| 5 | 0 | 65 | 57 |
| 5 | 1 | 65 | 52 |
| 5 | 2 | 65 | 42 |

- **Min effective fan-in across all 14 units = 38; max = 60.**
- **No unit is at or below 6** (the evolved network's maximum fan-in). Densified
  units are broadly mixed and individually non-nameable — the structural
  contrast the evolved network's 2–6-input units provide survives densification
  of the *same units at the same depths*.

## Methodological note (bug found and fixed mid-analysis)

Initial runs stalled at ~0.88 test AUC across four optimiser configurations. Root
cause was a **double-scaling bug** in this script: `common.load_model` already
returns z-normalised X, and the script re-applied the scaler. Fixed (use the
returned X directly). The transient "matched AUC unreachable" and a bogus
"linear ceiling 0.88" were entirely artifacts of that bug; all numbers above
are post-fix, with baselines re-verified (evolved net back to its known 0.9055).
