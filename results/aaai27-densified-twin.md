# AAAI-27 — densified twin of the Adult paper model (Section 5.5)

Generated 2026-07-14 on branch `analysis/aaai27-paper-results` (off `main`).
Script: `scripts/aaai27/run_c2_densified_twin.py`. Seed 42, stored split only,
paper genome not modified.

## Paper sentence (filled)

> we densified the evolved network itself **over its 54 connected inputs**
> (same 14 hidden units, same depth; **884** connections against the evolved
> 92), retrained it to matched performance (test AUC **0.905**), and **every
> hidden unit became broadly mixed — effective fan-in 38–60 inputs (first-layer
> mean 45 of 54), not one unit at or below the evolved network's maximum of 6.**

Wording note: say "over its 54 connected inputs" explicitly. "Every possible
connection" alone reads as including the 53 pruned inputs; the densification is
architecture-, input-, and performance-matched to the evolved network and
differs *only* in connectivity — densifying over all 107 would add a second
difference (access to features the evolved net discarded) and invite a
confound. Densification here = every feed-forward/skip edge among the 54
connected inputs + 14 hidden + output.

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

### ⚠ Path-length wording — needs a single metric across all three models

Verified with `scripts/aaai27/verify_path_lengths.py`. Metric stated once:
**max edges (hops) on a directed input→output path in the evolved phenotype**.
The three models' claims were **not** computed consistently:

| Model | Paper/draft claim | Actual (hops) | Note |
|---|---|---|---|
| Adult | 4 | **6** | claim matches neither hops (6) nor layers (7) |
| Heart | 3 | **3** | matches hops |
| MONK  | 3 | **2** | claim = `ExplaNEAT.depth()` layer count (= hops + 1) |

- Adult longest chain: `-63 → 3446 → 4739 → 2129 → 655 → 313 → 0` (6 hops).
- Heart: `-2 → 155 → 534 → 0` (3 hops). MONK: `-1 → 21 → 0` (2 hops).

**Recommendation:** adopt one metric — "hops on the longest active input→output
path" — and set Adult = 6, Heart = 3 (unchanged), MONK = 2. (Or, if the draft
prefers the layer-count convention MONK's "3" uses, then Adult = 7, Heart = 4,
MONK = 3 — but *don't* leave 4/3/3, which is three different conventions.)
Do not simply change Adult's "4" to "6" in isolation.

### ⚠ Adult capital-gain cascade (§5.2) — two concrete corrections

The same paragraph's structural description needs two fixes (verified against
the genome):

1. **"seven neurons" ✓** — 3446, 4739, 2129, 655, 313, 5102, 6421 all present.
2. **An extra neuron sits on the path the prose omits.** §5.2 says the cascade
   "converges on a single hub neuron, read out to the output" — implying
   hub → output. In fact `313`, `5102`, `6421` are **readout neurons fed by the
   hub 655**, each with its own edge to the output (655 also has a direct
   edge). So the longest path is hub → 313 → output, i.e. 6 hops / 5 hidden
   units (3446, 4739, 2129, 655, 313), not the cascade(3)+hub(1)=4 the "4"
   counts. This is why "max active path length 4" undercounts.
3. **Sign grouping is wrong.** The four edges into the output are
   313 **+0.63**, 655 **−0.95**, 5102 **−1.38**, 6421 **−0.98** — i.e. **one
   positive (313) vs three negative (655, 5102, 6421)**. The draft's
   "(655/313 vs 5102/6421)" groups the negative hub 655 with the positive 313;
   correct it to "313 (+) against 655, 5102, 6421 (−)".

These are the same structure the densification is built on, so the depth used
here (7-layer / 6-hop mapping) is the correct one regardless of the wording fix.

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
