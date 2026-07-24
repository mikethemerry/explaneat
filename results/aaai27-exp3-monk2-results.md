# AAAI-27 Experiment 3 candidate — monk2: results

Generated 2026-07-11 on branch `analysis/aaai27-exp3-candidates` (off `main`).
All scripts under `scripts/aaai27/`; seed 42 throughout; stored splits only.

## Verdict: STRONG candidate

A tiny sparse network learns the strongly nonlinear MONK-2 rule **exactly**
(test AUC 1.000) where logistic regression cannot (0.423), and does so with
only **2 hidden units** — small enough to explain to full coverage. This is
the cleanest possible "explained nonlinearity" demonstration.

## Ground truth (verified, scripts/aaai27/verify + ingest_monk2.py)

- PMLB monk2 (601 rows) deduplicated to the full input-space enumeration:
  **exactly 432 unique rows** (3·3·2·3·4·2). STOP condition passed.
- Ground-truth rule "positive iff EXACTLY TWO of {a1=1,…,a6=1}" matches the
  labels on **100%** of the 432 rows (142 positives). STOP condition passed.
- No structural penalty; fitness = AUC on the training split (paper pipeline).

## Ingested artifacts

| Record | ID | Notes |
|---|---|---|
| raw `monk2_dedup432` | `562fe825-6ad9-4a77-ab23-42149903a69e` | 432 rows, a1..a6 categorical, full metadata (source pinning, rule, dedup+split note) |
| prepared `monk2_onehot` | `c2340904-9db2-4830-99be-e9ed637ed400` | 17 one-hot cols, display names `a1=1`..`a6=2` |
| split | `63257a45-49c9-42a6-8b28-8e346eab2bee` | stratified 80:20 seed 42 (train 345 / test 87), StandardScaler attached (no-op on 0/1) |
| experiment | `45e998d7-dac1-4e6b-953e-3f1d32cbc59e` | "monk2 - OHE - 250-100-5", 100 generations |
| best genome | `23dde0fa-a635-4974-91aa-c05266ea0747` | for explanation |

**Source pinning:** PMLB via `pmlb` 1.0.1.post3, ingested 2026-07-10, URL
`github.com/EpistasisLab/pmlb/.../monk2`. The pmlb package does not expose the
underlying data-repo commit; recorded as such. [MIKE: pin the data commit if a
reviewer needs it.]

## Topology of the best genome (scripts/aaai27/topology_summary.py)

- nodes 3 (2 hidden ReLU + output), connection genes 27 (23 enabled)
- **connected inputs: 15 of 17** (only `a2:2`, `a4:3` pruned)
- **all 6 rule-relevant `=1` inputs connected** (a1:1…a6:1)
- hidden fan-in {1, 8}, max active path length 3
- **train AUC 1.000 (n=345), test AUC 1.000 (n=87)**

## Key result: the exact rule is learned

The network's thresholded predictions equal the ground-truth "exactly two of
six =1" rule on **all 432 configurations** (train + held-out test), i.e. it
recovers the true Boolean function and generalises perfectly to unseen inputs —
not a memorised fit (train/test are disjoint unique configurations). With 2
hidden units this is a compact "count == 2" detector over the six indicators,
squarely in annotation range.

## Baseline contrast (scripts/aaai27/run_a-style, computed on our split)

| Model | monk2 test AUC |
|---|---|
| Logistic regression (L2, CV-tuned) | 0.423 |
| Evolved sparse net (this run) | **1.000** |
| (PropNEAT-58 benchmark LR, for reference) | 0.444 |

LR is at/below chance; the nonlinearity is essential. This is the argument-
maximiser profile the task flagged.

## Training configuration (task-unspecified — my choice, flagged)

The task did not specify monk2 training hyperparameters. I used the paper's
Adult settings: **population 250, 100 generations, 5 backprop epochs, AUC
fitness, no structural penalty**. [MIKE: adjust if you want a different search
budget; ingest_monk2.py::TRAIN.]

## Known issues / provenance notes

1. **Foreign `monk2` collision (handled non-destructively).** An un-deduplicated
   `monk2` (601 rows, `attribute#1..6`, no metadata) already existed and is the
   parent of a pre-existing experiment `adcf1dc2` I did not create. I did not
   mutate/delete it; my authoritative raw is `monk2_dedup432`. [MIKE: if that
   foreign chain is stale, delete it + adcf1dc2 and rename mine to `monk2`.]
2. **Framework resume bug (discovered).** Resuming an interrupted experiment
   crashes: `DatabaseBackpropPopulation.resume_from_db` speciates before a
   reporter is attached → `AttributeError: 'NoneType' has no attribute 'info'`
   (neat/species.py). Affects the API resume route and the paper's interrupted
   experiments too. Out of scope here; worth a dedicated fix. Workaround used:
   fresh full run (the interrupted 58-gen run was discarded, not resumed).
3. Topology tool had a string-vs-int node-key bug in the rule-relevant metric
   (fixed); it briefly mis-reported 0/6 before the fix. Final numbers above are
   post-fix and independently re-verified.

## Scripts (all committed, reproducible)

- `ingest_monk2.py` — dedup+verify, ingest raw+prepared, split, launch training
- `topology_summary.py <experiment_id> [out.md]` — generic topology summary
- `resume_monk2.py` — resume helper (blocked by the framework bug above)
- `inspect_hypothyroid.py` — read-only structural inspection (hypothyroid blocked)
