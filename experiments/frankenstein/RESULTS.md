# Frankenstein union graft on Churn: results

> **Future work, not in the thesis** (exploratory, 28 Nov 2026). No thesis claim rests on it,
> and nothing here is exported to `thesis/`.

Design: `PLAN.md`, committed in 3e665070 before any outcome run. All numbers below are read from
the CSVs in `results/`, which the commands at the end regenerate. Each stage was run twice and
all 12 CSVs were byte-identical. Every database read is READ ONLY, and no genome was stored.

## Headline

1. **The union beats a single seed and the ensemble, by about one point of AUC.**
   - The standalone Frankenstein has the 71 direct edges plus all 21 pooled parts. With a
     readout refit it reaches test AUC 0.908 [0.876, 0.937].
   - Against the best-by-train seed (s7, 0.900): +0.008 [−0.009, 0.024].
   - Against the 10-seed logit ensemble (0.899): +0.009 [−0.005, 0.023].
   - Grafting into each seed gives a median of 0.910 over the ten unions.
     - Against the best seed: +0.010, 10/10 positive, but only 2/10 CIs above 0.
     - Against the ensemble: +0.011, 10/10 positive, 3/10 CIs above 0.
   - The gain is consistent across recipients but small, and each comparison is inside its own
     CI. Gradient boosting stays ahead: 0.925 (`benchmarks.csv`) and 0.934 (`baselines.json`).
2. **The trained part weights matter, not just the added capacity.**
   - Graft against the sham of the same topology:
     - standalone: +0.020 [0.002, 0.038];
     - unions: median +0.019, 10/10, 7/10 CIs above 0.
   - The sham union barely helps its recipient (median +0.004, 7/3), while the real union adds
     +0.024 (10/0, 9/10 CIs above 0).
3. **Most of the gain needs only three parts.**
   - The greedy deduplicated Frankenstein stops after 3 of the 21 features:
     - s8 `csc_lowusage_hinge`;
     - s7 `usagePathway`;
     - s5 `csc_threshold_low_day`.
   - It reaches test AUC 0.904, with 13 hidden units and 101 connections. That is about one
     seed's size, against 184 / 798 for the ensemble.
   - The 18 other features add about 0.005 AUC on test, with about 6× the hidden units (76 vs
     13).
4. **The union mostly duplicates the shared signals, and the readout spreads weight over the
   aliases rather than picking one.**
   - Within the two large families the effective number of realisations is high: 5.3 of 9 for
     `shape:day`, 3.4 of 6 for `shape:csc`.
   - The aliases largely cancel each other: the net / gross contribution ratio is 0.35 and
     0.45.
   - 10 of the 21 refit parts have their sign flipped: corr(refit contribution, donor signal)
     < 0.

## Test AUC (`results/summary_auc.csv`, `results/union_summary.csv`, `results/comparators.csv`)

Test AUC with its 95% bootstrap CI (10,000 resamples, seed 42). Δ values are paired, with the
same resamples.

| Model | Test AUC [95% CI] | Δ vs best seed | Δ vs ensemble | Δ vs sham | Hidden / connections |
|---|---|---|---|---|---|
| Base seeds (range) | 0.874–0.904 | | | | 13–27 / 59–94 |
| Best-by-train seed (s7) | 0.900 [0.866, 0.932] | | | | 14 / 59 |
| 10-seed ensemble (logit mean) | 0.899 [0.865, 0.930] | | | | 184 / 798 (total) |
| LR (L2), benchmarks / references | 0.819 [0.782, 0.854] | | | | |
| Gradient boosting, benchmarks | 0.925 [0.890, 0.957] | | | | |
| Gradient boosting, references | 0.934 [0.902, 0.962] | | | | |
| **Standalone, refit** | **0.908 [0.876, 0.937]** | +0.008 [−0.009, 0.024] | +0.009 [−0.005, 0.023] | +0.020 [0.002, 0.038] | 76 / 275 |
| Standalone, refit + ft E500 | 0.910 [0.877, 0.939] | +0.009 [−0.006, 0.026] | +0.010 [−0.003, 0.025] | +0.017 [−0.001, 0.038] | 76 / 275 |
| Standalone, refit (fit rows) + ft ES | 0.895 [0.859, 0.927] | −0.005 [−0.026, 0.014] | −0.004 [−0.023, 0.012] | +0.014 [−0.008, 0.036] | 76 / 275 |
| Standalone sham, refit | 0.889 [0.855, 0.919] | −0.012 | −0.011 | | 76 / 275 |
| **Deduplicated (3 parts), refit** | **0.904 [0.871, 0.933]** | +0.003 [−0.008, 0.016] | +0.004 [−0.006, 0.016] | | 13 / 101 |

Unions: one per recipient seed, each with 18–20 parts from the other nine seeds. Each cell gives
the median over the ten unions and the sign count (positive / negative), with the number of CIs
above 0 in parentheses.

| Condition | Test AUC median (range) | Δ vs base | Δ vs best seed | Δ vs ensemble | Δ vs sham |
|---|---|---|---|---|---|
| refit | 0.910 (0.901–0.920) | +0.024, 10/0 (9) | +0.010, 10/0 (2) | +0.011, 10/0 (3) | +0.019, 10/0 (7) |
| refit + ft E500 | 0.911 (0.898–0.924) | +0.022, 10/0 (7) | +0.011, 9/1 (1) | +0.012, 9/1 (2) | +0.016, 10/0 (4) |
| refit (fit rows) + ft ES | 0.899 (0.888–0.914) | +0.012, 9/1 (2) | −0.002, 5/5 (1) | −0.000, 5/5 (1) | +0.012, 10/0 (4) |
| sham refit | 0.894 (0.880–0.904) | +0.004, 7/3 (1) | −0.006, 3/7 | −0.005, 3/7 | |

- Union size: median 87 hidden units and 262 connections.
- **Fine-tuning adds little:**
  - E = 500 changes the standalone by +0.001 and the union median by +0.0005.
  - Early stopping is worse. It refits on 3,200 rows and stops on a noisy 800-row validation
    set (standalone best epoch 62), the same asymmetry D2 found.
- The chosen C is 0.1 for the standalone and 0.03 to 0.3 for the unions. It is 30 for the
  standalone sham, which has no aliases to shrink.

## What the readout does with duplicated features

The data are in `results/readout_standalone.csv`, `results/families_summary.csv` and
`results/pool.csv`.

**Signal families.** Each feature's family is the non-linear block of the churn signal catalogue
that carries the largest share of its donor contribution. The families were fixed on train
before evaluation (`PLAN.md`).

| Family (standalone refit) | Parts | Effective n | Net / gross | Largest part (share of gross) |
|---|---|---|---|---|
| `shape:day` heavy day-usage hinge | 9 | 5.34 | 0.35 | s7 `usagePathway` (0.34) |
| `shape:csc` service-call threshold | 6 | 3.39 | 0.45 | s5 `csc_threshold_low_day` (0.49) |
| `int:day*eve` combined bill | 3 | 2.47 | 0.82 | s1 `totalChargeChain` (0.55) |
| `int:csc*day` calls × low day use | 2 | 1.21 | 0.82 | s5 `csc_low_day_gate` (0.90) |
| `int:day*vmail` day use without voice mail | 1 | 1.00 | 1.00 | s6 `day_charge_no_voicemail` |

Definitions:
- **Effective n** = (Σ sd)² / Σ sd², over the parts' train contribution sds: 1 means one
  realisation carries the family, n means the weight is spread evenly over n.
- **Net / gross** = sd(Σ contributions) / Σ sd(contribution): 1 means the aliases add up, near
  0 means they cancel.

The unions look the same: median effective n 4.9 (`shape:day`) and 2.9 (`shape:csc`); net /
gross 0.37 and 0.50.

**What the readout keeps:**
- It does not keep one realisation per signal.
  - In the two families every seed has (heavy day use, service calls), it spreads weight over
    about half the copies.
  - Most of that weight cancels: 10 of the 21 refit parts now count *against* their own donor
    signal.
  - Only 4 parts keep at least 0.75 of their donor's contribution sd.
- Examples:
  - s7 `usagePathway` is scaled up to 1.36× its donor size, and s5 `day_minutes_hinge` is
    flipped (corr −1.00, 0.76×). Together they form a difference of two day hinges with
    different knots.
  - Across the 10 unions, 87 of 189 grafted parts flip.
- The two interaction families with few copies behave as one realisation plus a passenger.
- **The greedy deduplication shows the redundancy directly** (`results/dedup_steps.csv`):
  - The first part (s8 `csc_lowusage_hinge`) lifts validation AUC from 0.814 (LR) to 0.885.
  - The next two add 0.008 and 0.009.
  - The fourth would add 0.0004.
  - By family the kept three are two `shape:day` parts (s8 `csc_lowusage_hinge`, despite its
    name mostly a low-usage hinge, and s7 `usagePathway`) and one `shape:csc` part.

## What this says about explainability

**Is the union still readable?**
- **Formally it is explained.** The union's explanation is the donor explanations plus one
  linear readout: every part keeps its hypothesis, and the readout is a weighted sum.
- **In practice the readout is no longer readable.**
  - About half the parts enter with the opposite sign to their own explanation.
  - Within a family they cancel to a third of their gross size.
  - The model's heavy-day-usage effect is now a difference of several hinges with different
    knots. Each hinge's hypothesis is true of its donor network and misleading as a reading of
    its role here.
- **What it buys:** about one AUC point over the best seed and the ensemble, which neither the
  CIs nor gradient boosting say is much.
- **What the deduplicated model shows:** after deduplication, three parts hold nearly all of the
  gain.
  - It is a network the size of one seed, with at most two realisations per family.
  - Its readout can be read part by part again: in `shape:day`, the larger part, s7
    `usagePathway`, carries 0.66 of the family's gross sd.
- **The lesson:**
  - The union is a way to *find* which explained features are worth keeping.
  - It is not a better explained model by itself.
  - Composing explanations needs a deduplication step (one realisation per signal), or the
    readout turns aliases into cancelling pairs.

## Deviations and notes

- **Added after the first standalone run, before the union runs:**
  - the per-family net / gross ratio and corr(refit contribution, donor signal);
  - `nanmedian` in the family summary (one dead sham part has zero contribution).
  - Both are descriptive and change no selection. Every stage was then rerun from scratch twice,
    and the AUC numbers are those of the first run.
- **Family assignment:**
  - A first version used the catalogue *theme* with the largest summed share. Every part has a
    large linear day or service-call share, so that put 15/21 in "day usage".
  - It was replaced by the dominant non-linear block before `PLAN.md` was committed.
- **How the pool is built:**
  - It is rebuilt by `experiments/churn/run_transplant.candidates`, as in C2, and checked
    against C2's graftable donors.
  - That function reads the churn100 record JSON through `ext_common.read_record`. The record is
    used only for an assertion and a record-gain column; the pool and its hypotheses come from
    the Explanation operations in the database.

## Reproduce

```bash
export DATABASE_URL=postgresql://localhost/explaneat_dev PYTHONHASHSEED=0   # read only
uv run python experiments/frankenstein/frankenstein.py pool                   # results/pool.csv (train only)
uv run python experiments/frankenstein/frankenstein.py run --build standalone # conditions/readout/families_standalone, dedup_steps, comparators (~1.5 min)
uv run python experiments/frankenstein/frankenstein.py run --build union      # conditions/readout/families_union (~10 min)
uv run python experiments/frankenstein/frankenstein.py summary                # summary_auc, union_summary, families_summary
```

The byte-identity check repeats the four commands with `--out-dir <dir>` and runs `cmp` on each
CSV.
