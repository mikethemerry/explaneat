# Studies F and G: results

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

Design: `GRAFT_F_G_PLAN.md` (committed `1d3a403` before any run; amendments 1 and 2 before the
runs they cover). Selections (`earlygen_selection.csv`, `monk2graft_selection.csv`,
`monk2_val_split.json`) were committed in `3c6cea7` before any test evaluation.

**Conventions:**
- Test AUC deltas are paired bootstrap CIs (10,000 resamples, seed 42).
- Pooled rows give the median [bootstrap CI of the median], the sign count and a two-sided
  sign test, over cases, and per recipient (each recipient's median over its cases).
- "≈0" means the median's CI lies inside ±0.005.
- All pooled numbers are in `graft_fg_summary.csv` (`analyze_fg.py`).
- No genomes were written to the DB.

**Reproduction checks (P = 0):** with the protection code in place, P = 0 reproduces the
earlier results exactly:
- `transplant2_judged.csv`: 18 / 18 rows, |Δ test AUC| = 0;
- `rescue2.csv`: 142 / 142;
- `rescue2_es.csv`: 142 / 142.

Unit tests show the new kwargs are bit-identical to `1deebbf` when left at their defaults.

## F1: grafting the C2 pool onto earlier-generation churn recipients

**Setup:**
- 60 recipients: 10 seeds × generations 0, 5, 10, 25, 50, 99.
- 120 cases: per recipient, the max-train-inc donor and a seeded random donor.
- Files: `earlygen_es.csv` (primary, early stopping) and `earlygen_e500.csv`.

**Novelty falls as evolution proceeds** (train only, all 1,188 pairs). Median novelty by
generation:

| Generation | 0 | 5 | 10 | 25 | 50 | 99 |
|---|---|---|---|---|---|---|
| Median novelty | 0.47 | 0.45 | 0.33 | 0.21 | 0.19 | 0.10 |

- inc tracks novelty: Spearman ρ = 0.74 [0.71, 0.76].
- Median inc falls from +0.057 at generation 0 to +0.0001 at generation 99.

**Median test AUC by generation (ES):**

| Gen | nov | base | ft-only | graft + refit | graft + refit + ft | sham + refit + ft | graft + ft |
|---|---|---|---|---|---|---|---|
| 0 | 0.51 | 0.608 | 0.803 | 0.711 | 0.861 | 0.858 | 0.873 |
| 5 | 0.47 | 0.757 | 0.813 | 0.799 | 0.867 | 0.858 | 0.873 |
| 10 | 0.32 | 0.805 | 0.836 | 0.830 | 0.863 | 0.864 | 0.863 |
| 25 | 0.23 | 0.851 | 0.866 | 0.864 | 0.874 | 0.869 | 0.877 |
| 50 | 0.27 | 0.868 | 0.866 | 0.875 | 0.879 | 0.869 | 0.884 |
| 99 | 0.15 | 0.886 | 0.886 | 0.890 | 0.890 | 0.885 | 0.891 |

**Dose-response (ES, 120 cases).** Spearman ρ against novelty, with a case bootstrap CI and
a seed-cluster bootstrap CI.

| Δ | Median [CI], sign | ρ vs nov [case CI] (cluster CI) | ρ vs generation | partial ρ vs nov given generation |
|---|---|---|---|---|
| D1 graft + refit − base | +0.0145 [+0.008, +0.024], 107/12 | **+0.74 [0.62, 0.82]** (0.67, 0.81) | −0.76 | +0.42 |
| D2 graft + refit − ft-only | −0.0006 [−0.007, +0.002], 57/62 | −0.25 [−0.41, −0.06] | +0.57 | +0.29 |
| D3 graft + refit + ft − ft-only | +0.0227 [+0.010, +0.032], 107/12 | **+0.73 [0.63, 0.81]** (0.66, 0.80) | −0.74 | +0.43 |
| D4 graft + refit + ft − sham | +0.0030 [+0.002, +0.005], 84/36 | +0.17 [−0.01, +0.33] (+0.02, +0.30) | −0.03 | +0.21 |
| D5 graft + ft − ft-only | +0.0242 [+0.013, +0.035], 99/21 | +0.75 [0.63, 0.84] | −0.77 | +0.44 |

- **E500 gives the same pattern:** ρ(D1, nov) = +0.73, ρ(D3) = +0.72, ρ(D4) = +0.17
  [−0.01, +0.34].
- **D2 is more negative under E500:** median −0.009, 38/82. Fixed fine-tuning helps the plain
  recipient more than a single refit does.

**Per arm (ES):**
- maxinc donors: D1 +0.025 (60/0), D3 +0.030, D4 +0.003.
- random donors: D1 +0.006 (47/12), D3 +0.009, D4 +0.003.
- ρ(D1, nov) is +0.77 for maxinc and +0.66 for random. The unselected arm shows the same
  gradient.

**Low-novelty bin, nov < 0.1** (15 cases, 13 recipients, mostly generations 50 and 99). ES
medians:

| Δ | Median [CI] | Sign | ≈0 |
|---|---|---|---|
| D1 | +0.0004 [−0.0004, +0.0010] | 9/5, p = 0.42 | yes |
| D2 | +0.0000 [−0.017, +0.001] | 7/7 | no: CI too wide |
| D3 | +0.0010 [+0.0001, +0.0024] | 11/3, p = 0.06 | yes |
| D4 | −0.0000 [−0.0010, +0.0016] | 7/8 | yes |

E500 also gives ≈0 for D1, D4 and D5 in this bin. For D3 under E500 the CI reaches +0.0052,
just outside the margin.

**Early vs late (secondary):** the per-seed median D1 at generations 0–10, minus D1 at
generation 99, is +0.038 [+0.034, +0.041], 10/10 seeds (p = 0.002). D3 gives +0.043, 10/10.

**ES caveat:**
- At generation 99, ES fine-tuning stops at best epoch 0 for:
  - ft-only in 8/10 recipients;
  - graft + refit + ft in 18/20;
  - sham in 19/20.
- So the generation-99 fine-tuned rows are mostly the refit networks. The stopping-at-once
  asymmetry seen in B/D2/E is back for converged recipients.
- At generations 0–25, no run stops at epoch 0. Median best epochs are 500–1,200.

### Reading against the hypothesis

1. **Gain tracks novelty.**
   - D1, D3 and D5 rise with novelty (ρ ≈ 0.73–0.75; the CIs exclude 0 under both case and
     seed-cluster bootstraps).
   - About 40% of the association survives conditioning on generation (partial ρ ≈ 0.42).
   - So novelty predicts gain within a generation too, not only through "early networks are
     weak".
2. **Gain is about 0 when novelty is about 0.** In the pre-registered nov < 0.1 bin, D1, D3
   and D4 are within ±0.005 of 0, with medians of +0.0000 to +0.0010.
3. **Most of the early gain is not graft-specific.**
   - At generation 0–10 the refit graft alone (D2) is *worse* than just fine-tuning the
     recipient: −0.09 at generation 0.
   - After fine-tuning, the real graft beats a same-topology random graft by only +0.003
     (D4), with no reliable novelty gradient (ρ +0.17, CI touching 0).
   - So most of the early-generation gain from "graft + ft" is added capacity plus training.
     The donor's specific weights add a small, consistent extra (84/36 cases). That extra is
     ≈0 at low novelty.

## F2: MONK-2, converged but missing (`monk2graft_es.csv`, `monk2graft_e500.csv`)

**Setup:**
- Recipients: the 7 seeds with train AUC < 1 (s0, s2, s3, s4, s6, s8, s9).
- Donors:
  - qualifying: the s1 and s5 `few_indicators_gate` (the low-count veto, the "lower edge" of
    the exactly-two window);
  - ramps: restore gain 0 by definition, and no direct readout edge (identity-exit ramps);
  - s7: its two nonlinear-looking leaves have gain 0.
- 14 cases, all graftable.
- Test set: 87 rows.

**Median test AUC across the 14 cases:**

| | ES | E500 |
|---|---|---|
| base | 0.767 | 0.767 |
| ft-only | 0.710 | 0.709 |
| graft + refit | 0.920 | 0.920 |
| graft + refit + ft | **1.000** (12/14 cases at 1.000, the rest 0.996) | **1.000** (min 0.986) |
| sham + refit + ft | 0.993 | 0.897 |
| graft + ft (no refit) | 1.000 | 0.999 |

**Deltas (ES; E500 in brackets):**

| Δ | Median | Sign | Per recipient |
|---|---|---|---|
| D1 graft + refit − base | +0.092 [+0.095] | 14/0 | 7/7, p = 0.016 |
| D3 graft + refit + ft − ft-only | +0.287 [+0.286] | 12/0, 2 ties | 6/0 |
| D4 graft + refit + ft − sham | +0.007 [+0.099] | 7/1, 6 ties | 5/0 |

- ρ(D1, nov) = +0.90 [0.58, 1.00].
- **Ties:**
  - every tie in D4 is a case where the sham also reaches test AUC 1.0;
  - the sham also reaches test AUC 1.000 in r2 and r8, and for one donor each of r4, r6 and r9;
  - the real gate reaches it in every case.
- **D3 ties** are the two r8 cases, where ft-only already reaches 1.000.
- **Rule outcome:**
  - ft-only fixes only one wrong network (s8, test AUC 1.000). Otherwise it stays at or
    below base (median 0.71);
  - graft + refit + ft puts all 7 wrong networks at test AUC ≥ 0.996 in ES (≥ 0.986 in E500),
    and at 1.000 for 6/7 in ES (test accuracy at 0.5: median 1.0).
- **MONK-2's wrong models benefit strongly.** The missing piece is a signal they don't
  represent (nov 0.08–0.42), and the explained gate supplies it.
- **The sham sometimes also recovers the rule:** one random ReLU on the same inputs plus
  training is enough capacity in a 17-input Boolean problem. The real gate is worse than the
  sham in only 1/14 cases in ES (r6, −0.004) and 2/14 in E500 (r4, −0.014 and −0.003).
- **Caveats:** the ES validation set has 69 rows, and an 87-row test set.

## F3: does a graft speed evolution? (`evolve_graft_s{0,1,2}.csv`; descriptive, 3 seeds)

**Setup:**
- The stored generation-10 population continued in memory to generation 50, with the same seed
  per arm.
- Grafted elite: the generation-10 best + its maxinc donor, refit, replacing the worst genome.

**Results:**

| Seed | Arm | First generation reaching the original generation-50 best | Gen-50 best train / test AUC |
|---|---|---|---|
| s0 | control | 32 | 0.878 / 0.870 |
| s0 | graft | **14** | 0.901 / 0.872 |
| s0 | sham | 28 | 0.892 / 0.865 |
| s1 | control | never (by 50) | 0.884 / 0.862 |
| s1 | graft | **24** | 0.906 / 0.875 |
| s1 | sham | 27 | 0.898 / 0.874 |
| s2 | control | never | 0.888 / 0.883 |
| s2 | graft | **35** | 0.907 / 0.884 |
| s2 | sham | never | 0.885 / 0.863 |

- **The graft arm wins on every seed:**
  - it reaches the original generation-50 fitness first, 8–26 or more generations earlier than
    the control;
  - it ends with the highest train AUC on every seed;
  - its test AUC is the highest or tied highest (+0.002 to +0.013 over the control).
- **The sham also speeds things up on s0 and s1, but less.**
- **Supports the secondary hypothesis, weakly:**
  - 3 seeds;
  - one draw of evolution's randomness per arm;
  - the control is a continuation, not the original run. It reached the generation-50 level at
    32 on s0, so the restart itself changes the trajectory.

## G: a protection phase for grafts

**What protection does.** The graft's entry → graft and internal weights, plus its node
biases, are frozen for the first P epochs; the readout trains throughout. P = 25 is primary;
P = 100 is secondary and was run for contexts 1 and 2.

**Context 1, C2 judged arm (6 pairs; `protect_c2_*_P0-25-100.csv`):**
- **E500:**
  - graft P25 − P0: median −0.0001 (2/4); P100 − P0: −0.0002 (1/5);
  - graft − sham: +0.0015 at P0, +0.0014 at P25, +0.0012 at P100;
  - survival r ≈ 0.925 at every P.
- **ES:**
  - every grafted run's best epoch is ≤ 25 (4/6 at epoch 0), so protection cannot change the
    outcome;
  - all deltas are 0 to four decimals.

**Context 2, D2 alias cases (35 runnable; `protect_d2_*_P0-25-100.csv`):**

| Regime | Δ | P0 | P25 | P100 |
|---|---|---|---|---|
| ES | alias − sham | +0.0096 (31/4) | +0.0109 (31/4) | +0.0119 (32/3) |
| ES | alias − wrong | +0.0070 | +0.0079 | +0.0066 |
| ES | alias − relearn | +0.0192 | +0.0192 | +0.0192 |
| E500 | alias − sham | +0.0007 (24/11) | +0.0015 (24/11) | +0.0015 (26/9) |
| E500 | alias − wrong | −0.0008 | −0.0011 | −0.0013 |
| E500 | alias − relearn | +0.0104 | +0.0104 | +0.0101 |

- **Per-case P25 − P0 differences** of these deltas: all medians are ≈0, and no sign test
  has p < 0.3.
- **E500, P100 − P0 for alias − wrong:** −0.0007, 11/24, p = 0.04. Protecting the wrong-family
  graft for 100 epochs helps it slightly.
- **Graft survival.** The alias keeps its donor function:
  - r = 0.96 under E500 and 1.00 under ES, at every P;
  - P100 raises it by +0.001 (26/9).
- **Survival of the controls.** Wrong-family and sham features drift more: median r is 0.67
  (sham, E500) and 0.58 (wrong-family, ES) at P0. The medians across cases rise with P (sham
  E500 0.81 at P100; wrong-family ES 0.84). But the **paired** per-case changes are small:
  - sham E500, P100 − P0: +0.003 (21/14);
  - wrong-family E500, P100 − P0: +0.007 (25/11, p = 0.03);
  - ES pairs: ≈0.

  The rise in the medians comes from a few cases that move a lot.
- **Frozen sham as a random-feature layer:** protecting it changes test AUC by ≈0 (E500 P100
  −0.0002).

**Context 3, F1 recipients at generations 0, 10 and 99 (60 cases, P25):**
- graft P25 − P0 ≈0 in both regimes:
  - ES: +0.0000, 21/23;
  - E500: +0.0001, 35/25.
- graft − sham: P25 − P0 ≈0.
- **Survival (paired P25 − P0):**
  - graft under E500: +0.002 (43/17, p = 0.001); the level is about 0.86;
  - sham: +0.002 (E500) and ≈0 (ES).

**Reading.**
- Protection does not change test AUC in any context or regime. Every primary P25 − P0 median
  is within ±0.0002 and inside the ±0.005 margin.
- It very slightly increases how much of the grafted function survives training: paired
  survival r +0.0003 to +0.002 under E500.
- The explained graft's advantage after fine-tuning is the same with or without it:
  - under ES: alias − sham about +0.010;
  - under E500: about +0.001.
- **Why under ES:** the alias and C2 grafts stop at a best epoch inside the phase, so the
  phase is moot. This is the ES asymmetry again.
- **Why under E500:** the donor weights barely move in 500 epochs anyway (survival 0.93–0.96).
  Fine-tuning mostly retunes the rest of the network, so freezing the graft early changes
  little.

## Interpretation against the hypothesis

- **Does gain track novelty?** Yes:
  - Across 120 churn cases, the gain from a graft + refit (and graft + refit + ft) over the
    recipient rises with novelty (ρ ≈ 0.73), also within generation (partial ρ ≈ 0.42).
  - MONK-2 shows the same with ρ ≈ 0.9.
  - The graft-specific part (graft vs a same-topology random graft) is small on churn (+0.003)
    and has at most a weak novelty gradient.
  - So "help" is mostly: a novel signal + refit adds AUC, and capacity + training adds the
    rest.
- **Is gain about 0 at convergence?**
  - Yes: in the nov < 0.1 bin all primary deltas are within ±0.005 (medians ≤ +0.001).
  - At generation 99 the median gains are +0.002.
  - This matches C2's null result.
- **Do MONK-2's wrong models benefit?** Yes, strongly. The low-count veto from a correct seed,
  refit and fine-tuned, fixes all 7 wrong networks (test AUC 0.996–1.0), and ft-only fixes only one (s8).
  A random graft sometimes does too.
- **Does grafting capture signal faster during evolution?**
  - In 3/3 seeds the grafted population reached the original generation-50 fitness first and
    ended best.
  - The sham sped things up less.
  - Descriptive only.
- **Does protection help?** Not measurably in AUC, in any context or regime (P = 25 or 100). Its
  only detectable effect is a very small increase in how closely grafted (and wrong-family)
  features keep their initial function under fixed-epoch training.

## Deviations and incidents

- **Concurrency:**
  - The first F/G queue and the F3 queue raced and briefly took the machine to 6 heavy
    processes.
  - The modular study also rose to 4 processes on its own.
  - F3 seed 0 was killed at once. A governor (`fg_governor.sh`) then SIGSTOPped F/G processes
    whenever the modular study plus F/G exceeded 4, and resumed them as slots freed.
  - Pausing does not change results (deterministic, single-threaded).
- **Crashes and reruns:** D2 protection crashed twice and was rerun from scratch:
  1. A grafted node off the input → output path has no NeuralNeat parameters to protect. The
     fix (`cd9aa51`) skips such keys and records the count.
  2. The one non-runnable D2 case (alias graft collision, as in D2) is now reported instead of
     raising (`7dfff8a`).
- F3 seed 1 was first refused by the dirty-tree guard and relaunched.
- **Queue logs:** the queue scripts log `exit $?` after `$(date)`, so their exit codes are
  always 0. Completion was checked from the CSVs and logs.
- **F1 generation-99 recipients:** for s2 and s7 these are not the `runs_churn100.csv`
  genomes, because those seeds' best is at generation 98 (amendment 1).
- **`earlygen_selection.csv` is 65 MB:** the selection writer serialised an internal `_donor`
  column, the donor objects' repr. That column was not used in selection checks. The file was
  committed as pre-registered and already pushed (by another session's push), so it was not
  rewritten.
- F2 follows the plan, but the donors are only the two equivalent gates. s7 contributes none.
