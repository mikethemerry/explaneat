# Studies F and G: grafting onto unconverged / wrong recipients, and a protection phase

Written 2026-11-06, before any F or G run. Requested by Mike. Conventions follow C2 and D2
(`TRANSPLANT2_PLAN.md`, `RESCUE2_REINIT_PLAN.md`, `EARLY_STOPPING_PLAN.md`):
- masked fine-tuning (genome weights and biases only), Adadelta lr 1.5, full batch, BCE;
- study C's readout refit (donor readout weight(s) / identity-exit scale and the output bias,
  L2 1e-6 on the readout only);
- selections use train data only and are written to a CSV that is committed before any test
  evaluation;
- test AUC deltas are paired bootstrap intervals (10,000 resamples, seed 42, the shared
  index matrix of `experiments/10seed/compute_runs.bootstrap_indices`);
- fine-tune seed = sham seed = recipient seed.

## Hypothesis (Mike's)

- A *signal* is a target-relevant function of the inputs; a *representation* is a subgraph plus
  weights that computes it in a given model; *aliases* are different representations of the
  same signal.
- **Main:** grafting a feature helps only when the recipient has no representation of that
  signal yet. Testable form: the graft gain rises with the signal's novelty to the recipient,
  and is about 0 when novelty is about 0.
- **Secondary:** grafting captures signal faster during evolution.
- **Protection:** freezing a graft's weights for an adjustment period before letting them train
  changes how grafts behave.

## Shared definitions

- **Novelty** `nov(d, r)` (C2): 1 − R² of OLS (with intercept) of the donor's train logit
  contribution `s_d` (computed with the donor genome) on the recipient basis: its inputs plus
  the activation of every hidden node NeuralNeat computes for the recipient
  (`genome_surgery.signal_basis`, `linear_novelty`). Full train split.
- **Incremental value** `inc(d, r)` (C2): train AUC of the recipient logit plus the refit donor
  contribution, minus the recipient's train AUC (`genome_surgery.refit_added_signal`). Full
  train split.
- **Graftable:** none of the donor subgraph's connection keys exists in the recipient
  (`genome_surgery.graft_collisions`; only input → output keys can collide).
- **Fine-tuning regimes** (both run; ES is primary for F, both are co-primary for G):
  - **ES (primary):** readout refits on the fit rows only; fine-tuning on the fit rows with
    early stopping on the validation rows (`val_split.json`, patience 50, min_delta 1e-5,
    max 5000 epochs, best weights restored). Caveat, as before: the validation rows were part
    of evolution's training data, which favours stopping at epoch 0 for networks evolved on
    them. The number of runs with best epoch = 0 is reported per condition.
  - **E500 (secondary for F):** readout refits on the full train split; fixed 500 epochs on
    the full train split.
- **Conditions per graft case** (recipient r, donor d):
  1. `base`: the recipient as stored;
  2. `ftonly`: the recipient fine-tuned (one per recipient);
  3. `graft-refit`: graft + readout refit, no fine-tune (refit rows per regime);
  4. `graft-refit-ft`: 3 then fine-tuned;
  5. `sham-refit-ft`: same topology and entries, grafted weights and biases N(0, 1)
     (`genome_surgery.graft(..., sham_seed = recipient seed)`), refit, fine-tuned;
  6. `graft-ft`: graft without refit, fine-tuned (secondary).
- **Deltas** (test AUC, paired bootstrap CI per case):
  - `D1` = graft-refit − base;
  - `D2` = graft-refit − ftonly;
  - `D3` = graft-refit-ft − ftonly;
  - `D4` = graft-refit-ft − sham-refit-ft;
  - `D5` = graft-ft − ftonly (secondary).
- **Pooled statistics:** median, sign test (two-sided binomial), over cases and over
  recipients (each recipient's median over its cases), as in D2.
- **Rank correlations:** Spearman rho with a percentile 95% bootstrap CI over cases (10,000
  resamples, `np.random.RandomState(42)`); a cluster bootstrap over recipient seeds is a
  sensitivity check.
- **"About 0":** the equivalence margin is ±0.005 test AUC. A median gain is "about 0" when its
  95% bootstrap CI (10,000 resamples of the cases, seed 42) lies inside [−0.005, +0.005].

## Study F1: churn novelty gradient over generations

**Recipients.** For each churn100 seed 0–9 (experiments `churn100-s<seed>`, as
`runs_churn100.csv`), the best genome by stored train fitness in the population of generation
0, 5, 10, 25, 50 and 99. Generations are stored 0-indexed (0–99, 100 populations), so "100"
is generation 99. Ties: lowest NEAT genome key; NaN fitness ignored. 60 recipients. Check:
the generation-99 recipient should be the `runs_churn100.csv` genome (its
`generation_of_best` is 99); any mismatch is reported.

**Donors.** The C2 candidate pool: churn100 leaf annotations with all-input entries, the graft
preconditions, a readout edge and train restore gain ≥ 0.02 (`run_rescue2.load_pool`, 22
leaves; any other count stops the run). Only donors from seeds other than the recipient's.

**Selection (train only; `earlygen_selection.csv`, committed before test evaluation).** For
every recipient and every other-seed pool leaf: nov, inc, graftable. Two arms per recipient:
- **maxinc:** the graftable donor with the largest inc (ties: lowest donor seed, then name).
  Best-case estimate.
- **random:** one donor drawn uniformly from the remaining graftable donors with
  `numpy.random.default_rng([20261106, seed, generation]).choice` over the list sorted by
  (donor seed, name). Unselected estimate.

So 120 cases (60 recipients × 2 arms), unless a recipient has fewer than 2 graftable donors
(then the missing arm is reported, not replaced).

**Protected conditions (study G, context 3).** For recipients at generations 0, 10 and 99,
`graft-refit-ft` and `sham-refit-ft` are also run with a protection phase P = 25
(`-p25`), in both regimes.

**Outcomes.**
- **Primary (ES regime):**
  - Spearman rho of D1, D2, D3 and D4 against nov, over all cases (both arms), with bootstrap
    CIs; and against generation.
  - **Low-novelty bin, nov < 0.1:** median D1–D4 with bootstrap CI and sign test; "about 0"
    as defined above. If the bin has fewer than 5 cases, the bin is reported as too small and
    nov < 0.15 is reported as the secondary bin.
- **Secondary:**
  - the same in the E500 regime;
  - per arm;
  - partial Spearman of D vs nov controlling for generation (rank residuals);
  - cluster bootstrap by seed;
  - D5 (graft-ft);
  - **early vs late:** per seed, the median D1 over generations 0, 5 and 10 (both arms) minus
    D1 at generation 99 (median of its two arms); median, sign test over the 10 seeds, bootstrap
    CI over seeds;
  - train-only: Spearman(nov, inc) over every (recipient, donor) pair in the selection;
  - ES best-epoch = 0 counts per condition and generation.

## Study F2: MONK-2, converged but missing

**Recipients.** The 10seed MONK-2 runs (`experiments/10seed/runs.csv`, experiments
`10seed-monk2-s<seed>`, split `63257a45`): the seeds whose best genome has train AUC < 1.0.
From `runs.csv` these are s0, s2, s3, s4, s6, s8, s9 (7 seeds); s1, s5 and s7 learned the
rule (train AUC 1.0, test about 1.0; `results/10seed-consistency.md`, `rule_agreement` in
`explanations.csv`).

**Donors.** Leaf annotations of the correct seeds' explanations (`Explanation` rows of the
best genomes, as for churn) that meet the C2 criteria computed on MONK-2 train: all-input
entries, the graft preconditions, a readout edge, train restore gain ≥ 0.02. Linear leaves
(the count ramps) have restore gain 0 by definition, so only nonlinear gates can qualify.

**Selection (train only; `monk2graft_selection.csv`, committed before test evaluation).**
Every donor × recipient pair with nov, inc and graftable. **All graftable pairs are cases**
(the pool is small). The pair with the largest inc per recipient is flagged `maxinc`. If no
donor qualifies, or none is graftable, that is the result and no test evaluation follows.

**Validation split for ES.** The same rule as `val_split.json`, on the MONK-2 train split:
stratified 20%, `random_state = 42` (69 of 345 rows), written once to `monk2_val_split.json`.
With 69 validation rows the ES rule is noisy; that is a caveat, not a reason to change it.

**Conditions and outcomes.** As F1 (both regimes, conditions 1–6, D1–D5), over cases and per
recipient. Test has 87 rows, so per-case CIs will be wide. The novelty analysis is the same
(Spearman of D vs nov over cases, with bootstrap CI). Test accuracy at 0.5 is reported
alongside AUC.

## Study F3: faster capture during evolution (secondary; run only if budget remains)

- Seeds 0, 1, 2 of churn100. Population = the stored generation-10 genomes (250). The
  generation-10 **worst** genome is replaced by the grafted elite: the generation-10 best
  genome plus its F1 maxinc donor, graft + refit on full train.
- Continue PropNEAT in memory (`BackpropPopulation`, the experiment's stored config,
  `mask_phantom=False` as in the original runs) from generation 10 to generation 50, with
  `random.seed` / `torch.manual_seed` = seed for every arm.
- Arms: **control** (unmodified generation-10 population), **graft**, **sham** (sham graft +
  refit replacing the worst).
- Outcomes: generation at which the population's best train fitness first reaches the
  original run's generation-50 best; test AUC of the generation-50 best genome. Three seeds,
  descriptive only.
- This is not a reproduction of the original run from generation 10 (its RNG state at
  generation 10 is not stored); the control is the comparison, not the original run.

## Study G: a protection phase for grafts

**Implementation.** `genome_surgery.finetune(..., protect_conn_keys, protect_node_keys,
protect_epochs)`, additive kwargs. For epochs 1..P the protected connection weights and node
biases get zero gradient (as `frozen_*`), then train normally. Protected = the graft's added
connections that end at a grafted hidden node (entry → graft and internal edges) plus the
grafted nodes' biases (`genome_surgery.graft_protect_keys`); the readout and any input →
output edges stay trainable. Tests: defaults are bit-identical to commit 1deebbf, with fixed
epochs and with ES; protected parameters do not change during the phase while others do.

- **Fixed E500:** the phase is the first P of the 500 epochs.
- **ES:** the phase counts toward `max_epochs`. Validation BCE is tracked from epoch 0, so the
  best epoch may lie inside the phase; the patience counter cannot stop training before
  epoch P + 50 (stop at the first epoch with epoch − max(best epoch, P) ≥ 50). The number of
  runs whose best epoch is ≤ P is reported.

**P values.** P ∈ {0, 25} primary. P = 100 secondary, for contexts 1 and 2 only, if the budget
allows (decided before any test result of G is seen, from the timing of the P = 0/25 runs).

**Contexts.**
1. **C2 judged arm** (`transplant2_judged_selection.csv`, 6 pairs): ftonly, graft-refit-ft and
   sham-refit-ft at each P, both regimes. Check: P = 0, E500 must reproduce the test AUCs in
   `transplant2_judged.csv` (|Δ| < 1e-9), otherwise the run stops.
2. **D2 alias cases** (the committed `rescue2_selection.csv`, 35 runnable cases): KO + ft,
   and KO + alias / sham / wrong + refit + ft, each graft condition at each P, both regimes.
   KO + ft has no graft and is not protected. Check: P = 0 must reproduce `rescue2.csv`
   (E500) and `rescue2_es.csv` (ES) test AUCs (|Δ| < 1e-9).
3. **F1 recipients at generations 0, 10 and 99**: graft-refit-ft and sham-refit-ft at P = 25
   (P = 0 is F1's own condition), both regimes, both arms.

**Outcomes.**
- Δ test AUC protected − unprotected (P25 − P0; P100 − P0 secondary), per case with paired
  bootstrap CI; median and sign test over cases and per recipient (feature), per context, arm
  and regime.
- Whether protection changes the alias − sham and alias − relearn (KO + ft) deltas (D2): the
  per-case difference Δ(alias − sham | P25) − Δ(alias − sham | P0), median and sign test; the
  same for alias − relearn and alias − wrong; and for context 1 and 3, graft − sham.
- **Graft survival:** Pearson r between the grafted feature's train logit contribution after
  fine-tuning (its readout edges, `genome_surgery.readout_contribution` on the full train split)
  and the donor's original contribution `s_d`; P25 vs P0, median and sign test of the paired
  difference. For shams, r with the sham's own contribution at the start of fine-tuning (after
  refit).
- **The frozen sham as a random-feature layer:** during the phase a protected sham is a fixed
  random ReLU feature with a trainable readout; its P25 − P0 delta and survival are read that
  way.

## Code, outputs, budget

- Code: `experiments/churn/graft_fg.py` (shared), `run_earlygen.py` (F1 + G context 3),
  `run_monk2graft.py` (F2), `run_protect.py` (G contexts 1 and 2), `analyze_fg.py` (pooled
  statistics from the CSVs), tests in `tests/test_experiments/`.
- Outputs (new files only): `earlygen_selection.csv`, `earlygen_es.csv`, `earlygen_e500.csv`,
  `monk2graft_selection.csv`, `monk2graft_es.csv`, `monk2graft_e500.csv`, `monk2_val_split.json`,
  `protect_c2_{es,e500}.csv`, `protect_d2_{es,e500}.csv`, `graft_fg_summary.csv`, and the
  write-up `GRAFT_F_G_RESULTS.md`. Existing results files are not modified.
- **No DB writes.** Unlike C2/D2, derived genomes are not stored as experiments (about 1,500
  fine-tunes); every result is reproducible from the commit, the committed selections and the
  read-only DB.
- **Budget.** At most 4 heavy processes on the machine including the modular study, 1 torch
  thread each, `PYTHONHASHSEED=0`. Estimated fine-tunes: F1 ≈ 420 per regime + 120 per regime
  for G context 3; F2 ≈ 7 + 4 × (number of pairs) per regime on a 345-row problem; G context 1
  ≈ 30 per regime; G context 2 ≈ 229 per regime. Each runner prints a runtime estimate after its
  first recipient / case. Order of priority if time runs short: F1 ES, G contexts 1–3 (both
  regimes), F2, F1 E500, P = 100, F3. Anything not run is reported as not run.

## Amendment 1 (2026-11-06, after the F1 dry run, before any test evaluation of F or G)

- **Timing:** an ES fine-tune of an F1 recipient takes about 2 s, so the budget is not binding.
  P = 100 is therefore run for G contexts 1 and 2 (`--P 0,25,100`, outputs
  `protect_<context>_<regime>_P0-25-100.csv`). P = 0 and 25 remain primary.
- **Selection facts (train only):** the generation-99 best genome differs from the
  `runs_churn100.csv` genome for s2 and s7, whose overall best is at generation 98; F1 uses
  generation 99 as pre-registered. F2: the qualifying donors are s1 and s5
  `few_indicators_gate`; s7 has no qualifying leaf (its gate is not a graftable all-input
  leaf with gain ≥ 0.02), so F2 has 14 pairs (7 recipients × 2 donors), all graftable.

## Amendment 2 (2026-11-06, before any F3 run): F3 details

F3 is run, since the budget allows (`run_evolve_graft.py`, one process per seed, seeds 0–2,
outputs `evolve_graft_s<seed>.csv`). Details fixed now:
- the stored generation-10 genomes (already trained and evaluated in the original run) are
  the initial population; the continuation backpropagates and evaluates them again as its
  first generation (labelled 10) and runs 41 generations (labels 10–50);
- the grafted elite's nodes are renumbered above the population's largest node id; the
  genome config's node indexer starts 50 above that, and the reproduction genome indexer
  above the largest key, identically in every arm; the inserted genome takes the worst
  genome's key;
- species are rebuilt by speciating the generation-10 population (stagnation history is not
  restored);
- outcome reference: the original run's best stored fitness in generation 50;
- the final genome is the best (by train fitness) of the generation-50 population; its test
  AUC is read once.
