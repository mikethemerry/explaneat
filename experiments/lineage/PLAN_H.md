# Study H: capacity thresholds along lineages (pre-registration)

This file was committed before any lineage analysis. Its status at commit:
- **Running:** the replays for parent capture. Only a 3-generation smoke verification had been
  run (Heart s0, 750/750 genomes identical).
- **Not looked at:** no lineage fitness curve, Δ distribution, jump, or emergence curve.

Amendments will be dated and appended at the end, never edited in place.

## Hypothesis (Mike)

Performance jumps happen when a lineage first has the minimal set of nodes and connections
needed to realise a signal. Sometimes that set is a single node. The jump marks the boundary
of that realisation, and shows where the improvement comes from.

Terminology:
- **signal:** a function of the inputs that the network uses. In practice, a final leaf
  annotation's contribution.
- **realisation:** a subgraph and weights that compute it.
- **alias:** a different realisation of the same signal.

These follow `../churn/EXTENSIONS_RESULTS.md`, `../churn/GRAFT_F_G_RESULTS.md` and
`../xor/RESULTS.md`.

Background:
- Graft gain tracks novelty and is ≈0 at convergence (F1).
- XOR's minimal realisations are enumerated in `../xor/RESULTS.md`.

## Why structure and weights are separable here

In the churn, heart and XOR configs, NEAT weight and bias mutation are off
(`weight_mutate_rate = 0`, `bias_mutate_rate = 0`). Weights change through only two routes:
- the 5 epochs of backprop per generation, which is continuous;
- crossover's random choice between the two parents' values for a *matching* gene, which is
  discrete but adds no capacity.

Every change of structure is a discrete mutation event: node or connection added or
deleted, or an enabled flag flipped (by mutation or by crossover's choice between parents).

**MONK-2 is different:** `weight_mutate_rate = 0.8` and `bias_mutate_rate = 0.7`. Weights also
change by discrete mutation there, so MONK-2 is only analysed descriptively (Test 5).

## Data

| Set | Runs | Lineage end (pre-registered) |
|---|---|---|
| churn100 | 10 seeds, all explained (`../10seed/explanations/churn100-s*.json`) | the explained genome = the run's best genome |
| churn200 | 10 seeds, all explained | the explained genome |
| heart100 | 10 seeds, all explained | the explained genome |
| MONK-2 | the 3 seeds that learned the rule: s1, s5, s7 (train AUC 1.0, test AUC ≥ 0.998) | the explained / best genome |
| XOR | 30 seeds, in memory, full ancestry capture (`run_xor_lineage.py`, same config, seeds 0–29 and stopping rule as `../xor/run_xor.py`; check: every seed's solved generation equals `xor_results.json`) | the first genome that classifies all four rows (the solving genome) |

Parents:
- **Recorded runs:** the replay sidecars `parents/<experiment>.csv.gz`. These are written
  only after a replay matched every stored genome bit for bit (`replay_verification.csv`).
- **XOR:** `BackpropPopulation.parent_log`.

Every explained genome above is its run's best genome (checked against `runs_*.csv`).

**Lineage (pre-registered):** the final genome's chain in **`fitter_parent`** mode
(`explaneat.analysis.lineage.get_lineage`).
- **Crossover child:** follow the parent with higher train fitness, which is the parent NEAT
  takes disjoint and excess genes from.
- **Elite copy:** follow the same key one generation back.
- **Sensitivity:** `parent1` mode.

**Unit of analysis.** A lineage step is one generation, g − 1 → g. Lineages are independent
seeds, so **the seed (lineage) is the unit** for every CI and permutation.
- **churn100 and churn200 are not independent:** each churn200 run is a continuation of the
  same seed's churn100 run. They are reported separately, with churn200 as a longer-horizon
  replication, and are never pooled.
- **heart (40 generations)** is a prefix of heart100. It is not used.

## Quantities

For lineage genome a_g (fitness measured after that generation's backprop, exactly as
stored):
- **Δ_g = f(a_g) − f(a_{g−1})** with f = train fitness, which is train AUC in all recorded
  runs (`auc_fitness`).
- **Test AUC** is computed for every lineage genome (NeuralNeat, the split's stored scaler)
  and reported alongside. It is never used to define jumps.
- **S_g (structural step):** 1 iff `structure_signature` differs between a_{g−1} and a_g.
  The signature covers node keys and (connection key, enabled), and excludes weights.
- **Step type:**
  - *elite* (same key: backprop only, structure identical by construction);
  - *child-structural* (S = 1);
  - *child-weights* (new key, S = 0: crossover weight mixing plus backprop).

## Jump definition (primary, pre-registered)

**Primary:** a jump is a step with Δ_g ≥ τ_D. τ_D is the **95th percentile of all step Δ**
(all step types, negatives and zeros included), pooled over the 10 lineages of data set D
and computed separately for churn100, churn200 and heart100.

Sensitivity, all reported:
- τ at the 90th and 99th percentiles;
- absolute thresholds Δ ≥ 0.005, 0.01 and 0.02 (train AUC);
- a "beyond backprop" threshold: the 99th percentile of Δ over **elite** steps only, which is
  what backprop alone does in one generation;
- excluding generations 1–10, where everything moves fast.

## Tests

### Test 1: jumps coincide with structural change

- **Statistic:** RD = P(jump | S = 1) − P(jump | S = 0), pooled over the steps of data set D.
  - Also reported: the risk ratio, mean Δ by step type, and P(S = 1 | jump).
- **Null:** within each lineage, permute S over that lineage's steps. 10,000 permutations,
  `numpy.random.default_rng(42)`. One-sided p for RD > 0, with the two-sided p also given.
- **Secondary null (generation confound):** permute S within blocks of 10 consecutive
  generations of the same lineage.
- **CI:** bootstrap over lineages (resample the 10 seeds with replacement, 10,000 resamples,
  seed 42), percentile 95%.
- **Supported** if RD > 0 with permutation p < 0.05 in churn100. churn200 and heart100 are
  replications, reported with the same criterion.

### Test 2: jump genes are explained

**Genes:** the genes added at a lineage step, i.e. node keys and connection keys present in
a_g but not in a_{g−1}, kept separately by type (node, connection).
- **Jump genes:** added at a jump step.
- **Comparison genes:** added at non-jump structural steps.

**Outcome:** the gene lies inside a **leaf** annotation of the final genome's explanation.
This uses `map_genes_to_annotations`, which maps split and identity ids back to their base
genes. Two versions, both pre-registered:
- **(a) primary:** all added genes; a gene absent from the final genome counts as "not in a
  leaf";
- **(b) conditional on survival:** only genes present in the final genome.

(b) is expected to sit near ceiling for nodes, because coverage is full and every surviving
hidden node lies in some leaf. It is reported for transparency.

**Statistic:** difference in proportions, jump minus non-jump.
- **Type-stratified:** it is computed within node genes and within connection genes, and
  pooled as the Mantel–Haenszel-style weighted mean difference.
- **Permutation null:** permute the jump label among added genes within lineage × type.
  10,000 permutations, seed 42.
- **CI:** bootstrap over lineages.

### Test 3: emergence is a step

**Signal.** For each final leaf annotation L, s_L(x) is the leaf's contribution to the
output logit on the train rows. It is computed with `genome_surgery.readout_contribution`
from `annotation_to_genome_subgraph`, i.e. readout weights × exit activations, plus the
direct input-output edges for an identity exit.
- **Fallback:** for leaves whose entries are not all inputs, or which have no readout edge to
  the output, s_L is the activation of the leaf's exit node(s). Multi-exit leaves take the
  sum of R² numerators over the sum of denominators.
- **Exclusions:** leaves with a constant s_L are excluded, and counted.

**Decodability along the lineage:**
- d_L(g) = R² of s_L regressed (OLS with intercept) on the basis B(a_g).
- B(a_g) is a_g's inputs plus every hidden activation, `genome_surgery.signal_basis`.
- So d_L(g) = 1 − novelty, exactly as in C2 and F1 (`genome_surgery.linear_novelty`).

**Step criterion (pre-registered).** Let rise R_L = d_L(final) − min_{g ≤ final} d_L(g).
- **Eligible:** a leaf "needs to emerge" if R_L ≥ 0.10. Others, such as linear leaves already
  decodable from the inputs, are reported but not tested.
- **Step-like:** an eligible leaf is step-like if its largest single-step rise
  max_g [d_L(g) − d_L(g−1)] is **≥ 50% of R_L**.
- **Report:** the fraction step-like, with a seed-cluster bootstrap CI.
- **Sensitivity:** 33% and 67% cut-offs, and an R_L cut-off of 0.05 and 0.2.

**Alignment with the realisation's completion:**
- Let G_L be the leaf's genes in the final genome: node genes and enabled connection genes,
  as mapped back from the model-state subgraph.
- Let g_c be the earliest generation from which every gene of G_L is present (connections
  enabled) in every lineage genome up to the final one.
- Let g* be the generation of the largest single-step rise.
- **Primary:** P(g* = g_c), against the chance rate Σ_L 1/T_L, where T_L is the number of
  steps with a positive rise. A permutation test draws g* uniformly from the positive-rise
  steps (10,000 draws, seed 42).
- **Secondary:** |g* − g_c| ≤ 2. Also: whether step g* was structural, and whether genes of
  G_L were added at g*.
- **Size of the subgraph present at the step:** the number of G_L's node and connection
  genes present in a_{g*}, and the full |G_L|.
  - This is the leaf's final subgraph intersected with the ancestor, not a pruned minimum.

### Test 4: XOR

- **Solve point:** the lineage's last step is the jump to solved, because the run stops at
  the first generation whose best genome classifies all four rows.
- **Feasible units:** for every lineage genome, each phenotype hidden unit has an active set
  P (rows where its pre-activation is > 0) and, if it has a direct edge to the output, the
  sign u of that weight.
  - A unit is *nonlinear* if P has 1–3 rows.
  - It is *feasible* if (P, u) is a feasible ReLU realisation pair in Part 1's LP (any u if
    there is no direct output edge).
- **g_F:** the first lineage generation with a feasible nonlinear unit.
- **Prediction:** g_F = g_solve. The jump to solved coincides with the first feasible
  nonlinear unit.
- **Report:**
  - the distribution of g_solve − g_F;
  - the fraction with g_F = g_solve, with a binomial 95% CI;
  - whether the solving step was structural;
  - whether the unit that is feasible at g_solve was added at that step or was present
    earlier with a different active set (weights moved it into the class).

### Test 5: MONK-2, seeds 1, 5 and 7 (descriptive)

- **g_learn:** the first lineage generation with train AUC ≥ 0.99.
- **Report at the step g_learn − 1 → g_learn:**
  - Δ train and Δ test AUC;
  - the gene diff: added and removed genes, enabled flips, weight and bias mutations;
  - which annotations of the final explanation contain the added genes.
- **Which gene caused it:** in a_{g_learn}, revert each changed gene alone, with no
  retraining, and record the train AUC lost:
  - an added connection is removed;
  - an added node is removed with its edges;
  - a flipped enabled flag is restored;
  - a mutated weight or bias takes the parent's value.

  The gene whose reversion loses the most is reported, with the size of the loss.
- **Caveat:** MONK-2 mutates weights, so "weights only" is not continuous there.

## Controls and caveats (stated in advance)

- **Backprop is continuous.** The elite-step Δ distribution is the gradual baseline, and the
  "beyond backprop" threshold separates step changes from it.
- **Elitism.** Many lineage steps are elite copies. They are kept as S = 0 steps, because they
  are the cleanest weights-only steps, and RD is also reported with elite steps excluded.
- **Selection.** Selection is on train fitness, so jumps are defined on train. Test AUC Δ at
  jumps is reported alongside.
- **Survivorship.** The lineage is the winner's chain, and the final explanation is the
  winner's. Genes added in non-final lineages are not studied.
- **The generation confound.** Early generations have both large Δ and many structural
  mutations. This is handled by the block permutation and by excluding generations 1–10.

## Statistics

- Bootstrap: 10,000 resamples, seed 42, percentile 95% CIs, resampling lineages (seeds).
- Permutations: 10,000, `numpy.random.default_rng(42)`.
- Unit of analysis: the lineage (seed). No pooling across data sets.

## Outputs

- **CSVs:**
  - `h_steps.csv`: every lineage step with Δ, step type, S, jump flags and test AUC;
  - `h_test1.csv`, `h_test2.csv`;
  - `h_emergence.csv`: one row per leaf × generation;
  - `h_leaves.csv`: one row per leaf, with the step criterion, g*, g_c and sizes;
  - `h_xor.csv`, `h_monk2.csv`.
- **Figures (PDF and PNG):**
  - lineage fitness curves per data set, with jumps and structural events marked;
  - emergence curves per leaf.
- **Write-up:** `RESULTS_H.md`, an honest write-up including nulls, and a pointer section
  in `../churn/EXTENSIONS_RESULTS.md`.
