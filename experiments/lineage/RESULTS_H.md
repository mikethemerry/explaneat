# Study H: capacity thresholds along lineages: results

The design is in `PLAN_H.md`, committed `b8682a2` before any lineage analysis. No amendments.

Scripts:
- `h_build.py`: per-lineage caches;
- `h_analyze.py`: Tests 1–3, CSVs and figures;
- `h_monk2.py`: Test 5;
- `run_xor_lineage.py`: Test 4;
- `h_explore.py`: exploratory, not pre-registered.

Conventions:
- **Bootstrap:** 10,000 resamples over lineages (seeds), seed 42, percentile 95% CIs.
- **Permutations:** 10,000, `default_rng(42)`, within lineage.
- **Unit:** the lineage (seed). churn200 is a continuation of churn100, so it is a replication,
  never pooled.

## Data and verification

Every lineage comes from a parent map verified bit for bit (`replay_verification.csv`):
- every stored genome of every generation;
- 50 experiments, 1,145,000 genomes;
- matched on key, fitness and md5 of the serialised genome.

Every lineage genome's recomputed train AUC equals its stored fitness exactly; the maximum
absolute difference is 0.

| Set | Lineages | Steps | Elite steps | Child steps (all structural*) | τ95 (train AUC Δ) |
|---|---|---|---|---|---|
| churn100 | 10 | 988 | 303 | 685 | 0.0177 |
| churn200 | 10 | 1,956 | 871 | 1,085 | 0.0093 |
| heart100 | 10 | 974 | 313 | 651 (+10 weights-only) | 0.0047 |

\* **Every child step is structural, which the plan did not anticipate.** In churn, *every*
non-elite lineage step changes structure (node or connection added or removed, or an
enabled flag flipped): with `conn_add_prob` 0.68 and `node_add_prob` 0.7, a child that
differs from its fitter parent by weight mixing alone almost never survives onto the
winner's line. Heart has 10 such steps.

So "weights only" (S = 0) is, in practice, "elite copy: backprop only". **Test 1 therefore
compares new children with re-trained elites.** It cannot separate "structure changed" from
"a new genome was selected".

## Test 1: jumps coincide with structural change. Formally supported; substantively weak

Primary analysis: τ95, `fitter_parent` lineage (`h_test1.csv`).

| Set | P(jump \| S=1) | P(jump \| S=0) | RD [95% CI] | perm p (1-sided) | block-10 perm p | P(S=1 \| jump) |
|---|---|---|---|---|---|---|
| churn100 | 0.073 | 0.000 | **+0.073 [0.066, 0.082]** | 0.0001 | 0.026 | 50/50 |
| churn200 | 0.090 | 0.000 | +0.090 [0.076, 0.108] | 0.0001 | 0.0001 | 98/98 |
| heart100 | 0.074 | 0.003 | +0.071 [0.050, 0.092] | 0.0001 | 0.0008 | 48/49 |

The pre-registered criterion (RD > 0, p < 0.05 in churn100) is met and replicates. The
`parent1` lineage gives the same result: churn100 RD +0.065, p = 0.0001.

**But the jumps are almost all in the initial climb.** For churn100 and heart100, the τ95
jumps sit at generations 1–10. Excluding those generations, as pre-registered:

| Set | Jumps (τ95) after generation 10 | RD | p |
|---|---|---|---|
| churn100 | 2 of 50 | +0.003 | 0.43 |
| heart100 | 3 of 49 | +0.005 | 0.26 |
| churn200 | 27 of 98 | +0.027 | 0.0001 |

churn200's second climb after generation 100 is visible in the figures.

Sensitivity:
- The absolute thresholds and the "beyond backprop" threshold (99th percentile of elite Δ,
  0.0013–0.0034) all give RD > 0, p ≤ 0.002.
- Δ ≥ 0.02 after generation 10 leaves 0–1 jumps per set.

**Elite steps almost never jump.** Backprop alone moves a lineage genome by a median Δ of
+0.0001 to +0.0003 per generation (means +0.0002 to +0.0004).

**Exploratory E1: where the gain comes from** (`h_explore.csv`). The median share of the
lineage's total train-AUC gain carried by the τ95 jump steps (about 5% of steps) is:

| Set | Train-AUC gain share | Test-AUC gain share |
|---|---|---|
| churn100 | 0.62 | 0.61 |
| churn200 | 0.77 | 0.79 |
| heart100 | 0.84 | 0.75 |

Elite (backprop-only) steps carry 0.03–0.05. So improvement *is* concentrated in a few
discrete child steps. At jumps, the median Δ test AUC is +0.012 to +0.027.

**Exploratory E2: node additions do not predict jumps.** Among child-structural steps,
P(jump | a node was added) − P(jump | none):
- churn100: −0.030, p = 0.94;
- heart100: +0.006, p = 0.47;
- churn200: +0.020, p = 0.19 (+0.025, p = 0.018 after generation 10).

## Test 2: jump genes are explained. Not supported; inconsistent across data sets

Outcome: the added gene lies in a leaf of the final explanation, with type-stratified MH
differences, jump minus non-jump (`h_test2.csv`).

| Set | All added genes: diff [CI], p (1-sided) | Survivors only: diff [CI], p | Jump genes survive less: diff [CI] |
|---|---|---|---|
| churn100 | +0.069 [−0.033, +0.167], 0.041 | +0.285 [0.218, 0.354], 0.0001 | −0.148 [−0.272, −0.025] |
| churn200 | +0.052 [−0.005, +0.116], 0.037 | +0.199 [0.093, 0.310], 0.0001 | −0.120 [−0.182, −0.069] |
| heart100 | **−0.144** [−0.223, −0.073], 1.0 | −0.197 [−0.397, +0.092], 0.99 | −0.243 [−0.376, −0.087] |

On the primary measure, the churn estimates are small, with lineage CIs that include 0.
Heart goes the other way.

Two effects pull in opposite directions:
- Genes added at jumps are *less* likely to survive to the final genome; they are early, and
  churn of the early network is high.
- Among those that do survive, churn jump genes, especially **nodes** (churn100: 0.95 vs
  0.38), are much more often inside a leaf.

That fits "early jump nodes become the core of later signals" in churn. It is confounded
with generation: jumps are early, and early survivors are core. It does not hold in heart.

## Test 3: emergence is a step. About half step-like; the step rarely aligns with completion or with a fitness jump

`h_leaves.csv`, `h_test3.csv`, `figures/h_emergence_*`.

- **Leaf signals:** 184 leaves.
  - Most are each leaf's logit contribution (readout contribution).
  - 14 use the exit-activation fallback (heart 12, churn200 2).
  - Four are excluded as constant on the data.
- **Below the rise cut-off:** 89 leaves were already decodable from the earliest ancestor
  (rise R < 0.10). They are mostly linear backbones and per-state offsets, i.e. one-hot
  functions that any network decodes from its inputs.

| Set | Tested leaves | Step-like (max single rise ≥ 50% of R) [CI] | at 33% | at 67% | g* = g_c | Largest rise at a fitness jump | Genes of G_L added at g* |
|---|---|---|---|---|---|---|---|
| churn100 | 36 | **0.53 [0.43, 0.63]** | 0.89 | 0.25 | 0.14 (5/36; chance 0.5 leaves, p < 0.001) | 0.11 | 0.36 |
| churn200 | 38 | 0.58 [0.42, 0.69] | 0.79 | 0.32 | 0.11 (4/38; chance 0.3) | 0.29 | 0.16 |
| heart100 | 17 | 0.59 [0.38, 0.79] | 0.88 | 0.47 | 0.35 (6/17; chance 0.3) | 0.00 | 0.65 |

Reading:
- **Shape.** The median leaf's largest single step carries 50–61% of its total rise. So
  emergence is neither smooth nor a clean single step. The curves show a few discrete jumps
  each, all at structural (child) steps,
  plus dips when ancestors lose and regain partial realisations.
- **Alignment with completion.** Above chance, but low in absolute terms. The final leaf's
  full gene set G_L is typically completed late (median g_c: 90, 173 and 77, against g*: 18,
  25 and 48). Late connections refine a signal that was already mostly decodable. Signals
  emerge from a *partial* realisation.
- **Size of what is present at the step.** At g*, the median ancestor holds 1 of the leaf's
  2 hidden nodes and 1–2 of its 3–5 connections. Small subgraphs, consistent with "sometimes
  a single node".
- **Alignment with fitness.** The largest emergence step is a train-fitness jump (τ95) for
  only 0–29% of leaves. Most signals gain decodability at steps that don't move fitness much.

The **plan's G_L** is "the leaf's node and enabled connection genes". It was implemented as
the leaf's hidden nodes plus internal and input-output connections from
`genome_surgery.annotation_to_genome_subgraph`, excluding the readout edge to the output.
For leaves that function cannot resolve, the lineage resolver's mapping is used instead.

## Test 4: XOR. The pre-registered prediction fails (0/30)

`h_xor.csv`, `h_xor_lineages.csv`.
- **Reproduction:** all 30 seeds reproduce `../xor/xor_results.json`'s solved generation
  exactly.
- **g_F** is the first lineage generation with a nonlinear hidden unit whose (active set,
  output sign) is a feasible ReLU realisation pair.
- **Result:**
  - g_F = g_solve in **0/30** (Clopper–Pearson 95% CI [0, 0.12]);
  - g_solve − g_F has a median of 15 generations (range 3–27);
  - g_F is generation 1–6.
- **Why it fails:** units with a feasible active set appear almost immediately. Feasibility
  of one unit's active set is necessary, not sufficient: the rest of the network (skip
  weights and output bias) must also be in the right region.

Descriptive:

| Measure | Count |
|---|---|
| Solving step structural | 26/30 |
| Solving step an elite copy (backprop alone finished it) | 3/30 |
| Predecessor already had 3 of 4 rows correct | 24/30 |
| A feasible unit is new at the solving step | 10/30 |
| … added at that step | 5/10 |
| … an existing unit moved into a feasible class by weights | 5/10 |

**Reading.** In XOR the jump to solved is usually a structural step, but its timing is not
set by when a feasible-class nonlinear unit first exists.

## Test 5: MONK-2, seeds 1, 5 and 7 (descriptive)

`h_monk2.csv`, `h_monk2_ablation.csv`, `h_monk2_largest_steps.csv`.

**Pre-registered: g_learn**, the first lineage generation with train AUC ≥ 0.99.

| Seed | g_learn | Δ train | Δ test | Step | Largest single-gene reversion loss |
|---|---|---|---|---|---|
| s1 | 80 | +0.003 | +0.001 | elite (no structural change) | weight −7→773, 0.0004 |
| s5 | 54 | +0.048 | +0.045 | child, weights only (MONK-2 mutates weights) | weight −16→0, 0.020 |
| s7 | 75 | +0.009 | −0.005 | 3 enabled flips into the output | bias of 389, 0.033 (flips: 0.000) |

The 0.99 crossing is a small, mostly non-structural step. It is not where the rule is learned.

**Exploratory: the largest steps on these lineages.** Two of the three seeds' biggest
jumps add one input connection into the leaf the final explanation names as the rule gate:
- **s1, generation 86:** adding −2→773 into `few_indicators_gate` takes train AUC from 0.913
  to **0.9996** (test +0.10). The s1 lineage had reached 0.99 at generation 80 and fallen back
  before this.
- **s5, generation 46:** −9→217, also into `few_indicators_gate`: +0.088.
- **s5, generation 33:** −12→217 into the same gate: +0.062.
- **s7:** the biggest step adds −7→1311, but that gene is not in a final leaf.

This is the hypothesis in its pure form: one connection completes the count gate, and
fitness jumps. But it is three seeds, chosen after the fact, in a config that also mutates
weights.

## Verdict against the hypothesis

> Performance jumps happen when a lineage first has the minimal set of nodes and connections
> needed to realise a signal.

| Part of the hypothesis | Verdict | Basis |
|---|---|---|
| Improvement comes in discrete steps | **Yes** | τ95 jump steps (about 5% of steps) carry 62–84% of the lineage gain; backprop-only steps carry 3–5% |
| Those steps are structural | **Yes, but uninformatively** | Every non-elite step on these lineages is structural, so "jump ⇒ structural" holds trivially. Beyond the initial climb (generations 1–10), RD is ≈0 in churn100 and heart100 (2 and 3 jumps); only churn200's second climb shows it |
| Jump genes become the explained signals | **Inconsistent** | Positive in churn (only for surviving genes, and confounded with generation), negative in heart |
| Signals emerge as a step, at the completion of their realisation | **Partly; no** | About half of the leaves are step-like. The largest emergence step coincides with the completion of the leaf's final subgraph in 11–35% of leaves (above chance), and with a fitness jump in 0–29%. Signals become decodable from small partial subgraphs, often well before their final realisation is complete and without a fitness jump |
| XOR: solve when the first feasible nonlinear unit appears | **No** | 0/30 |
| MONK-2: the rule arrives with a gene | **Anecdotally yes** | 2/3 seeds: one added connection into the gate gives the largest jump |

**Overall:**
- The study supports a weaker statement: *fitness improves in discrete structural steps,
  mostly early.*
- In the churn and heart networks, signal emergence and fitness jumps are largely
  decoupled, and neither is timed by the completion of the final realisation.
- The clean "one gene completes the realisation and fitness jumps" event exists (MONK-2 s1
  and s5) but is not the typical mechanism in these data.
- A possible reading, consistent with F1's convergence result: on churn and heart, most
  signals are reachable by many partial realisations and aliases. Capacity is rarely the
  binding constraint after the first few generations, so there is no threshold to cross.
  On MONK-2, the count rule needs specific inputs wired into one gate, and the threshold is
  real.

## Caveats

- **Winner's line only.** Lineages are the winner's chain and explanations are the winner's.
  Jumps on non-surviving lines are not studied.
- **Train-based definitions.** Jumps are defined on train, by selection. Test Δ at jumps is
  positive (medians +0.012 to +0.027).
- **Elitism and S = 0.** Elite copies make up 31–45% of steps. In churn, S = 0 is exactly the
  elite steps.
- **Test 3's measure.** Decodability is linear R² on the ancestor's inputs plus its hidden
  activations. It measures whether the signal is *linearly available*, not whether the
  ancestor *uses* it.
- **Adult** was not replayed: about 47 h per run as recorded. It has no parent map and is
  not in Study H.
