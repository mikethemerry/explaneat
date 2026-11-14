# Jumps and plateaus in PropNEAT lineages

> **Exploratory secondary analysis, not pre-registered.** The definitions in section 1 were
> written and committed before any trajectory was examined under them. Study H
> (`experiments/lineage/RESULTS_H.md`) had already been read, so two of its findings shaped the
> design and are named where they apply: almost every non-elite lineage step is structural, and
> jumps cluster in the first ten generations.

**Question (Mike's hypothesis).** Training shows jumps and plateaus: long flat stretches of best
fitness broken by sudden steps. Jumps are linked to structural changes in the lineage (a new node
or connection, a crossover bringing in structure); plateaus are weight tuning.

## 1. Definitions (fixed before results)

### 1.1 Runs and studies

The unit of analysis is the **run**. A study is a group of runs with one dataset and config.

| Study | Runs | Generations | Genomes | Parents |
|---|---|---|---|---|
| heart100 | 10seed-heart100-s0..9 | 100 | pruned; replayed | sidecar |
| monk2 | 10seed-monk2-s0..9 | 100 | pruned; replayed | sidecar |
| churn200 | churn200-s0..9 | 200 | pruned; replayed | sidecar |
| shape-A, shape-B | shape-{A,B}-s0..4 | 100 | full in DB | DB |
| shape-C-g300 | shape-C-s0..4-g300 | 300 | full in DB | DB |
| shape-C-sig | shape-C-sig-s0..4 | 300 | full in DB | DB |
| churn-case | churn-case-s10 | 200 | full in DB | DB |
| adult | 10seed-adult-s0..9 | 100 | full in DB | none |
| old | runs created before 2026-11-01 with at least 40 generations and full populations | 40-500 | full in DB | some |

Prefixes are not pooled: churn100-s*, 10seed-heart-s* (40 generations) and shape-C-s* (100
generations) are bit-identical prefixes of churn200, heart100 and shape-C-g300. They are used only
where their own explained champion is needed (section 1.6). Pooled results use the studies with a
lineage (all but adult and the old runs without parents).

### 1.2 Trajectory

- `f_g`: the best training fitness in generation g (maximum over the population; for every study
  except the old runs this is train AUC). Test fitness is reported where logged and never used to
  define anything.
- `B_g = max_{h<=g} f_h`: best-so-far.
- Noise scale `m`: the median over generations of `|f_g - f_{g-1}|`.
- Total gain `T = B_last - B_0`.

### 1.3 Jumps

- Threshold `tau = max(3 m, 0.10 T)`.
- A **jump window** is a pair (a, b) with `1 <= b - a <= 3` and `B_b - B_a >= tau`. Each window is
  trimmed to the generations where B actually rises; overlapping or adjacent trimmed windows are
  merged into one **event** with onset o (first rising generation), end e (last rising
  generation) and size `B_e - B_{o-1}`.
- **Plateau condition:** an event is a jump only if it is followed by a plateau,
  `B_{min(e+5, last)} - B_e < tau / 2`. An event at the last generation is kept. Events that fail
  the condition are **climbs**, counted but not used in the enrichment.
- **Early:** onset `o <= 10`. Study H found that jumps cluster there, so every enrichment is
  reported for all jumps and for late jumps (`o > 10`).
- **Sensitivity** (two alternatives): strict `tau = max(5 m, 0.20 T)`; loose
  `tau = max(2 m, 0.05 T)`.

### 1.4 Lineage and step classes

- **Champion:** the best-by-train genome of the run (highest training fitness, earliest generation,
  lowest key). Its lineage is `get_lineage(mode="fitter_parent")`: one ancestor `L_g` per
  generation, elite copies followed to the same key.
- **Step g** is `L_{g-1} -> L_g`, classified from the gene diff (`diff_genome_data`) against the
  followed parent:
  - **elite**: an elite copy (no genetic change; backprop retrains it). Weight-only.
  - **child, structural (S)**: a node added or removed, a connection added or removed, an enabled
    flag flipped, or a **gene from the other parent** (a node or connection key of the child absent
    from the followed parent and present in the other parent of a crossover).
  - **child, weights only**: a new genome with none of these.
  - Sub-flags: node added, connection added, connection enabled (false to true), crossover (two
    distinct parents), gene from the other parent.
- Study H showed that almost every child step on these lineages is structural, so "structural vs
  weight-only" is close to "new child vs re-trained elite". The sub-flags (node added in
  particular) are the sharper test and are reported alongside.

### 1.5 Linking jumps to structure

- A jump (o, e) is **carried** by the champion lineage when the lineage's own fitness rises over
  the same generations by at least half the jump: `max_{g in [o, e]} F(L_g) - F(L_{o-1}) >= 0.5 size`.
  Primary analyses use carried jumps; all jumps are reported too.
- The jump's **lineage window** is steps o..e. The jump is **structural** when any step in the
  window is S (likewise for each sub-flag).
- **Null:** every lineage window of the same width that does not overlap any event window (jump or
  climb), in the same run. `P(S | non-jump)` is the share of these that contain an S step.
- **Enrichment:** `RD = P(S | jump) - P(S | non-jump)`, per study and pooled (pooled over jumps,
  each run's null weighted by its number of jumps). 95% CI: percentile bootstrap over runs within
  study (10,000 resamples, `default_rng(42)`). One-sided permutation p: each run's jump windows are
  re-placed at random non-overlapping positions of the same widths, 10,000 times; p is the share of
  placements with a pooled structural share at least the observed one.
- **Lag:** for a carried jump, `lag = max(0, o - g_s)`, with g_s the generation of the latest S
  step at or before e (0: the structural change arrives inside the jump window). The same quantity
  at every non-jump generation is the null. The same lag is computed for node additions only.

### 1.6 Link to meaning

For runs whose champion is explained (10-seed v4 records in `experiments/10seed/explanations_v4/`,
annotations read from `explaneat_v4`; `experiments/ushape_saddle/explanations/`;
`experiments/churn_case/explanations/`), the genes added along the lineage inside each carried
jump window that survive into the champion are mapped with `map_genes_to_annotations` to the
explanation's leaves. Case studies (4-6): the largest late carried jump in each explained study,
then the most vivid by size; chosen by that rule, not by outcome.

### 1.7 Runs without a lineage

adult and old runs without parents get trajectories and jumps, and a weaker structural comparison:
the **generation-best line**, the structure signature of the generation's best genome at `o - 1`
vs at e, against the same comparison over non-jump windows. A change there can also mean another
line took the lead.

## 2. Answer in brief

- **Jumps and plateaus are real but rarer than the picture suggests.** Under the fixed rule, the
  51 lineage runs have 42 jumps and 40 climbs. Most of each run's gain comes in a steep initial
  climb over 5-15 generations that keeps rising afterwards, which the rule calls a climb, not a
  jump. Clean stair steps (a rise within 3 generations, then a flat stretch) are common only in
  **MONK-2** (22 jumps in 10 runs, 19 of them after generation 10) and occasionally in the
  synthetic shape runs. Heart, churn, Adult and the old runs mostly show a fast climb and then a
  slow creep.
- **What is enriched at jumps is "a new child genome", not "a new node".** No carried jump is
  made of elite copies alone (0 of 34; 1 of 42 counting jumps the champion line did not carry), against 22% of
  same-width non-jump windows (RD -0.22 [-0.27, -0.17], permutation p < 0.001). So plateaus
  can be backprop-only stretches; jumps never are. Any structural step: 0.88 vs 0.75, RD +0.13
  [+0.02, +0.23], p 0.056. A node added: 0.59 vs 0.44, RD +0.15 [+0.05, +0.29], p 0.040.
- **The node signal does not survive the generation confound.** Late jumps only: node added 0.40
  vs 0.30, RD +0.10 [-0.10, +0.40]. Against non-jump windows within ±10 generations of the jump
  (exploratory), structural +0.06 [-0.11, +0.18] and node added +0.05 [-0.17, +0.38]. Only "not
  elite-only" holds up (RD -0.12 [-0.19, -0.02]).
- **Lag: the structural change arrives inside the jump window, not before it.** 30 of 32 carried
  jumps have a structural step within the window (lag 0; non-jump generations 52%). For node
  additions, lag 0 in 69% of jumps (22% at non-jump generations); for late jumps, 40% at lag 0
  and 27% at lag 1. The pattern "structure first, jump some generations later as backprop tunes
  it" is not seen; the window is wide enough to hold both.
- **Mike's claim** (supervision meeting, 19 June 2025; not in the thesis or any repo document)
  holds in its weak form: a jump comes with a new genome (in these configs, almost always a
  structural child), and flat stretches can be backprop alone. It does not hold in the strong
  form that jumps are specifically *node or connection additions* beyond their base rate.
- **Meaning.** Of the 25 carried jumps in explained champions (primary rule, not counting
  prefixes), 17 left surviving genes in the champion and 16 of those have a gene inside a named
  leaf. Late jumps: 8 of 12, all 8 in named leaves.

## 3. Data and replay notes

- **Replays.** heart100, monk2 and churn200 (30 runs) were rerun in memory with
  `replay_dump.py` (the `replay_lineage.py` harness, database stubbed; nothing written to any
  database), and every genome's md5 was checked against the verified replay CSVs in
  `experiments/lineage/replay_db/`: 30/30 runs, about 1,000,000 genomes, all identical
  (`../lineage/replay_verification.csv`). Dumps are in scratch, not in the repo. Wall time: heart
  and MONK-2 about 1 min, churn200 10-16 min per run.
- **Prefixes** (churn100, 10seed-heart, shape-C 100-generation) were cut from the same
  replays or read from the DB; reported in the CSVs, not pooled.
- **Checks.** Every replayed lineage equals Study H's cached lineage (keys and structural flags,
  `../lineage/replay_db/h_cache`, 33 runs), and every DB lineage equals
  `lineage.get_lineage` (3 spot checks). Every champion equals the explained genome where one
  exists. Every lineage genome's recomputed train AUC equals its stored fitness (max abs diff
  < 1e-6, all 76 runs with a lineage), so lineage test AUC is reported for all of them.
- **No lineage.** 10seed-adult (10 runs; full genomes, no parents; replay about 47 h per run)
  and the 21 old runs (2025-08 to 2026-07; 40-500 generations). Six old runs have DB parents, but
  the old ancestry reporter linked about two thirds of children to genomes of the *same*
  generation (an off-by-one), so their chains are unusable. They get the generation-best line
  (section 1.7). "Heart" (2026-04-19, constant fitness -1000) was dropped.
- **Not used:** Connect-4, MNIST and KRK runs (only the final generation is stored), the Cox
  cases (one generation stored), XOR (Study H's Test 4 covers it).
- **Fitness.** Train AUC everywhere except two old runs ("Heart - paper", "No pressure - long
  run", a log-likelihood-like score up to 3.2); the rule is relative, so they are comparable.

## 4. Trajectories (`figures/trajectories_*.png`, `runs.csv`)

Per study, primary rule (medians over runs; `T`: total gain in train AUC, `tau`: threshold,
"flat": share of generations where best-so-far does not rise; step shares are lineage steps):

| Study | Runs | T | tau | Jumps (late) | Climbs | Gain share in jumps | Flat | Structural steps | Node steps | Elite steps |
|---|---|---|---|---|---|---|---|---|---|---|
| heart100 | 10 | 0.094 | 0.009 | 5 (0) | 5 | 0.34 | 0.36 | 0.67 | 0.23 | 0.33 |
| monk2 | 10 | 0.319 | 0.032 | 22 (19) | 10 | 0.35 | 0.35 | 0.51 | 0.07 | 0.36 |
| churn200 | 10 | 0.292 | 0.029 | 1 (0) | 10 | 0 | 0.36 | 0.55 | 0.39 | 0.45 |
| shape-A | 5 | 0.154 | 0.015 | 5 (3) | 3 | 0.16 | 0.34 | 0.69 | 0.22 | 0.21 |
| shape-B | 5 | 0.171 | 0.017 | 2 (0) | 6 | 0 | 0.10 | 0.71 | 0.25 | 0.27 |
| shape-C-g300 | 5 | 0.223 | 0.022 | 5 (1) | 3 | 0.12 | 0.28 | 0.36 | 0.14 | 0.63 |
| shape-C-sig | 5 | 0.197 | 0.020 | 1 (0) | 3 | 0 | 0.03 | 0.44 | 0.14 | 0.54 |
| churn-case | 1 | 0.256 | 0.026 | 1 (0) | 0 | 0.77 | 0.47 | 0.26 | 0.19 | 0.74 |
| adult | 10 | 0.188 | 0.019 | 1 (0) | 10 | 0 | 0.23 | - | - | - |
| old | 21 | 0.167 | 0.017 | 5 (3) | 24 | 0 | 0.46 | - | - | - |

- In all studies the noise scale m (0.0001-0.001) is far below 10% of the gain, so `tau` is set
  by the gain term. Rises of 0.02-0.05 spread over 5-12 generations, which the eye reads as
  steps, are climbs under the rule (for example shape-C-s0-g300 at generations 66-71, +0.030).
- MONK-2 is the study that looks like the textbook picture: long flat stretches, then
  near-vertical rises of 0.04-0.11 (a counting rule completed or not). Its lineages are also the
  most jagged (the lineage ancestor often sits well below the population best), because MONK-2's
  config mutates weights (`weight_mutate_rate` 0.8; the others do not).
- The old runs look flatter (46% flat generations), mainly because several had small
  populations; "Working Backache 50gen" (2025-08-28, 30 genomes, 50 generations) is the clearest
  staircase in the whole set: its generation best fluctuates by ±0.08 while best-so-far moves
  in four flat steps, which is what a population that loses and refinds its best produces.

## 5. Enrichment: structure at jumps vs non-jumps (`enrichment.csv`, `figures/enrichment.png`)

Primary rule, carried jumps. Cells: P(flag | jump) vs P(flag | non-jump), RD [95% bootstrap CI
over runs], one-sided permutation p (for "elite only": the lower tail).

| Study | Runs with a jump | Jumps | Any structural step | A node added | Elite copies only |
|---|---|---|---|---|---|
| heart100 | 5 | 5 | 1.00 vs 0.94, RD +0.06 [+0.00, +0.19], p 0.71 | 1.00 vs 0.86, RD +0.14 [+0.03, +0.28], p 0.41 | 0.00 vs 0.06, RD -0.06 [-0.19, +0.00], p 0.71 |
| monk2 | 9 | 14 | 0.71 vs 0.70, RD +0.02 [-0.34, +0.25], p 0.67 | 0.14 vs 0.22, RD -0.07 [-0.29, +0.26], p 0.86 | 0.00 vs 0.23, RD -0.23 [-0.32, -0.12], p 0.036 |
| churn200 | 1 | 1 | 1.00 vs 0.76 | 1.00 vs 0.76 | 0.00 vs 0.24 |
| shape-A | 4 | 5 | 1.00 vs 0.97, RD +0.03 [+0.00, +0.05], p 0.88 | 1.00 vs 0.68, RD +0.32 [+0.25, +0.42], p 0.17 | 0.00 vs 0.03, p 0.88 |
| shape-B | 2 | 2 | 1.00 vs 0.80, p 0.65 | 0.00 vs 0.37, p 1.0 | 0.00 vs 0.19, p 0.67 |
| shape-C-g300 | 5 | 5 | 1.00 vs 0.54, RD +0.46 [+0.37, +0.56], p 0.048 | 1.00 vs 0.37, RD +0.63 [+0.53, +0.70], p 0.008 | 0.00 vs 0.46, RD -0.46 [-0.55, -0.36], p 0.049 |
| shape-C-sig | 1 | 1 | 1.00 vs 0.68 | 1.00 vs 0.34 | 0.00 vs 0.30 |
| churn-case | 1 | 1 | 1.00 vs 0.48 | 1.00 vs 0.48 | 0.00 vs 0.52 |
| **pooled** | 28 | 34 | **0.88 vs 0.75, RD +0.13 [+0.02, +0.23], p 0.056** | **0.59 vs 0.44, RD +0.15 [+0.05, +0.29], p 0.040** | **0.00 vs 0.22, RD -0.22 [-0.27, -0.17], p < 0.001** |

Other flags, pooled: connection added 0.68 vs 0.60 (RD +0.07 [-0.03, +0.18]); a crossover
child in the window 1.00 vs 0.78 (RD +0.22 [+0.18, +0.27]); a gene from the other parent 0.03
vs 0.02 (fitter-parent crossover almost never imports structure from the weaker parent; it
does in 1 of 34 jumps). All jumps, carried or not (42): structural 0.88 vs 0.74, RD +0.14
[+0.05, +0.24], p 0.026; elite only 0.02 vs 0.21, p < 0.001.

Sensitivity and timing (pooled):

| Rule | Jumps | Jumps | Any structural step | A node added | Elite copies only |
|---|---|---|---|---|---|
| primary | all | 34 | 0.88 vs 0.75, RD +0.13 [+0.02, +0.23], p 0.056 | 0.59 vs 0.44, RD +0.15 [+0.05, +0.29], p 0.040 | 0.00 vs 0.22, RD -0.22 [-0.27, -0.17], p < 0.001 |
| primary | late | 15 | 0.87 vs 0.75, RD +0.12 [-0.13, +0.28], p 0.30 | 0.40 vs 0.30, RD +0.10 [-0.10, +0.40], p 0.32 | 0.00 vs 0.21, RD -0.21 [-0.29, -0.10], p 0.040 |
| strict | all | 8 | 0.88 vs 0.80, RD +0.07 [-0.17, +0.26], p 0.51 | 0.38 vs 0.52, RD -0.15 [-0.37, +0.08], p 0.91 | 0.00 vs 0.18, RD -0.18 [-0.25, -0.11], p 0.16 |
| strict | late | 0 | - | - | - |
| loose | all | 30 | 0.83 vs 0.72, RD +0.12 [-0.09, +0.29], p 0.17 | 0.50 vs 0.40, RD +0.10 [-0.05, +0.23], p 0.11 | 0.00 vs 0.26, RD -0.26 [-0.33, -0.18], p 0.001 |
| loose | late | 17 | 0.94 vs 0.70, RD +0.24 [+0.03, +0.39], p 0.028 | 0.41 vs 0.36, RD +0.05 [-0.19, +0.30], p 0.32 | 0.00 vs 0.27, RD -0.27 [-0.37, -0.16], p 0.007 |
| primary, null within ±10 generations (exploratory) | all | 31 | 0.87 vs 0.88, RD -0.01 [-0.12, +0.09] | 0.55 vs 0.51, RD +0.04 [-0.09, +0.21] | 0.00 vs 0.07, RD -0.07 [-0.11, -0.02] |
| same | late | 15 | 0.87 vs 0.81, RD +0.06 [-0.11, +0.18] | 0.40 vs 0.35, RD +0.05 [-0.17, +0.38] | 0.00 vs 0.12, RD -0.12 [-0.19, -0.02] |

Reading:
- The only result that survives every threshold, the late-only restriction and the
  generation-matched null is **"no jump is backprop alone"**: every jump window contains at
  least one new child genome. In heart, churn and the shape runs a child is nearly always
  structural (Study H: "every child step is structural"), so this is also "every jump contains a
  structural step". In MONK-2, 4 of 14 jumps contain only weights-only children (weight
  mutation), so there the jump is a new genome but not new structure.
- Node additions are over-represented at jumps in the pooled primary analysis, but the effect
  sits in the early climb (where nodes are added at the highest rate) and disappears against
  windows at the same stage of the run. The strongest single study is shape-C-g300 (5 of 5 jumps
  with a node, against 37% of windows), with 5 jumps.
- **Generation-best line** (every run, `genbest.csv`): the best genome's structure changed
  across 100% of primary jumps in every study except MONK-2 (86%), against 49-100% of non-jump
  windows (MONK-2 55%; old runs 5/5 vs 70%; Adult 1/1 vs 52%). The old runs and Adult agree with
  the lineage studies, but the comparison cannot separate "new structure" from "another genome
  took the lead".

## 6. Lag (`lag.csv`, `figures/lag.png`)

Generations from the last structural step (or node addition) to the jump's onset; 0 means
inside the window. Null: the same at every non-jump generation of the same lineages.

| | Lag 0 | 1 | 2-5 | 6+ | n |
|---|---|---|---|---|---|
| Any structural step, jumps | 0.94 | 0 | 0.06 | 0 | 32 |
| Any structural step, non-jump generations | 0.52 | 0.09 | 0.14 | 0.25 | 7,545 |
| Node added, jumps | 0.69 | 0.14 | 0.03 | 0.14 | 29 |
| Node added, late jumps | 0.40 | 0.27 | 0.07 | 0.27 | 15 |
| Node added, non-jump generations | 0.22 | 0.13 | 0.25 | 0.40 | 7,486 |

A jump window spans 1-12 generations while the null is one generation wide, so lag 0 is
favoured at jumps by construction; the node lag for late jumps (3-generation windows) is the
fairer reading. It points to the new node arriving in the jump window or one generation before
(67% of late jumps), not to a delayed payoff. The 4 late jumps whose last node is 6 or more
generations old are MONK-2 jumps carried by new connections, enabled flags or weights.

## 7. Case studies (`meaning.csv`, `figures/case_*.png`)

Rule (section 1.6): the largest late carried jump in each explained study (MONK-2, shape-A;
churn needs the loose rule; heart, shape-B, shape-C-s4 and the churn case have none), then the
most vivid by size; plus Mike's motivating example from the ushape study.

1. **monk2-s7, generations 18-20 and 34-36 (+0.064 and +0.062).** The first jump adds node 648
   and the connection -7→288 into node 288, the unit the v4 explanation names
   `few_first_values_gate`; the second adds node 1311 and -9→389, which ends up in
   `a4_a5_first_value_unit`. Two named parts of the MONK-2 rule arrive at two stair steps
   (`case_monk2-s7.png`).
2. **monk2-s1, generations 73-79 (+0.106, the largest late jump).** No new gene survives: the
   steps are crossover children with large weight changes (up to 3.1) and a connection removed,
   then elite copies. A jump the explanation cannot attribute to a part: the counting rule was
   already wired, and the jump is weights.
3. **shape-A-s0, generations 17-19 (+0.016).** The connection -5→357 (sodium into node 357)
   arrives at generation 18; it is the input edge of `low_sodium_hinge`, the left arm of the U.
   Node 1789 arrives at 19 (no leaf). Small in size, but it is the step at which the U's low
   arm became available.
4. **churn200-s7, generations 14-16 (loose rule, +0.020).** Node 2297 and -21→2297 arrive at 14
   and land in `day_charge_gate`; the churn100-s7 explanation of the same lineage puts the same
   genes in no leaf (`case_churn200-s7.png`).
5. **shape-C-s1-g300, generations 100-200 (the saddle, ushape RESULTS section 11).** The +0.040
   train-AUC gain between the 100 and 200 checkpoints is not one jump. Its steepest part is a
   12-generation climb (generations 110-122, +0.030 best-so-far) during which the lineage added
   **six nodes in nine generations** (110, 112, 114, 115, 117, 118), test AUC moving from 0.835 to
   0.861; under the rule it is a climb, not a jump (`case_shape-C-s1-g300.png`). The champion
   is not explained, so the parts cannot be named.
6. **monk2-s4, three equal steps (generations 22-24, 41-43, 56-58; +0.043 each).** The second
   adds -11→1223 and the third -16→1717 plus two direct edges; both new connections end up in
   `a5_not_first_lift`, so one named part was built over two jumps. The first step adds no gene
   that survives (an enabled flag and weights) (`case_monk2-s4.png`).

Over all explained champions: of 25 carried jumps (primary), 17 left genes that survive into the
champion (59 of 307 new genes survive) and 16 of them have a gene in a named leaf; for the 12
late jumps, 8 left surviving genes, all 8 in named leaves.

## 8. Old vs new, and Mike's claims

**Where the claim was made.** The thesis (`~/dev/ExplaNEAT-Thesis/sections/*.tex`) and the repo
documents make no claim of jumps and plateaus in fitness. The nearest thesis statements are
design arguments in `sections/6-PropNEAT.tex`: lines 249-250 ("any topological change is
associated with a fine-tuning of the model before evaluation, and any improvement can be
immediately seen"), 259-261 ("backpropagation is equivalent to multiple generations of genetic
improvements of weights") and 269-273 (topological mutation as exploration, backprop as
exploitation). The phenomenon itself is stated in a supervision meeting (PhD meeting, 19 June
2025, Granola notes): Mike described a change in the second derivative of performance at a
generation, followed by a plateau, and said that in earlier work such changes were typically
associated with a structural change and the plateau with weight optimisation. Study H
(`experiments/lineage/PLAN_H.md` lines 12-14, `RESULTS_H.md`) is the only previous test; it
found jumps carry 62-84% of the lineage gain and are all structural child steps, but
"uninformatively", and mostly in generations 1-10. The ushape study (section 11) reports a
+0.04 train-AUC jump with the saddle between the 100 and 200 checkpoints.

**Does it hold up?**

| Claim | Verdict | Basis |
|---|---|---|
| Training shows long plateaus broken by sudden steps | **Partly** | Flat best-so-far in 3-47% of generations (median per study). Sudden steps (≤ 3 generations, then flat) are common in MONK-2 and rare elsewhere; most gain comes in steep multi-generation climbs, early |
| A jump comes with a structural change | **Yes, weakly** | Every jump contains a new child genome (0/34 elite-only vs 22%); 88% contain a structural step vs 75% (RD +0.13 [+0.02, +0.23]). Not above what is usual at the same stage of the run |
| Specifically a new node or connection | **No** beyond the generation confound | Node added +0.15 [+0.05, +0.29] overall, +0.10 [-0.10, +0.40] late, +0.04 [-0.09, +0.21] against nearby windows |
| Plateaus are weight tuning | **Yes, as a possibility, not a rule** | Elite-only (backprop-only) windows make up 21-27% of non-jump windows and none of the jumps; but most non-jump windows also contain structural children that did not move fitness |
| The structural change precedes the jump, backprop then tunes it | **Not seen** | The structural step is inside the jump window in 94% of jumps; the last node is in the window or one generation before in 67% of late jumps |
| The saddle arrived as a +0.04 jump (ushape §11) | **Refined** | Between checkpoints, yes; per generation, a 12-generation climb with six node additions on the lineage |
| Old runs show it more clearly | **Yes, where the population was small** | The 2025 "Working Backache" run with 30 genomes is the cleanest staircase; old 250-genome runs look like the new ones (1 early climb, slow creep) |

## 9. Caveats

- **Exploratory and small.** 34 carried jumps in 28 runs; per-study cells have 1-14 jumps.
  The thresholds were fixed in advance, but the late-only and local-null analyses are exactly
  where the effects shrink.
- **Selection.** The champion lineage is a survivor; jumps on lines that died are not counted
  (8 of 42 jumps were not carried by the champion line), and the null windows are from the same
  surviving line.
- **Structural is nearly "child".** In configs without weight mutation every child step is
  structural (Study H), so "structural vs weight-only" mostly separates new children from
  re-trained elites. The node flag is the sharper test, and it is weak.
- **Full-batch training** makes elite plateaus very flat (backprop moves an elite by about
  1e-4 per generation), which helps the rule find plateaus and makes the noise term negligible:
  `tau` is set by 10% of the gain in every run.
- **Single species.** The shape and churn populations mostly stay in one or two species, so
  "crossover" is within one line; the other parent almost never contributes structure.
- **Lag windows differ in width** between jumps and the null (section 6).
- **Meaning** uses the explanation's leaves as given; a gene in no leaf may sit in a composition
  or be an unexplained edge. The ushape saddle runs (g300) are not explained.

## 10. Files and reproduce

- `RESULTS.md` (this file; section 1 committed before any result), `replay_dump.py`, `build.py`,
  `analyze.py`, `meaning.py`, `figures.py`.
- `jumps.csv` (every event, three rules), `runs.csv`, `enrichment.csv`, `genbest.csv`,
  `lag.csv`, `meaning.csv`; `figures_data.json` (trajectories, events, lineage flags and test
  AUC per run, enrichment, lag counts, case strips) for a web report.
- Figures: `figures/trajectories_10seed.png`, `trajectories_synthetic.png`,
  `trajectories_other.png`, `enrichment.png`, `lag.png`, `case_monk2-s7.png`,
  `case_monk2-s1.png`, `case_monk2-s4.png`, `case_shape-A-s0.png`, `case_churn200-s7.png`,
  `case_shape-C-s1-g300.png`.

```bash
# replays (scratch only; at most 3 at a time), then per-run caches
PYTHONHASHSEED=0 uv run python experiments/lineage_jumps/replay_dump.py heart100 0 --out $SCRATCH/dump
uv run python experiments/lineage_jumps/build.py --replay heart100 0 1 2 3 4 5 6 7 8 9
uv run python experiments/lineage_jumps/build.py --db shape-A-s0 shape-C-s1-g300 churn-case-s10
uv run python experiments/lineage_jumps/analyze.py
uv run python experiments/lineage_jumps/meaning.py
uv run python experiments/lineage_jumps/figures.py monk2-s1 monk2-s7 monk2-s4 shape-A-s0 \
    churn200-s7:loose shape-C-s1-g300:95-135
```
