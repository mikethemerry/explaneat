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

## 2. Data and replay notes

(pending)
