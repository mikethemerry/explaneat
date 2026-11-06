# Modular crossover: results

Design: `PLAN.md` (committed in 0518e38 before any outcome run). All 80
pre-registered runs completed: 2 tasks × 4 arms × seeds 0–9, 4 islands of
150, 400 generations. Two runs had to be restarted after a bug fix; see
Deviations. CSVs are in `results/`, made by `analyze.py`:
- `runs.csv`: one row per run.
- `arm_summary.csv`
- `comparisons.csv`: paired by seed, with sign tests, bootstrap CIs (10,000
  resamples, seed 42) and Holm p.
- `library.csv`: every library variant.

Per-run summaries and elites are in `results/main/*.json` and `*.elites.pkl`.
Per-generation, operator and library logs, including the module weights, are
in `results/details_main.tar.gz`.

Arms:
1. **plain**: whole-elite migration.
2. **modules**: explained weights plus refit.
3. **sham**: N(0,1) weights plus refit.
4. **topology**: zero readout, no refit.

## Headline

- **MUX-11 curriculum.**
  - Module sharing beat whole-genome migration: arm 2 > arm 1 on 9/10 seeds.
  - The *sham* did just as well: arm 2 vs arm 3 split 5/5.
  - So the gain came from grafting refit chunk *topologies*, which add
    capacity with an optimal readout. It did not come from the explained
    weights the chunks carried.
  - Topology-only grafts (no transferred weights, no refit) gave no gain over plain.
- **MONK-2** was at ceiling for every arm under this budget, so it can't tell the arms apart.
  - Every arm learned the rule in 9–10/10 runs, and median test AUC was 1.0.
  - That makes the "runs diverge" premise from `experiments/10seed` false here. That
    study used one population of 250 for 100 generations, against 4 × 150 for 400 here.
- **Generations-to-solve** (pre-registered) was uninformative:
  - MUX-11: no run in any arm reached train AUC 1.0 on stage 2, so all were censored at 400.
  - MONK-2: every comparison was close to a 5/5 split.

## Primary outcomes

Final genome: the highest train AUC in the last generation across islands.
Test AUC is on the final stage's test rows. Δ is paired by seed; "a/b/tie"
counts the seeds where arm a or arm b was better.

| Task | Comparison | Median AUC a vs b | a/b/tie | Sign p (Holm) | Median Δ [95% CI] | Mean Δ [95% CI] |
|---|---|---|---|---|---|---|
| MUX-11 | modules vs plain | 0.814 vs 0.783 | 9/1/0 | 0.021 (0.19) | +0.041 [+0.009, +0.079] | +0.043 [+0.021, +0.067] |
| MUX-11 | modules vs sham | 0.814 vs 0.838 | 5/5/0 | 1.0 (1.0) | −0.007 [−0.046, +0.040] | −0.003 [−0.029, +0.024] |
| MUX-11 | modules vs topology | 0.814 vs 0.787 | 9/1/0 | 0.021 (0.19) | +0.033 [+0.018, +0.052] | +0.037 [+0.022, +0.053] |
| MONK-2 | modules vs plain | 1.000 vs 1.000 | 1/1/8 | 1.0 (1.0) | 0 [0, 0] | +0.009 [−0.003, +0.030] |
| MONK-2 | modules vs sham | 1.000 vs 1.000 | 1/1/8 | 1.0 (1.0) | 0 [0, 0] | −0.001 [−0.003, 0.000] |
| MONK-2 | modules vs topology | 1.000 vs 1.000 | 2/1/7 | 1.0 (1.0) | 0 [0, 0] | +0.011 [−0.003, +0.036] |

**Holm.** Holm was applied over the finite primary p-values, 8 of 12. The four
MUX-11 generations-to-solve tests have no p-value, because every pair is tied
at the censoring value. After adjustment nothing is significant at 0.05. The
two 9/1 results are the strongest signals (raw p = 0.021).

**Generations-to-solve on MONK-2** (median among solved runs; "censored" counts unsolved runs as 400):

| Arm | Solved | Median generation (solved runs) |
|---|---|---|
| plain | 9/10 | 88 |
| modules | 10/10 | 106 |
| sham | 10/10 | 104 |
| topology | 9/10 | 105 |

Paired comparisons with modules (sign tests):

| Comparison | Modules faster / slower | p |
|---|---|---|
| modules vs plain | 6 / 4 | 0.75 |
| modules vs sham | 5 / 5 | 1.0 |
| modules vs topology | 6 / 4 | 0.75 |

The censored mean differences have very wide CIs, in `comparisons.csv`.

**Test accuracy at p > 0.5** (co-primary, descriptive). On MUX-11 the medians were:

| Arm | Median test accuracy |
|---|---|
| plain | 0.697 |
| modules | 0.731 |
| sham | 0.751 |
| topology | 0.718 |

Paired comparisons with modules:

| Comparison | Modules better / worse / tied |
|---|---|
| modules vs plain | 9/1 |
| modules vs sham | 4/6 |
| modules vs topology | 6/3/1 |

Secondary pairs (no adjustment), MUX-11 test AUC:

| Comparison | a/b | Median Δ [95% CI] |
|---|---|---|
| sham vs plain | 9/1 | +0.049 [+0.022, +0.074] |
| topology vs plain | 5/5 | −0.008 |
| sham vs topology | 7/3 | +0.055 [−0.009, +0.079] |

## Secondary outcomes

**Network size and wall time.** Medians of the final genome: live hidden nodes and live enabled connections.

| Task | Arm | Hidden | Connections | Total CPU-h (10 runs) |
|---|---|---|---|---|
| MONK-2 | plain | 5.5 | 29 | 1.44 |
| MONK-2 | modules | 9 | 42 | 2.15 |
| MONK-2 | sham | 9 | 42.5 | 1.97 |
| MONK-2 | topology | 4.5 | 28.5 | 1.59 |
| MUX-11 | plain | 4.5 | 23.5 | 1.58 |
| MUX-11 | modules | 13 | 50 | 2.51 |
| MUX-11 | sham | 15 | 58.5 | 2.49 |
| MUX-11 | topology | 7 | 29 | 1.81 |

- **The refit arms build networks 2–3× larger.** The MUX-11 accuracy gain of arms 2
  and 3 comes with this extra capacity.
- **Wall time:**
  - The whole study used 15.5 CPU-hours.
  - It took about 6.2 h elapsed on 4 processes, 11:10–17:21. That figure includes
    the restart of the two hung runs.

**Module library and reuse.**

- **Library size:** the median number of modules per run was:

  | Task | modules | sham | topology |
  |---|---|---|---|
  | MONK-2 | 10.5 | 14.5 | 8.5 |
  | MUX-11 | 25 | 34.5 | 24.5 |

- **Chunks are small:** the median chunk has 2 hidden nodes.
  - Median train restore gain was 0.33–0.37 on MONK-2 and 0.06–0.08 on MUX-11.
  - These numbers come from `library.csv`.
- **Alias families form:** independent aliases are later variants of an existing
  module id whose nodes did not already carry that id. The median per run was:

  | Task | modules | sham | topology |
  |---|---|---|---|
  | MONK-2 | 5 | 9.5 | 6.5 |
  | MUX-11 | 13 | 12 | 8 |

  Most re-found variants (50–73%) are re-derivations of genes that already carry the id.
- **Operators were applied** to about 1,000–1,700 offspring per run.
  - Almost every request was applicable: fallbacks to insertion are logged.
  - "Not applied" was 0–53 per run.
  - In the topology arm most replacement requests fell back to insertion.
- **Lineage survival:**
  - In the modules and sham arms, at the final generation almost the whole
    population carries a module tag that was first found on another island. The
    median was 570–599 of 600.
  - All 4 island bests carry modules.
  - In the topology arm about half the population does (median 230–290).
- **Caveat:** "carrying" means having a tagged node. A tag survives even when the
  module's genes have stopped mattering. For example, the s9 genome below carries
  two modules with no live readout.

**Ground-truth features (MUX-11).** These counts are library variants whose
probe-set contribution has r ≥ 0.9 with some g_k or D_k.

| Arm | Variants matching | Runs with a match | Feature matched | Generations of the matches |
|---|---|---|---|---|
| modules | 7 | 3/10 | g2 or g3 | 299–374 |
| sham | 3 | 2/10 | g3 | 99, 349, 374 |
| topology | 13 | 4/10 | g1, g2 or g3 (up to r = 0.998) | all ≥ 224 |

- No address decoder D_k was ever matched.
- Nearly all variants were partial mixtures: median best r was 0.35–0.42.
- **The evolved chunks are mostly not the textbook selector units.** When they
  are, they appear mostly after the stage switch. Most are re-derivations of one
  persistent unit, for example topology s7 module 23 (g1, r ≈ 0.99) at 7
  successive checkpoints.

## Exploratory (not pre-registered): MUX-11 stage 1

Stage 1 is the 6-multiplexer, generations 0–199. This is the first generation at
which the run's best train AUC reached 1.0, with 200 meaning not reached.

| Arm | Runs that solved stage 1 | Median generation |
|---|---|---|
| plain | 0/10 | 200 |
| modules | 7/10 | 168.5 |
| sham | 7/10 | 163 |
| topology | 2/10 | 200 |

- **Grafting refit chunks speeds up the easier stage strongly.**
- **The effect is the same with random weights:** modules vs sham was 5 earlier,
  3 later and 2 tied.
- **The advantage partly washes out after the switch.** At the first stage-2
  generation every arm drops to about 0.75–0.77 train AUC.

## Final genomes and their modules

These descriptions come from `describe_genome.py`. Node ids are unique within a run.
Grafted nodes get fresh, consecutive ids, so a block of consecutive ids that is in
no library variant marks a graft.

**MUX-11 modules s9.** This genome had the best test AUC of the modules arm, 0.864.
It is on island 2 and has 14 live hidden nodes. It carries 7 module ids.

- **Module 9:** a 7-node chunk with one readout.
  - It was first found on island 0 at generation 99, with gain 0.148, and re-found there at 124 and 149.
  - Its nodes in the final genome (9113–9119) are consecutive and in no variant. So
    they are a graft of island 0's variant: 3 insertions and 5 replacements from
    island 0 hit island 2. They were then inherited, with 7 module crossovers on
    island 2 passing it on.
  - All 7 nodes are still live.
- **Module 6:** 2 nodes, first found on island 1 at generation 49.
  - It arrived the same way, by replacements and an insertion from island 1. Its
    nodes 4580/4581 are a graft.
- **Module 1:** first found on island 0 at generation 24 and inserted 16 times into island 2.
  - Its live node 2258 is island 2's own re-derived copy, from the generation-149 variant.
- **Modules 10 and 20:** native to island 2.
- **Modules 3 and 12:** still tagged, but no node is live any more.
- **No module matches a ground-truth feature.** The best was module 10 vs the decoder
  D1, with r = 0.49.

**MUX-11 modules s2.** Test AUC 0.858, island 1, 9 live hidden nodes.

- All three of its modules (4, 19, 20) were first found on its own island.
- Module 4's two nodes come from island 1's own generation-249 variant. That held
  even though 27 replacements on island 1 swapped in island 3's alias, so the
  lineage that kept the native version won.
- This run's library holds a g2 selector (module 12, r = 0.947, island 0,
  generation 374), but the final genome does not carry it.

**Reading.** Cross-island module transfer does survive into winning genomes, as
s9 shows with a 7-node chunk from another island, live and inherited. But the
sham result shows that the same benefit is available from random-weight chunks of
the same shape with a refit readout.

## Interpretation

1. **What helped on the curriculum was refit grafts of chunk-shaped subgraphs**,
   explained or random. Arms 2 and 3 both beat arm 1 (9/10 each) and arm 4.
   - The explained weights added nothing measurable: 5/5, median Δ −0.007 AUC.
   - This repeats the churn finding that random-weight capacity catches up. Here
     it catches up completely, because every graft gets a logistic readout refit and
     then backprop, and on MUX-11 random ReLU features plus a fitted readout are
     already useful.
2. **Topology alone did not help** (5/5 vs plain). Zero-readout grafts that only
   backprop trains add structure without fitness gain, so selection does not keep
   them: only about half the population carries a foreign module at the end.
3. **Whole-genome migration is the weakest form of sharing here.** Plain islands
   converged on one solution: at generation 399 the four island bests have
   the same train AUC in 17/20 plain runs (logs), and no plain run solved stage 1 in 200 generations.
4. **MONK-2 gives no information.** Under 4 × 150 × 400 every arm solves it.
   Differences are single seeds, for example plain s2 at test AUC 0.899 and
   topology s0 at 0.881.
5. **The modules are rarely clean ground-truth features.** So the "explained
   chunk" this algorithm finds is a statistical boundary unit, not a semantic one.

**Null result for the study's central hypothesis**: transferring *explained*
weights in modules gave no advantage over the sham on either task. The sham was
pre-registered as the essential control, and it absorbs the whole effect.

## Deviations from PLAN

1. **Two runs hung (bug).**
   - The runs were `mux11-sham-s0` (stuck at the generation-199 operator step from
     11:27) and `monk2-sham-s8` (stuck at generation 149).
   - **Cause:** a module crossover can create a directed cycle through edges the
     child already had. `refit` then built NeuralNeat on the cyclic genome, and
     NeuralNeat's layer parsing never terminates.
   - **Fix** (d4d9890): `operators.refit` checks acyclicity first and raises. The
     PLAN rule "an operator that raises is treated like an invalid result" then
     keeps the NEAT child.
   - **Why other runs are unchanged:** the check runs after all operator rng draws,
     so runs that never produced a cycle are unaffected. `verify_rerun.sh` re-ran
     two completed runs with the fixed code: `monk2-sham-s0` and `mux11-modules-s1`.
     Their detail logs are byte-identical to the originals (see `logs/verify`,
     `results/verify`).
   - The two hung runs were killed and re-run from scratch with the fixed code; their
     logs from the hang are kept as `logs/main/*.hung.log`. The other 78 runs used
     commit 0518e38.
2. **Holm adjustment:** the PLAN says 12 primary tests. Four of them have no p-value
   (all ties at the censoring value), so Holm was applied over the 8 finite ones.
3. **The MUX-11 stage-1 analysis above is exploratory** and was not in the PLAN.
4. **The PLAN expected MONK-2 runs to diverge.** They did not under the
   pre-registered budget. This is a finding, not a protocol change; no runs
   were added.
