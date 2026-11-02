# Pre-registration: 10-seed consistency experiment (thesis Ch 7)

Committed 2026-11-02, before any 10-seed result was seen. Owner: Mike Merry.
Brief: vault `PhD/drafts/2026-10-22 - Brief for Claude Code - 10-seed consistency experiment.md`.
Results freeze: Thu 19 Nov 2026.

## Purpose

Measure, honestly, how often PropNEAT produces networks that can be explained, and at
what cost. The thesis's existence claim does not depend on this experiment. No seed is
selected, dropped or re-run because of its outcome.

## Design

| Item | Setting |
|---|---|
| Datasets | Adult `adult_ohe_v2` (`46c63b17-42c5-4142-9fcf-3b5f369319c5`, split `1df50169-8095-40d1-b61a-30bb541cd313`); Heart `Heart Disease (binary, prepared)` (`eea247c2-bdb7-41af-8ec1-9e26dcb5050d`, split `165cebbd-6609-46ca-867e-c09e17b2d6ec`); MONK-2 `monk2_onehot` (`c2340904-9db2-4830-99be-e9ed637ed400`, split `63257a45-49c9-42a6-8b28-8e346eab2bee`). Split ids confirmed against the paper runs. |
| Seeds | 0–9 per dataset (30 runs). The paper runs are a reference, not one of the 10. |
| Split | The paper's split for every seed; only evolutionary randomness varies. |
| Population / epochs / fitness | 250 / 5 backprop epochs per generation / AUC-ROC on the training split. |
| Generations | Adult 100, Heart 40, MONK-2 100 (as in the paper). |
| Training | Adadelta, lr 1.5, full-batch BCE (unchanged `backproppop.py`). |
| Structural penalty | None (no penalty exists in the code path; as in the paper). |
| Selection | Best genome by training-split fitness across all generations. The test split is never used for selection or tuning. |
| Naming | Experiments `10seed-<dataset>-s<seed>`, description "Thesis Ch7 consistency experiment, 2026-10". |
| Hardware | Apple M4 Pro (14 cores, 48 GB), CPU, float64, 1 torch thread per run. CUDA is absent; MPS is not used because it lacks float64 (the paper runs also ran on CPU). |

### Configs

NEAT configs are in `configs/{adult,heart,monk2}.ini`. They are the reproducibility-bundle
configs (`reproducibility/*/neat_config.ini`), with one cosmetic fix: the
`activation_options`/`aggregation_options` lines are written as `relu`/`sum` instead of
`['relu']`/`['sum']`. Both options are unused, because the mutation rates are 0 and
the defaults are set explicitly. With that fix, each config's serialised form is
**identical** to the stored `Experiment.config_json` of its paper run (checked
programmatically on 2026-11-02).

| | Adult | Heart | MONK-2 |
|---|---|---|---|
| Paper experiment | `dbab685c` "No pressure" | `c8f3805e` "Heart Binary" | `45e998d7` "monk2 - OHE - 250-100-5" |
| Paper genome | `4bc8fa07` | `9022a9eb` | `23dde0fa` |
| Paper code commit | `0bd2e54` | `b21658f` | `fff5bb0` |
| Config origin | template `bb576521` + overrides | template `bb576521` | Default template values |
| conn add / delete | 0.68 / 0.15 | 0.5 / 0.3 | 0.3 / 0.1 |
| node add / delete | 0.7 / 0.1 | 0.4 / 0.2 | 0.15 / 0.05 |
| enabled_mutate_rate | 0.1 | 0.1 | 0.01 |
| weight mutate rate / power / replace | 0 / 0 / 0 | 0 / 0 / 0 | 0.8 / 0.5 / 0.1 |
| bias mutate rate / power / replace | 0 / 0 / 0 | 0 / 0 / 0 | 0.7 / 0.5 / 0.1 |
| compatibility threshold / disjoint coeff | 5.0 / 2.0 | 3.0 / 1.0 | 3.0 / 1.0 |
| split stratified | no | no | yes |

Common to all three: inputs 107 / 26 / 17, 1 output, `full_direct`, ReLU, sum, weight
and bias init N(0,1) clipped to ±30, compat weight coeff 0.5, max stagnation 15, species
elitism 2, elitism 2, min species size 2, survival threshold 0.2.

**Code.** None of the training-path files (`core/backproppop.py`, `core/neuralneat.py`,
`db/population.py`, `evaluators/`, `api/experiment_runner.py`) changed between any paper
commit and the commit used here (`git diff` is empty). The only addition is the seeded
harness `run_seed.py`. It seeds `random`, `numpy` and `torch` before the population is
built, runs one seed per process with `PYTHONHASHSEED=0`, and calls the same
`DatabaseBackpropPopulation.run(auc_fitness, ...)` as the UI runner. It fits the
StandardScaler in memory and asserts that it equals the split's stored params, without
writing to the split. The code commit is recorded in each Experiment's
`hardware_info.run_params.code_commit` and in `runs.csv`.

**Determinism check (2026-11-02).** Two concurrent runs of seed 1000 on Heart for
3 generations produced bit-identical genomes and fitnesses (750/750). These dry runs are
named `10seed-dryrun-*` and are not part of the results.

### Heart reference genome

The paper's Heart network is `9022a9eb` (experiment `c8f3805e`, generation 38; train AUC
0.9455, test AUC 0.8966, matching the paper). It is not `9d3e0b7a` "Heart - paper" (Default
template, BCE-style fitness, 100 generations, unannotated) or `85c403c5` "Heart - Paper
- AUC" (same config, 100 generations, unannotated; best train AUC by generation 39
was 0.947). Note: `c8f3805e` is marked `interrupted`. It was launched from a template
specifying 100 generations, and it stopped about a minute after generation 39 was
saved, at the same instant as a concurrent run, so the 40-generation budget is where
that run was cut rather than a planned budget. The 10-seed Heart runs use 40
generations by design (decided 2 Nov 2026).

## Amendments

- **2026-11-03 — tie-break for best genome.** Heart seeds 2 and 3 each have two distinct
  genomes that tie exactly on best training fitness in the same generation, and
  `get_best_genome` orders ties arbitrarily. Rule: highest training fitness; ties are
  broken by earliest generation, then lowest NEAT genome key (born first). This gives
  heart-s2 = `0057df7a` (key 9083, not 9578) and heart-s3 = `c426b094` (key 9854, not
  9877). The rule was fixed before any genome structure was inspected or explained.
  Only genome-level node and connection counts had been seen. The same rule applies to
  all datasets.
- **2026-11-03 — explanation protocol.** Explanations follow `EXPLAIN_PROTOCOL.md`.
  Claude subagents call the MCP tool functions through `xn.py`, with
  function-preserving operations only. All evidence uses the training split. Each agent
  has a 3-hour wall-clock box. Heart is explained first, at Mike's request (2 Nov),
  starting with heart-s5 as a pilot, then the remaining Heart seeds in their
  pre-registered order. 100-generation Heart runs will follow later as a separate,
  additional analysis.

- **2026-11-03 — completeness criterion (after the heart-s5 pilot).** The pilot showed
  that the tool's `compositional_coverage` is computed as compositions / (annotations −
  leaves), so it is 1.0 whenever any composition exists. It does not test whether the
  hierarchy reaches the output. "Full structural and compositional coverage" in the
  threshold therefore means: tool structural coverage = 1.0 AND tool compositional
  coverage = 1.0 AND `check_explanation.py` `hierarchy.complete` (a single root annotation
  whose subgraph contains the output node, with every annotation a descendant of it).
  Explanations must also preserve function: max |Δ output| < 1e-9 on train. heart-s5
  (pilot) meets this. It is reported with the others and flagged as the pilot. The
  protocol was updated with the pilot's tooling lessons before the remaining genomes
  were explained.

## Metrics (per run, `runs.csv`, all 30 best genomes)

- `dataset, seed, experiment_id, genome_id, code_commit, hardware`, plus a `rerun` note
  if a crashed run was repeated with the same seed.
- Performance: train AUC; test AUC with a 95% CI from 10,000 paired bootstrap resamples
  of the test set (seed 42), as in the paper.
- Size: connected input nodes and unused input nodes; hidden nodes; enabled phenotype
  connections; genome genes including disabled ones.
- Shape: depth (longest input→output path, in edges); skip connections = enabled
  phenotype edges with target depth > source depth + 1, where depth = longest path from
  any input; direct input→output edges reported separately; mean and max fan-in of
  hidden nodes.
- Structural monosemanticity proxy (a structural proxy, **not** an annotation result):
  for each hidden node, the number of distinct original input features upstream of it,
  where one-hot columns map back to their source feature. Report the share of hidden
  nodes with ≤2 such features, and the median.
- Cost: wall-clock training time, the generation of the best genome, total generations.
- Context: the paper's L2 logistic-regression test AUCs (Adult 0.905, Heart 0.919,
  MONK-2 0.423).

## Explanations (`explanations.csv`)

- Every one of the 30 best genomes gets a Claude-assisted explanation via the ExplaNEAT
  MCP, following the paper workflow: analyse the phenotype → propose candidate subgraphs
  → function-preserving refactoring only (splits, identity nodes) → leaf annotations
  (closed form, hypothesis, evidence) → composition annotations → `get_coverage`. All
  operations go into the event stream.
- **Sampling/order:** all 30, in the pre-registered order below; no skipping or
  reordering because of how a network looks.
- **Time box:** 3 hours of explanation work per network. Failure to reach full coverage
  in the box is recorded as "not explained within budget", with the partial coverage.
- Automatic checks: `get_coverage`, and the maximum absolute error between each leaf's
  closed form and the subgraph outputs on the training split.
- Recorded per network: structural and compositional coverage and whether both reach
  1.0; number of leaf and composition annotations; refactor operations (splits,
  identities); annotation-based monosemanticity (share of hidden nodes covered by a
  single leaf annotation without a split); closed-form max error; Claude wall clock;
  Mike's verification time; `verified_by_mike`; Mike's changes; notes on anything that
  resisted explanation.
- Claude-only and Mike-verified results are reported in separate columns and never
  merged.

### Verification / explanation order

A permutation of all 30 (dataset, seed) pairs, drawn with
`numpy.random.default_rng(20261102).permutation(30)` over
`[f'{d}-s{s}' for d in ('adult','heart','monk2') for s in range(10)]`:

```
 1 adult-s0    7 monk2-s4   13 adult-s7   19 heart-s4   25 heart-s7
 2 heart-s5    8 adult-s1   14 monk2-s7   20 heart-s2   26 monk2-s6
 3 monk2-s2    9 monk2-s5   15 monk2-s8   21 adult-s3   27 heart-s3
 4 heart-s0   10 adult-s2   16 monk2-s0   22 adult-s9   28 adult-s8
 5 heart-s1   11 monk2-s9   17 adult-s4   23 adult-s5   29 adult-s6
 6 monk2-s3   12 heart-s6   18 heart-s9   24 heart-s8   30 monk2-s1
```

Claude explains in this order, among the genomes whose runs have finished. Mike verifies
down the same order, as far as he gets.

## Threshold and wording

- **Support:** the consistency claim is supported for a dataset if **≥6 of its 10 best
  genomes reach full structural and compositional coverage (both 1.0) within the time
  box**. Claude-explained and Mike-verified counts are reported separately. Only actual
  coverage counts (no "predicted from structural metrics" route).
- **Fallback wording if below threshold:** "Existence holds. Explainable networks arise
  often enough that a train-until-explainable loop is practical, at a median cost of X
  hours of training plus Y hours of explanation."

## Guardrails

Every run is reported. A crashed run is re-run with the same seed and flagged in
`runs.csv`. The config is identical across seeds. The paper experiments, genomes and
explanations are never modified. The test split is never used for selection.

## Deferred

Label-permutation test: future work (design paragraph for Ch 7/9 to follow; not run).
