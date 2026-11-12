# Connect-4 capacity study

Exploratory, not pre-registered. Training completed on 2026-11-11/12; the preserved
runs were audited and the selected network annotated on 2026-11-12.

**Capacity and training closed much of the linear-to-GBM gap.** The three saved
main champions score **0.9071, 0.9149 and 0.9109 test AUC**: mean **0.9110**, sample
SD **0.0039**. Each exceeds the stated 0.90 target. The mean closes **60.4%** of the
gap from ordinal logistic regression (0.8611) to GBM (0.9437) on this shared split.
The selected champion closes 65.2% of that gap.

**Explanation outcome: covered and described, not tactically explained.** The
selected network has a complete function-preserving hierarchy, but its distributed
gates were not recovered as threats, winning lines or parity concepts. Coverage
does not resolve that failure of meaning. The initial annotation attempt took
about eleven minutes; independent review and a narrow terminology correction
brought the final elapsed time to about twenty-one minutes, within the three-hour
maximum. This includes a brief environment-diagnosis pause; the budget was not exhausted.

## Data, training and selection audit

The data and encoding are unchanged from `../complex_real/README.md`: 67,557
eight-ply Connect-4 positions, 42 ordinal squares (o = -1, blank = 0, x = +1), x to
move; target is x wins with perfect play versus loss/draw. The stratified 80/20
split uses seed 42: **54,045 training / 13,512 test rows**. Training inputs are
standardised with parameters fitted on the training split.

- **Pilots:** 17 completed runs, population 150, seed 0, 50 generations. Each fits
  48,640 rows and monitors the same held-out 5,405 training rows (stratified split,
  seed 777). Pilot configuration selection uses saved validation AUC; the maximum
  is `p18-h10-w015-e50` at 0.9188. There is no replication or correction for this
  adaptive reuse of validation data.
- **Preprocessing limit:** the scaler was fitted on all 54,045 training rows
  before the pilot fit/validation subdivision. Validation covariates therefore
  influence scaling. No test rows influence scaling or fitted weights.
- **Test use:** `prepare()` materialises the test arrays even in pilot mode, so
  the older runner description saying test is "not read at all" is too strong.
  The pilot fitness, monitor and final metrics do not evaluate those arrays;
  pilot JSONs contain validation metrics and no test metrics. Main monitoring
  uses training rows, and test is evaluated after each main run finishes.
  The inherited CSV/log label `champ_test_auc` means validation in pilots and
  training in mains. Baseline and previous-study test results on this split
  were already known. This is an exploratory comparison, not a pristine new
  independent holdout claim.
- **Main budget:** population 150, 80 generations, 50 full-batch backprop epochs
  per generation, seeds 0/1/2, static threshold 3.0. The original broad task did
  not preregister a main budget; this differs from the 100-generation earlier
  complex-real study and from the 50-generation pilots. It was already launched
  before the handover; no runs or parameters were changed during consolidation.
- **Selection:** training fitness selects the saved genome within a run; the
  largest saved **training** AUC selects seed 1 for explanation. Test AUC does
  not enter either selection rule. Pilot validation is a harness monitor, not
  the population's validation-selection branch.
- **Historical-best caveat:** the training-only branch of
  `BackpropPopulation.run()` stores `best_genome` by reference. Elites continue
  training and can decline. Thus the files named `best_genome` are **saved
  champions, not guaranteed frozen historical optima**. For example, seed 1's
  logged generation 78 AUC is 0.919006, versus saved generation 79 AUC 0.918850.
  The reporter's `best_generation` records the last evaluation of that object,
  not necessarily when it first won. `runs.csv` preserves both saved fitness
  and maximum logged fitness. Existing results were not repaired or rerun.

Backprop uses BCE, full-batch Adadelta at learning rate 1.5, float64, a sigmoid
output, AUC fitness, `mask_phantom=True`, `skip_invalid_genomes=True` and no
complexity penalty. Weight/bias mutation is disabled as in the inherited config;
topology mutations remain enabled. Initial hidden IDs are aligned across genomes
to make crossover homologous. All starts use `full_direct`, including direct
input-to-output connections. LR warm starts fit direct weights on the same fit
rows. Run JSONs record source commits, dirty status, settings and hardware.

## What retained useful capacity

The completed pilot table is reproduced in [tables.md](tables.md); exact metrics
and the historical-best audit are in [runs.csv](runs.csv).

| Comparison | Validation AUC | Saved hidden / connections |
|---|---:|---:|
| Default small pilot (`p0`) | 0.8619 | 1 / 35 |
| Start ten hidden, original weight SD 1 (`p1`) | 0.8595 | 2 / 53 |
| Start ten hidden, weight SD 0.15 (`p12`) | 0.8828 | 8 / 373 |
| Same small initial weights, twenty epochs (`p13`) | 0.9121 | 11 / 444 |
| Same, fifty epochs (`p18`, selected) | 0.9188 | 10 / 472 |
| Twenty hidden, small initial weights, twenty epochs (`p16`) | 0.9148 | 20 / 902 |
| Five hidden, small initial weights, twenty epochs (`p19`) | 0.8927 | 5 / 227 |

Starting dense hidden capacity, shrinking the initial weights, and providing more
backprop training gave the clearest gains in these single-seed comparisons.
Growth probabilities alone produced sparse networks with low validation AUC;
the small-step species controller increased species without improving the matched
`p13`/`p17` result. LR warm starts and the growth/species combinations did not
beat `p18`. The disjoint-distance pilot had 73 final species and validation AUC
0.8652. Higher species count was not sufficient here.

This does **not** isolate an evolutionary benefit. Main seed 1's saved genome key
33 was present in the initial population and retains the dense 472-edge topology;
its gains largely reflect continued optimisation of an initial network. Dense
initialisation and extra optimisation are confounded. Optional plain-MLP code is
present in `baselines.py`, but no completed MLP results are in the preserved
baseline JSON; none are invented here. Queue definitions and a config without a
completed result (for example `p2`) are prospective artifacts, not extra runs.

## Main runs and baselines

| Run | Training AUC | Test AUC | Hidden | Connections | Species mean / last |
|---|---:|---:|---:|---:|---:|
| Seed 0 | 0.9123 | 0.9071 | 10 | 472 | 1.0 / 1 |
| **Seed 1, selected by training** | **0.9189** | **0.9149** | 10 | 472 | 1.0 / 1 |
| Seed 2 | 0.9139 | 0.9109 | 10 | 437 | 1.0 / 1 |
| Ordinal-cell LR | 0.8570 | 0.8611 | — | — | — |
| Ordinal-cell LR + line/threat/parity/centre features | 0.8875 | 0.8878 | — | — | — |
| One-hot-cell LR + those features | 0.8923 | 0.8922 | — | — | — |
| HistGradientBoosting | 0.9471 | 0.9437 | — | — | — |

GBM metrics are inherited from `complex_real/baselines.json`, not refitted. The
three seeds share one dataset split; the sample SD describes seed variation, not
a confidence interval over datasets or train/test splits. All three saved mains
have ten hidden units and use every input. The selected network's ten units switch
on the full training split, including under a one-percent activity tolerance.

## Frozen authoring-v4 explanation

Only the selected run was imported into PostgreSQL `explaneat_dev`:

- Experiment: `154143d6-dae7-454f-9d6d-f9649ebfad19`
- Genome: `f0eb7635-7f85-437b-815b-993c5ad6cc14`
- Existing dataset: `c3ef4429-660b-40fc-9613-4c13629bd699`
- Existing split: `69849f7e-1ff7-4c06-9784-1a7bc79e358f`

A pre-write inventory is preserved locally under `artifacts/`. The import added
a new experiment and its final population of 150 genomes, reusing the existing
dataset/split. No schema operations or migrations were run; `explaneat_v4` was
not accessed. No existing experiment or explanation was edited.

The DB genome's NeuralNeat predictions are bit-equal to the stored in-memory
genome. StructureNetwork differs by at most **1.78e-15** on test predictions and
reproduces the stored metrics (BCE difference 5.55e-17).

Authoring used the frozen worktree's `xn.py` tools and authoring-v4 skill, with
full training data throughout explanation queries. Start: **2026-11-11
20:46:23 UTC**; initial author record: **20:57:04 UTC**; initial own recheck:
**20:57:15 UTC**. A later independent review narrowed the root's terminology;
the final record and recheck retain the updated timestamps. All operations
preserve the function: 42 input splits, one
identity collector for the direct edges, eleven leaves and one root. Hidden
nodes were not split. Each leaf and the root has a stored constructed self-probe;
the largest ablation effects are included. The initial probes left two gates
inactive, so additional active-case probes were attached instead.

- Structural and compositional coverage: **1.0 / 1.0**; hierarchy complete.
- Maximum function-preservation error: **1.665e-15**.
- Maximum numeric leaf-evaluation error: **5.329e-15**.
- **Rounded displayed leaf formulas are approximate:** their largest output
  error is **0.0239586**, measured on training rows. This is distinct from the
  unchanged exact network function; the displayed coefficients must not be
  presented as exact identities.
- Own recheck: **0 mismatches, 0 needs-review**, with 25 free-text sentences that
  the checker does not verify. Hypotheses and summary use 86 numeric references
  to 35 stored queries; formulas are attached verbatim from `get_formula`.
- Semantic assessment: **0 meaningful, 0 partial, 11 described leaves**. Specific
  per-square contribution directions were probed, but no tactical concepts were
  recovered. The constructed edits can violate gravity or piece counts and are
  explicitly not legal move sequences.

The stopping point was **semantic identification**, after decomposition,
numerical checks and positional probes, rather than coverage, tool capacity or
the time limit. This is a bounded unsuccessful tactical explanation attempt,
not proof that the network cannot be explained with more analysis or another
approach. The canonical independent-review target is
[`verification/item/`](verification/item/), preserved as reviewed, with rendered
narratives, operations, queries and the author record. The original independent
verdict is in [`verification/independent/`](verification/independent/): it checked
49 distinct factual units across twelve parts, found no numeric refutations,
upheld the numerical/function checks, and weakened eleven already-disclosed
semantic limitations plus the root's ambiguous "threat" wording. The root
narrative and its existing subgroup evidence were amended in place with the
open-line/gravity/sampling distinction; the old evidence is retained in its
amendment history. The corrected snapshot is
[`verification/revised_item/`](verification/revised_item/). The semantic outcome
remains unresolved, and the original verdict is not rewritten as a clean pass.

## Does the nonlinearity match lines, threats or parity?

The stored full-training linearisation query `q_9c824faecf` changes AUC from
**0.918850 to 0.855376** when switching hidden gates are replaced by least-squares
affine fits. This demonstrates substantial learnt nonlinearity, unlike the earlier
near-linear Connect-4 runs. Individual ablation losses overlap and are not
additive shares of predictive performance.

The separate training-only diagnostic `domain_check.py` reports:

- A linear fit of the logit on cells accounts for **74.73%** of logit variance.
  A linear fit of aggregate line/threat/parity/centre features to the remaining
  residual accounts for **6.21%** of residual variance.
- Adding those features to the network logit in an in-sample LR changes training
  AUC from **0.91885 to 0.92233**. This is an exploratory in-sample diagnostic,
  not a cross-validated estimate of additional predictive value.
- The largest hidden-unit feature correlation is about **0.33 in magnitude**
  (gate 1 versus centre-x count). Weak correlations with aggregate counts do not
  exclude location-specific or distributed tactical computation.

Stored subgroup queries provide a more concrete remaining error:

| Training subgroup | Positions | Observed x wins | Mean predicted x-win probability | Stored query |
|---|---:|---:|---:|---|
| Open three-o line with a blank completion square | 765 | 26.3% | 41.8% | `q_b16d796df5` |
| Open three-x line with a blank completion square | 660 | 83.9% | 81.6% | `q_95e16a4c6b` |
| o completion square on an even-numbered row | 665 | 22.9% | 39.0% | `q_13887ddf41` |
| x completion square on an odd-numbered row | 66 | 93.9% | 79.8% | `q_db727482d2` |

The script's feature name "threat" means distinct empty cells completing an
otherwise open line; it does not require gravity support. The
[UCI dataset description](https://archive.ics.uci.edu/dataset/26/connect-4)
explicitly excludes positions where the next move is forced. Independent review
confirmed that the o-completion squares in this training subgroup are not
immediately playable. Thus this is a calibration check for open line patterns,
not a test of detecting an immediate winning move. These subgroups overlap and
are not a partition. The chosen network still substantially overpredicts x's
chances in the open-three-o subgroup. The diagnostics establish a remaining
limitation; they do not justify calling the network tactic-blind or claiming
that no hidden unit encodes any tactical feature.

## Artifacts, verification and reproduction

Small raw JSON/CSV logs, derived configs, baseline metrics, the selected import
and parity checks, and explanation/provenance snapshots are preserved with this
report. Full-precision selected weights are also preserved in
`artifacts/selected_genome.json`; serialization/deserialization was checked for
exact equality. Large run binaries remain under `~/.explaneat/c4_capacity_runs/<name>/`.
`artifact_manifest.json` records SHA-256 hashes of measured sources and locally
available binary artifacts. `summary.json`, `runs.csv` and `tables.md` are
deterministically rebuilt from the preserved measurements:

```sh
uv run python experiments/c4_capacity/consolidate.py
uv run python experiments/c4_capacity/summarize.py --test
```

To repeat training intentionally, use a separate checkout/output location to
preserve these frozen logs. `run_pilots.sh` records candidate settings;
`run_main.sh` records the three main runs. Run with `PYTHONHASHSEED=0`, the
existing environment and one numerical thread. The completed mains took about
120–133 minutes each on the recorded machine. Do not rerun import on this name:
the importer refuses an existing experiment.

The sandbox initially blocked OpenMP shared-memory creation. Numerical verification
resumed with the approved execution setting that permits it; no model computation
or frozen library code was changed to suppress the failure. `KMP_USE_SHM` was not
treated as a verified runtime fix. No additional training was launched.

Earlier study commits are `bbb4986`, `bea1f30`, `8b673ba` and `7e7e391`. The
completion commit is the commit introducing this report and the consolidated
artifacts on `exp/c4-capacity`; no push or merge was performed during completion.
